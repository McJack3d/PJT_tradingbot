"""Stage 0 — raw text ingestion.

Pulls news from RSS feeds (and, when enabled, SEC filing indexes) and
emits `NewsItem`s. The HTTP layer is pluggable so tests can inject a
fake fetcher and avoid network calls entirely.

Symbol detection is intentionally simple — a $TICKER prefix or a
matching uppercase token. The downstream LLM stage is responsible for
the harder semantic check ("does this article actually move AAPL?").
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from src.ibkr_sentiment.sentiment.models import NewsItem
from src.logging_setup import log

Fetcher = Callable[[str], Awaitable[str]]


_DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_BARE_TICKER = re.compile(r"\b([A-Z]{2,5})\b")
# RFC822 ish; falls back to ingested_at on parse failure.
_RFC822_FMTS = (
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
)


def detect_symbols(text: str, universe: Iterable[str]) -> tuple[str, ...]:
    """Return universe tickers found in `text`.

    A symbol matches if it appears as `$TICK` anywhere, or as a bare
    uppercase token AND is in the configured universe. The universe
    filter is what keeps "AT" or "ALL" from being mistaken for tickers
    in normal prose.
    """
    universe_set = {s.upper() for s in universe}
    dollar_hits = {m.group(1) for m in _DOLLAR_TICKER.finditer(text)}
    bare_hits = {m.group(1) for m in _BARE_TICKER.finditer(text)} & universe_set
    return tuple(sorted(dollar_hits | bare_hits))


def parse_rss(xml_text: str) -> list[NewsItem]:
    """Parse a generic RSS 2.0 feed. Atom is supported via the same
    code path because we only read `title`, link / id, and a date field.

    Returns an empty list on parse failure rather than raising — a
    broken feed should never knock the bot over.
    """
    try:
        # Feeds are operator-configured RSS/Atom URLs, not arbitrary
        # user input; stdlib parsing is acceptable here.
        root = ET.fromstring(xml_text)  # noqa: S314
    except ET.ParseError:
        return []

    items: list[NewsItem] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    # RSS 2.0
    for entry in root.iter("item"):
        items.append(_make_item_from_rss(entry))
    # Atom
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        items.append(_make_item_from_atom(entry, ns))
    return items


def _make_item_from_rss(entry: ET.Element) -> NewsItem:
    title = (entry.findtext("title") or "").strip()
    link = (entry.findtext("link") or "").strip()
    body = (entry.findtext("description") or "").strip()
    pub = entry.findtext("pubDate") or ""
    return NewsItem(
        source="rss",
        url=link,
        title=title,
        body=body,
        published_at=_parse_date(pub),
    )


def _make_item_from_atom(entry: ET.Element, ns: dict[str, str]) -> NewsItem:
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    link_el = entry.find("atom:link", ns)
    link = link_el.attrib.get("href", "") if link_el is not None else ""
    body = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    pub = entry.findtext("atom:updated", default="", namespaces=ns) or ""
    return NewsItem(
        source="atom",
        url=link,
        title=title,
        body=body,
        published_at=_parse_date(pub),
    )


def _parse_date(text: str) -> datetime:
    text = text.strip()
    if not text:
        return datetime.now(UTC)
    for fmt in _RFC822_FMTS:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    return datetime.now(UTC)


@dataclass
class _DedupRecord:
    key: tuple[str, str]
    seen_at: datetime


class Deduper:
    """Sliding-window dedup keyed on (source, url).

    URL alone is too fragile — the same Bloomberg headline can show up
    on two distinct URL paths (mobile vs. desktop) — but combined with
    a short window it's enough to stop the obvious "same item polled
    twice" case. Anything more sophisticated belongs downstream.
    """

    def __init__(self, window: timedelta):
        self.window = window
        self._records: deque[_DedupRecord] = deque()
        self._index: set[tuple[str, str]] = set()

    def _evict(self, now: datetime) -> None:
        cutoff = now - self.window
        while self._records and self._records[0].seen_at < cutoff:
            old = self._records.popleft()
            self._index.discard(old.key)

    def is_new(self, item: NewsItem) -> bool:
        now = datetime.now(UTC)
        self._evict(now)
        key = (item.source, item.url)
        if key in self._index:
            return False
        self._index.add(key)
        self._records.append(_DedupRecord(key=key, seen_at=now))
        return True


class _HttpFetcher(Protocol):
    async def __call__(self, url: str) -> str: ...


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_PLACEHOLDER_DOMAINS = ("example.com", "example.org", "example.net", ".invalid", "localhost")


def is_sec_url(url: str) -> bool:
    return "sec.gov" in url


def valid_sec_user_agent(ua: str | None) -> bool:
    """SEC EDGAR requires a User-Agent naming who you are with a real
    contact email, and throttles or blocks requests without one."""
    if not ua or "@" not in ua:
        return False
    return not any(d in ua.lower() for d in _PLACEHOLDER_DOMAINS)


def expand_feed_urls(feeds: Iterable[str], universe: list[str], chunk: int = 20) -> list[str]:
    """Expand a `{symbols}` placeholder into one URL per chunk of the
    universe (comma-joined), e.g. Yahoo Finance's per-ticker headline
    feed. URLs without the placeholder pass through unchanged."""
    out: list[str] = []
    for url in feeds:
        if "{symbols}" not in url:
            out.append(url)
            continue
        for i in range(0, len(universe), chunk):
            out.append(url.replace("{symbols}", ",".join(universe[i : i + chunk])))
    return out


def make_httpx_fetcher(sec_user_agent: str | None = None) -> _HttpFetcher:
    async def fetch(url: str) -> str:
        import httpx

        ua = _BROWSER_UA
        if is_sec_url(url):
            if not valid_sec_user_agent(sec_user_agent):
                raise ValueError("SEC EDGAR needs a real contact User-Agent")
            ua = sec_user_agent  # type: ignore[assignment]
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": ua})
            resp.raise_for_status()
            return resp.text

    return fetch


_httpx_fetcher = make_httpx_fetcher()


@dataclass
class FeedHealth:
    consecutive_errors: int = 0
    consecutive_empty: int = 0
    last_ok: datetime | None = None
    last_error: str = ""


class IngestionService:
    """Polls a list of feeds and emits new `NewsItem`s.

    The service is intentionally tiny — fetch, parse, tag with the
    universe-restricted symbol set, dedup, hand off. Everything else
    happens in the pipeline.
    """

    def __init__(
        self,
        feeds: list[str],
        universe: list[str],
        *,
        poll_interval_s: int = 60,
        max_items_per_poll: int = 50,
        dedup_window_minutes: int = 240,
        fetcher: _HttpFetcher | None = None,
        sec_user_agent: str | None = None,
        sec_enabled: bool = True,
        unhealthy_after: int = 3,
    ):
        self.universe = list(universe)
        self.disabled_feeds: dict[str, str] = {}
        self.feeds = []
        for url in expand_feed_urls(feeds, self.universe):
            if is_sec_url(url) and not sec_enabled:
                self.disabled_feeds[url] = "sec_filings_enabled is false"
            elif is_sec_url(url) and not valid_sec_user_agent(sec_user_agent):
                self.disabled_feeds[url] = (
                    "SEC requires a real contact User-Agent: set ingestion.sec_user_agent "
                    "or SEC_USER_AGENT to e.g. 'trad-bot you@yourdomain.com'"
                )
            else:
                self.feeds.append(url)
        for url, why in self.disabled_feeds.items():
            log.warning("ingestion.feed_disabled", url=url, reason=why)
        self.poll_interval_s = poll_interval_s
        self.max_items_per_poll = max_items_per_poll
        self.fetcher = fetcher or make_httpx_fetcher(sec_user_agent)
        self.unhealthy_after = unhealthy_after
        self.health: dict[str, FeedHealth] = {url: FeedHealth() for url in self.feeds}
        self.deduper = Deduper(timedelta(minutes=dedup_window_minutes))
        self._task: asyncio.Task | None = None
        self._on_item: Callable[[NewsItem], Awaitable[None]] | None = None
        self._stopping = asyncio.Event()

    async def fetch_once(self) -> list[NewsItem]:
        out: list[NewsItem] = []
        for url in self.feeds:
            h = self.health.setdefault(url, FeedHealth())
            try:
                body = await self.fetcher(url)
            except Exception as e:
                h.consecutive_errors += 1
                h.last_error = f"{type(e).__name__}: {e}"
                if h.consecutive_errors == self.unhealthy_after:
                    log.warning(
                        "ingestion.feed_failing",
                        url=url,
                        consecutive_errors=h.consecutive_errors,
                        error=h.last_error,
                    )
                continue
            parsed = parse_rss(body)
            h.consecutive_errors = 0
            h.consecutive_empty = 0 if parsed else h.consecutive_empty + 1
            if parsed:
                h.last_ok = datetime.now(UTC)
            elif h.consecutive_empty == self.unhealthy_after:
                log.warning("ingestion.feed_empty", url=url, polls=h.consecutive_empty)
            for item in parsed[: self.max_items_per_poll]:
                item.symbols = detect_symbols(
                    f"{item.title}\n{item.body}", self.universe
                )
                if not item.symbols:
                    continue
                if not self.deduper.is_new(item):
                    continue
                out.append(item)
        return out

    def unhealthy_feeds(self) -> dict[str, FeedHealth]:
        return {
            url: h
            for url, h in self.health.items()
            if h.consecutive_errors >= self.unhealthy_after
            or h.consecutive_empty >= self.unhealthy_after
        }

    async def start(
        self, on_item: Callable[[NewsItem], Awaitable[None]]
    ) -> None:
        self._on_item = on_item
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        assert self._on_item is not None
        while not self._stopping.is_set():
            try:
                items = await self.fetch_once()
                for item in items:
                    await self._on_item(item)
            except Exception as e:
                log.warning("ingestion.poll_error", error=f"{type(e).__name__}: {e}")
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self.poll_interval_s
                )
            except TimeoutError:
                pass
