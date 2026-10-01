"""News ingestion for the crypto sentiment bot.

Two sources, both producing the shared `NewsItem` the funnel consumes:

  * **CryptoPanic** (`parse_cryptopanic`) — a news aggregator with a free
    tier that tags each post with the coins it concerns. The structured
    `currencies[].code` tags are far more reliable than scraping tickers
    out of free text, which is why this is the primary small-cap source.
  * **Generic RSS** — reused via `parse_rss` from the IBKR ingestion
    module. Symbols are detected from the text against the universe,
    which is noisy for short tickers; treat RSS as supplementary.

Both go through a `fetcher(url) -> str` callable so the live HTTP path
and a deterministic test fake are interchangeable.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from src.ibkr_sentiment.sentiment.ingestion import detect_symbols, parse_rss
from src.ibkr_sentiment.sentiment.models import NewsItem
from src.logging_setup import log

Fetcher = Callable[[str], Awaitable[str]]


def parse_cryptopanic(json_text: str, universe: set[str]) -> list[NewsItem]:
    """Parse a CryptoPanic v1 `/posts/` JSON response into NewsItems.

    Only posts tagged with at least one in-universe currency are kept.
    A broken/blank payload yields an empty list rather than raising.
    """
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError):
        return []
    universe = {s.upper() for s in universe}
    out: list[NewsItem] = []
    for r in data.get("results", []) or []:
        codes = tuple(
            str(c.get("code", "")).upper()
            for c in (r.get("currencies") or [])
            if c.get("code")
        )
        syms = tuple(c for c in codes if c in universe)
        if not syms:
            continue
        out.append(
            NewsItem(
                source="cryptopanic",
                url=str(r.get("url", "")),
                title=str(r.get("title", "")),
                body=str(r.get("title", "")),  # CryptoPanic free tier has no body
                published_at=_parse_iso(r.get("published_at")),
                symbols=syms,
            )
        )
    return out


def _parse_iso(s) -> datetime:  # noqa: ANN001
    if not s:
        return datetime.now(UTC)
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)


def cryptopanic_url(auth_token: str, currencies: list[str]) -> str:
    codes = ",".join(c.upper() for c in currencies)
    return (
        "https://cryptopanic.com/api/v1/posts/"
        f"?auth_token={auth_token}&kind=news&currencies={codes}"
    )


class CryptoNewsGatherer:
    """Pulls and tags fresh NewsItems from the configured sources.

    `fetcher` is injected so tests can feed canned payloads with no
    network. `cryptopanic_token` enables the structured source when set.
    """

    def __init__(
        self,
        rss_feeds: list[str],
        *,
        fetcher: Fetcher,
        cryptopanic_token: str = "",
        max_items_per_feed: int = 50,
    ) -> None:
        self.rss_feeds = list(rss_feeds)
        self.fetcher = fetcher
        self.cryptopanic_token = cryptopanic_token
        self.max_items_per_feed = max_items_per_feed

    async def gather(self, universe_bases: set[str]) -> list[NewsItem]:
        items: list[NewsItem] = []

        if self.cryptopanic_token:
            url = cryptopanic_url(self.cryptopanic_token, sorted(universe_bases))
            try:
                body = await self.fetcher(url)
                items.extend(parse_cryptopanic(body, universe_bases))
            except Exception as e:  # noqa: BLE001 - one bad feed must not stop the rest
                log.warning("crypto_sentiment.cryptopanic.fetch_failed", error=str(e))

        for feed in self.rss_feeds:
            try:
                body = await self.fetcher(feed)
            except Exception as e:  # noqa: BLE001
                log.warning("crypto_sentiment.rss.fetch_failed", feed=feed, error=str(e))
                continue
            for item in parse_rss(body)[: self.max_items_per_feed]:
                syms = detect_symbols(f"{item.title}\n{item.body}", universe_bases)
                if not syms:
                    continue
                item.symbols = syms
                items.append(item)
        return items
