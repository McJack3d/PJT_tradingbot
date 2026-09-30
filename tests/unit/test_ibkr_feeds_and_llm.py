"""Feeds and LLM model (audit defect #12).

Before: the configured Reuters feeds had been dead since 2020 and
failed silently; the SEC feed used a placeholder User-Agent (the
configured one was ignored); and the Claude gatekeeper was pinned to a
stale model and sent `temperature`, which current Claude models reject
— every call failed and was silently turned into a "noise" verdict.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.ibkr_sentiment.config import IbkrSentimentConfig, LLMConfig, UniverseEntry
from src.ibkr_sentiment.sentiment import ingestion as ing
from src.ibkr_sentiment.sentiment.ingestion import (
    IngestionService,
    expand_feed_urls,
    valid_sec_user_agent,
)
from src.ibkr_sentiment.sentiment.llm_gatekeeper import (
    ANTHROPIC_DEFAULT_MODEL,
    AnthropicLLMGatekeeper,
    build_gatekeeper,
)
from src.ibkr_sentiment.sentiment.models import FinBertScore, NewsItem

SEC = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom"
YAHOO = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbols}&region=US"
RSS = """<rss><channel><item><title>AAPL beats estimates</title>
<link>https://x/1</link><description>growth</description></item></channel></rss>"""


# ---- feeds ----------------------------------------------------------


def test_symbols_placeholder_expands_in_chunks():
    urls = expand_feed_urls([YAHOO, "https://static/feed"], [f"S{i}" for i in range(25)], chunk=20)
    assert len(urls) == 3
    assert "s=S0,S1," in urls[0] and urls[0].count(",") == 19
    assert "s=S20,S21,S22,S23,S24&" in urls[1]
    assert urls[2] == "https://static/feed"


@pytest.mark.parametrize(
    ("ua", "ok"),
    [
        ("trad-bot research contact@example.com", False),  # old placeholder
        ("", False),
        (None, False),
        ("trad-bot research", False),
        ("Jane Doe jane@test.invalid", False),
        ("trad-bot jane@realdomain.fr", True),
    ],
)
def test_sec_user_agent_validation(ua, ok):
    assert valid_sec_user_agent(ua) is ok


def test_sec_feed_is_disabled_until_enabled_with_real_user_agent():
    kw = dict(universe=["AAPL"], fetcher=lambda u: None)
    off = IngestionService([SEC], sec_enabled=False, sec_user_agent="a b@realdomain.fr", **kw)
    placeholder = IngestionService(
        [SEC], sec_enabled=True, sec_user_agent="x contact@example.com", **kw
    )
    on = IngestionService([SEC], sec_enabled=True, sec_user_agent="a b@realdomain.fr", **kw)
    assert off.feeds == [] and "sec_filings_enabled" in off.disabled_feeds[SEC]
    assert placeholder.feeds == [] and "User-Agent" in placeholder.disabled_feeds[SEC]
    assert on.feeds == [SEC]


@pytest.mark.asyncio
async def test_fetcher_sends_configured_sec_user_agent(monkeypatch):
    seen: dict[str, str] = {}

    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers):
            seen[url] = headers["User-Agent"]
            return SimpleNamespace(text=RSS, raise_for_status=lambda: None)

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    fetch = ing.make_httpx_fetcher("trad-bot jane@realdomain.fr")
    await fetch(SEC)
    await fetch("https://feeds.example/rss")
    assert seen[SEC] == "trad-bot jane@realdomain.fr"
    assert seen["https://feeds.example/rss"] != seen[SEC]
    with pytest.raises(ValueError):
        await ing.make_httpx_fetcher(None)(SEC)


@pytest.mark.asyncio
async def test_dead_and_empty_feeds_are_flagged_then_recover():
    state = {"dead": 0}

    async def fetcher(url: str) -> str:
        if url == "dead":
            state["dead"] += 1
            raise ConnectionError("tunnel failed")
        if url == "empty":
            return "<rss><channel></channel></rss>"
        return RSS

    svc = IngestionService(["dead", "empty", "ok"], universe=["AAPL"], fetcher=fetcher)
    for _ in range(3):
        await svc.fetch_once()
    bad = svc.unhealthy_feeds()
    assert set(bad) == {"dead", "empty"}
    assert "tunnel failed" in bad["dead"].last_error
    assert svc.health["ok"].last_ok is not None

    async def healed(url: str) -> str:
        return RSS

    svc.fetcher = healed
    await svc.fetch_once()
    assert svc.unhealthy_feeds() == {}


def test_shipped_config_has_no_reuters_and_no_placeholder_agent():
    cfg = IbkrSentimentConfig.from_yaml("config/ibkr_sentiment.yaml")
    assert not any("reuters.com" in u for u in cfg.ingestion.rss_feeds)
    assert "example.com" not in cfg.ingestion.sec_user_agent


# ---- LLM ------------------------------------------------------------


class _FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    async def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def _gatekeeper(resp=None, exc=None) -> tuple[AnthropicLLMGatekeeper, _FakeMessages]:
    gk = AnthropicLLMGatekeeper(api_key="k")
    msgs = _FakeMessages(resp, exc)
    gk._client = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return gk, msgs


def _item_and_score():
    item = NewsItem(title="AAPL beats", body="growth", symbols=("AAPL",))
    return item, FinBertScore(item_id=item.id, polarity="positive", score=0.9, confidence=0.9)


VERDICT = json.dumps(
    {
        "verdict": "bullish",
        "conviction": 0.8,
        "temporal_impact": "short",
        "structural": False,
        "source_credibility": 0.7,
        "rationale": "beat",
        "asset_score": {"AAPL": 0.6},
    }
)


@pytest.mark.asyncio
async def test_claude_request_uses_current_model_and_no_sampling_params():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=VERDICT)],
    )
    gk, msgs = _gatekeeper(resp)
    verdict = await gk.analyze(*_item_and_score())
    [kw] = msgs.calls
    assert kw["model"] == ANTHROPIC_DEFAULT_MODEL == "claude-opus-5-5"
    assert "temperature" not in kw and "top_p" not in kw
    assert kw["output_config"] == {"effort": "low"}
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["max_tokens"] >= 4000  # thinking counts toward it
    assert verdict.verdict == "bullish" and verdict.asset_score == {"AAPL": 0.6}


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
async def test_incomplete_response_becomes_noise(stop):
    resp = SimpleNamespace(
        stop_reason=stop,
        stop_details=SimpleNamespace(category=None),
        content=[SimpleNamespace(type="text", text=VERDICT)],
    )
    gk, _ = _gatekeeper(resp)
    assert (await gk.analyze(*_item_and_score())).verdict == "noise"


@pytest.mark.asyncio
async def test_api_error_becomes_noise():
    gk, _ = _gatekeeper(exc=RuntimeError("400 invalid_request_error"))
    assert (await gk.analyze(*_item_and_score())).verdict == "noise"


def test_factory_defaults_per_provider():
    cfg = LLMConfig()
    assert cfg.model == ""
    claude = build_gatekeeper("anthropic", anthropic_key="k", model=cfg.model, effort=cfg.effort)
    gpt = build_gatekeeper("openai", openai_key="k", model=cfg.model)
    assert claude.model == "claude-opus-5-5" and claude.effort == "low"
    assert gpt.model == "gpt-4o-mini"  # was handed the Claude model id before


def test_config_rejects_unknown_effort():
    with pytest.raises(ValueError):
        LLMConfig(effort="turbo")
    IbkrSentimentConfig(universe=[UniverseEntry(symbol="AAPL")], llm=LLMConfig(effort="high"))
