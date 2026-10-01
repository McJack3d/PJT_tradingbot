"""Tests for the small-cap crypto sentiment bot.

Everything runs deterministically with no network and no LLM: the
funnel uses the existing StubFinBertScorer + StubLLMGatekeeper, news is
canned JSON through an injected fetcher, and trading goes through
FakeExchange. These tests pin the bot's *decisions* (universe filter,
entry/exit/guards), not the cleverness of the sentiment model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.adapters.fake import FakeExchange
from src.crypto_sentiment.bot import CryptoSentimentBot
from src.crypto_sentiment.config import CryptoSentimentConfig
from src.crypto_sentiment.feeds import CryptoNewsGatherer, parse_cryptopanic
from src.crypto_sentiment.positions import OpenPosition, PositionStore
from src.crypto_sentiment.universe import MarketInfo, build_universe
from src.ibkr_sentiment.sentiment.finbert import StubFinBertScorer
from src.ibkr_sentiment.sentiment.llm_gatekeeper import StubLLMGatekeeper
from src.ibkr_sentiment.sentiment.pipeline import PipelineConfig, SentimentPipeline

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


# ---- universe filtering ---------------------------------------------


def _mkt(base, vol, last=Decimal("1"), quote="USDT", active=True, spread=Decimal("0.001")):
    half = last * spread / 2
    return MarketInfo(
        symbol=f"{base}/{quote}", base=base, quote=quote, active=active,
        quote_volume_24h=Decimal(str(vol)), last=last,
        bid=last - half, ask=last + half,
    )


def test_universe_filters_majors_illiquid_and_caps():
    cfg = CryptoSentimentConfig(
        min_quote_volume_24h=Decimal("100000"),
        max_quote_volume_24h=Decimal("10000000"),
        max_universe=2,
    )
    markets = [
        _mkt("BTC", 9_000_000),       # major → excluded
        _mkt("FOO", 5_000_000),       # in band
        _mkt("BAR", 1_000_000),       # in band
        _mkt("BAZ", 2_000_000),       # in band (but cap=2 drops the smallest)
        _mkt("TINY", 50_000),         # below floor → excluded
        _mkt("WHALE", 50_000_000),    # above ceiling → excluded
        _mkt("DEAD", 3_000_000, active=False),  # inactive → excluded
        _mkt("EURX", 4_000_000, quote="EUR"),   # wrong quote → excluded
    ]
    uni = build_universe(markets, cfg)
    bases = [m.base for m in uni]
    assert bases == ["FOO", "BAZ"]  # top-2 by volume, majors/illiquid/inactive gone


# ---- cryptopanic parsing --------------------------------------------


def test_parse_cryptopanic_keeps_only_in_universe():
    payload = """
    {"results": [
      {"title": "FOO network sees record adoption", "url": "http://x/1",
       "published_at": "2026-06-15T11:50:00Z", "currencies": [{"code": "FOO"}]},
      {"title": "Unrelated major news", "url": "http://x/2",
       "published_at": "2026-06-15T11:55:00Z", "currencies": [{"code": "BTC"}]}
    ]}
    """
    items = parse_cryptopanic(payload, {"FOO", "BAR"})
    assert len(items) == 1
    assert items[0].symbols == ("FOO",)
    assert items[0].source == "cryptopanic"


def test_parse_cryptopanic_handles_garbage():
    assert parse_cryptopanic("not json", {"FOO"}) == []
    assert parse_cryptopanic('{"results": null}', {"FOO"}) == []


# ---- bot harness ----------------------------------------------------


class _FakeProvider:
    def __init__(self, markets):
        self._markets = markets

    async def fetch_markets(self):
        return self._markets


def _fetcher_for(payload: str):
    async def _f(url: str) -> str:
        # Only the cryptopanic call matters here; RSS feeds return empty.
        return payload if "cryptopanic" in url else "<rss></rss>"
    return _f


def _bot(tmp_path, *, markets, news_payload, cfg=None, starting_usdt=Decimal("1000")):
    cfg = cfg or CryptoSentimentConfig(cryptopanic_enabled=True)
    ex = FakeExchange(starting_usdt=starting_usdt)
    # FakeExchange splits starting_usdt 50/50 spot/perp; ensure spot has room.
    ex._balances["spot:USDT"] = type(ex._balances["spot:USDT"])(
        "USDT", starting_usdt, Decimal("0"), starting_usdt
    )
    for m in markets:
        ex.set_ticker(m.symbol, "spot", m.last)
    pipeline = SentimentPipeline(
        scorer=StubFinBertScorer(),
        gatekeeper=StubLLMGatekeeper(),
        cfg=PipelineConfig(min_conviction=cfg.min_conviction, signal_window=cfg.signal_window),
    )
    news = CryptoNewsGatherer([], fetcher=_fetcher_for(news_payload),
                              cryptopanic_token="test-token")
    store = PositionStore(str(tmp_path / "pos.json"))
    bot = CryptoSentimentBot(
        exchange=ex, pipeline=pipeline, market_provider=_FakeProvider(markets),
        news=news, store=store, cfg=cfg,
    )
    return bot, ex, store


def _bullish(base: str, when=NOW) -> str:
    return f"""
    {{"results": [
      {{"title": "{base} surges on record breakthrough approval, strong growth",
        "url": "http://x/{base}", "published_at": "{when.isoformat()}",
        "currencies": [{{"code": "{base}"}}]}}
    ]}}
    """


@pytest.mark.asyncio
async def test_bullish_signal_opens_long(tmp_path):
    markets = [_mkt("FOO", 5_000_000, last=Decimal("2"))]
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=_bullish("FOO"))

    report = await bot.tick(now=NOW)

    assert "FOO" in report.opened
    assert store.is_open("FOO")
    pos = store.get("FOO")
    # ~$10 notional / $2 ask ≈ 5 units (minus tiny slippage/rounding).
    assert pos.qty > 0
    # Spent quote on the exchange.
    assert ex._balances["spot:FOO"].total == pos.qty


@pytest.mark.asyncio
async def test_noise_opens_nothing(tmp_path):
    markets = [_mkt("FOO", 5_000_000, last=Decimal("2"))]
    neutral = """{"results": [
      {"title": "FOO holds a conference next week", "url": "http://x/1",
       "published_at": "2026-06-15T11:50:00Z", "currencies": [{"code": "FOO"}]}]}"""
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=neutral)

    report = await bot.tick(now=NOW)

    assert report.opened == []
    assert not store.is_open("FOO")


@pytest.mark.asyncio
async def test_max_concurrent_positions_enforced(tmp_path):
    markets = [_mkt("FOO", 5_000_000, Decimal("2")), _mkt("BAR", 4_000_000, Decimal("2"))]
    payload = """{"results": [
      {"title": "FOO surges record breakthrough approval strong growth wins",
       "url": "u1", "published_at": "2026-06-15T11:50:00Z", "currencies": [{"code": "FOO"}]},
      {"title": "BAR surges record breakthrough approval strong growth wins",
       "url": "u2", "published_at": "2026-06-15T11:50:00Z", "currencies": [{"code": "BAR"}]}]}"""
    cfg = CryptoSentimentConfig(cryptopanic_enabled=True, max_concurrent_positions=1)
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=payload, cfg=cfg)

    report = await bot.tick(now=NOW)

    assert len(report.opened) == 1
    assert ("max_concurrent" in [r for _, r in report.skipped]) or len(report.skipped) >= 1
    assert store.open_count() == 1


@pytest.mark.asyncio
async def test_wide_spread_skips_entry(tmp_path):
    # 5% spread, guard is 1.5%.
    markets = [_mkt("FOO", 5_000_000, Decimal("2"), spread=Decimal("0.05"))]
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=_bullish("FOO"))

    report = await bot.tick(now=NOW)

    assert report.opened == []
    assert ("FOO", "spread") in report.skipped


@pytest.mark.asyncio
async def test_stop_loss_closes_position(tmp_path):
    markets = [_mkt("FOO", 5_000_000, last=Decimal("2"))]
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=_bullish("FOO"))
    await bot.tick(now=NOW)
    assert store.is_open("FOO")

    # Price craters 10% — past the 3% stop. Re-price the market + ticker.
    crashed = [_mkt("FOO", 5_000_000, last=Decimal("1.8"))]
    bot.market_provider = _FakeProvider(crashed)
    bot._universe_at = None  # force refresh
    ex.set_ticker("FOO/USDT", "spot", Decimal("1.8"))
    # No fresh bullish news so the only action is the stop.
    bot.news = CryptoNewsGatherer([], fetcher=_fetcher_for('{"results": []}'),
                                  cryptopanic_token="t")

    report = await bot.tick(now=NOW + timedelta(minutes=5))

    assert ("FOO", "stop_loss") in report.closed
    assert not store.is_open("FOO")


@pytest.mark.asyncio
async def test_cooloff_blocks_immediate_reentry(tmp_path):
    markets = [_mkt("FOO", 5_000_000, last=Decimal("2"))]
    cfg = CryptoSentimentConfig(cryptopanic_enabled=True, asset_cooloff_minutes=60)
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=_bullish("FOO"), cfg=cfg)

    # Simulate an exit 10 minutes ago.
    store.record_exit("FOO", Decimal("2"), now=NOW - timedelta(minutes=10))
    report = await bot.tick(now=NOW)

    assert report.opened == []
    assert ("FOO", "cooloff") in report.skipped


@pytest.mark.asyncio
async def test_daily_loss_stop_halts_entries(tmp_path):
    markets = [_mkt("FOO", 5_000_000, last=Decimal("2"))]
    cfg = CryptoSentimentConfig(cryptopanic_enabled=True, daily_loss_stop_usd=Decimal("5"))
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=_bullish("FOO"), cfg=cfg)

    # Bank a -$6 day (worse than the -$5 stop). Align the store's "today"
    # with the injected clock so the daily roll doesn't reset it.
    store._today = NOW.date().isoformat()
    store._realized_today = Decimal("-6")
    report = await bot.tick(now=NOW)

    assert report.halted is True
    assert report.opened == []


@pytest.mark.asyncio
async def test_evaluate_is_read_only(tmp_path):
    markets = [_mkt("FOO", 5_000_000, last=Decimal("2"))]
    bot, ex, store = _bot(tmp_path, markets=markets, news_payload=_bullish("FOO"))

    signals = await bot.evaluate(now=NOW)

    assert any(s.symbol.upper() == "FOO" and s.score > 0 for s in signals)
    assert store.open_count() == 0  # evaluate placed nothing


# ---- position store -------------------------------------------------


def test_position_store_roundtrip_and_pnl(tmp_path):
    path = str(tmp_path / "p.json")
    s = PositionStore(path)
    s.record_entry(OpenPosition("FOO", "FOO/USDT", Decimal("10"), Decimal("2"),
                                "USDT", NOW))
    # Reload from disk — state persists.
    s2 = PositionStore(path)
    assert s2.is_open("FOO")
    pnl = s2.record_exit("FOO", Decimal("2.5"), fees=Decimal("0.05"), now=NOW)
    assert pnl == Decimal("4.95")  # (2.5-2)*10 - 0.05
    assert s2.realized_today(NOW) == Decimal("4.95")
    assert not s2.is_open("FOO")
