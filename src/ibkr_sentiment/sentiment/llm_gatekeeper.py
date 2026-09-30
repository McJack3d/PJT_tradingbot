"""Stage 2 — generative LLM gatekeeper.

Takes high-conviction items the discriminative screen surfaced and
forces an instruction-tuned model to reason about them. Output is a
structured `LLMVerdict` (verdict, conviction, time horizon, structural
flag, asset score per ticker, rationale).

The prompt is a deliberate chain-of-thought template:

  1. Identify the *primary* asset(s).
  2. Rate the source credibility 0..1.
  3. Estimate the temporal impact (intraday / short / medium / long).
  4. Decide if the change is structural or a transient overreaction.
  5. Give a final verdict + a numeric asset score per ticker.

Three implementations:

  * `StubLLMGatekeeper` — deterministic, no network, used by tests
    and by paper-mode dry runs.
  * `AnthropicLLMGatekeeper` — Claude (anthropic SDK), lazy-imported.
  * `OpenAILLMGatekeeper` — GPT-4-class (openai SDK), lazy-imported.

`build_gatekeeper()` picks the right one from `LLMConfig.provider`.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from src.ibkr_sentiment.sentiment.models import (
    FinBertScore,
    LLMVerdict,
    NewsItem,
    TimeHorizon,
    Verdict,
)
from src.logging_setup import log

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
# Server-side refusal fallback: on a policy decline the API re-runs the
# request on a fallback model it picks by refusal category.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

PROMPT_TEMPLATE = """You are a senior buy-side analyst acting as a qualitative
gatekeeper on a news-driven trading signal. Reason step by step, then output
ONLY a JSON object with the schema below.

News item:
  Source: {source}
  Title: {title}
  Body: {body}

Candidate tickers (from upstream filters): {symbols}
FinBERT polarity: {polarity} (score={score:+.3f}, confidence={confidence:.2f})

Reasoning steps (do not include in the output):
  1. Which tickers (if any) are this item materially about?
  2. How credible is the source? (0..1)
  3. What is the likely time horizon of the impact?
     (intraday | short_term | medium_term | long_term | unknown)
  4. Is the change structural or a transient market overreaction?
  5. Given (1)-(4), what is the final verdict for each ticker?

Return a single JSON object (no markdown, no commentary):
{{
  "verdict": "bullish" | "bearish" | "neutral" | "noise",
  "conviction": 0.0..1.0,
  "temporal_impact": "intraday" | "short_term" | "medium_term" | "long_term" | "unknown",
  "structural": true | false,
  "source_credibility": 0.0..1.0,
  "rationale": "one-sentence summary",
  "asset_score": {{"TICKER": -1.0..1.0, ...}}
}}
"""


class LLMGatekeeper(Protocol):
    async def analyze(
        self, item: NewsItem, finbert: FinBertScore
    ) -> LLMVerdict: ...

    async def analyze_many(
        self, pairs: Iterable[tuple[NewsItem, FinBertScore]]
    ) -> list[LLMVerdict]: ...


# ---- shared parsing -----------------------------------------------------


_VERDICTS: set[Verdict] = {"bullish", "bearish", "neutral", "noise"}
_HORIZONS: set[TimeHorizon] = {
    "intraday",
    "short_term",
    "medium_term",
    "long_term",
    "unknown",
}


def parse_verdict_json(
    item: NewsItem, raw: str, *, default_credibility: float = 0.5
) -> LLMVerdict:
    """Parse an LLM response into a typed `LLMVerdict`.

    Tolerant of common nonsense — extra prose around the JSON, missing
    fields, out-of-range scores. Anything unrecoverable becomes a
    `noise` verdict with zero conviction so it gets filtered out
    downstream rather than crashing the pipeline.
    """
    payload = _extract_json(raw)
    if payload is None:
        return _noise(item.id)

    verdict_raw = str(payload.get("verdict", "noise")).strip().lower()
    verdict: Verdict = verdict_raw if verdict_raw in _VERDICTS else "noise"

    horizon_raw = str(payload.get("temporal_impact", "unknown")).strip().lower()
    horizon: TimeHorizon = horizon_raw if horizon_raw in _HORIZONS else "unknown"

    conviction = _clamp01(payload.get("conviction", 0.0))
    credibility = _clamp01(
        payload.get("source_credibility", default_credibility)
    )
    structural = bool(payload.get("structural", False))
    rationale = str(payload.get("rationale", ""))[:1024]

    asset_score_raw = payload.get("asset_score", {}) or {}
    asset_score: dict[str, float] = {}
    if isinstance(asset_score_raw, dict):
        for k, v in asset_score_raw.items():
            try:
                asset_score[str(k).upper()] = max(-1.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                continue

    return LLMVerdict(
        item_id=item.id,
        verdict=verdict,
        conviction=conviction,
        temporal_impact=horizon,
        structural=structural,
        source_credibility=credibility,
        rationale=rationale,
        asset_score=asset_score,
    )


def _noise(item_id: str) -> LLMVerdict:
    return LLMVerdict(
        item_id=item_id,
        verdict="noise",
        conviction=0.0,
        temporal_impact="unknown",
        structural=False,
        source_credibility=0.0,
        rationale="LLM output unparseable",
    )


def _clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Grab the first {...} block from `raw` and json.loads it."""
    if not raw:
        return None
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---- stub gatekeeper ----------------------------------------------------


@dataclass
class StubLLMGatekeeper:
    """Deterministic gatekeeper used in tests and stub mode.

    Mirrors the FinBERT polarity and assumes a moderate conviction,
    medium-term horizon, and 50% credibility. It's intentionally
    boring — the test suite reasons about the pipeline plumbing, not
    about the cleverness of this stub.
    """

    async def analyze(
        self, item: NewsItem, finbert: FinBertScore
    ) -> LLMVerdict:
        polarity = finbert.polarity
        verdict: Verdict = (
            "bullish"
            if polarity == "positive"
            else "bearish"
            if polarity == "negative"
            else "neutral"
        )
        asset_score: dict[str, float] = {
            sym: finbert.score for sym in item.symbols
        }
        return LLMVerdict(
            item_id=item.id,
            verdict=verdict,
            conviction=min(1.0, abs(finbert.score) * finbert.confidence + 0.4),
            temporal_impact="short_term",
            structural=abs(finbert.score) > 0.7,
            source_credibility=0.5,
            rationale=f"stub mirror of finbert polarity={polarity}",
            asset_score=asset_score,
        )

    async def analyze_many(
        self, pairs: Iterable[tuple[NewsItem, FinBertScore]]
    ) -> list[LLMVerdict]:
        return [await self.analyze(item, fb) for item, fb in pairs]


# ---- Anthropic backend --------------------------------------------------


class AnthropicLLMGatekeeper:
    """Claude-backed gatekeeper. Lazy SDK import; safe to construct
    even without the `anthropic` package — until you call
    `analyze()` it never imports the SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = ANTHROPIC_DEFAULT_MODEL,
        max_tokens: int = 16000,
        effort: str = "low",
        max_concurrent: int = 4,
        request_timeout_s: float = 30.0,
    ):
        # No temperature: current Claude models reject sampling params.
        # Thinking is always on and its tokens count toward max_tokens,
        # so max_tokens is a generous ceiling, not the verdict size;
        # `effort` is the cost/depth control (low suits classification).
        if not api_key:
            raise ValueError("AnthropicLLMGatekeeper requires an API key")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._sem = asyncio.Semaphore(max_concurrent)
        self.request_timeout_s = request_timeout_s
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "AnthropicLLMGatekeeper requires the 'llm' extra. "
                "Install with: pip install -e '.[llm]'"
            ) from e
        self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def analyze(
        self, item: NewsItem, finbert: FinBertScore
    ) -> LLMVerdict:
        prompt = PROMPT_TEMPLATE.format(
            source=item.source,
            title=item.title,
            body=item.body[:4_000],
            symbols=", ".join(item.symbols) or "(none)",
            polarity=finbert.polarity,
            score=finbert.score,
            confidence=finbert.confidence,
        )
        client = self._client_lazy()
        async with self._sem:
            try:
                resp = await asyncio.wait_for(
                    client.beta.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        output_config={"effort": self.effort},
                        betas=[_FALLBACK_BETA],
                        fallbacks="default",
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=self.request_timeout_s,
                )
            except Exception as e:
                log.warning(
                    "llm_gatekeeper.anthropic_error",
                    item_id=item.id,
                    model=self.model,
                    error=f"{type(e).__name__}: {e}",
                )
                return _noise(item.id)
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason in ("refusal", "max_tokens"):
            log.warning(
                "llm_gatekeeper.anthropic_incomplete",
                item_id=item.id,
                stop_reason=stop_reason,
                category=getattr(getattr(resp, "stop_details", None), "category", None),
            )
            return _noise(item.id)
        text = ""
        for block in getattr(resp, "content", []) or []:
            chunk = getattr(block, "text", None)
            if chunk:
                text += chunk
        return parse_verdict_json(item, text)

    async def analyze_many(
        self, pairs: Iterable[tuple[NewsItem, FinBertScore]]
    ) -> list[LLMVerdict]:
        coros = [self.analyze(i, fb) for i, fb in pairs]
        return await asyncio.gather(*coros)


# ---- OpenAI backend -----------------------------------------------------


class OpenAILLMGatekeeper:
    """GPT-class gatekeeper. Same lazy-import contract as the
    Anthropic backend."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 800,
        temperature: float = 0.0,
        max_concurrent: int = 4,
        request_timeout_s: float = 30.0,
    ):
        if not api_key:
            raise ValueError("OpenAILLMGatekeeper requires an API key")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrent)
        self.request_timeout_s = request_timeout_s
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAILLMGatekeeper requires the 'llm' extra."
            ) from e
        self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def analyze(
        self, item: NewsItem, finbert: FinBertScore
    ) -> LLMVerdict:
        prompt = PROMPT_TEMPLATE.format(
            source=item.source,
            title=item.title,
            body=item.body[:4_000],
            symbols=", ".join(item.symbols) or "(none)",
            polarity=finbert.polarity,
            score=finbert.score,
            confidence=finbert.confidence,
        )
        client = self._client_lazy()
        async with self._sem:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                    ),
                    timeout=self.request_timeout_s,
                )
            except Exception:
                return _noise(item.id)
        text = resp.choices[0].message.content or "" if resp.choices else ""
        return parse_verdict_json(item, text)

    async def analyze_many(
        self, pairs: Iterable[tuple[NewsItem, FinBertScore]]
    ) -> list[LLMVerdict]:
        coros = [self.analyze(i, fb) for i, fb in pairs]
        return await asyncio.gather(*coros)


def build_gatekeeper(provider: str, *, anthropic_key: str = "", openai_key: str = "",
                     model: str = "", max_concurrent: int = 4,
                     max_tokens: int = 800, temperature: float = 0.0,
                     request_timeout_s: float = 30.0,
                     effort: str = "low") -> LLMGatekeeper:
    """Factory: pick a gatekeeper backend based on `provider` string."""
    provider = (provider or "stub").lower()
    if provider == "stub":
        return StubLLMGatekeeper()
    if provider == "anthropic":
        return AnthropicLLMGatekeeper(
            api_key=anthropic_key,
            model=model or ANTHROPIC_DEFAULT_MODEL,
            max_tokens=max_tokens,
            effort=effort,
            max_concurrent=max_concurrent,
            request_timeout_s=request_timeout_s,
        )
    if provider == "openai":
        return OpenAILLMGatekeeper(
            api_key=openai_key,
            model=model or "gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
            request_timeout_s=request_timeout_s,
        )
    if provider == "fingpt":
        # FinGPT exposes an OpenAI-compatible HTTP endpoint via the
        # AI4Finance project; reuse the OpenAI client pointed at it.
        return OpenAILLMGatekeeper(
            api_key=openai_key or "fingpt",
            model=model or "fingpt-7b",
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
            request_timeout_s=request_timeout_s,
        )
    raise ValueError(f"unknown LLM provider: {provider}")
