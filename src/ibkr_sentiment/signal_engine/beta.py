"""Market beta estimates for beta-neutral hedging.

"Dollar-neutral" is not market-neutral: $10k long NVDA (beta ~1.8)
against $10k short XOM (beta ~0.6) is still ~$12k long the market. The
basket builder and risk overlay use these betas to balance, and cap,
beta-weighted exposure instead.

Beta = cov(r_asset, r_bench) / var(r_bench) over daily close-to-close
returns on the days both series traded, then shrunk toward 1 with the
Blume adjustment (0.67 * raw + 0.33) — raw betas are noisy and mean-
revert toward the market — and clamped to a sane range.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from src.ibkr_sentiment.bar_cache import bar_day
from src.ibkr_sentiment.broker.base import Bar

MIN_OBSERVATIONS = 20
BETA_FLOOR = 0.2
BETA_CAP = 3.0


def estimate_beta(
    asset: Sequence[Bar],
    bench: Sequence[Bar],
    *,
    lookback: int = 60,
    blume: bool = True,
) -> Decimal | None:
    """Beta of `asset` vs `bench` over the last `lookback` common daily
    returns, or None with fewer than MIN_OBSERVATIONS of them."""
    a_close = {bar_day(b.ts): float(b.close) for b in asset if b.close > 0}
    b_close = {bar_day(b.ts): float(b.close) for b in bench if b.close > 0}
    days = sorted(a_close.keys() & b_close.keys())[-(lookback + 1) :]
    ra: list[float] = []
    rb: list[float] = []
    for prev, cur in zip(days, days[1:], strict=False):
        ra.append(a_close[cur] / a_close[prev] - 1)
        rb.append(b_close[cur] / b_close[prev] - 1)
    n = len(rb)
    if n < MIN_OBSERVATIONS:
        return None
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    var_b = sum((x - mean_b) ** 2 for x in rb) / (n - 1)
    if var_b <= 0:
        return None
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True)) / (n - 1)
    beta = cov / var_b
    if blume:
        beta = 0.67 * beta + 0.33
    beta = min(BETA_CAP, max(BETA_FLOOR, beta))
    return Decimal(str(round(beta, 4)))
