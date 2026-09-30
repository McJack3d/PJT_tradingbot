"""Dollar-neutral basket construction.

Given a list of `SymbolDecision` (LONG, SHORT, FLAT) and an account
NLV, build a target portfolio that:

  1. Caps gross exposure at `max_gross_pct * equity`.
  2. Caps each name at `max_position_pct * equity`.
  3. Sizes long and short legs to be as close to dollar-equal as
     possible (the "dollar-neutral" half of "long/short equity") so
     that broad market drops don't sink the book.

The output is a list of `TargetPosition` objects. The execution engine
turns target deltas (target - current) into orders.

Pure function — easy to unit-test by hand.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from src.ibkr_sentiment.signal_engine.mapping import Side, SymbolDecision


@dataclass(slots=True)
class TargetPosition:
    symbol: str
    side: Side
    target_qty: Decimal  # signed: + long, - short, 0 = flat
    notional: Decimal
    reason: str


def _quantize(qty: Decimal, step: Decimal) -> Decimal:
    """Floor `qty` to the nearest multiple of `step`. Flooring (not
    rounding) is important: any other choice can push the resulting
    notional above the caller's per-name cap."""
    if step <= 0:
        return qty
    n = (qty / step).quantize(Decimal("1"), rounding=ROUND_DOWN)
    return n * step


def _scale(
    t: TargetPosition, factor: Decimal, min_qty: dict[str, Decimal], label: str
) -> TargetPosition | None:
    """Scale a target DOWN by `factor`, flooring |qty| to the symbol's lot
    step (rounding toward zero, so a cap is never exceeded). None if it
    rounds to nothing."""
    qty = _quantize(abs(t.target_qty) * factor, min_qty.get(t.symbol, Decimal("1")))
    if qty <= 0:
        return None
    price = t.notional / abs(t.target_qty)
    return TargetPosition(
        symbol=t.symbol,
        side=t.side,
        target_qty=qty if t.target_qty > 0 else -qty,
        notional=qty * price,
        reason=t.reason + f"; {label} x{float(factor):.2f}",
    )


def _beta_balance(
    targets: list[TargetPosition],
    betas: dict[str, Decimal],
    min_qty: dict[str, Decimal],
) -> list[TargetPosition]:
    one = Decimal("1")
    longs = [t for t in targets if t.target_qty > 0]
    shorts = [t for t in targets if t.target_qty < 0]
    beta_long = sum((t.notional * betas.get(t.symbol, one) for t in longs), Decimal("0"))
    beta_short = sum((t.notional * betas.get(t.symbol, one) for t in shorts), Decimal("0"))
    if beta_long <= 0 or beta_short <= 0 or beta_long == beta_short:
        return targets  # one-sided books are capped elsewhere
    heavy = longs if beta_long > beta_short else shorts
    factor = min(beta_long, beta_short) / max(beta_long, beta_short)
    heavy_syms = {t.symbol for t in heavy}
    out: list[TargetPosition] = []
    for t in targets:
        if t.symbol in heavy_syms:
            scaled = _scale(t, factor, min_qty, "beta-balance")
            if scaled is not None:
                out.append(scaled)
        else:
            out.append(t)
    return out


def beta_exposure(
    targets: Iterable[TargetPosition], betas: dict[str, Decimal]
) -> Decimal:
    """Signed beta-weighted notional of a set of targets."""
    return sum(
        (
            t.notional * betas.get(t.symbol, Decimal("1")) * (1 if t.target_qty > 0 else -1)
            for t in targets
            if t.target_qty != 0
        ),
        Decimal("0"),
    )


def build_dollar_neutral_basket(
    decisions: Iterable[SymbolDecision],
    *,
    nlv: Decimal,
    max_gross_pct: Decimal,
    max_position_pct: Decimal,
    min_qty: dict[str, Decimal] | None = None,
    sector_of: dict[str, str] | None = None,
    max_sector_pct: Decimal | None = None,
    betas: dict[str, Decimal] | None = None,
) -> list[TargetPosition]:
    """Convert decisions → target positions, respecting caps.

    `min_qty` and `sector_of` are optional dicts indexed by symbol.
    Symbols absent from `min_qty` quantize to 1 share. Sector caps are
    applied AFTER per-name sizing (anything over the sector cap gets
    proportionally trimmed).

    With `betas`, the legs are then balanced on BETA-weighted notional
    instead of dollars: the leg with more beta-dollars is scaled down
    until sum(beta * long $) ~= sum(beta * short $). Scaling only ever
    shrinks positions, so every cap above still holds. Symbols missing
    from `betas` count as beta 1.
    """
    min_qty = min_qty or {}
    sector_of = sector_of or {}
    max_sector_pct = max_sector_pct or Decimal("1")

    longs = [d for d in decisions if d.side == Side.LONG and not d.rejected]
    shorts = [d for d in decisions if d.side == Side.SHORT and not d.rejected]
    flats = [d for d in decisions if d.side == Side.FLAT or d.rejected]

    gross_budget = nlv * max_gross_pct
    # Split gross 50/50 long/short; each leg's budget gets cut further
    # if there are few names on that side.
    half = gross_budget / Decimal("2")
    long_budget = half if longs else Decimal("0")
    short_budget = half if shorts else Decimal("0")
    if longs and not shorts:
        # No shorts available — cut long budget to keep net exposure
        # bounded. We aim for net <= max_position_pct * 4 ~ "modest".
        long_budget = min(long_budget, nlv * max_position_pct * Decimal("4"))
    if shorts and not longs:
        short_budget = min(short_budget, nlv * max_position_pct * Decimal("4"))

    cap_per_name = nlv * max_position_pct

    def _size_side(rows: list[SymbolDecision], budget: Decimal, side: Side) -> list[TargetPosition]:
        if not rows or budget <= 0:
            return []
        # Weight by |composite_score| * conviction so highest-conviction
        # names get more capital.
        weights = [
            (d, max(0.0, abs(d.composite_score)) * max(0.0, d.conviction))
            for d in rows
        ]
        total_w = sum(w for _, w in weights)
        if total_w <= 0:
            return []
        out: list[TargetPosition] = []
        for d, w in weights:
            share = Decimal(str(w / total_w))
            notional = min(budget * share, cap_per_name)
            if d.last_price <= 0:
                continue
            step = min_qty.get(d.symbol, Decimal("1"))
            raw_qty = notional / d.last_price
            qty = _quantize(raw_qty, step)
            if qty <= 0:
                continue
            signed = qty if side == Side.LONG else -qty
            out.append(
                TargetPosition(
                    symbol=d.symbol,
                    side=side,
                    target_qty=signed,
                    notional=qty * d.last_price,
                    reason=(
                        f"weight {float(share):.2f} * budget {budget} "
                        f"(score {d.composite_score:+.2f}, conv {d.conviction:.2f}, "
                        f"tech: {d.technical_reason})"
                    ),
                )
            )
        return out

    long_targets = _size_side(longs, long_budget, Side.LONG)
    short_targets = _size_side(shorts, short_budget, Side.SHORT)

    # Sector cap: trim within each sector if total notional > cap.
    targets = long_targets + short_targets
    if sector_of and max_sector_pct < Decimal("1"):
        sector_total: dict[str, Decimal] = {}
        for t in targets:
            sec = sector_of.get(t.symbol)
            if sec is None:
                continue
            sector_total[sec] = sector_total.get(sec, Decimal("0")) + t.notional
        sector_cap = nlv * max_sector_pct
        scale: dict[str, Decimal] = {}
        for sec, notional in sector_total.items():
            if notional > sector_cap and notional > 0:
                scale[sec] = sector_cap / notional
        if scale:
            scaled: list[TargetPosition] = []
            for t in targets:
                sec = sector_of.get(t.symbol)
                if sec in scale:
                    trimmed = _scale(t, scale[sec], min_qty, "sector-trim")
                    if trimmed is not None:
                        scaled.append(trimmed)
                else:
                    scaled.append(t)
            targets = scaled

    if betas is not None:
        targets = _beta_balance(targets, betas, min_qty)

    # Emit zero targets for FLAT symbols so the execution engine knows
    # to close any open position in them.
    for d in flats:
        targets.append(
            TargetPosition(
                symbol=d.symbol,
                side=Side.FLAT,
                target_qty=Decimal("0"),
                notional=Decimal("0"),
                reason=(
                    f"flat (score {d.composite_score:+.2f}, "
                    f"reason: {d.rejected_reason or 'dead_band'})"
                ),
            )
        )
    return targets


def diff_targets(
    current: dict[str, Decimal], targets: list[TargetPosition]
) -> list[TargetPosition]:
    """Return the delta orders needed to move from `current` to `targets`.

    `target_qty` on the returned positions is the SIGNED order quantity
    (positive = buy, negative = sell). Symbols absent from `targets`
    but present in `current` get a fully-closing delta.
    """
    out: list[TargetPosition] = []
    seen: set[str] = set()
    for t in targets:
        seen.add(t.symbol)
        cur = current.get(t.symbol, Decimal("0"))
        delta = t.target_qty - cur
        if delta == 0:
            continue
        out.append(
            TargetPosition(
                symbol=t.symbol,
                side=t.side,
                target_qty=delta,
                notional=t.notional,
                reason=t.reason,
            )
        )
    for sym, qty in current.items():
        if sym in seen or qty == 0:
            continue
        out.append(
            TargetPosition(
                symbol=sym,
                side=Side.FLAT,
                target_qty=-qty,
                notional=Decimal("0"),
                reason="close — no signal in current cycle",
            )
        )
    return out
