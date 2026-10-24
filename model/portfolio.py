"""One target vector: deadband, limits, scenario caps, ETF mark gap.

PORT, GRID, FERRY, and ARCH share one Cascadia budget. ARCH is not an island
hedge. GEN is not given a damage term here; that constraint lives in value.py.
"""

from __future__ import annotations

import numpy as np

from model.config import (
    ARCH_BASKET,
    CASCADIA_BUDGET,
    DISAGREE_SCALE,
    EDGE_DOLLARS_FOR_FULL_SIZE,
    MIN_ETF_GAP,
    NAMES,
    REBLD_BASKET,
    SCENARIO_CAP_DOLLARS,
)

IDX = {name: i for i, name in enumerate(NAMES)}
CASCADIA_INDEX = np.array([IDX[name] for name in CASCADIA_BUDGET], dtype=int)


def shares_from_edge(edge: np.ndarray, limits: np.ndarray, fraction: float, scale: float = EDGE_DOLLARS_FOR_FULL_SIZE) -> np.ndarray:
    out = np.tanh(edge / scale) * limits * fraction
    out[IDX["ARCH"]] = 0.0
    out[IDX["REBLD"]] = 0.0
    return out


def deadband_holds(current: np.ndarray, new: np.ndarray, limits: np.ndarray, fraction: float) -> bool:
    return bool(np.all(np.abs(new - current) <= fraction * limits + 1e-9))


def etf_mark_sleeve(mark: np.ndarray, limits: np.ndarray, fraction: float, gap_min: float = MIN_ETF_GAP) -> np.ndarray:
    """Short the rich side of a house-mark gap. This is a mark, not a book fill."""
    shares = np.zeros(len(NAMES), dtype=float)
    for etf, basket in (("ARCH", ARCH_BASKET), ("REBLD", REBLD_BASKET)):
        basket_px = sum(qty * mark[IDX[name]] for name, qty in basket.items())
        gap = float(mark[IDX[etf]] - basket_px)
        if abs(gap) <= gap_min:
            continue
        quantity = -np.sign(gap) * limits[IDX[etf]] * fraction
        shares[IDX[etf]] += quantity
        for name, qty in basket.items():
            shares[IDX[name]] += -quantity * qty
    return shares


def fit_inside_limits(fundamental: np.ndarray, extra: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Scale `extra` so fundamental + extra stays inside the position limits."""
    alpha = 1.0
    for i, limit in enumerate(limits):
        leg = extra[i]
        if abs(leg) < 1e-12:
            continue
        if leg > 0:
            room = limit - fundamental[i]
            alpha = min(alpha, max(0.0, room / leg))
        else:
            room = fundamental[i] + limit
            alpha = min(alpha, max(0.0, room / (-leg)))
    return extra * alpha


def _scale_to_match(sleeves: dict[str, np.ndarray], target: np.ndarray) -> dict[str, np.ndarray]:
    total = np.zeros_like(target)
    for shares in sleeves.values():
        total += shares
    scale = np.ones_like(target)
    mask = np.abs(total) > 1e-8
    scale[mask] = target[mask] / total[mask]
    # Names the sleeves do not hold stay at the target only if target is ~0.
    return {name: shares * scale for name, shares in sleeves.items()}


def apply_scenario_caps(
    sleeves: dict[str, np.ndarray],
    mark: np.ndarray,
    scenario_values: dict[str, np.ndarray],
    direct_value: np.ndarray,
    limits: np.ndarray,
    edges: np.ndarray,
    cap: float = SCENARIO_CAP_DOLLARS,
) -> dict[str, np.ndarray]:
    """Shrink the book until each scenario loss is inside the cap.

    The direct-hit cap treats PORT, GRID, FERRY, and ARCH as one bet.
    Further cuts hit the lowest edge-times-room name first.
    """
    shares = sum(sleeves.values())
    direct_pnl = float((direct_value[CASCADIA_INDEX] - mark[CASCADIA_INDEX]) @ shares[CASCADIA_INDEX])
    if direct_pnl < -cap:
        shares[CASCADIA_INDEX] *= cap / abs(direct_pnl)

    def worst(candidate: np.ndarray) -> float:
        losses = []
        for value in scenario_values.values():
            losses.append(float((value - mark) @ candidate))
        return min(losses) if losses else 0.0

    for _ in range(40):
        if worst(shares) >= -cap:
            break
        room = np.maximum(limits - np.abs(shares), 1.0)
        score = np.abs(edges) * room
        score = np.where(np.abs(shares) > 1e-6, score, np.inf)
        which = int(np.argmin(score))
        shares[which] *= 0.85
    shortfall = worst(shares)
    if shortfall < -cap:
        shares *= cap / abs(shortfall)
    shares = np.clip(shares, -limits, limits)
    return _scale_to_match(sleeves, shares)


def disagree_scale(shares: np.ndarray, physics: np.ndarray, statistical: np.ndarray, mark: np.ndarray) -> np.ndarray:
    """Cut a name when the structured value and the flat linear value disagree on the sign."""
    out = shares.copy()
    phys = np.sign(physics - mark)
    stat = np.sign(statistical - mark)
    cut = (phys != stat) & (phys != 0) & (stat != 0)
    out[cut] *= DISAGREE_SCALE
    return out
