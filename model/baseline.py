"""The example-bot rule: sell an island business once per storm when an official
advisory puts an island at or above the published odds. Same next-mark fill as our book.
"""

from __future__ import annotations

from model.config import BASELINE_CLIP, BASELINE_ODDS, PHYSICAL


def symbols_to_sell(advisory: dict, sites_by_company: dict[str, list[str]], sold: set[str], odds: float = BASELINE_ODDS) -> list[str]:
    if advisory.get("type") != "advisory" or advisory.get("source") != "official":
        return []
    conditions = advisory.get("hurricane_conditions") or {}
    hit = {island for island, prob in conditions.items() if float(prob) >= odds}
    if not hit:
        return []
    out = []
    for symbol in PHYSICAL:
        if symbol in sold:
            continue
        islands = sites_by_company.get(symbol, [])
        if any(island in hit for island in islands):
            out.append(symbol)
    return out


def clip_shares(symbol: str, limits: dict[str, float], clip: float = BASELINE_CLIP) -> float:
    """Negative shares: the example bot sells."""
    return -clip * float(limits[symbol])
