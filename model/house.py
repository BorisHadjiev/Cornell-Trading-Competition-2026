"""Half-life of the house mark after a public event.

The horizon is a later mark, not season-end settlement. Wiggles inside the
quote-noise band are ignored.
"""

from __future__ import annotations

import numpy as np

from model.config import HOUSE_HORIZON_REAL_SECONDS, NAMES, QUOTE_NOISE_DOLLARS

EVENT_TYPES = ("official_advisory", "wire", "impact", "preliminary", "final")
BAR_SECONDS = 2.0  # market.csv steps one market minute, which is 2 real seconds


def event_key(event: dict) -> str | None:
    kind = event["type"]
    if kind == "advisory" and event.get("source") == "official":
        return "official_advisory"
    if kind == "advisory" and event.get("source") == "wire":
        return "wire"
    if kind == "impact":
        return "impact"
    if kind == "damage_report" and event.get("stage") == "preliminary":
        return "preliminary"
    if kind == "damage_report" and event.get("stage") == "final":
        return "final"
    return None


def half_life_table(seasons: list, horizon_seconds: float = HOUSE_HORIZON_REAL_SECONDS) -> dict:
    horizon_bars = max(1, int(round(horizon_seconds / BAR_SECONDS)))
    samples: dict[str, list[float]] = {key: [] for key in EVENT_TYPES}
    mean_move: dict[str, list[float]] = {key: [] for key in EVENT_TYPES}
    for season in seasons:
        times = season.times
        prices = season.prices
        for event in season.events:
            key = event_key(event)
            if key is None:
                continue
            start = int(np.searchsorted(times, event["time"], side="left"))
            end = start + horizon_bars
            if start >= len(times) or end >= len(times):
                continue
            for col in range(len(NAMES)):
                p0 = prices[start, col]
                p1 = prices[end, col]
                move = p1 - p0
                if abs(move) < QUOTE_NOISE_DOLLARS:
                    continue
                path = prices[start:end + 1, col] - p0
                target = 0.5 * move
                hit = np.where((np.sign(path) == np.sign(move)) & (np.abs(path) >= abs(target) - 1e-9))[0]
                lag = int(hit[0]) if len(hit) else horizon_bars
                samples[key].append(lag * BAR_SECONDS)
                mean_move[key].append(float(move))
    table = {}
    for key in EVENT_TYPES:
        vals = samples[key]
        moves = mean_move[key]
        table[key] = {
            "estimated": True,
            "half_life_seconds": float(np.median(vals)) if vals else None,
            "mean_signed_move": float(np.mean(moves)) if moves else None,
            "n": len(vals),
            "horizon_seconds": horizon_seconds,
        }
    if "settlement" in table:
        raise RuntimeError("house table must not score settlement")
    return table
