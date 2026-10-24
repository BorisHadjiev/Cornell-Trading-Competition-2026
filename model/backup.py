"""Backup book. The starter example's rule, plus pre-set loss bounds.

Sell about 10% of the position limit when an official advisory puts one of a
company's islands at the example bot's odds. At most one new short per name
per storm. No severity, revision, known-damage, ETF, overnight carry, or add
after the name is open.

NAME_LOSS_CAP, BOOK_SWITCH_LOSS, and SEASON_STOP_LOSS are pre-set. They are
not fit on seasons 1–500 or 501–600, and the replay below is not used to move
them. The replay does not read manifest class, settlement, or final damage
until the scoring price at the last bar.
"""

from __future__ import annotations

import json

import numpy as np

from model import baseline
from model.config import (
    BASELINE_CLIP,
    BASELINE_ODDS,
    BOOK_SWITCH_LOSS,
    DEADBAND_FRACTION,
    NAME_LOSS_CAP,
    NAMES,
    PHYSICAL,
    SEASON_STOP_LOSS,
)
from model.data import HISTORY, limits_from_case, load_case, load_season, sites_from_case

IDX = {name: i for i, name in enumerate(NAMES)}
MAIN = "main"
BACKUP = "backup"
STOPPED = "stopped"


def backup_quantity(limit: float, clip: float = BASELINE_CLIP) -> int:
    """Shares to sell. Same rounding as the starter example."""
    return max(1, int(round(float(clip) * float(limit))))


def backup_names(advisory: dict, sites_by_company: dict, opened: set[str], blocked: set[str]) -> list[str]:
    """Names the example rule would sell, skipping a name already opened or capped."""
    return baseline.symbols_to_sell(advisory, sites_by_company, set(opened) | set(blocked), BASELINE_ODDS)


def next_book(mode: str, season_mark: float) -> str:
    """Season mark is exchange realized plus unrealized. Do not switch back.

    −800 flattens and stops, from either book. −400, while the location book
    is on, hands the rest of the season to the backup. Both numbers are the
    pre-set bounds in config.py.
    """
    if season_mark <= -SEASON_STOP_LOSS:
        return STOPPED
    if mode == STOPPED:
        return STOPPED
    if mode == BACKUP:
        return BACKUP
    if season_mark <= -BOOK_SWITCH_LOSS:
        return BACKUP
    return MAIN


def name_capped(pnl: float) -> bool:
    """A backup name at the pre-set per-name bound is closed for the season."""
    return float(pnl) <= -NAME_LOSS_CAP


def fill_delay(name: str, position: float, target: float, mode: str) -> int:
    """Marks to wait. Crowded delays only the obvious island short."""
    if mode == "crowded":
        if name in PHYSICAL and target < -1e-9 and target < float(position) - 1e-9:
            return 5
        return 1
    return int(mode)


def replay_backup(
    prices: np.ndarray,
    rounds: np.ndarray,
    times: np.ndarray,
    events: list[dict],
    limits: np.ndarray,
    sites_by_company: dict,
    settlement: np.ndarray,
    mode: str,
    *,
    carry: bool = False,
    use_caps: bool = True,
) -> float:
    """Season pnl of the backup. `mode` is 1, 2, 5, or crowded.

    A decision at bar t fills at t+k. The deadband sees shares that have
    already filled. A new target outside the deadband replaces the working
    order. Round 1 and round 2 close at the previous round's last mark unless
    `carry` is set. Round 3 is marked at settlement. Settlement is not read
    before that line.
    """
    n = len(times)
    position = np.zeros(len(NAMES))
    cash = np.zeros(len(NAMES))
    pending_bar = np.full(len(NAMES), -1, dtype=int)
    pending_qty = np.zeros(len(NAMES))
    desired = np.zeros(len(NAMES))
    opened: set[str] = set()
    blocked: set[str] = set()
    stopped = False
    round_id = int(rounds[0])
    by_bar: list[list[dict]] = [[] for _ in range(n)]
    for event in events:
        if event.get("type") != "advisory":
            continue
        bar = int(np.searchsorted(times, event["time"], side="left"))
        if bar < n:
            by_bar[bar].append(event)

    for bar in range(n):
        rnd = int(rounds[bar])
        if rnd != round_id:
            pending_bar[:] = -1
            if not carry and bar > 0:
                prev = prices[bar - 1]
                for index in range(len(NAMES)):
                    trade = -position[index]
                    cash[index] -= trade * float(prev[index])
                    position[index] = 0.0
            opened = set()
            desired[:] = 0.0
            round_id = rnd
        for index in range(len(NAMES)):
            if pending_bar[index] != bar:
                continue
            trade = float(pending_qty[index] - position[index])
            cash[index] -= trade * float(prices[bar, index])
            position[index] = pending_qty[index]
            pending_bar[index] = -1
        mark = prices[bar]
        if use_caps:
            for index, name in enumerate(NAMES):
                pnl = float(cash[index] + position[index] * mark[index])
                if name_capped(pnl):
                    blocked.add(name)
            season_pnl = float(cash.sum() + position @ mark)
            if season_pnl <= -SEASON_STOP_LOSS:
                stopped = True
        if not stopped:
            for event in by_bar[bar]:
                if int(event.get("round") or rnd) != rnd:
                    continue
                for name in backup_names(event, sites_by_company, opened, blocked):
                    opened.add(name)
        target = np.zeros(len(NAMES))
        if not stopped:
            for name in opened:
                if name in blocked:
                    continue
                target[IDX[name]] = -backup_quantity(limits[IDX[name]])
        for index, name in enumerate(NAMES):
            if target[index] == desired[index]:
                continue
            desired[index] = target[index]
            if abs(desired[index] - position[index]) <= DEADBAND_FRACTION * float(limits[index]) + 1e-9:
                pending_bar[index] = -1
                continue
            pending_bar[index] = bar + fill_delay(name, position[index], desired[index], mode)
            pending_qty[index] = desired[index]
    return float(cash.sum() + position @ settlement)


def _sites_by_company(case: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for site in sites_from_case(case):
        grouped.setdefault(site.company, []).append(site.island)
    return grouped


def score_backup(start: int = 1, stop: int = 500) -> dict[str, dict[str, float]]:
    """Mean, median, and 10th percentile on seasons `start`–`stop`.

    Four fills: the next mark, two marks, five marks, and the crowded rule.
    The caps stay at the pre-set bounds. Seasons after `stop` are not loaded.
    """
    case = load_case()
    limits = limits_from_case(case)
    sites = _sites_by_company(case)
    manifest = json.loads((HISTORY / "manifest.json").read_text())
    modes = ("1", "2", "5", "crowded")
    pnl = {mode: [] for mode in modes}
    for entry in manifest["seasons"]:
        number = int(entry["season"])
        if number < start or number > stop:
            continue
        season = load_season(number, entry)
        for mode in modes:
            pnl[mode].append(
                replay_backup(
                    season.prices,
                    season.rounds,
                    season.times,
                    season.events,
                    limits,
                    sites,
                    season.settlement,
                    mode,
                )
            )
        if number % 100 == 0:
            print(f"scored backup through season {number}", flush=True)
    out = {}
    for mode, values in pnl.items():
        column = np.asarray(values, dtype=float)
        out[mode] = {
            "mean": float(np.mean(column)),
            "median": float(np.median(column)),
            "p10": float(np.percentile(column, 10)),
            "n": int(column.size),
        }
    return out


def _print_scores(table: dict[str, dict[str, float]]) -> None:
    print(f"{'fill':<10} {'mean':>10} {'median':>10} {'p10':>10}")
    for mode in ("1", "2", "5", "crowded"):
        row = table[mode]
        print(f"{mode:<10} {row['mean']:10.2f} {row['median']:10.2f} {row['p10']:10.2f}")


if __name__ == "__main__":
    _print_scores(score_backup())
