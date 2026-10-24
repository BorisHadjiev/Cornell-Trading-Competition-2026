#!/usr/bin/env python3
"""Per-name mark-to-market, sign reversals and an estimate of spread paid, from book-log.jsonl.

    ./venv/bin/python scripts/attribution.py <start HH:MM> <end HH:MM>     (local time, today)

Reads only. MTM is position x change in mid, second by second; a fill is charged half the quoted
spread per share (the touch is not known, so this is an upper bound for a rest and a lower bound
for a cross that walks the book).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.config import NAMES  # noqa: E402


def stamp(hhmm: str) -> float:
    now = datetime.now()
    h, m = (int(x) for x in hhmm.split(":"))
    return now.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def main() -> None:
    start, end = stamp(sys.argv[1]), stamp(sys.argv[2])
    rows = []
    for line in (ROOT / "book-log.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if start <= r["t"] <= end:
            rows.append(r)
    mtm = {n: 0.0 for n in NAMES}
    cost = {n: 0.0 for n in NAMES}
    traded = {n: 0.0 for n in NAMES}
    flips = {n: 0 for n in NAMES}
    prev = None
    for r in rows:
        names = r["names"]
        if prev is not None:
            for n in NAMES:
                a, b = prev["names"][n], names[n]
                if a["bid"] is None or a["ask"] is None or b["bid"] is None or b["ask"] is None:
                    continue
                mid0 = 0.5 * (a["bid"] + a["ask"]); mid1 = 0.5 * (b["bid"] + b["ask"])
                mtm[n] += a["position"] * (mid1 - mid0)
                d = b["position"] - a["position"]
                if abs(d) >= 1:
                    traded[n] += abs(d)
                    cost[n] += abs(d) * 0.5 * (b["ask"] - b["bid"])
                if a["position"] * b["position"] < 0:
                    flips[n] += 1
        prev = r
    print(f"rows {len(rows)} from {time.strftime('%H:%M', time.localtime(rows[0]['t']))} to {time.strftime('%H:%M', time.localtime(rows[-1]['t']))}")
    print(f"{'name':6s} {'mtm':>8s} {'spread$':>8s} {'shares':>7s} {'flips':>5s}")
    for n in sorted(NAMES, key=lambda k: mtm[k]):
        print(f"{n:6s} {mtm[n]:8.0f} {cost[n]:8.0f} {traded[n]:7.0f} {flips[n]:5d}")
    print(f"total mtm {sum(mtm.values()):.0f}, spread paid ~{sum(cost.values()):.0f}, shares traded {sum(traded.values()):.0f}, flips {sum(flips.values())}")
    print("last season mark in window:", rows[-1].get("season_mark"))


if __name__ == "__main__":
    main()
