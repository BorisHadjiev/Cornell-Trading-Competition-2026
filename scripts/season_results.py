#!/usr/bin/env python3
"""Per-season results of the live bot from book-log.jsonl and overnight.log. Reads only.

    ./venv/bin/python scripts/season_results.py

A season boundary in the book log is where the exchange's season mark and every position
go back to zero. The last row before that is the season's close.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.config import NAMES  # noqa: E402


def rows(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def main() -> None:
    book = ROOT / "book-log.jsonl"
    if not book.is_file():
        print("no book-log.jsonl yet")
        return
    seasons = []
    current = None
    prev = None
    spreads = {name: [] for name in NAMES}
    for row in rows(book):
        mark = row.get("season_mark")
        names = row.get("names") or {}
        flat = all(abs(float(v.get("position") or 0.0)) < 1 for v in names.values())
        for name, v in names.items():
            if v.get("bid") is not None and v.get("ask") is not None and v["ask"] > v["bid"] > 0:
                spreads[name].append((v["ask"] - v["bid"]) / (0.5 * (v["ask"] + v["bid"])) * 1e4)
        if prev is not None and flat and (mark is None or abs(float(mark)) < 1e-6) and prev["mark"] not in (None, 0.0) and abs(prev["mark"]) > 1e-6:
            # the previous row closed a season
            seasons.append({"closed_at": prev["t"], "final_mark": prev["mark"], "rows": prev["n"]})
            current = None
        if current is None:
            current = {"start": row["t"], "n": 0}
        current["n"] += 1
        prev = {"t": row["t"], "mark": None if mark is None else float(mark), "n": current["n"]}
    print("completed seasons:")
    for s in seasons:
        print(f"  closed {time.strftime('%H:%M', time.localtime(s['closed_at']))}  final mark {s['final_mark']:+9.2f}")
    if prev is not None:
        print(f"current season: mark {(prev['mark'] or 0.0):+9.2f} at {time.strftime('%H:%M:%S', time.localtime(prev['t']))}")
    if seasons:
        marks = [s["final_mark"] for s in seasons]
        print(f"mean {sum(marks)/len(marks):+.2f} over {len(marks)} seasons, losing {sum(1 for m in marks if m < 0)}")
    med = {n: sorted(v)[len(v) // 2] for n, v in spreads.items() if v}
    print("median spread, bps:", {n: round(v, 1) for n, v in med.items()})
    log = ROOT / "overnight.log"
    if log.is_file():
        kinds = {}
        statuses = {}
        for row in rows(log):
            k = row.get("kind")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "order":
                statuses[row.get("status")] = statuses.get(row.get("status"), 0) + 1
        print("log kinds:", kinds)
        print("order statuses at send:", statuses)


if __name__ == "__main__":
    main()
