#!/usr/bin/env python3
"""One-screen health check of the live bot: process, clock, season, book, recent log.

    SSL_CERT_FILE=... ./venv/bin/python scripts/health_check.py

Reads only. Places no orders and changes nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import live  # noqa: E402
from model.config import NAMES  # noqa: E402


def main() -> None:
    running = subprocess.run(["pgrep", "-f", "model.live"], capture_output=True, text=True).stdout.split()
    print(f"bot processes: {len(running)}")
    key, url = live.api_credentials()
    host = live.resolve_host(url, key)
    ex = live.Exchange(host, key)
    clock = ex.get("/api/v1/competitions/time")
    print("clock:", json.dumps(clock))
    matched = live.match_season(str(clock.get("fictional_time")))
    print("research season:", matched[0] if matched else None)
    payload = ex.get("/api/v1/positions?include_zero=true")
    positions = live.parse_positions(payload)
    marks = live.parse_marks(payload)
    held = {name: int(qty) for name, qty in positions.items() if abs(qty) >= 1}
    print("positions:", held)
    print("season mark (realized+unrealized):", round(live.season_mark(marks) or 0.0, 2))
    orders = live.parse_orders(ex.get("/api/v1/orders/open"))
    print(f"open orders: {len(orders)}")
    log = ROOT / "overnight.log"
    if log.is_file():
        lines = log.read_text().splitlines()
        kinds: dict[str, int] = {}
        for line in lines:
            if line.startswith("{"):
                try:
                    kinds[json.loads(line).get("kind", "?")] = kinds.get(json.loads(line).get("kind", "?"), 0) + 1
                except json.JSONDecodeError:
                    pass
        print("log lines:", len(lines), "by kind:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:10]))
        errors = [line for line in lines if '"kind": "error"' in line or '"disconnected"' in line]
        print(f"errors/disconnects: {len(errors)}")
        for line in errors[-3:]:
            print("  ", line[:200])
        print("last 3 lines:")
        for line in lines[-3:]:
            print("  ", line[:200])
    book = ROOT / "book-log.jsonl"
    if book.is_file():
        last = json.loads(book.read_text().splitlines()[-1])
        print("book log at", time.strftime("%H:%M:%S", time.localtime(last["t"])), "book", last.get("book"), "round_open", last.get("round_open"))


if __name__ == "__main__":
    main()
