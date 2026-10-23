"""Announcement stub.

Research seasons do not emit these. The packet's 45 real seconds is not a fit.
"""

from __future__ import annotations


def record(message: dict, log: list) -> None:
    if message.get("type") != "announcement":
        return
    log.append({
        "symbols": list(message.get("symbols") or []),
        "headline": message.get("headline"),
        "time": message.get("time"),
    })
