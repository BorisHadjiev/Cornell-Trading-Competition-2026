"""Load the Hurricane Season research data: seasons, the feed, the house prices and the settlement
values.

    from history import load
    data = load("../history")              # the folder with manifest.json
    season = data.seasons[0]
    season.storms                          # manifest entries: storm, pattern, damage_pct, ...
    season.messages(type="advisory", source="official")
    season.track(1)                        # [(age, x, y, radius), ...] of the storm in round 1
    season.prices["PORT"]                  # house consensus price at each sample
    season.settlement["PORT"]              # settlement value

This module uses only the standard library. Each season loads its files when you first use it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any


@dataclass
class Season:
    """One research season: three storms, the feed, the house prices and the settlement values."""

    number: int
    folder: Path
    storms: list[dict[str, Any]]
    season_loss_musd: float

    @cached_property
    def feed(self) -> list[dict[str, Any]]:
        """All feed messages in `seq` order, exactly as the exchange published them."""
        with (self.folder / "feed.jsonl").open() as f:
            return [json.loads(line) for line in f]

    @cached_property
    def times(self) -> list[str]:
        """The fictional time of each price sample."""
        return [row["time"] for row in self._rows]

    @cached_property
    def rounds(self) -> list[int]:
        """The round of each price sample."""
        return [int(row["round"]) for row in self._rows]

    @cached_property
    def prices(self) -> dict[str, list[float]]:
        """The house consensus price of each symbol at each sample, with a small delay and noise."""
        symbols = [k for k in self._rows[0] if k not in ("time", "round")]
        return {s: [float(row[s]) for row in self._rows] for s in symbols}

    @cached_property
    def settlement(self) -> dict[str, float]:
        """The settlement value of each symbol at the end of the season."""
        return json.loads((self.folder / "settlement.json").read_text())

    def messages(self, round: int | None = None, **match: Any) -> list[dict[str, Any]]:
        """The feed messages in `round` (or in all rounds) that match all the given fields."""
        return [m for m in self.feed
                if (round is None or m["round"] == round) and all(m.get(k) == v for k, v in match.items())]

    def track(self, round: int) -> list[tuple[int, float, float, float]]:
        """The observed track of the storm in a round: `(age, x, y, radius)` for each message."""
        return [(m["age"], m["x"], m["y"], m["radius"]) for m in self.messages(round, type="storm")]

    @cached_property
    def _rows(self) -> list[dict[str, str]]:
        with (self.folder / "market.csv").open() as f:
            return list(csv.DictReader(f))


@dataclass
class History:
    """The research data: all seasons, in sequence."""

    root: Path
    seasons: list[Season] = field(default_factory=list)

    def season_of(self, date: str) -> Season | None:
        """The season with storms on `date` (`YYYY-MM-DD`), to find the season of a recording."""
        for s in self.seasons:
            if any(storm["open"].startswith(date) for storm in s.storms):
                return s
        return None


def load(root: str | Path) -> History:
    """Read `manifest.json` in `root`. Each season loads its files when you first use it."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    seasons = [Season(s["season"], root / s["dir"], s["storms"], s["season_loss_musd"]) for s in manifest["seasons"]]
    return History(root, seasons)
