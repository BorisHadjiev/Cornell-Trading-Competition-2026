"""Load the research pack into compact season objects.

Manifest fields (pattern, flip age, settlement) are available for fitting and
for scoring. Decision code must not read them. Public inputs are the feed
messages and the house marks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.config import NAMES, PHYSICAL
from model.labels import label_at

ROOT = Path(__file__).resolve().parent.parent / "hurricane-season-research"
HISTORY = ROOT / "history"


@dataclass
class Site:
    company: str
    name: str
    island: str
    x: float
    y: float
    weight: float


@dataclass
class StormObs:
    round: int
    name: str
    pattern: str
    flip: float | None
    obs: np.ndarray  # age, x, y, vx, vy, radius, intensity
    times: np.ndarray  # datetime64[s], one per observation
    prelim_damage: dict[str, float] | None = None
    final_damage: dict[str, float] | None = None
    prelim_loss: float | None = None
    final_loss: float | None = None

    def labels(self) -> np.ndarray:
        return np.array([label_at(self.pattern, self.flip, float(a)) for a in self.obs[:, 0]])


@dataclass
class Season:
    number: int
    storms: list[StormObs]
    times: np.ndarray
    prices: np.ndarray  # (T, 12) in NAMES order
    rounds: np.ndarray
    settlement: np.ndarray
    events: list[dict] = field(default_factory=list)
    season_loss: float = 0.0

    @property
    def open_prices(self) -> np.ndarray:
        return self.prices[0]


def load_case(root: Path = ROOT) -> dict:
    return json.loads((root / "case.json").read_text())


def sites_from_case(case: dict) -> list[Site]:
    out: list[Site] = []
    for company in PHYSICAL:
        for site in case["companies"][company]["sites"]:
            out.append(Site(company, site["name"], site["island"], float(site["x"]), float(site["y"]), float(site["weight"])))
    return out


def limits_from_case(case: dict) -> np.ndarray:
    return np.array([float(case["limits"][name]) for name in NAMES], dtype=float)


def _time(value: str) -> np.datetime64:
    return np.datetime64(value, "s")


def load_season(number: int, entry: dict, case_sites: list[Site] | None = None) -> Season:
    folder = HISTORY / entry["dir"]
    frame = pd.read_csv(folder / "market.csv")
    times = frame["time"].map(lambda s: _time(str(s))).to_numpy()
    prices = frame.loc[:, list(NAMES)].to_numpy(dtype=float)
    rounds = frame["round"].to_numpy(dtype=int)
    settlement_raw = json.loads((folder / "settlement.json").read_text())
    settlement = np.array([float(settlement_raw[name]) for name in NAMES], dtype=float)

    by_round: dict[int, dict] = {}
    events: list[dict] = []
    for line in (folder / "feed.jsonl").read_text().splitlines():
        if not line:
            continue
        message = json.loads(line)
        kind = message["type"]
        when = _time(message["time"])
        rnd = int(message["round"])
        if kind == "storm":
            bucket = by_round.setdefault(rnd, {"name": message["storm"], "rows": [], "times": []})
            bucket["name"] = message["storm"]
            bucket["rows"].append((
                float(message["age"]), float(message["x"]), float(message["y"]),
                float(message["vx"]), float(message["vy"]), float(message["radius"]),
                float(message["intensity"]),
            ))
            bucket["times"].append(when)
        elif kind == "storm_formed":
            bucket = by_round.setdefault(rnd, {"name": message["storm"], "rows": [], "times": []})
            bucket["name"] = message["storm"]
            events.append({"type": kind, "time": when, "round": rnd, "storm": message["storm"]})
        elif kind == "advisory":
            events.append({
                "type": kind,
                "time": when,
                "round": rnd,
                "storm": message.get("storm"),
                "source": message.get("source"),
                "outlook": message.get("outlook"),
                "age": float(message.get("age", 0.0)),
                "hurricane_conditions": message.get("hurricane_conditions") or {},
                "cone": message.get("cone") or [],
            })
        elif kind == "damage_report":
            events.append({
                "type": kind,
                "time": when,
                "round": rnd,
                "storm": message.get("storm"),
                "stage": message.get("stage"),
                "age": None,
                "damage_pct": message.get("damage_pct") or {},
                "industry_loss_musd": float(message.get("industry_loss_musd", 0.0)),
            })
        elif kind in ("impact", "storm_dissipated", "news", "announcement"):
            events.append({
                "type": kind,
                "time": when,
                "round": rnd,
                "storm": message.get("storm"),
                "island": message.get("island"),
                "age": float(message["age"]) if "age" in message else None,
                "symbols": message.get("symbols") or [],
                "headline": message.get("headline"),
            })

    meta = {int(storm["round"]): storm for storm in entry["storms"]}
    storms: list[StormObs] = []
    for rnd in (1, 2, 3):
        info = meta[rnd]
        bucket = by_round[rnd]
        obs = np.asarray(bucket["rows"], dtype=float)
        obs_times = np.asarray(bucket["times"], dtype="datetime64[s]")
        order = np.argsort(obs[:, 0], kind="mergesort")
        storms.append(StormObs(
            round=rnd,
            name=info["storm"],
            pattern=info["pattern"],
            flip=None if info["turned_erratic_at"] is None else float(info["turned_erratic_at"]),
            obs=obs[order],
            times=obs_times[order],
        ))
    for event in events:
        if event["type"] != "damage_report":
            continue
        storm = storms[int(event["round"]) - 1]
        if event["stage"] == "preliminary":
            storm.prelim_damage = {k: float(v) for k, v in event["damage_pct"].items()}
            storm.prelim_loss = float(event["industry_loss_musd"])
        elif event["stage"] == "final":
            storm.final_damage = {k: float(v) for k, v in event["damage_pct"].items()}
            storm.final_loss = float(event["industry_loss_musd"])
    events.sort(key=lambda item: (item["time"], item["type"]))
    return Season(
        number=number,
        storms=storms,
        times=times.astype("datetime64[s]"),
        prices=prices,
        rounds=rounds,
        settlement=settlement,
        events=events,
        season_loss=float(entry["season_loss_musd"]),
    )


def load_seasons(progress: bool = True) -> list[Season]:
    manifest = json.loads((HISTORY / "manifest.json").read_text())
    seasons: list[Season] = []
    entries = manifest["seasons"]
    for index, entry in enumerate(entries, start=1):
        seasons.append(load_season(int(entry["season"]), entry))
        if progress and index % 100 == 0:
            print(f"loaded {index}/{len(entries)} seasons", flush=True)
    return seasons


def damage_vector(payload: dict[str, float] | None) -> np.ndarray:
    if not payload:
        return np.zeros(len(PHYSICAL), dtype=float)
    return np.array([float(payload.get(name, 0.0)) for name in PHYSICAL], dtype=float)
