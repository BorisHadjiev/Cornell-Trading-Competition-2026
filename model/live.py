"""Live prototype. Trades the fitted book on the testnet.

The research backtest is unchanged. This runner does not turn the reserve off,
does not turn the location scale off, and does not raise revision or known
damage above PRE_TURN_FRACTION. Severity is the fitted sleeve. The ETF mark
sleeve is not sent as a directional order. Create or redeem is quantity 1,
and only after both books still show a locked gap.

Run from a clone, after the research pack is fetched:

    export CTC_API_KEY=...
    python -m model.live

The key is the CTC_API_KEY environment variable. It is never printed. A cloud
check may fall back to /tmp/ctc-bot.env only when that variable is unset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from model import portfolio
from model.announce import record as record_announcement
from model.backtest import Models, _state_at, targets_from_state
from model.config import (
    ARCH_BASKET,
    CLASSES,
    DEADBAND_FRACTION,
    LIVE_ANNOUNCEMENT_STAND_DOWN,
    LIVE_SPREAD_CROSSING,
    NAMES,
    PHYSICAL,
    PRE_TURN_FRACTION,
    REBLD_BASKET,
    SEED,
)
from model.damage import DamageFit
from model.data import Season, StormObs, limits_from_case, load_case, load_season, sites_from_case
from model.filter import FilterFit
from model.paths import PathLibrary
from model.value import ValueFit

USER_AGENT = "hurricane-research/1.0"
# Packet: the house takes about 45 real seconds to finish an announcement.
ANNOUNCEMENT_STAND_DOWN_SECONDS = 45.0
# Leave a few actions under the team cap of 180 per minute.
TEAM_ACTION_BUDGET = 170
# A missed pong must not close the socket during the last two real minutes of a round.
PING_INTERVAL_SECONDS = 20.0
PING_TIMEOUT_SECONDS = 120.0
ENV_PATH = Path("/tmp/ctc-bot.env")
FALLBACK_HOST = "https://cornelltradingcompetition.org"
ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "hurricane-season-research" / "history"
BASKETS = {"ARCH": ARCH_BASKET, "REBLD": REBLD_BASKET}


def api_credentials() -> tuple[str, str]:
    """Laptop path is CTC_API_KEY. The env file is only used when that variable is unset."""
    key = os.environ.get("CTC_API_KEY", "").strip()
    url = os.environ.get("CTC_API_URL", "").strip()
    if not key and ENV_PATH.is_file():
        file_env = load_env(ENV_PATH)
        key = (file_env.get("CTC_API_KEY") or file_env.get("CTC_BOT_KEY") or "").strip()
        if not url:
            url = file_env.get("CTC_API_URL", "").strip()
    if not key:
        raise SystemExit("CTC_API_KEY is not set")
    return key, url


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "<redacted>")
    return text


def limit_price(side: str, bid: float, ask: float, fair: float) -> float | None:
    """Rest on our side. Cross only when the edge through the spread beats the spread.

    LIVE_SPREAD_CROSSING is the research flag and stays false. Crossing is the
    exception in this function, not a second mode.
    """
    if bid is None or ask is None or not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
        return None
    spread = float(ask - bid)
    if side == "buy":
        if not LIVE_SPREAD_CROSSING and float(fair - ask) > spread:
            return float(ask)
        return float(bid)
    if side == "sell":
        if not LIVE_SPREAD_CROSSING and float(bid - fair) > spread:
            return float(bid)
        return float(ask)
    return None


def locked_etf_edges(quotes: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Create uses the ETF bid and the stock asks. Redeem uses the stock bids and the ETF ask."""
    out: dict[str, dict[str, float]] = {}
    for etf, basket in BASKETS.items():
        book = quotes.get(etf) or {}
        bid, ask = book.get("bid"), book.get("ask")
        if bid is None or ask is None:
            continue
        stock_ask = 0.0
        stock_bid = 0.0
        complete = True
        for name, qty in basket.items():
            leg = quotes.get(name) or {}
            if leg.get("bid") is None or leg.get("ask") is None:
                complete = False
                break
            stock_ask += qty * float(leg["ask"])
            stock_bid += qty * float(leg["bid"])
        if not complete:
            continue
        out[etf] = {
            "create": float(bid) - stock_ask,
            "redeem": stock_bid - float(ask),
        }
    return out


def best_locked_conversion(edges: dict[str, dict[str, float]]) -> tuple[str, str, float] | None:
    """One conversion. Dust at or under one cent is not a locked gap."""
    best: tuple[str, str, float] | None = None
    for etf, sides in edges.items():
        for side, edge in sides.items():
            if edge <= 0.01:
                continue
            if best is None or edge > best[2]:
                best = (etf, side, edge)
    return best


def order_target(total: np.ndarray, sleeves: dict[str, np.ndarray]) -> np.ndarray:
    """Drop the ETF mark sleeve. It is a house-mark gap, not a book fill."""
    return total - sleeves["etf_gap"]


def room_for(limit: float, position: float, resting_buy: float, resting_sell: float, side: str) -> float:
    if side == "buy":
        return float(limit - position - resting_buy)
    return float(limit + position - resting_sell)


class ActionBudget:
    def __init__(self, cap: int = TEAM_ACTION_BUDGET) -> None:
        self.cap = cap
        self.times: list[float] = []

    def remaining(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        self.times = [stamp for stamp in self.times if now - stamp < 60.0]
        return self.cap - len(self.times)

    def take(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self.remaining(now) < 1:
            return False
        self.times.append(now)
        return True


def reconnect_after_read_failure(state: str | None) -> bool:
    """Stay up while the competition is running. One failed read is not a stop."""
    return state is None or state == "running"


def standing_down(log: list[dict], now: float) -> set[str] | None:
    """Symbols to leave alone. None means the whole book, empty set means trade."""
    if not LIVE_ANNOUNCEMENT_STAND_DOWN:
        return set()
    blocked: set[str] = set()
    for item in log:
        stamp = item.get("received_at")
        if stamp is None or now - float(stamp) > ANNOUNCEMENT_STAND_DOWN_SECONDS:
            continue
        symbols = [str(symbol) for symbol in item.get("symbols") or []]
        if not symbols:
            return None
        blocked.update(symbols)
    return blocked


def _time(value: str) -> np.datetime64:
    return np.datetime64(value, "s")


def blank_season(opened: np.ndarray, now: np.datetime64) -> Season:
    """A season that has only what the socket has delivered. Used when the research pack is absent."""
    storms = [
        StormObs(
            round=rnd,
            name="",
            pattern="poison",
            flip=None,
            obs=np.zeros((0, 7)),
            times=np.array([], dtype="datetime64[s]"),
        )
        for rnd in (1, 2, 3)
    ]
    return Season(
        number=0,
        storms=storms,
        times=np.array([now, now], dtype="datetime64[s]"),
        prices=np.vstack([opened, opened.copy()]),
        rounds=np.array([1, 1], dtype=int),
        settlement=np.zeros(len(NAMES), dtype=float),
        events=[],
        season_loss=0.0,
    )


def match_season(fictional_time: str) -> tuple[int, dict] | None:
    """Match a replay to a research directory by the public open date. Do not read damage."""
    manifest_path = HISTORY / "manifest.json"
    if not manifest_path.is_file():
        return None
    day = str(fictional_time)[:10]
    manifest = json.loads(manifest_path.read_text())
    found = None
    for entry in manifest["seasons"]:
        for storm in entry["storms"]:
            if str(storm.get("open", ""))[:10] == day:
                found = (int(entry["season"]), entry)
                break
        if found is not None:
            break
    return found


def public_view(number: int, entry: dict, now: np.datetime64) -> tuple[object, np.ndarray]:
    """Season object a decision may see. Future rows and manifest labels are removed."""
    season = load_season(number, entry)
    opened = season.prices[0].copy()
    open_time = season.times[0]
    rnd = 1
    for storm in season.storms:
        storm.pattern = "poison"
        storm.flip = None
        storm.final_damage = None
        storm.prelim_damage = None
        storm.final_loss = None
        storm.prelim_loss = None
        keep = storm.times <= now
        storm.obs = storm.obs[keep]
        storm.times = storm.times[keep]
        if len(storm.times):
            rnd = int(storm.round)
    season.events = [event for event in season.events if event["time"] <= now]
    season.prices = np.vstack([opened, opened.copy()])
    season.times = np.array([open_time, now], dtype="datetime64[s]")
    season.rounds = np.array([1, rnd], dtype=int)
    season.settlement = np.zeros(len(NAMES), dtype=float)
    season.season_loss = 0.0
    return season, opened


def apply_news(season, message: dict) -> None:
    kind = message.get("type")
    if kind not in ("storm", "storm_formed", "advisory", "damage_report", "impact", "storm_dissipated", "announcement"):
        return
    when = _time(str(message["time"]))
    rnd = int(message["round"])
    if when > season.times[1]:
        season.times[1] = when
    season.rounds[1] = rnd
    if kind == "storm":
        storm = next(item for item in season.storms if item.round == rnd)
        row = np.array([
            float(message["age"]), float(message["x"]), float(message["y"]),
            float(message["vx"]), float(message["vy"]), float(message["radius"]),
            float(message["intensity"]),
        ], dtype=float)
        if len(storm.obs) == 0 or float(storm.obs[-1, 0]) < row[0]:
            storm.obs = np.vstack([storm.obs, row]) if len(storm.obs) else row.reshape(1, 7)
            storm.times = np.concatenate([storm.times, np.array([when], dtype="datetime64[s]")])
        return
    event = {
        "type": kind,
        "time": when,
        "round": rnd,
        "storm": message.get("storm"),
        "source": message.get("source"),
        "outlook": message.get("outlook"),
        "age": float(message["age"]) if message.get("age") is not None else None,
        "hurricane_conditions": message.get("hurricane_conditions") or {},
        "cone": message.get("cone") or [],
        "stage": message.get("stage"),
        "damage_pct": message.get("damage_pct") or {},
        "industry_loss_musd": float(message.get("industry_loss_musd") or 0.0),
        "island": message.get("island"),
        "symbols": message.get("symbols") or [],
        "headline": message.get("headline"),
    }
    ident = (kind, str(when), rnd, message.get("source"), message.get("stage"), message.get("island"), message.get("outlook"))
    seen = getattr(season, "event_ids", None)
    if seen is None:
        seen = set()
        season.event_ids = seen
    if ident in seen:
        return
    seen.add(ident)
    season.events.append(event)
    season.events.sort(key=lambda item: (item["time"], str(item["type"])))


def fair_from_state(state: dict, models: Models) -> np.ndarray:
    if state["final"]:
        vector = state["known"]
    elif state["location"]:
        vector = state["loc"]
    else:
        vector = state["sev"]
    return models.value.predict(state["open"], *vector)


def load_library(skip_season: int | None) -> PathLibrary:
    """Training paths shipped with the model. The live season is left out so we do not draw its own future."""
    packed = np.load(ROOT / "model" / "path_library.npz")
    seasons = packed["seasons"]
    offsets = packed["offsets"]
    obs = packed["obs"]
    labels = packed["labels"]
    labeled = []
    for index, season_number in enumerate(seasons):
        if skip_season is not None and int(season_number) == int(skip_season):
            continue
        start, stop = int(offsets[index]), int(offsets[index + 1])
        labeled.append((obs[start:stop], labels[start:stop]))
    print(f"path library {len(labeled)} storms, excludes season {skip_season}", flush=True)
    return PathLibrary.from_labeled(labeled)


def load_models(fitted: dict, case: dict, skip_season: int | None) -> Models:
    """Apply the published fit. Anchors live in fitted.json; paths are the shipped library."""
    runtime = fitted["runtime"]
    published = float(fitted["value"]["insurance"]["retention_musd"])
    if abs(float(runtime["retention"]) - published) > 1e-6:
        raise RuntimeError("runtime retention does not match fitted.json")
    sites = sites_from_case(case)
    damage = DamageFit(
        fitted["damage"]["form"],
        float(fitted["damage"]["dt_seconds"]),
        np.array([fitted["damage"]["intercept"][name] for name in PHYSICAL]),
        np.array([fitted["damage"]["slope"][name] for name in PHYSICAL]),
        float(fitted["damage"]["outage_threshold"]),
    )
    value = ValueFit(
        physical={name: np.asarray(coef, dtype=float) for name, coef in runtime["physical"].items()},
        recon=(float(runtime["recon"][0]), float(runtime["recon"][1])),
        mat=(float(runtime["mat"][0]), float(runtime["mat"][1])),
        gen=(float(runtime["gen"][0]), float(runtime["gen"][1])),
        retention=float(runtime["retention"]),
        width=float(runtime["width"]),
        haven_slope=float(runtime["haven_slope"]),
        rein_slope=float(runtime["rein_slope"]),
        mean_damage=float(runtime["mean_damage"]),
        mean_damage_vec=np.asarray(runtime["mean_damage_vec"], dtype=float),
        mean_outage=np.asarray(runtime["mean_outage"], dtype=float),
        mean_recon_curve=float(runtime["mean_recon_curve"]),
        mean_mat_curve=float(runtime["mean_mat_curve"]),
        mean_gen_curve=float(runtime["mean_gen_curve"]),
        mean_haven_pay=float(runtime["mean_haven_pay"]),
        mean_rein_pay=float(runtime["mean_rein_pay"]),
        mean_season_loss=float(runtime["mean_season_loss"]),
        stat_coef={name: np.asarray(coef, dtype=float) for name, coef in runtime["stat_coef"].items()},
    )
    filt_raw = fitted["filter"]
    filt = FilterFit(
        mean=np.asarray(filt_raw["feature_mean"], dtype=float),
        std=np.asarray(filt_raw["feature_std"], dtype=float),
        weight=np.asarray(filt_raw["weight"], dtype=float),
        bias=np.asarray(filt_raw["bias"], dtype=float),
        hazard_weight=np.asarray(filt_raw["hazard_weight"], dtype=float),
        hazard_bias=float(filt_raw["hazard_bias"]),
        prior=np.array([filt_raw["prior"][name] for name in CLASSES]),
        storm_seconds=float(case["timing"]["storm"]),
    )
    means = fitted["class_conditional_means"]
    sites_by_company: dict[str, list[str]] = {}
    for site in sites:
        sites_by_company.setdefault(site.company, []).append(site.island)
    return Models(
        value=value,
        damage=damage,
        sites=sites,
        revision_intercept=np.array([fitted["revision"]["intercept"][name] for name in PHYSICAL]),
        revision_slope=np.array([fitted["revision"]["slope"][name] for name in PHYSICAL]),
        loss_intercept=float(fitted["revision"]["loss_intercept"]),
        loss_slope=float(fitted["revision"]["loss_slope"]),
        loss_coef=np.asarray(fitted["insured_loss_site_coef"]["coef"], dtype=float),
        filt=filt,
        library=load_library(skip_season),
        limits=limits_from_case(case),
        mean_damage=np.asarray(runtime["storm_mean_damage"], dtype=float),
        mean_outage=np.asarray(runtime["storm_mean_outage"], dtype=float),
        mean_loss=float(runtime["storm_mean_loss"]),
        median_ri=float(fitted["median_radius_times_intensity"]),
        by_class_damage={name: np.asarray(means["damage_pct"][name], dtype=float) for name in CLASSES},
        by_class_outage={name: np.asarray(means["outage_hours"][name], dtype=float) for name in CLASSES},
        by_class_loss={name: float(means["industry_loss_musd"][name]) for name in CLASSES},
        storm_seconds=float(case["timing"]["storm"]),
        dissipate_seconds=float(case["storm"]["dissipate"]),
        sites_by_company=sites_by_company,
    )


class Exchange:
    def __init__(self, host: str, key: str) -> None:
        self.host = host.rstrip("/")
        self.key = key
        self.headers = {"User-Agent": USER_AGENT, "X-API-Key": key}

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        headers = dict(self.headers)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.host + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = redact(exc.read().decode()[:400], self.key)
            raise RuntimeError(f"{method} {path} -> {exc.code} {body}") from None

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    def post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, payload)

    def delete(self, path: str) -> dict:
        return self._request("DELETE", path)


def _order_id(response: dict) -> str | None:
    if not isinstance(response, dict):
        return None
    found = response.get("id") or response.get("order_id")
    if found:
        return str(found)
    inner = response.get("order")
    if isinstance(inner, dict):
        found = inner.get("id") or inner.get("order_id")
        return str(found) if found else None
    return None


def _signed_qty(row: dict) -> float:
    qty = row.get("quantity", row.get("qty", row.get("position")))
    if qty is None:
        return 0.0
    value = float(qty)
    side = str(row.get("side") or "").lower()
    if side == "sell":
        return -abs(value)
    if side == "buy":
        return abs(value)
    return value


def parse_positions(payload: dict) -> dict[str, float]:
    rows = payload.get("positions") if isinstance(payload, dict) else payload
    out = {name: 0.0 for name in NAMES}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if symbol in out:
            out[symbol] = _signed_qty(row)
    return out


def parse_orders(payload: dict) -> list[dict]:
    rows = payload.get("orders") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def decide_orders(
    target: np.ndarray,
    fair: np.ndarray,
    position: dict[str, float],
    orders: list[dict],
    quotes: dict[str, dict],
    limits: np.ndarray,
    blocked: set[str] | None,
) -> list[dict]:
    """Desired resting orders. A blocked book (None) sends nothing."""
    if blocked is None:
        return []
    resting: dict[str, list[dict]] = {name: [] for name in NAMES}
    for order in orders:
        symbol = order.get("symbol")
        if symbol in resting:
            resting[symbol].append(order)
    planned = []
    for index, name in enumerate(NAMES):
        if name in blocked or name not in quotes:
            continue
        quote = quotes[name]
        bid, ask = quote.get("bid"), quote.get("ask")
        if bid is None or ask is None:
            continue
        pos = float(position.get(name, 0.0))
        buy_rest = sum(abs(_signed_qty(order)) for order in resting[name] if str(order.get("side")) == "buy")
        sell_rest = sum(abs(_signed_qty(order)) for order in resting[name] if str(order.get("side")) == "sell")
        signed_rest = buy_rest - sell_rest
        have = pos + signed_rest
        delta = float(target[index] - have)
        if abs(delta) < 1.0:
            continue
        side = "buy" if delta > 0 else "sell"
        qty = int(min(abs(delta), max(0.0, room_for(limits[index], pos, buy_rest, sell_rest, side))))
        if qty < 1:
            continue
        price = limit_price(side, float(bid), float(ask), float(fair[index]))
        if price is None:
            continue
        same = [
            order for order in resting[name]
            if str(order.get("side")) == side
            and abs(float(order.get("price", 0.0)) - price) <= 0.05
            and abs(abs(_signed_qty(order)) - qty) < 1
        ]
        if same:
            continue
        planned.append({
            "symbol": name,
            "side": side,
            "order_type": "limit",
            "quantity": qty,
            "price": round(price, 6),
            "cancel": [order.get("id") or order.get("order_id") for order in resting[name]],
        })
    return planned


class Runner:
    def __init__(self, exchange: Exchange, models: Models, season, opened: np.ndarray, record_path: Path, max_seconds: float, open_from_feed: bool = False) -> None:
        self.exchange = exchange
        self.models = models
        self.season = season
        self.opened = opened
        self.open_from_feed = open_from_feed
        self.open_locked = not open_from_feed
        self.record_path = record_path
        self.max_seconds = max_seconds
        self.quotes: dict[str, dict] = {}
        self.seen_seq: set[int] = set()
        self.announcements: list[dict] = []
        self.budget = ActionBudget()
        self.rng = np.random.default_rng(SEED + int(season.number))
        self.started = time.time()
        self.last_decision = 0.0
        self.etf_done = False
        self.log: list[dict] = []
        self.competition_id = None
        self.connected = False
        self.errors: list[str] = []
        self.orders_sent: list[dict] = []
        self.position = {name: 0.0 for name in NAMES}
        self.open_orders: list[dict] = []

    def note(self, kind: str, **fields) -> None:
        row = {"kind": kind, "at": time.time(), **fields}
        self.log.append(row)
        safe = {key: value for key, value in row.items() if key != "at"}
        print(redact(json.dumps(safe, default=str), self.exchange.key), flush=True)

    def record_frame(self, frame: dict) -> None:
        with self.record_path.open("a") as handle:
            handle.write(json.dumps({**frame, "received_at": time.time()}) + "\n")

    def on_news(self, message: dict) -> bool:
        seq = message.get("seq")
        if seq is not None:
            if int(seq) in self.seen_seq:
                return False
            self.seen_seq.add(int(seq))
        if message.get("type") == "announcement":
            record_announcement(message, self.announcements)
            if self.announcements:
                self.announcements[-1]["received_at"] = time.time()
            self.note("announcement", symbols=message.get("symbols") or [], headline=message.get("headline"))
        if message.get("type") in ("storm", "storm_formed", "advisory", "damage_report", "impact", "storm_dissipated", "announcement") and message.get("time"):
            apply_news(self.season, message)
        return message.get("type") in ("advisory", "impact", "damage_report", "announcement", "storm_formed", "storm_dissipated")

    def refresh_account(self) -> None:
        self.position = parse_positions(self.exchange.get("/api/v1/positions"))
        self.open_orders = parse_orders(self.exchange.get("/api/v1/orders/open"))

    def act(self, reason: str) -> None:
        if not any(item.get("bid") is not None for item in self.quotes.values()):
            return
        mids = self.opened.copy()
        quoted = 0
        for index, name in enumerate(NAMES):
            quote = self.quotes.get(name) or {}
            if quote.get("bid") is not None and quote.get("ask") is not None:
                mids[index] = 0.5 * (float(quote["bid"]) + float(quote["ask"]))
                quoted += 1
        if self.open_from_feed and not self.open_locked:
            if quoted < len(NAMES):
                return
            self.opened = mids.copy()
            self.season.prices[0] = self.opened
            self.open_locked = True
            self.note("open", source="first quotes")
        self.season.prices[1] = mids
        try:
            self.refresh_account()
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="account", message=str(exc))
            return
        state = _state_at(self.season, 1, self.models, self.rng)
        total, sleeves = targets_from_state(state, self.models, self.season, 1)
        target = order_target(total, sleeves)
        current = np.array([self.position[name] for name in NAMES], dtype=float)
        if portfolio.deadband_holds(current, target, self.models.limits, DEADBAND_FRACTION):
            self.note("hold", reason=reason, round=int(self.season.rounds[1]))
            self._maybe_etf()
            return
        fair = fair_from_state(state, self.models)
        blocked = standing_down(self.announcements, time.time())
        planned = decide_orders(target, fair, self.position, self.open_orders, self.quotes, self.models.limits, blocked)
        self.note(
            "decision",
            reason=reason,
            round=int(self.season.rounds[1]),
            orders=len(planned),
            post_turn=bool(state.get("post_turn")),
            stand_down=blocked is None or sorted(blocked),
        )
        for order in planned:
            self._replace(order)
        self._maybe_etf()

    def _replace(self, order: dict) -> None:
        for order_id in order["cancel"]:
            if not order_id or not self.budget.take():
                if order_id:
                    self.note("budget", action="cancel")
                continue
            try:
                self.exchange.delete(f"/api/v1/orders/{order_id}")
                self.note("cancel", id=order_id, symbol=order["symbol"])
            except RuntimeError as exc:
                self.errors.append(str(exc))
                self.note("error", where="cancel", message=str(exc))
        if not self.budget.take():
            self.note("budget", action="place", symbol=order["symbol"])
            return
        payload = {key: order[key] for key in ("symbol", "side", "order_type", "quantity", "price")}
        try:
            response = self.exchange.post("/api/v1/orders", payload)
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="order", message=str(exc))
            return
        sent = {**payload, "status": response.get("status") if isinstance(response, dict) else None, "id": _order_id(response)}
        self.orders_sent.append(sent)
        self.note("order", **sent)

    def _maybe_etf(self) -> None:
        edges = locked_etf_edges(self.quotes)
        if edges:
            self.note("etf_gap", edges=edges, already=self.etf_done)
        if self.etf_done:
            return
        choice = best_locked_conversion(edges)
        if choice is None or not self.budget.take():
            return
        etf, side, edge = choice
        path = f"/api/v1/etf/{etf}/{side}"
        try:
            response = self.exchange.post(path, {"quantity": 1})
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="etf", message=str(exc))
            return
        self.etf_done = True
        self.note("etf", symbol=etf, side=side, quantity=1, edge=edge, status=response.get("status"))

    async def _act(self, reason: str) -> None:
        """One decision at a time, off the read loop so a slow order burst cannot stall the socket."""
        async with self._act_lock:
            if self._halt_state is not None:
                return
            await asyncio.to_thread(self.act, reason)

    def _track(self, task: asyncio.Task) -> None:
        self._tasks.append(task)

        def _done(done: asyncio.Task) -> None:
            if done.cancelled():
                return
            exc = done.exception()
            if exc is None:
                return
            message = redact(str(exc), self.exchange.key)
            self.errors.append(message)
            self.note("error", where="task", message=message)

        task.add_done_callback(_done)

    def _kick(self, reason: str) -> None:
        self._track(asyncio.create_task(self._act(reason)))

    async def _clock(self) -> None:
        try:
            clock = await asyncio.to_thread(self.exchange.get, "/api/v1/competitions/time")
        except Exception as exc:
            message = redact(str(exc), self.exchange.key)
            self.errors.append(message)
            self.note("error", where="clock", message=message)
            return
        self.competition_id = clock.get("competition_id")
        self.note(
            "clock",
            state=clock.get("state"),
            fictional_time=clock.get("fictional_time"),
            elapsed=clock.get("elapsed_active_seconds"),
        )
        if clock.get("state") != "running":
            self._halt_state = clock.get("state")
            return
        await self._act("timer")

    async def run(self) -> None:
        import websockets

        subscribe = {"action": "subscribe", "symbols": list(NAMES), "channels": ["news", "quotes"]}
        ws_url = self.exchange.host.replace("https://", "wss://", 1) + "/ws/v1/market-data"
        headers = {"X-API-Key": self.exchange.key, "User-Agent": USER_AGENT}
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self._act_lock = asyncio.Lock()
        self._halt_state = None
        self._tasks: list[asyncio.Task] = []
        while time.time() - self.started < self.max_seconds:
            try:
                async with websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    open_timeout=15,
                    ping_interval=PING_INTERVAL_SECONDS,
                    ping_timeout=PING_TIMEOUT_SECONDS,
                    max_queue=None,
                ) as ws:
                    self.connected = True
                    await ws.send(json.dumps(subscribe))
                    self.note("connected", host=self.exchange.host)
                    async for frame in ws:
                        if time.time() - self.started >= self.max_seconds:
                            self.note("stop", why="time")
                            return
                        data = json.loads(frame)
                        self.record_frame(data)
                        if self._halt_state is not None:
                            self.note("stop", why="state", state=self._halt_state)
                            return
                        kind = data.get("type")
                        if kind == "quote":
                            self.quotes[data["symbol"]] = data
                            if len(self.quotes) >= len(NAMES) and self.last_decision == 0.0:
                                self.last_decision = time.time()
                                self._kick("quotes")
                        elif kind == "news" and isinstance(data.get("message"), dict):
                            if self.on_news(data["message"]):
                                self._kick(str(data["message"].get("type") or "news"))
                        now = time.time()
                        if now - self.last_decision >= 60.0 and self.quotes:
                            self.last_decision = now
                            self._track(asyncio.create_task(self._clock()))
            except Exception as exc:
                message = redact(str(exc), self.exchange.key)
                self.errors.append(message)
                self.note("disconnected", message=message)
                if self._halt_state is not None:
                    self.note("stop", why="state", state=self._halt_state)
                    return
                try:
                    clock = await asyncio.to_thread(self.exchange.get, "/api/v1/competitions/time")
                except Exception as clock_exc:
                    self.note("error", where="clock", message=redact(str(clock_exc), self.exchange.key))
                    continue
                self.competition_id = clock.get("competition_id")
                state = clock.get("state")
                self.note(
                    "clock",
                    state=state,
                    fictional_time=clock.get("fictional_time"),
                    elapsed=clock.get("elapsed_active_seconds"),
                )
                if not reconnect_after_read_failure(state):
                    self.note("stop", why="state", state=state)
                    return
                continue
        self.note("stop", why="time")


def resolve_host(url: str, key: str) -> str:
    hosts = []
    for host in (url, FALLBACK_HOST):
        host = (host or "").rstrip("/")
        if host and host not in hosts:
            hosts.append(host)
    last = "no host"
    for host in hosts:
        req = urllib.request.Request(
            host + "/api/v1/competitions/time",
            headers={"User-Agent": USER_AGENT, "X-API-Key": key},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    return host
        except urllib.error.HTTPError as exc:
            last = f"{host} {exc.code}"
        except OSError as exc:
            last = f"{host} {exc.__class__.__name__}"
    raise RuntimeError(f"exchange rejected the connection ({last})")


def main() -> None:
    if not LIVE_ANNOUNCEMENT_STAND_DOWN or LIVE_SPREAD_CROSSING:
        raise SystemExit("live flags must stay on: stand down, and do not cross unless the gap beats the spread")
    if PRE_TURN_FRACTION > 0.35 + 1e-12:
        raise SystemExit("revision and known damage must stay at PRE_TURN_FRACTION 0.35")
    parser = argparse.ArgumentParser(description="Trade the hurricane prototype on the testnet.")
    parser.add_argument("--max-seconds", type=float, default=50 * 60)
    parser.add_argument("--record", default="recording.jsonl")
    args = parser.parse_args()
    key, url = api_credentials()
    host = resolve_host(url, key)
    case = load_case()
    fitted = json.loads((ROOT / "model" / "fitted.json").read_text())
    clock = Exchange(host, key).get("/api/v1/competitions/time")
    print(json.dumps({
        "host": host,
        "state": clock.get("state"),
        "fictional_time": clock.get("fictional_time"),
        "elapsed_active_seconds": clock.get("elapsed_active_seconds"),
        "competition_id": clock.get("competition_id"),
        "stand_down": LIVE_ANNOUNCEMENT_STAND_DOWN,
        "spread_crossing": LIVE_SPREAD_CROSSING,
        "pre_turn_fraction": PRE_TURN_FRACTION,
    }), flush=True)
    matched = match_season(str(clock.get("fictional_time")))
    now = _time(str(clock["fictional_time"]))
    if matched is None:
        number = 0
        season = blank_season(np.zeros(len(NAMES)), now)
        opened = season.prices[0].copy()
        open_from_feed = True
        print("no local research season; the open will be the first quotes", flush=True)
    else:
        number, entry = matched
        season, opened = public_view(number, entry, now)
        open_from_feed = False
        print(f"replay season {number} public feed through {clock.get('fictional_time')}", flush=True)
    models = load_models(fitted, case, skip_season=number or None)
    runner = Runner(
        Exchange(host, key), models, season, opened, Path(args.record), args.max_seconds, open_from_feed,
    )
    runner.competition_id = clock.get("competition_id")
    try:
        asyncio.run(runner.run())
    finally:
        try:
            runner.exchange.delete("/api/v1/orders/all")
            runner.note("cancel_all")
        except RuntimeError as exc:
            runner.errors.append(str(exc))
            runner.note("error", where="cancel_all", message=str(exc))
        try:
            runner.refresh_account()
        except RuntimeError as exc:
            runner.errors.append(str(exc))
        summary = {
            "connected": runner.connected,
            "host": host,
            "competition_id": runner.competition_id,
            "season": number,
            "orders": runner.orders_sent,
            "position": runner.position,
            "errors": runner.errors,
            "log": runner.log,
        }
        Path("/tmp/ctc-live-summary.json").write_text(json.dumps(summary, default=str))
        print("summary written", flush=True)


if __name__ == "__main__":
    main()
