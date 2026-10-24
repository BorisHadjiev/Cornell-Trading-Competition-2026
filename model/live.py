"""Live book. Location sleeve from the first order, then the backup if it is losing.

The research backtest is unchanged. Live orders are the location residual on
the physical names. Severity, the insurer kink, composition, revision, known
damage, and ETF orders stay off for the whole season. That is not a refit.

If the exchange season mark (realized plus unrealized) reaches -400 while the
location book is on, cancel, flatten, and only the backup trades for the rest
of the season. At -800, flatten and send no more risk. A backup name that
loses 150 is closed for the season. Those three numbers are pre-set bounds.

Both books size an order as target minus the exchange position minus the
working leaves. A cancel that comes back already filled (409) is a fill, and
the account is reread before another order. A rest is priced at the live
touch. A reduction crosses. An announcement cancels resting orders. Every
name is flattened when the round leaves running.

Every second the process appends bid, ask, house mark, target, exchange
position, and working quantity for each name. That loop does not place orders.

Run from a clone:

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
from model.backup import BACKUP, MAIN, STOPPED, backup_names, backup_quantity, name_capped, next_book
from model.house import event_key
from model.config import (
    ARCH_BASKET,
    CLASSES,
    DEADBAND_FRACTION,
    LIVE_ANNOUNCEMENT_STAND_DOWN,
    LIVE_ETF_ORDERS,
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


def limit_price(side: str, bid: float, ask: float, fair: float, reducing: bool = False) -> float | None:
    """Rest on our side. Cross only when the edge through the spread beats the spread.

    A reduction does not use that gate. A sell that shrinks a long hits the bid.
    A buy that shrinks a short lifts the ask. Opening or adding still rests
    unless the gap beats the spread. LIVE_SPREAD_CROSSING stays false.
    """
    if bid is None or ask is None or not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
        return None
    spread = float(ask - bid)
    if side == "buy":
        if reducing or (not LIVE_SPREAD_CROSSING and float(fair - ask) > spread):
            return float(ask)
        return float(bid)
    if side == "sell":
        if reducing or (not LIVE_SPREAD_CROSSING and float(bid - fair) > spread):
            return float(bid)
        return float(ask)
    return None


_HALF_LIVES: dict[str, float] | None = None


def published_half_lives() -> dict[str, float]:
    """House half-lives published in fitted.json. Not refit."""
    global _HALF_LIVES
    if _HALF_LIVES is None:
        table = json.loads((ROOT / "model" / "fitted.json").read_text())["house_half_life"]
        _HALF_LIVES = {
            key: float(table[key]["half_life_seconds"])
            for key in ("official_advisory", "wire", "impact", "preliminary", "final")
        }
    return _HALF_LIVES


def stale_order_ids(orders: list[dict], meta: dict[str, dict], now: float, half_lives: dict[str, float]) -> list[str]:
    """Resting ids older than the half-life of the event that justified them."""
    found = []
    for order in orders:
        oid = str(order.get("id") or order.get("order_id") or "")
        info = meta.get(oid)
        if not oid or not info:
            continue
        half = half_lives.get(info.get("event_key"))
        if half is None:
            continue
        if float(now) - float(info["placed_at"]) > float(half):
            found.append(oid)
    return found


def targets_for_live(state: dict, models: Models, season, bar: int):
    """Location sleeve only, from the first order.

    Severity, the insurer kink, composition, revision, known damage, and the
    ETF sleeve stay off for the whole season. Research calls targets_from_state
    without this flag and is unchanged. This is not a refit.
    """
    live_state = dict(state)
    live_state["location_only"] = True
    return targets_from_state(live_state, models, season, bar)


def target_is_dead(target_qty: float, limit: float) -> bool:
    """The name is inside the deadband of flat, so the target no longer wants it."""
    return abs(float(target_qty)) <= DEADBAND_FRACTION * float(limit) + 1e-9


def marketable_flatten(name: str, position: float, resting: list[dict], quote: dict | None) -> dict | None:
    """Cancel this name's rests and, if a position is open, cross out to flat."""
    cancels = [order.get("id") or order.get("order_id") for order in resting if order.get("id") or order.get("order_id")]
    qty = int(abs(float(position)) + 1e-9)
    if qty < 1:
        if not cancels:
            return None
        return {"symbol": name, "cancel": cancels, "flatten": True}
    side = "sell" if float(position) > 0 else "buy"
    quote = quote or {}
    bid, ask = quote.get("bid"), quote.get("ask")
    if bid is None or ask is None:
        if not cancels:
            return None
        return {"symbol": name, "cancel": cancels, "flatten": True}
    price = limit_price(side, float(bid), float(ask), 0.0, reducing=True)
    if price is None:
        return {"symbol": name, "cancel": cancels, "flatten": True}
    price = round(float(price), 6)
    working = [
        order for order in resting
        if str(order.get("side")) == side
        and abs(float(order.get("price", 0.0)) - price) <= 0.05
        and abs(abs(_signed_qty(order)) - qty) < 1
    ]
    if working:
        extras = [
            order.get("id") or order.get("order_id")
            for order in resting
            if order not in working and (order.get("id") or order.get("order_id"))
        ]
        if not extras:
            return None
        return {"symbol": name, "cancel": extras, "flatten": True}
    return {
        "symbol": name,
        "side": side,
        "order_type": "limit",
        "quantity": qty,
        "price": price,
        "cancel": cancels,
        "flatten": True,
        "reducing": True,
    }


def pure_reduce(position: float, side: str, quantity: int) -> bool:
    """A sell that stops at flat from a long, or a buy that stops at flat from a short."""
    if quantity < 1 or abs(position) < 1e-9:
        return False
    if side == "sell" and position > 0:
        return quantity <= int(position + 1e-9)
    if side == "buy" and position < 0:
        return quantity <= int(-position + 1e-9)
    return False


def allows_round_order(
    opened_side: int,
    saw_inventory: bool,
    position: float,
    side: str,
    quantity: int,
) -> bool:
    """One entry per name per round. After inventory prints, only a pure reduce.

    The settled season bought HAVEN back and sold it again inside the same round.
    An add or a flip does not go out. Replacing an entry that has not printed yet,
    on the same side, still does.
    """
    if saw_inventory and abs(position) < 1:
        return False
    pos_side = 1 if position > 1e-9 else -1 if position < -1e-9 else 0
    side_sign = 1 if side == "buy" else -1
    if pos_side == 0 and opened_side == 0:
        return True
    locked = pos_side or opened_side
    if side_sign == locked and pos_side == 0:
        return True
    if side_sign == locked:
        return False
    return pure_reduce(position, side, quantity)


def screen_round_orders(
    plans: list[dict],
    position: dict[str, float],
    opened_side: dict[str, int],
    saw_inventory: dict[str, bool],
) -> list[dict]:
    kept = []
    for order in plans:
        if order.get("flatten") or "side" not in order:
            kept.append(order)
            continue
        name = str(order["symbol"])
        if allows_round_order(
            int(opened_side.get(name, 0)),
            bool(saw_inventory.get(name, False)),
            float(position.get(name, 0.0)),
            str(order["side"]),
            int(order["quantity"]),
        ):
            kept.append(order)
    return kept


def flatten_all_orders(position: dict[str, float], orders: list[dict], quotes: dict[str, dict]) -> list[dict]:
    """Marketable exit for every open name. Used when a round ends and at the next open."""
    resting: dict[str, list[dict]] = {name: [] for name in NAMES}
    for order in orders:
        symbol = order.get("symbol")
        if symbol in resting:
            resting[symbol].append(order)
    plans = []
    for name in NAMES:
        plan = marketable_flatten(name, float(position.get(name, 0.0)), resting[name], quotes.get(name))
        if plan:
            plans.append(plan)
    return plans


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


def session_continues(state: str | None) -> bool:
    """Stay up through a round and through the ready gap between rounds.

    Ready is flat and sends no risk. The season caps have to survive that gap.
    Any other clock state ends the session.
    """
    return state in (None, "running", "ready")


def reconnect_after_read_failure(state: str | None) -> bool:
    return session_continues(state)


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


class ExchangeError(RuntimeError):
    def __init__(self, method: str, path: str, code: int, body: str) -> None:
        super().__init__(f"{method} {path} -> {code} {body}")
        self.code = int(code)
        self.body = body


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
            raise ExchangeError(method, path, exc.code, body) from None

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


def _order_legs(position: float, side: str, quantity: int) -> list[tuple[int, bool]]:
    """Split one gap into a reducing leg and an adding leg. Only the reducing leg crosses."""
    closable = 0
    if side == "sell" and position > 0:
        closable = int(position + 1e-9)
    elif side == "buy" and position < 0:
        closable = int(-position + 1e-9)
    reduce_qty = min(int(quantity), max(0, closable))
    add_qty = int(quantity) - reduce_qty
    legs = []
    if reduce_qty >= 1:
        legs.append((reduce_qty, True))
    if add_qty >= 1:
        legs.append((add_qty, False))
    return legs


def _number(value: object) -> float | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def house_mark_from_quote(quote: dict | None) -> float | None:
    """House mark on the live quote. The mid is not a stand-in for it."""
    if not isinstance(quote, dict):
        return None
    for key in ("house_mark", "mark", "house_price", "last", "price"):
        found = _number(quote.get(key))
        if found is not None:
            return found
    return None


def signed_leaves(order: dict) -> float:
    """Signed shares still working. Filled shares are already in the position."""
    raw = order.get("quantity", order.get("qty"))
    if raw is None:
        return 0.0
    qty = abs(float(raw))
    filled = order.get("filled_quantity", order.get("filled"))
    if filled is not None:
        qty = max(0.0, qty - abs(float(filled)))
    side = str(order.get("side") or "").lower()
    if side == "sell":
        return -qty
    if side == "buy":
        return qty
    return 0.0


def working_quantity(orders: list[dict]) -> float:
    return float(sum(signed_leaves(order) for order in orders))


def execution_gap(target: float, position: float, working: float) -> float:
    """Signed shares still to send. Positive is a buy."""
    return float(target) - float(position) - float(working)


def after_cancel(results: list[str]) -> str:
    """A 409 is a fill. Do not send the next order until the account is reread.

    A cancel that did remove the order changes the working quantity, so the
    size is recomputed from the exchange before the next send.
    """
    if "filled" in results:
        return "resync"
    if "cancelled" in results:
        return "resize"
    if any(item in ("failed", "budget") for item in results):
        return "stop"
    return "send"


def cancel_result_from_error(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if code == 409 or "already filled" in text or "already-filled" in text:
        return "filled"
    return "failed"


def parse_marks(payload: dict) -> dict[str, float]:
    """Per-name exchange realized plus unrealized. Missing fields are omitted."""
    rows = payload.get("positions") if isinstance(payload, dict) else payload
    out: dict[str, float] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if symbol not in NAMES:
            continue
        if "realized_pnl" not in row and "unrealized_pnl" not in row:
            continue
        out[str(symbol)] = float(row.get("realized_pnl") or 0.0) + float(row.get("unrealized_pnl") or 0.0)
    return out


def season_mark(marks: dict[str, float]) -> float | None:
    if not marks:
        return None
    return float(sum(marks.values()))


def book_snapshot(
    quotes: dict[str, dict],
    target: np.ndarray | None,
    position: dict[str, float],
    orders: list[dict],
) -> dict[str, dict]:
    """One second of the book log: bid, ask, house mark, target, position, working."""
    resting: dict[str, list[dict]] = {name: [] for name in NAMES}
    for order in orders:
        symbol = order.get("symbol")
        if symbol in resting:
            resting[symbol].append(order)
    out: dict[str, dict] = {}
    for index, name in enumerate(NAMES):
        quote = quotes.get(name) or {}
        out[name] = {
            "bid": _number(quote.get("bid")),
            "ask": _number(quote.get("ask")),
            "house_mark": house_mark_from_quote(quote),
            "target": None if target is None else float(target[index]),
            "position": float(position.get(name, 0.0)),
            "working": working_quantity(resting[name]),
        }
    return out


def _split_need(position: float, side: str, quantity: float) -> tuple[float, float]:
    closable = 0.0
    if side == "sell" and position > 0:
        closable = float(position)
    elif side == "buy" and position < 0:
        closable = float(-position)
    reduce_qty = min(float(quantity), max(0.0, closable))
    return reduce_qty, float(quantity) - reduce_qty


def decide_orders(
    target: np.ndarray,
    fair: np.ndarray,
    position: dict[str, float],
    orders: list[dict],
    quotes: dict[str, dict],
    limits: np.ndarray,
    blocked: set[str] | None,
    skip: set[str] | None = None,
) -> list[dict]:
    """Orders sized as target minus the exchange position minus working leaves.

    Leaves we keep are orders at the touch we still want. A wrong-price order
    is cancelled and is not part of the working quantity. A reduction crosses.
    A new order rests at the live touch. `fair` only decides whether an add
    crosses. A blocked book (None) sends nothing.
    """
    if blocked is None:
        return []
    skip = skip or set()
    resting: dict[str, list[dict]] = {name: [] for name in NAMES}
    for order in orders:
        symbol = order.get("symbol")
        if symbol in resting:
            resting[symbol].append(order)
    planned = []
    for index, name in enumerate(NAMES):
        if name in blocked or name in skip or name not in quotes:
            continue
        quote = quotes[name]
        bid, ask = quote.get("bid"), quote.get("ask")
        if bid is None or ask is None:
            continue
        pos = float(position.get(name, 0.0))
        need = execution_gap(float(target[index]), pos, 0.0)
        name_orders = resting[name]
        if abs(need) < 1.0:
            cancel = [
                order.get("id") or order.get("order_id")
                for order in name_orders
                if (order.get("id") or order.get("order_id")) and abs(signed_leaves(order)) >= 1
            ]
            if cancel:
                planned.append({"symbol": name, "cancel": cancel})
            continue
        side = "buy" if need > 0 else "sell"
        reduce_qty, add_qty = _split_need(pos, side, abs(need))
        reduce_px = limit_price(side, float(bid), float(ask), float(fair[index]), reducing=True) if reduce_qty >= 1 else None
        add_px = limit_price(side, float(bid), float(ask), float(fair[index]), reducing=False) if add_qty >= 1 else None
        kept_reduce = 0.0
        kept_add = 0.0
        cancel = []
        for order in name_orders:
            oid = order.get("id") or order.get("order_id")
            leaves = abs(signed_leaves(order))
            same = str(order.get("side") or "") == side and leaves >= 1.0
            price = float(order.get("price") or 0.0)
            if same and reduce_px is not None and abs(price - reduce_px) <= 0.05 and kept_reduce + leaves <= reduce_qty + 1e-6:
                kept_reduce += leaves
                continue
            if same and add_px is not None and abs(price - add_px) <= 0.05 and kept_add + leaves <= add_qty + 1e-6:
                kept_add += leaves
                continue
            if oid:
                cancel.append(oid)
        kept_buy = (kept_reduce + kept_add) if side == "buy" else 0.0
        kept_sell = (kept_reduce + kept_add) if side == "sell" else 0.0
        room = room_for(float(limits[index]), pos, kept_buy, kept_sell, side)
        legs = []
        send_reduce = int(round(reduce_qty - kept_reduce))
        send_add = int(round(add_qty - kept_add))
        if send_reduce >= 1 and reduce_px is not None and room >= 1:
            take = min(send_reduce, int(room))
            if take >= 1:
                legs.append((take, True, reduce_px))
                room -= take
        if send_add >= 1 and add_px is not None and room >= 1:
            take = min(send_add, int(room))
            if take >= 1:
                legs.append((take, False, add_px))
        if not legs and not cancel:
            continue
        if not legs:
            planned.append({"symbol": name, "cancel": cancel})
            continue
        for leg_index, (qty, reducing, price) in enumerate(legs):
            planned.append({
                "symbol": name,
                "side": side,
                "order_type": "limit",
                "quantity": int(qty),
                "price": round(float(price), 6),
                "reducing": reducing,
                "cancel": cancel if leg_index == 0 else [],
            })
    return planned


class Runner:
    def __init__(self, exchange: Exchange, models: Models, season, opened: np.ndarray, record_path: Path, max_seconds: float, open_from_feed: bool = False, book_log: Path | None = None) -> None:
        self.exchange = exchange
        self.models = models
        self.season = season
        self.opened = opened
        self.open_from_feed = open_from_feed
        self.open_locked = not open_from_feed
        self.record_path = record_path
        self.book_log_path = book_log
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
        self.order_meta: dict[str, dict] = {}
        self.justifying_event: str | None = None
        self.flattened_on_exit = False
        self.flatten_order_ids: set[str] = set()
        self.round_side: dict[str, int] = {}
        self.saw_inventory: dict[str, bool] = {}
        self.carry_done = False
        self.book = MAIN
        self.blocked: set[str] = set()
        self.backup_open: set[str] = set()
        self.marks: dict[str, float] = {}
        self.last_target: np.ndarray | None = None
        self.announce_cancel = False
        self.round_open = False
        self.clock_state: str | None = None
        self.tracked_round: int | None = None
        self._warned_mark = False
        self._clock_error = False
        self._book_error = False
        self._halt_state: str | None = None

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
        if message.get("type"):
            key = event_key(message)
            if key:
                self.justifying_event = key
        if message.get("type") == "announcement":
            record_announcement(message, self.announcements)
            if self.announcements:
                self.announcements[-1]["received_at"] = time.time()
            self.announce_cancel = True
            self.note("announcement", symbols=message.get("symbols") or [], headline=message.get("headline"))
        if message.get("type") in ("storm", "storm_formed", "advisory", "damage_report", "impact", "storm_dissipated", "announcement") and message.get("time"):
            apply_news(self.season, message)
            rnd = int(self.season.rounds[1])
            if self.tracked_round is not None and rnd != self.tracked_round:
                self.backup_open = set()
                self.round_side = {}
                self.saw_inventory = {}
            self.tracked_round = rnd
        return message.get("type") in ("advisory", "impact", "damage_report", "announcement", "storm_formed", "storm_dissipated")

    def refresh_account(self) -> None:
        payload = self.exchange.get("/api/v1/positions?include_zero=true")
        self.position = parse_positions(payload)
        self.marks = parse_marks(payload)
        self.open_orders = parse_orders(self.exchange.get("/api/v1/orders/open"))

    def _books_ready(self) -> bool:
        return any(item.get("bid") is not None for item in self.quotes.values())

    def _sync_mark(self) -> bool:
        """Write the latest mids. False means the open is not locked yet."""
        mids = self.opened.copy()
        quoted = 0
        for index, name in enumerate(NAMES):
            quote = self.quotes.get(name) or {}
            if quote.get("bid") is not None and quote.get("ask") is not None:
                mids[index] = 0.5 * (float(quote["bid"]) + float(quote["ask"]))
                quoted += 1
        if self.open_from_feed and not self.open_locked:
            if quoted < len(NAMES):
                return False
            self.opened = mids.copy()
            self.season.prices[0] = self.opened
            self.open_locked = True
            self.note("open", source="first quotes")
        self.season.prices[1] = mids
        return True

    def _expired_ids(self, now: float) -> list[str]:
        orders = [{"id": oid} for oid in self.order_meta]
        return stale_order_ids(orders, self.order_meta, now, published_half_lives())

    def _cancel_id(self, oid: str, why: str) -> str:
        if not oid:
            return "skip"
        if not self.budget.take():
            self.note("budget", action="cancel")
            return "budget"
        try:
            self.exchange.delete(f"/api/v1/orders/{oid}")
            self.note("cancel", id=oid, why=why)
        except RuntimeError as exc:
            self.order_meta.pop(oid, None)
            kind = cancel_result_from_error(exc)
            if kind == "filled":
                self.note("fill", id=oid, why="cancel_409")
                return "filled"
            self.errors.append(str(exc))
            self.note("error", where="cancel", message=str(exc))
            return "failed"
        self.order_meta.pop(oid, None)
        return "cancelled"

    def _cancel_listed(self, ids: list[str], why: str) -> str:
        results = [self._cancel_id(str(oid), why) for oid in ids if oid]
        return after_cancel(results)

    def _load_target(self):
        state = _state_at(self.season, 1, self.models, self.rng)
        total, sleeves = targets_for_live(state, self.models, self.season, 1)
        return state, order_target(total, sleeves), fair_from_state(state, self.models)

    def _touch_fair(self) -> np.ndarray:
        """Mid of the live touch. It is the cross gate for a backup rest, not the order price."""
        fair = np.zeros(len(NAMES))
        for index, name in enumerate(NAMES):
            quote = self.quotes.get(name) or {}
            bid, ask = quote.get("bid"), quote.get("ask")
            if bid is not None and ask is not None:
                fair[index] = 0.5 * (float(bid) + float(ask))
        return fair

    def _has_position(self) -> bool:
        if self.open_orders:
            return True
        return any(abs(float(qty)) >= 1.0 for qty in self.position.values())

    def _reset_season(self) -> None:
        self.book = MAIN
        self.blocked = set()
        self.backup_open = set()
        self.marks = {}
        self.last_target = None
        self.note("season_reset")

    def _begin_round(self) -> None:
        self.flattened_on_exit = False
        self.carry_done = False
        self.round_side = {}
        self.saw_inventory = {}
        self.backup_open = set()
        self.note("round_open", round=int(self.season.rounds[1]), book=self.book)

    def _apply_clock(self, clock: dict) -> None:
        new_id = clock.get("competition_id")
        if self.competition_id and new_id and str(new_id) != str(self.competition_id):
            self._reset_season()
        if new_id:
            self.competition_id = new_id
        state = clock.get("state")
        if state != self.clock_state:
            self.clock_state = state
            self.note(
                "clock",
                state=state,
                fictional_time=clock.get("fictional_time"),
                elapsed=clock.get("elapsed_active_seconds"),
            )
        if state == "running":
            if not self.round_open:
                self._begin_round()
            self.round_open = True
            return
        retry = self.round_open or not self.flattened_on_exit
        self.round_open = False
        if retry:
            self.flatten_dead_targets()
        if not session_continues(state):
            self._halt_state = state

    def _apply_caps(self) -> str:
        """flatten, stop, or trade. The backup does not hand the book back."""
        mark = season_mark(self.marks)
        if mark is None:
            if not self._warned_mark:
                self._warned_mark = True
                self.note("mark_missing")
            return "stop" if self.book == STOPPED else "trade"
        updated = next_book(self.book, mark)
        if updated != self.book:
            self.note("book", previous=self.book, book=updated, mark=mark)
            self.book = updated
            if updated == STOPPED or self._has_position():
                return "stop" if updated == STOPPED else "flatten"
        if self.book == STOPPED:
            return "stop"
        if self.book == BACKUP:
            for name, pnl in self.marks.items():
                if name_capped(pnl) and name not in self.blocked:
                    self.blocked.add(name)
                    self.note("name_cap", symbol=name, pnl=pnl)
        return "trade"

    def _scan_backup(self) -> None:
        rnd = int(self.season.rounds[1])
        sites = self.models.sites_by_company
        for event in self.season.events:
            if int(event.get("round") or -1) != rnd:
                continue
            for name in backup_names(event, sites, self.backup_open, self.blocked):
                self.backup_open.add(name)

    def _backup_target(self) -> np.ndarray:
        target = np.zeros(len(NAMES))
        for name in self.backup_open:
            if name in self.blocked:
                continue
            index = NAMES.index(name)
            target[index] = -float(backup_quantity(self.models.limits[index]))
        return target

    def _plans(self, target: np.ndarray, fair: np.ndarray, only: set[str] | None) -> tuple[list[dict], bool]:
        """Flatten a dead target. Trade the rest. `only` limits a stale replace to those names."""
        current = np.array([self.position[name] for name in NAMES], dtype=float)
        dead = {
            name for index, name in enumerate(NAMES)
            if target_is_dead(target[index], self.models.limits[index])
        }
        plans: list[dict] = []
        resting: dict[str, list[dict]] = {name: [] for name in NAMES}
        for order in self.open_orders:
            symbol = order.get("symbol")
            if symbol in resting:
                resting[symbol].append(order)
        for name in NAMES:
            if name not in dead:
                continue
            if only is not None and name not in only:
                continue
            plan = marketable_flatten(name, self.position.get(name, 0.0), resting[name], self.quotes.get(name))
            if plan:
                plans.append(plan)
        adjusted = target.copy()
        for index, name in enumerate(NAMES):
            if name in dead:
                adjusted[index] = current[index]
        holding = portfolio.deadband_holds(current, adjusted, self.models.limits, DEADBAND_FRACTION)
        if not holding:
            blocked = standing_down(self.announcements, time.time())
            planned = decide_orders(
                target, fair, self.position, self.open_orders, self.quotes, self.models.limits, blocked, skip=dead,
            )
            if only is not None:
                planned = [order for order in planned if order["symbol"] in only]
            plans.extend(planned)
        plans = screen_round_orders(plans, self.position, self.round_side, self.saw_inventory)
        return plans, holding

    def _mark_inventory(self) -> None:
        for name, qty in self.position.items():
            if abs(float(qty)) < 1:
                continue
            self.saw_inventory[name] = True
            if name not in self.round_side:
                self.round_side[name] = 1 if float(qty) > 0 else -1

    def _note_entry(self, order: dict) -> None:
        if "side" not in order or order.get("flatten"):
            return
        name = str(order["symbol"])
        if name in self.round_side:
            return
        pos = float(self.position.get(name, 0.0))
        if abs(pos) >= 1:
            self.round_side[name] = 1 if pos > 0 else -1
        elif not order.get("reducing"):
            self.round_side[name] = 1 if order["side"] == "buy" else -1

    def _submit_plans(self, plans: list[dict]) -> bool:
        """False means a cancel changed the account and the caller must recompute."""
        for order in plans:
            order["event_key"] = order.get("event_key") or self.justifying_event
            self._note_entry(order)
            if self._replace(order) == "resync":
                try:
                    self.refresh_account()
                except RuntimeError as exc:
                    self.errors.append(str(exc))
                    self.note("error", where="resync", message=str(exc))
                return False
        return True

    def _order_ids(self) -> list[str]:
        return [
            str(order.get("id") or order.get("order_id"))
            for order in self.open_orders
            if order.get("id") or order.get("order_id")
        ]

    def act(self, reason: str) -> None:
        if not self.round_open:
            return
        if reason == "stale" and not self._expired_ids(time.time()):
            return
        try:
            self.refresh_account()
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="account", message=str(exc))
            return
        mode = self._apply_caps()
        if mode in ("flatten", "stop"):
            self.last_target = np.zeros(len(NAMES))
            if self._books_ready():
                self._sync_mark()
                for _ in range(4):
                    plans = flatten_all_orders(self.position, self.open_orders, self.quotes)
                    self.note("flatten", why=self.book, orders=len(plans))
                    if self._submit_plans(plans):
                        break
            if mode == "stop" or self._has_position():
                return
        if self.announce_cancel or reason == "announcement":
            self.announce_cancel = False
            ids = self._order_ids()
            if ids:
                status = self._cancel_listed(ids, "announcement")
                try:
                    self.refresh_account()
                except RuntimeError as exc:
                    self.errors.append(str(exc))
                    self.note("error", where="announcement", message=str(exc))
                    return
                if status == "resync":
                    return
        if not self._books_ready() or not self._sync_mark():
            return
        expired = self._expired_ids(time.time())
        only: set[str] | None = None
        if expired:
            symbols = {
                str(order.get("symbol"))
                for order in self.open_orders
                if str(order.get("id") or order.get("order_id") or "") in set(expired) and order.get("symbol")
            }
            status = self._cancel_listed(expired, "half_life")
            try:
                self.refresh_account()
            except RuntimeError as exc:
                self.errors.append(str(exc))
                self.note("error", where="half_life", message=str(exc))
                return
            if status == "resync":
                return
            if reason == "stale":
                only = symbols
        elif reason == "stale":
            return
        self._mark_inventory()
        if not self.carry_done:
            self.carry_done = True
            if any(self.saw_inventory.values()):
                carried = flatten_all_orders(self.position, self.open_orders, self.quotes)
                self.note("flatten", why="carry", orders=len(carried))
                self._submit_plans(carried)
                return
        state: dict = {}
        try:
            if self.book == BACKUP:
                self._scan_backup()
                target = self._backup_target()
                fair = self._touch_fair()
            elif self.book == STOPPED:
                target = np.zeros(len(NAMES))
                fair = self._touch_fair()
            else:
                state, target, fair = self._load_target()
        except Exception as exc:
            self.errors.append(str(exc))
            self.note("error", where="target", message=str(exc))
            return
        self.last_target = np.asarray(target, dtype=float)
        sent = False
        holding = False
        plans: list[dict] = []
        for _ in range(4):
            plans, holding = self._plans(self.last_target, fair, only)
            only = None
            if self._submit_plans(plans):
                sent = True
                break
        if not sent:
            return
        if holding and not plans:
            if reason != "stale":
                self.note("hold", reason=reason, round=int(self.season.rounds[1]), book=self.book)
            return
        if reason != "stale":
            blocked = standing_down(self.announcements, time.time())
            self.note(
                "decision",
                reason=reason,
                round=int(self.season.rounds[1]),
                orders=len(plans),
                book=self.book,
                post_turn=bool(state.get("post_turn")),
                stand_down=blocked is None or sorted(blocked),
            )
        else:
            self.note("stale", orders=len(plans))

    def flatten_dead_targets(self) -> None:
        """Round left `running`. Flatten every open name, not only a dead target.

        Nothing is held across the gap. A 409 on the way out is a fill, and
        the next pass reads the position again.
        """
        if self.flattened_on_exit:
            return
        if not self._books_ready():
            self.note("flatten", why="no quotes")
            return
        self.flattened_on_exit = True
        self._sync_mark()
        try:
            self.refresh_account()
        except Exception as exc:
            self.errors.append(str(exc))
            self.note("error", where="flatten", message=str(exc))
            self.flattened_on_exit = False
            return
        for _ in range(4):
            plans = flatten_all_orders(self.position, self.open_orders, self.quotes)
            self.note("flatten", orders=len(plans), why="round_end")
            if self._submit_plans(plans):
                return
        self.flattened_on_exit = False

    def _replace(self, order: dict) -> str:
        results = [
            self._cancel_id(str(order_id), "flatten" if order.get("flatten") else "replace")
            for order_id in order.get("cancel") or []
            if order_id
        ]
        decision = after_cancel(results)
        if decision in ("resync", "resize"):
            return "resync"
        if decision == "stop" or "side" not in order:
            return "dropped"
        if not self.budget.take():
            self.note("budget", action="place", symbol=order["symbol"])
            return "dropped"
        payload = {key: order[key] for key in ("symbol", "side", "order_type", "quantity", "price")}
        try:
            response = self.exchange.post("/api/v1/orders", payload)
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="order", message=str(exc))
            return "dropped"
        sent = {**payload, "status": response.get("status") if isinstance(response, dict) else None, "id": _order_id(response)}
        self.orders_sent.append(sent)
        if sent["id"]:
            self.order_meta[sent["id"]] = {
                "placed_at": time.time(),
                "event_key": order.get("event_key") or self.justifying_event,
            }
            if order.get("flatten"):
                self.flatten_order_ids.add(sent["id"])
        self.note("order", **sent)
        return "sent"

    def _maybe_etf(self) -> None:
        """ETF create and redeem stay off. There is no live exception."""
        if not LIVE_ETF_ORDERS:
            return

    def write_book_log(self) -> None:
        """One line per second. This does not place an order."""
        if self.book_log_path is None:
            return
        try:
            self.refresh_account()
        except RuntimeError as exc:
            if not self._book_error:
                self._book_error = True
                self.note("error", where="book_log", message=str(exc))
        row = {
            "t": time.time(),
            "book": self.book,
            "round_open": self.round_open,
            "season_mark": season_mark(self.marks),
            "names": book_snapshot(self.quotes, self.last_target, self.position, self.open_orders),
        }
        self.book_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.book_log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    def _second(self) -> None:
        try:
            self._apply_clock(self.exchange.get("/api/v1/competitions/time"))
        except RuntimeError as exc:
            if not self._clock_error:
                self._clock_error = True
                self.note("error", where="clock", message=str(exc))
        if self._halt_state:
            return
        if self.round_open and self.book == STOPPED and self._has_position():
            self.act("timer")
        self.write_book_log()

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
        async with self._act_lock:
            if self._halt_state is not None:
                return
            self._apply_clock(clock)
            if self._halt_state is not None or not self.round_open:
                return
        await self._act("timer")

    async def run(self) -> None:
        subscribe = {"action": "subscribe", "symbols": list(NAMES), "channels": ["news", "quotes"]}
        ws_url = self.exchange.host.replace("https://", "wss://", 1) + "/ws/v1/market-data"
        headers = {"X-API-Key": self.exchange.key, "User-Agent": USER_AGENT}
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self._act_lock = asyncio.Lock()
        self._halt_state = None
        self._tasks: list[asyncio.Task] = []
        stale_task = asyncio.create_task(self._stale_loop())
        book_task = asyncio.create_task(self._book_loop())
        try:
            await self._run_socket(ws_url, headers, subscribe)
        finally:
            stale_task.cancel()
            book_task.cancel()

    async def _book_loop(self) -> None:
        """Log the book once a second. The write does not place an order."""
        try:
            while time.time() - self.started < self.max_seconds and self._halt_state is None:
                async with self._act_lock:
                    if self._halt_state is not None:
                        return
                    await asyncio.to_thread(self._second)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    async def _stale_loop(self) -> None:
        """Reprice a rest once it is older than the event that justified it. Do not churn the rest of the book."""
        try:
            while time.time() - self.started < self.max_seconds and self._halt_state is None:
                await asyncio.sleep(1.0)
                if self._halt_state is not None:
                    return
                if not self._expired_ids(time.time()):
                    continue
                await self._act("stale")
        except asyncio.CancelledError:
            return

    async def _run_socket(self, ws_url: str, headers: dict, subscribe: dict) -> None:
        import websockets

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
                async with self._act_lock:
                    self._apply_clock(clock)
                if self._halt_state is not None:
                    self.note("stop", why="state", state=self._halt_state)
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
    if not LIVE_ANNOUNCEMENT_STAND_DOWN or LIVE_SPREAD_CROSSING or LIVE_ETF_ORDERS:
        raise SystemExit("live flags must stay on: stand down, rest at the touch, and no ETF orders")
    if PRE_TURN_FRACTION > 0.35 + 1e-12:
        raise SystemExit("revision and known damage must stay at PRE_TURN_FRACTION 0.35")
    parser = argparse.ArgumentParser(description="Trade the location book, then the backup, on the testnet.")
    parser.add_argument("--max-seconds", type=float, default=4 * 60 * 60)
    parser.add_argument("--record", default="recording.jsonl")
    parser.add_argument("--book-log", default="book-log.jsonl")
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
        "etf_orders": LIVE_ETF_ORDERS,
        "book": "location",
        "book_log": args.book_log,
        "name_loss_cap": 150,
        "switch_loss": 400,
        "stop_loss": 800,
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
        book_log=Path(args.book_log),
    )
    runner.competition_id = clock.get("competition_id")
    runner.clock_state = clock.get("state")
    runner.round_open = clock.get("state") == "running"
    try:
        asyncio.run(runner.run())
    finally:
        keep = set(runner.flatten_order_ids)
        try:
            if keep:
                runner.refresh_account()
                for order in runner.open_orders:
                    oid = str(order.get("id") or order.get("order_id") or "")
                    if oid and oid not in keep:
                        runner.exchange.delete(f"/api/v1/orders/{oid}")
                runner.note("cancel_resting", kept=sorted(keep))
            else:
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
