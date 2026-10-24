"""Live book. Sends the research book (config.LIVE_BOOK) to the exchange.

The research backtest is unchanged. Which book goes out, whether positions
carry across rounds, and the loss bounds are flags in config.py, each with
the holdout replay that chose its default (model/reports/live_rules.md).

Season state survives a restart and a reconnect: every news frame is appended
to state/<competition_id>.jsonl and replayed on start, and the exchange's
30-second snapshot fills any gap (damage reports from earlier rounds, the
latest advisory, the track). The value anchor is the season open: the matched
research season on the testnet, the first quotes at a round-1 open, or the
case packet's starting prices. An announcement re-anchors the named symbols by
the part of the house's 45-second move that the storm model does not explain,
so the book does not fade a repricing it cannot see.

If the season mark reaches -BOOK_SWITCH_LOSS, flatten and only the backup
trades. At -SEASON_STOP_LOSS, flatten and stop. A round-end or cap exit uses
the last two-sided quote when the live quote is empty, and it crosses.

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
    BOOK_SWITCH_LOSS,
    CLASSES,
    DEADBAND_FRACTION,
    LIVE_ANNOUNCEMENT_STAND_DOWN,
    LIVE_BOOK,
    LIVE_ETF_ORDERS,
    LIVE_FLATTEN_AT_ROUND_END,
    LIVE_ONE_ENTRY_PER_ROUND,
    LIVE_PER_NAME_DEADBAND,
    LIVE_SPREAD_CROSSING,
    MIN_REST_SECONDS,
    NAME_LOSS_CAP,
    NAMES,
    PACKET_OPEN,
    PHYSICAL,
    PRE_TURN_FRACTION,
    REBLD_BASKET,
    REST_IMPROVE,
    SEASON_STOP_LOSS,
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
STATE_DIR = ROOT / "state"
BASKETS = {"ARCH": ARCH_BASKET, "REBLD": REBLD_BASKET}
STOCKS = tuple(name for name in NAMES if name not in BASKETS)
IDX = {name: index for index, name in enumerate(NAMES)}
# The fictional clock runs 30 times real time. Storm age is in real seconds.
FICTIONAL_PER_REAL = 30
NEWS_TYPES = ("storm", "storm_formed", "advisory", "damage_report", "impact", "storm_dissipated", "announcement")


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


def rest_tick(bid: float, ask: float) -> float:
    """How far inside the touch a rest sits: REST_IMPROVE, but never more than a quarter spread."""
    spread = float(ask) - float(bid)
    tick = min(float(REST_IMPROVE), 0.25 * spread)
    return round(tick, 6) if tick >= 1e-6 else 0.0


def rest_price(side: str, bid: float, ask: float) -> float:
    """Where an opening order rests: one tick inside the house touch, first in the queue."""
    tick = rest_tick(bid, ask)
    return round(float(bid) + tick, 6) if side == "buy" else round(float(ask) - tick, 6)


def limit_price(side: str, bid: float, ask: float, fair: float, reducing: bool = False) -> float | None:
    """Rest inside the touch. Cross only when the edge through the spread beats the spread.

    A reduction does not use that gate. A sell that shrinks a long hits the bid.
    A buy that shrinks a short lifts the ask. Opening or adding rests one tick
    inside the touch unless the gap beats the spread. LIVE_SPREAD_CROSSING stays false.
    """
    if bid is None or ask is None or not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
        return None
    spread = float(ask - bid)
    if side == "buy":
        if reducing or (not LIVE_SPREAD_CROSSING and float(fair - ask) > spread):
            return float(ask)
        return rest_price("buy", bid, ask)
    if side == "sell":
        if reducing or (not LIVE_SPREAD_CROSSING and float(bid - fair) > spread):
            return float(bid)
        return rest_price("sell", bid, ask)
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
        if float(now) - float(info["placed_at"]) > max(float(half), float(MIN_REST_SECONDS)):
            found.append(oid)
    return found


def still_at_touch(order: dict, quote: dict | None) -> bool:
    """A rest that still sits where a fresh rest would go is not stale, whatever its age."""
    if not two_sided(quote):
        return False
    side = str(order.get("side") or "")
    if side not in ("buy", "sell"):
        return False
    want = rest_price(side, float(quote["bid"]), float(quote["ask"]))
    return abs(float(order.get("price") or 0.0) - want) <= 0.0051


def targets_for_live(state: dict, models: Models, season, bar: int):
    """The research book, or the variant named by config.LIVE_BOOK.

    Research calls targets_from_state without `live_book` and is unchanged.
    """
    live_state = dict(state)
    live_state["live_book"] = LIVE_BOOK
    return targets_from_state(live_state, models, season, bar)


def hold_inside_band(target: np.ndarray, current: np.ndarray, limits: np.ndarray, dead: set[str],
                     band: float | None = LIVE_PER_NAME_DEADBAND) -> np.ndarray:
    """Per-name deadband: a name whose change is inside `band` of its limit keeps its position.

    A dead name (target inside the deadband of flat) is left alone so the flatten still goes out.
    """
    if band is None:
        return target
    out = target.copy()
    for index, name in enumerate(NAMES):
        if name in dead:
            continue
        if abs(float(target[index]) - float(current[index])) <= float(band) * float(limits[index]) + 1e-9:
            out[index] = current[index]
    return out


def target_is_dead(target_qty: float, limit: float) -> bool:
    """The name is inside the deadband of flat, so the target no longer wants it."""
    return abs(float(target_qty)) <= DEADBAND_FRACTION * float(limit) + 1e-9


def two_sided(quote: dict | None) -> bool:
    """A quote we can cross: both sides present, and the ask is not through the bid."""
    if not isinstance(quote, dict):
        return False
    bid, ask = _number(quote.get("bid")), _number(quote.get("ask"))
    return bid is not None and ask is not None and np.isfinite(bid) and np.isfinite(ask) and ask >= bid


def exit_quotes(current: dict[str, dict], last: dict[str, dict]) -> dict[str, dict]:
    """Live touch when it is two-sided. Otherwise the last two-sided quote.

    Round-end and cap exits use this. An empty print at the bell does not
    cancel the exit. New risk still requires the live quote.
    """
    out: dict[str, dict] = {}
    for name in set(current) | set(last):
        live = current.get(name)
        chosen = live if two_sided(live) else last.get(name)
        if two_sided(chosen):
            out[name] = chosen
    return out


def names_past_cap(marks: dict[str, float], blocked: set[str]) -> list[str]:
    """Names whose exchange mark is through the pre-set per-name bound.

    The book mode is not an input. Location and backup both use this.
    """
    return [name for name, pnl in marks.items() if name not in blocked and name_capped(pnl)]


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
    """Apply the one-entry rule when LIVE_ONE_ENTRY_PER_ROUND is on. Off, every plan goes out."""
    if not LIVE_ONE_ENTRY_PER_ROUND:
        return list(plans)
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


def packet_open_vector() -> np.ndarray:
    """Case packet section 3, the house price at the first open of the live season."""
    return np.array([float(PACKET_OPEN[name]) for name in NAMES], dtype=float)


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


def _insert_obs(storm, row: np.ndarray, when: np.datetime64) -> bool:
    """Insert one observation in age order. A row whose age is already there is dropped."""
    ages = storm.obs[:, 0] if len(storm.obs) else np.zeros(0)
    at = int(np.searchsorted(ages, row[0], side="left"))
    if at < len(ages) and abs(float(ages[at]) - float(row[0])) < 1e-9:
        return False
    storm.obs = np.insert(storm.obs, at, row.reshape(1, 7), axis=0) if len(storm.obs) else row.reshape(1, 7)
    storm.times = np.insert(storm.times, at, np.datetime64(when, "s"))
    return True


def apply_track(season, message: dict) -> int:
    """Fill observation gaps from a track message. Returns the number of rows added.

    Track points are [age, x, y, radius] every 4 s of age and carry no intensity or velocity.
    Velocity is the finite difference along the track. Intensity is interpolated from the
    observations we do have, or the formation value of 1.0 before any exist. After a
    reconnect the exchange sends only the latest storm frame, so the track is the only record
    of the wind the storm delivered in the gap; the exposure integral uses per-row age gaps,
    so a 4-second row counts for 4 seconds.
    """
    points = message.get("points") or []
    if message.get("round") is None or len(points) < 2:
        return 0
    rnd = int(message["round"])
    storm = next((item for item in season.storms if item.round == rnd), None)
    if storm is None:
        return 0
    pts = np.asarray([[float(v) for v in point[:4]] for point in points], dtype=float)
    msg_age = float(message.get("age", pts[-1, 0]))
    msg_time = _time(str(message["time"]))
    known_ages = storm.obs[:, 0] if len(storm.obs) else np.zeros(0)
    added = 0
    for index, (age, x, y, radius) in enumerate(pts):
        if len(known_ages) and np.any(np.abs(known_ages - age) < 1e-9):
            continue
        prev = pts[index - 1] if index > 0 else None
        nxt = pts[index + 1] if index + 1 < len(pts) else None
        a, b = (prev, pts[index]) if nxt is None else ((pts[index], nxt) if prev is None else (prev, nxt))
        span = float(b[0] - a[0])
        vx = float((b[1] - a[1]) / span) if span > 0 else 0.0
        vy = float((b[2] - a[2]) / span) if span > 0 else 0.0
        if len(storm.obs):
            intensity = float(np.interp(age, storm.obs[:, 0], storm.obs[:, 6]))
        else:
            intensity = 1.0
        when = msg_time - np.timedelta64(int(round((msg_age - age) * FICTIONAL_PER_REAL)), "s")
        row = np.array([age, x, y, vx, vy, radius, intensity], dtype=float)
        if _insert_obs(storm, row, when):
            added += 1
            known_ages = storm.obs[:, 0]
    if msg_time > season.times[1]:
        season.times[1] = msg_time
    season.rounds[1] = rnd
    return added


def apply_snapshot(season, snap: dict) -> dict:
    """Catch up from the 30-second snapshot: damage reports of every round, the latest
    advisory, the impacts, the latest observation and the track. Returns what it added.

    A bot that connects late, or reconnects after a gap, gets no replay of old messages.
    Without this the book would value round 2 as if round 1 had done no damage.
    """
    rnd = snap.get("round")
    if rnd is None:
        advisory = snap.get("advisory") or {}
        reports = snap.get("damage") or []
        rnd = advisory.get("round") or (reports[-1].get("round") if reports else None)
    storm_name = snap.get("storm")
    stamp = str(snap.get("time"))
    added = {"damage": 0, "advisory": 0, "impact": 0, "obs": 0, "track": 0}
    before = len(season.events)
    for report in snap.get("damage") or []:
        if not isinstance(report, dict) or report.get("round") is None or not report.get("stage"):
            continue
        apply_news(season, {**report, "type": "damage_report", "time": report.get("time") or stamp})
    added["damage"] = len(season.events) - before
    before = len(season.events)
    advisory = snap.get("advisory")
    if isinstance(advisory, dict) and advisory.get("outlook"):
        apply_news(season, {**advisory, "type": "advisory", "source": advisory.get("source") or "official",
                            "time": advisory.get("time") or stamp,
                            "round": int(advisory.get("round") or rnd or 1), "storm": advisory.get("storm") or storm_name})
    added["advisory"] = len(season.events) - before
    before = len(season.events)
    if rnd is not None:
        for impact in snap.get("impacts") or []:
            if not isinstance(impact, dict) or not impact.get("island"):
                continue
            apply_news(season, {
                "type": "impact", "round": int(rnd), "storm": storm_name, "time": impact.get("time") or stamp,
                "island": impact["island"], "age": impact.get("age"),
                "headline": f"Hurricane conditions reported on {impact['island']}.",
            })
        added["impact"] = len(season.events) - before
        obs = snap.get("obs")
        if isinstance(obs, dict) and all(key in obs for key in ("age", "x", "y", "vx", "vy", "radius", "intensity")):
            storm = next((item for item in season.storms if item.round == int(rnd)), None)
            n_before = len(storm.obs) if storm is not None else 0
            apply_news(season, {**obs, "type": "storm", "round": int(rnd), "storm": storm_name, "time": obs.get("time") or stamp})
            added["obs"] = (len(storm.obs) if storm is not None else 0) - n_before
        track = snap.get("track")
        if track:
            age = float(obs["age"]) if isinstance(obs, dict) and "age" in obs else float(track[-1][0])
            added["track"] = apply_track(season, {"type": "track", "round": int(rnd), "storm": storm_name,
                                                   "time": (obs.get("time") if isinstance(obs, dict) else None) or stamp,
                                                   "age": age, "points": track})
        season.rounds[1] = int(rnd)
    when = _time(stamp) if snap.get("time") else None
    if when is not None and when > season.times[1]:
        season.times[1] = when
    return added


def apply_news(season, message: dict) -> None:
    kind = message.get("type")
    if kind == "track":
        apply_track(season, message)
        return
    if kind == "snapshot":
        apply_snapshot(season, message)
        return
    if kind not in NEWS_TYPES:
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
        _insert_obs(storm, row, when)
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


class StateStore:
    """Append-only season state, one JSON line per record, keyed by competition id.

    Records are the raw news frames the socket delivered, the locked open, and anchor shifts.
    On start the file is replayed, so a restart in round 2 knows round 1's final damage, the
    season-to-date insured loss, the open it is anchored on, and any announcement repricing.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def append(self, record: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def load(self) -> list[dict]:
        if self.path is None or not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def anchor_shift_for(mark_before: dict[str, float], mark_after: dict[str, float],
                     fair_before: np.ndarray, fair_after: np.ndarray, symbols: list[str]) -> dict[str, float]:
    """The part of the house's move on `symbols` that the storm model did not produce.

    After an announcement the house reprices for about 45 real seconds. Our value model knows
    nothing about announcements, so without this it would fade that repricing for the rest of
    the season. The shift is added to the season-open anchor of each named stock: the house's
    move minus our own fair-value change over the same window, so storm news inside the window
    is not mistaken for the announcement. ETFs are valued from their baskets and take no shift.
    """
    out: dict[str, float] = {}
    for name in symbols:
        if name not in IDX or name in BASKETS:
            continue
        if name not in mark_before or name not in mark_after:
            continue
        house = float(mark_after[name]) - float(mark_before[name])
        model = float(fair_after[IDX[name]]) - float(fair_before[IDX[name]])
        out[name] = house - model
    return out


class Runner:
    def __init__(self, exchange: Exchange, models: Models, season, opened: np.ndarray, record_path: Path, max_seconds: float, open_from_feed: bool = False, book_log: Path | None = None, state_store: StateStore | None = None) -> None:
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
        self.last_two_sided: dict[str, dict] = {}
        self._flatten_missing: tuple[str, ...] | None = None
        self._cap_quote_warned: set[str] = set()
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
        self.state_store = state_store or StateStore(None)
        self.replaying = False
        self.anchor_shift = np.zeros(len(NAMES), dtype=float)
        self.pending_anchor: list[dict] = []
        self.fictional_time: str | None = None

    def _mids(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in NAMES:
            quote = self.quotes.get(name) or {}
            if two_sided(quote):
                out[name] = 0.5 * (float(quote["bid"]) + float(quote["ask"]))
        return out

    def _fair_now(self) -> np.ndarray | None:
        try:
            state = _state_at(self.season, 1, self.models, np.random.default_rng(SEED))
            return fair_from_state(state, self.models)
        except Exception as exc:
            self.note("error", where="fair", message=redact(str(exc), self.exchange.key))
            return None

    def _apply_anchor(self) -> None:
        self.season.prices[0] = self.opened + self.anchor_shift

    def replay(self, records: list[dict]) -> None:
        """Rebuild season state from the store. Nothing is re-appended while replaying."""
        self.replaying = True
        try:
            for record in records:
                kind = record.get("kind")
                if kind == "open":
                    self.opened = np.asarray(record["opened"], dtype=float)
                    self.open_locked = True
                    self.open_from_feed = False
                    self._apply_anchor()
                    continue
                if kind == "anchor":
                    self.anchor_shift = np.asarray(record["shift"], dtype=float)
                    self._apply_anchor()
                    continue
                if record.get("type"):
                    self.on_news(record)
        finally:
            self.replaying = False
        self.pending_anchor = []
        self.note("replayed", records=len(records), round=int(self.season.rounds[1]),
                  events=len(self.season.events), obs=[int(len(storm.obs)) for storm in self.season.storms])

    def _settle_anchors(self) -> None:
        """Once the 45-second stand-down has passed, move the anchor by the unexplained house move."""
        if not self.pending_anchor:
            return
        now = time.time()
        due = [item for item in self.pending_anchor if now - float(item["at"]) >= ANNOUNCEMENT_STAND_DOWN_SECONDS]
        if not due:
            return
        self.pending_anchor = [item for item in self.pending_anchor if item not in due]
        fair_after = self._fair_now()
        mids = self._mids()
        if fair_after is None:
            return
        for item in due:
            shift = anchor_shift_for(item["mark_before"], mids, np.asarray(item["fair_before"]), fair_after, item["symbols"])
            for name, value in shift.items():
                self.anchor_shift[IDX[name]] += value
            self._apply_anchor()
            self.note("anchor", symbols=sorted(shift), shift={k: round(v, 3) for k, v in shift.items()}, headline=item.get("headline"))
        self.state_store.append({"kind": "anchor", "shift": self.anchor_shift.tolist(), "at": now})

    def note(self, kind: str, **fields) -> None:
        row = {"kind": kind, "at": time.time(), **fields}
        self.log.append(row)
        safe = {key: value for key, value in row.items() if key != "at"}
        print(redact(json.dumps(safe, default=str), self.exchange.key), flush=True)

    def record_frame(self, frame: dict) -> None:
        with self.record_path.open("a") as handle:
            handle.write(json.dumps({**frame, "received_at": time.time()}) + "\n")

    def on_news(self, message: dict) -> bool:
        kind = message.get("type")
        seq = message.get("seq")
        if seq is not None:
            if int(seq) in self.seen_seq:
                return False
            self.seen_seq.add(int(seq))
        if kind:
            key = event_key(message)
            if key:
                self.justifying_event = key
        if kind == "snapshot":
            added = apply_snapshot(self.season, message)
            if not self.replaying:
                self.state_store.append(message)
            if any(added.values()):
                self.note("snapshot", added=added, phase=message.get("phase"), round=int(self.season.rounds[1]))
            self.tracked_round = int(self.season.rounds[1])
            return any(added.values())
        if kind == "track":
            added = apply_track(self.season, message)
            if not self.replaying:
                self.state_store.append(message)
            if added:
                self.note("track_fill", rows=added, round=int(self.season.rounds[1]))
            return False
        if kind == "announcement":
            record_announcement(message, self.announcements)
            if self.announcements:
                self.announcements[-1]["received_at"] = time.time()
            self.announce_cancel = True
            self.note("announcement", symbols=message.get("symbols") or [], headline=message.get("headline"))
            if not self.replaying:
                symbols = [str(symbol) for symbol in (message.get("symbols") or []) if str(symbol) in IDX] or list(STOCKS)
                fair_before = self._fair_now()
                if fair_before is not None:
                    self.pending_anchor.append({
                        "symbols": symbols, "at": time.time(), "mark_before": self._mids(),
                        "fair_before": fair_before.tolist(), "headline": message.get("headline"),
                    })
        if kind in NEWS_TYPES and message.get("time"):
            apply_news(self.season, message)
            if not self.replaying:
                self.state_store.append(message)
            rnd = int(self.season.rounds[1])
            if self.tracked_round is not None and rnd != self.tracked_round:
                self.backup_open = set()
                self.round_side = {}
                self.saw_inventory = {}
            self.tracked_round = rnd
        return kind in ("advisory", "impact", "damage_report", "announcement", "storm_formed", "storm_dissipated")

    def refresh_account(self) -> None:
        payload = self.exchange.get("/api/v1/positions?include_zero=true")
        self.position = parse_positions(payload)
        self.marks = parse_marks(payload)
        self.open_orders = parse_orders(self.exchange.get("/api/v1/orders/open"))

    def _books_ready(self) -> bool:
        return any(item.get("bid") is not None for item in self.quotes.values())

    def _at_season_open(self) -> bool:
        """Round 1, before the first storm has formed: the quotes on screen are the season open."""
        if int(self.season.rounds[1]) != 1:
            return False
        if any(len(storm.obs) for storm in self.season.storms):
            return False
        return not any(event["type"] in ("storm_formed", "advisory", "damage_report") for event in self.season.events)

    def _sync_mark(self) -> bool:
        """Write the latest mids. False means the open is not locked yet.

        The value model is anchored on the season open. With no research match the anchor
        starts at the case packet's starting prices; it is replaced by the first full set of
        quotes only when those quotes are a round-1 open. In round 2 or 3 the packet table
        stays, because the quotes on screen already carry round-1 damage.
        """
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
            if self._at_season_open():
                self.opened = mids.copy()
                source = "first quotes"
            else:
                source = "case packet (not at a round-1 open)"
            self._apply_anchor()
            self.open_locked = True
            self.open_from_feed = False
            self.state_store.append({"kind": "open", "opened": self.opened.tolist(), "source": source, "at": time.time()})
            self.note("open", source=source)
        self.season.prices[1] = mids
        return True

    def _expired_ids(self, now: float) -> list[str]:
        orders = [{"id": oid} for oid in self.order_meta]
        expired = stale_order_ids(orders, self.order_meta, now, published_half_lives())
        if not expired:
            return []
        by_id = {str(order.get("id") or order.get("order_id") or ""): order for order in self.open_orders}
        out = []
        for oid in expired:
            order = by_id.get(oid)
            if order is not None and still_at_touch(order, self.quotes.get(str(order.get("symbol")))):
                self.order_meta[oid]["placed_at"] = now
                continue
            out.append(oid)
        return out

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

    def _new_season(self, clock: dict) -> None:
        """A new competition id is a new season: new feed state, new anchor, new store.

        The testnet starts the next research season under a new id while the socket stays up.
        Keeping the old season's damage reports and observations would value every name
        against the wrong storms.
        """
        self._reset_season()
        fictional = str(clock.get("fictional_time") or self.fictional_time or "")
        now = _time(fictional) if fictional else self.season.times[1]
        matched = match_season(fictional) if fictional else None
        if matched is None:
            number = 0
            self.season = blank_season(packet_open_vector(), now)
            self.opened = self.season.prices[0].copy()
            self.open_from_feed = True
            self.open_locked = False
        else:
            number, entry = matched
            self.season, self.opened = public_view(number, entry, now)
            self.open_from_feed = False
            self.open_locked = True
        try:
            self.models.library = load_library(number or None)
        except Exception as exc:
            self.note("error", where="library", message=redact(str(exc), self.exchange.key))
        self.rng = np.random.default_rng(SEED + int(number))
        self.anchor_shift = np.zeros(len(NAMES), dtype=float)
        self.pending_anchor = []
        self.seen_seq = set()
        self.announcements = []
        self.order_meta = {}
        self.flatten_order_ids = set()
        self.round_side = {}
        self.saw_inventory = {}
        self.tracked_round = None
        self.carry_done = False
        self.flattened_on_exit = False
        self.etf_done = False
        new_id = clock.get("competition_id")
        self.state_store = StateStore(STATE_DIR / f"{new_id}.jsonl") if new_id else StateStore(None)
        self.replay(self.state_store.load())
        self.note("new_season", competition_id=new_id, research_season=number, open_source="research" if matched else "packet")

    def close_round(self) -> None:
        """The round left `running`. Flatten if the flag says so; otherwise cancel rests and carry.

        Research carries positions across rounds, and the 600 research seasons show no
        systematic house move between a close and the next open (std under a dime per stock),
        so the carried book has the same expectation it had at the bell without paying two
        spreads. Resting orders are cancelled so nothing stale works into the next open.
        """
        if LIVE_FLATTEN_AT_ROUND_END:
            self.flatten_dead_targets()
            return
        if self.flattened_on_exit:
            return
        self.flattened_on_exit = True
        try:
            self.refresh_account()
        except Exception as exc:
            self.errors.append(str(exc))
            self.note("error", where="round_close", message=str(exc))
            return
        ids = self._order_ids()
        if ids:
            self._cancel_listed(ids, "round_end")
        self.note("round_close", carried={name: qty for name, qty in self.position.items() if abs(float(qty)) >= 1})

    def _begin_round(self) -> None:
        self.flattened_on_exit = False
        self.carry_done = False
        self.round_side = {}
        self.saw_inventory = {}
        self.backup_open = set()
        self.note("round_open", round=int(self.season.rounds[1]), book=self.book)

    def _apply_clock(self, clock: dict) -> None:
        new_id = clock.get("competition_id")
        if clock.get("fictional_time"):
            self.fictional_time = str(clock["fictional_time"])
        if self.competition_id and new_id and str(new_id) != str(self.competition_id):
            self.competition_id = new_id
            self._new_season(clock)
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
            self.close_round()
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
        return "trade"

    def _take_quote(self, data: dict) -> None:
        symbol = data.get("symbol")
        if not symbol:
            return
        symbol = str(symbol)
        self.quotes[symbol] = data
        if two_sided(data):
            self.last_two_sided[symbol] = {"bid": float(data["bid"]), "ask": float(data["ask"])}

    def _exit_quotes(self) -> dict[str, dict]:
        return exit_quotes(self.quotes, self.last_two_sided)

    def _flatten_capped_names(self) -> None:
        """Flatten a name the moment its exchange mark is through −150, on either book."""
        for name in names_past_cap(self.marks, self.blocked):
            self.blocked.add(name)
            self.note("name_cap", symbol=name, pnl=self.marks.get(name), book=self.book)
        quotes = self._exit_quotes()
        resting: dict[str, list[dict]] = {name: [] for name in NAMES}
        for order in self.open_orders:
            symbol = order.get("symbol")
            if symbol in resting:
                resting[symbol].append(order)
        plans = []
        missing = []
        for name in sorted(self.blocked):
            pos = float(self.position.get(name, 0.0))
            if abs(pos) < 1.0 and not resting[name]:
                self._cap_quote_warned.discard(name)
                continue
            if quotes.get(name) is None and abs(pos) >= 1.0:
                missing.append(name)
            plan = marketable_flatten(name, pos, resting[name], quotes.get(name))
            if plan:
                plans.append(plan)
        newly_missing = [name for name in missing if name not in self._cap_quote_warned]
        if newly_missing:
            self._cap_quote_warned.update(newly_missing)
            self.note("flatten", why="name_cap_no_quote", symbols=newly_missing)
        if not plans:
            return
        self.note("flatten", why="name_cap", orders=len(plans), symbols=sorted({str(order["symbol"]) for order in plans}))
        for _ in range(4):
            if self._submit_plans(plans):
                break
            quotes = self._exit_quotes()
            resting = {name: [] for name in NAMES}
            for order in self.open_orders:
                symbol = order.get("symbol")
                if symbol in resting:
                    resting[symbol].append(order)
            plans = []
            for name in sorted(self.blocked):
                plan = marketable_flatten(name, float(self.position.get(name, 0.0)), resting[name], quotes.get(name))
                if plan:
                    plans.append(plan)
            if not plans:
                break
        try:
            self.refresh_account()
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="name_cap", message=str(exc))

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
            plan = marketable_flatten(name, self.position.get(name, 0.0), resting[name], self._exit_quotes().get(name))
            if plan:
                plans.append(plan)
        target = hold_inside_band(target, current, self.models.limits, dead)
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
        self._flatten_capped_names()
        mode = self._apply_caps()
        if mode in ("flatten", "stop"):
            self.last_target = np.zeros(len(NAMES))
            self._sync_mark()
            for _ in range(4):
                plans = flatten_all_orders(self.position, self.open_orders, self._exit_quotes())
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
        self._settle_anchors()
        if not self.carry_done:
            self.carry_done = True
            if LIVE_FLATTEN_AT_ROUND_END and any(self.saw_inventory.values()):
                carried = flatten_all_orders(self.position, self.open_orders, self._exit_quotes())
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
            for name in self.blocked:
                target[NAMES.index(name)] = 0.0
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
        try:
            self.refresh_account()
        except Exception as exc:
            self.errors.append(str(exc))
            self.note("error", where="flatten", message=str(exc))
            return
        quotes = self._exit_quotes()
        open_names = [name for name in NAMES if abs(float(self.position.get(name, 0.0))) >= 1.0]
        missing = tuple(name for name in open_names if name not in quotes)
        if missing:
            if missing != self._flatten_missing:
                self._flatten_missing = missing
                self.note("flatten", why="no quotes", symbols=list(missing))
            if not any(name in quotes for name in open_names):
                return
        else:
            self._flatten_missing = None
        self._sync_mark()
        for _ in range(4):
            plans = flatten_all_orders(self.position, self.open_orders, self._exit_quotes())
            self.note("flatten", orders=len(plans), why="round_end")
            if self._submit_plans(plans):
                break
        try:
            self.refresh_account()
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.note("error", where="flatten", message=str(exc))
            return
        if missing:
            return
        resting: dict[str, list[dict]] = {name: [] for name in NAMES}
        for order in self.open_orders:
            symbol = order.get("symbol")
            if symbol in resting:
                resting[symbol].append(order)
        quotes = self._exit_quotes()
        still_open = []
        for name in NAMES:
            pos = float(self.position.get(name, 0.0))
            if abs(pos) < 1.0:
                continue
            plan = marketable_flatten(name, pos, resting[name], quotes.get(name))
            if plan and "side" in plan:
                still_open.append(name)
        if not still_open:
            self.flattened_on_exit = True

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
        try:
            self.refresh_account()
        except RuntimeError as exc:
            if not self._book_error:
                self._book_error = True
                self.note("error", where="book_log", message=str(exc))
        else:
            self._flatten_capped_names()
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
        """Once a second, log the book and flatten any name through the name cap."""
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
                            self._take_quote(data)
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
    parser = argparse.ArgumentParser(description="Trade the research book (config.LIVE_BOOK) on the exchange; the backup only after a season stop.")
    parser.add_argument("--max-seconds", type=float, default=4 * 60 * 60)
    parser.add_argument("--record", default="recording.jsonl")
    parser.add_argument("--book-log", default="book-log.jsonl")
    parser.add_argument("--state-dir", default=str(STATE_DIR), help="season state files, one per competition id, replayed on start")
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
        "book": LIVE_BOOK,
        "flatten_at_round_end": LIVE_FLATTEN_AT_ROUND_END,
        "one_entry_per_round": LIVE_ONE_ENTRY_PER_ROUND,
        "name_loss_cap": NAME_LOSS_CAP,
        "switch_loss": BOOK_SWITCH_LOSS,
        "stop_loss": SEASON_STOP_LOSS,
    }), flush=True)
    matched = match_season(str(clock.get("fictional_time")))
    now = _time(str(clock["fictional_time"]))
    if matched is None:
        number = 0
        season = blank_season(packet_open_vector(), now)
        opened = season.prices[0].copy()
        open_from_feed = True
        print("no local research season; anchor is the case packet open, replaced by the first quotes at a round-1 open", flush=True)
    else:
        number, entry = matched
        season, opened = public_view(number, entry, now)
        open_from_feed = False
        print(f"replay season {number} public feed through {clock.get('fictional_time')}", flush=True)
    models = load_models(fitted, case, skip_season=number or None)
    competition_id = clock.get("competition_id")
    store = StateStore(Path(args.state_dir) / f"{competition_id}.jsonl") if competition_id else StateStore(None)
    runner = Runner(
        Exchange(host, key), models, season, opened, Path(args.record), args.max_seconds, open_from_feed,
        book_log=Path(args.book_log), state_store=store,
    )
    runner.competition_id = competition_id
    runner.clock_state = clock.get("state")
    runner.fictional_time = str(clock.get("fictional_time") or "")
    runner.round_open = clock.get("state") == "running"
    records = store.load()
    if records:
        runner.replay(records)
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
