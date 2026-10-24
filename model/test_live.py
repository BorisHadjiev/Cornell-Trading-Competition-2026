"""Order rules for the live prototype. No network and no key."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model import live
from model import backup
from model.config import NAMES


class CredentialTests(unittest.TestCase):
    def test_environment_variable_is_enough(self) -> None:
        with patch.dict(os.environ, {"CTC_API_KEY": "from-env", "CTC_API_URL": "https://example.test"}):
            key, url = live.api_credentials()
        self.assertEqual(key, "from-env")
        self.assertEqual(url, "https://example.test")

    def test_missing_variable_and_missing_file_stops(self) -> None:
        with patch.dict(os.environ, {"CTC_API_KEY": ""}, clear=False):
            with patch.object(live, "ENV_PATH", Path("/tmp/ctc-bot-env-absent")):
                with self.assertRaises(SystemExit) as caught:
                    live.api_credentials()
        self.assertIn("CTC_API_KEY", str(caught.exception))


class OrderRuleTests(unittest.TestCase):
    def test_rests_inside_the_spread_and_crosses_only_when_the_gap_wins(self) -> None:
        # A rest sits one tick inside the house touch, so it is first in the queue for house flow.
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.3), 10.01)
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.5), 10.2)
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 9.9), 10.19)
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 9.5), 10.0)
        # A tight book gets no improvement: a half-cent price is rejected by the exchange, so the
        # rest joins the touch. Four cents of spread is the first width that earns a one-cent tick.
        self.assertEqual(live.rest_tick(10.0, 10.02), 0.0)
        self.assertEqual(live.limit_price("buy", 10.0, 10.02, 10.0), 10.0)
        self.assertEqual(live.rest_tick(10.0, 10.03), 0.0)
        self.assertEqual(live.rest_tick(10.0, 10.04), 0.01)
        self.assertEqual(live.rest_tick(10.0, 10.0), 0.0)
        # Every price the runner sends is a whole cent.
        self.assertEqual(live.round_price(49.155), 49.16)
        self.assertEqual(live.round_price(49.1549), 49.15)
        self.assertEqual(live.limit_price("sell", 49.153, 49.177, 40.0, reducing=True), 49.15)

    def test_etf_mark_sleeve_is_not_an_order(self) -> None:
        total = np.arange(len(NAMES), dtype=float)
        sleeves = {"etf_gap": np.ones(len(NAMES))}
        target = live.order_target(total, sleeves)
        self.assertTrue(np.allclose(target, total - 1))

    def test_locked_gap_is_after_both_spreads_and_dust_is_ignored(self) -> None:
        quotes = {name: {"bid": 10.0, "ask": 10.0} for name in NAMES}
        quotes["ARCH"] = {"bid": 80.0, "ask": 81.0}
        quotes["REBLD"] = {"bid": 50.0, "ask": 50.0}
        edges = live.locked_etf_edges(quotes)
        # ARCH basket is 1+1+2+1+2+1 = 8 names at 10, so the bid is even with the asks.
        self.assertAlmostEqual(edges["ARCH"]["create"], 0.0)
        self.assertIsNone(live.best_locked_conversion(edges))
        quotes["ARCH"]["bid"] = 80.5
        choice = live.best_locked_conversion(live.locked_etf_edges(quotes))
        self.assertEqual(choice[0], "ARCH")
        self.assertEqual(choice[1], "create")
        self.assertGreater(choice[2], 0.01)

    def test_announcement_stand_down(self) -> None:
        now = 1_000.0
        self.assertEqual(live.standing_down([], now), set())
        whole = live.standing_down([{"received_at": now - 10, "symbols": []}], now)
        self.assertIsNone(whole)
        named = live.standing_down([{"received_at": now - 10, "symbols": ["PORT"]}], now)
        self.assertEqual(named, {"PORT"})
        stale = live.standing_down([{"received_at": now - 90, "symbols": ["PORT"]}], now)
        self.assertEqual(stale, set())

    def test_a_ping_timeout_cannot_drop_the_last_two_minutes(self) -> None:
        self.assertGreaterEqual(live.PING_TIMEOUT_SECONDS, 120.0)
        self.assertTrue(live.session_continues("running"))
        self.assertTrue(live.session_continues(None))
        self.assertTrue(live.session_continues("ready"))
        self.assertFalse(live.session_continues("draft"))
        self.assertFalse(live.session_continues("ended"))

    def test_action_budget_stops_under_the_team_cap(self) -> None:
        budget = live.ActionBudget(cap=2)
        self.assertTrue(budget.take(0.0))
        self.assertTrue(budget.take(1.0))
        self.assertFalse(budget.take(2.0))
        self.assertTrue(budget.take(61.0))

    def test_a_reducing_order_is_marketable_and_an_add_still_respects_the_gate(self) -> None:
        # The gap does not beat the spread, so an add rests and a reduction crosses.
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 10.1), 10.19)
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 10.1, reducing=True), 10.0)
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.1), 10.01)
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.1, reducing=True), 10.2)

        mat = NAMES.index("MAT")
        aqua = NAMES.index("AQUA")
        fair = np.full(len(NAMES), 10.1)
        limits = np.full(len(NAMES), 700.0)
        quotes = {name: {"bid": 10.0, "ask": 10.2} for name in NAMES}
        position = {name: 0.0 for name in NAMES}

        shrink = np.zeros(len(NAMES))
        shrink[mat] = 100.0
        position["MAT"] = 506.0
        sell = [order for order in live.decide_orders(shrink, fair, position, [], quotes, limits, set()) if order["symbol"] == "MAT"]
        self.assertEqual(len(sell), 1)
        self.assertEqual(sell[0]["side"], "sell")
        self.assertTrue(sell[0]["reducing"])
        self.assertEqual(sell[0]["price"], 10.0)
        self.assertEqual(sell[0]["quantity"], 406)

        opened = np.zeros(len(NAMES))
        opened[mat] = 100.0
        position["MAT"] = 0.0
        buy = [order for order in live.decide_orders(opened, fair, position, [], quotes, limits, set()) if order["symbol"] == "MAT"]
        self.assertEqual(buy[0]["side"], "buy")
        self.assertFalse(buy[0]["reducing"])
        self.assertEqual(buy[0]["price"], 10.01)

        fair_cross = fair.copy()
        fair_cross[mat] = 10.5
        crossed = [order for order in live.decide_orders(opened, fair_cross, position, [], quotes, limits, set()) if order["symbol"] == "MAT"]
        self.assertEqual(crossed[0]["price"], 10.2)

        cover_target = np.zeros(len(NAMES))
        cover_target[aqua] = -100.0
        position["AQUA"] = -500.0
        cover = [order for order in live.decide_orders(cover_target, fair, position, [], quotes, limits, set()) if order["symbol"] == "AQUA"]
        self.assertEqual(cover[0]["side"], "buy")
        self.assertTrue(cover[0]["reducing"])
        self.assertEqual(cover[0]["price"], 10.2)

        # The shares that close the long cross. The shares that open a short still rest.
        flip = np.zeros(len(NAMES))
        flip[mat] = -30.0
        position["MAT"] = 100.0
        position["AQUA"] = 0.0
        legs = [order for order in live.decide_orders(flip, fair, position, [], quotes, limits, set()) if order["symbol"] == "MAT"]
        self.assertEqual(
            [(order["quantity"], order["price"], order["reducing"]) for order in legs],
            [(100, 10.0, True), (30, 10.19, False)],
        )

    def test_working_leaves_shrink_the_next_order(self) -> None:
        limits = np.full(len(NAMES), 700.0)
        target = np.zeros(len(NAMES))
        target[NAMES.index("PORT")] = -50.0
        fair = np.full(len(NAMES), 10.1)
        quotes = {"PORT": {"bid": 10.0, "ask": 10.2}}
        resting = [{
            "symbol": "PORT", "side": "sell", "quantity": 40, "filled_quantity": 10,
            "price": 10.2, "id": "keep",
        }]
        planned = live.decide_orders(target, fair, {"PORT": 0.0}, resting, quotes, limits, set())
        self.assertEqual([(item["quantity"], item["cancel"]) for item in planned], [(20, [])])
        self.assertEqual(live.execution_gap(-50, 0, -30), -20)

    def test_a_wrong_price_is_cancelled_and_not_counted(self) -> None:
        limits = np.full(len(NAMES), 700.0)
        target = np.zeros(len(NAMES))
        target[NAMES.index("PORT")] = -50.0
        fair = np.full(len(NAMES), 10.1)
        quotes = {"PORT": {"bid": 10.0, "ask": 10.2}}
        resting = [{"symbol": "PORT", "side": "sell", "quantity": 40, "price": 10.0, "id": "old"}]
        planned = live.decide_orders(target, fair, {"PORT": 0.0}, resting, quotes, limits, set())
        self.assertEqual(planned[0]["quantity"], 50)
        self.assertEqual(planned[0]["price"], 10.19)
        self.assertEqual(planned[0]["cancel"], ["old"])

    def test_a_409_is_a_fill(self) -> None:
        self.assertEqual(live.after_cancel(["filled"]), "resync")
        self.assertEqual(live.after_cancel(["cancelled"]), "resize")
        self.assertEqual(live.after_cancel([]), "send")
        filled = live.ExchangeError("DELETE", "/api/v1/orders/1", 409, "already filled")
        self.assertEqual(live.cancel_result_from_error(filled), "filled")
        other = live.ExchangeError("DELETE", "/api/v1/orders/1", 400, "no")
        self.assertEqual(live.cancel_result_from_error(other), "failed")

    def test_the_book_log_has_bid_ask_mark_target_position_and_working(self) -> None:
        quotes = {"PORT": {"bid": 10.0, "ask": 10.2, "mark": 10.05}}
        target = np.zeros(len(NAMES))
        target[0] = -20.0
        orders = [{"symbol": "PORT", "side": "sell", "quantity": 20, "filled_quantity": 5, "price": 10.2}]
        row = live.book_snapshot(quotes, target, {"PORT": -5.0}, orders)
        self.assertEqual(row["PORT"], {
            "bid": 10.0,
            "ask": 10.2,
            "house_mark": 10.05,
            "target": -20.0,
            "position": -5.0,
            "working": -15.0,
        })
        self.assertIsNone(row["ISLE"]["house_mark"])
        self.assertIsNone(row["ISLE"]["bid"])

    def test_live_book_full_is_the_research_book_and_location_is_the_residual(self) -> None:
        from model.backtest import targets_from_state
        from model.config import CLASSES

        n = len(NAMES)
        aqua = NAMES.index("AQUA")
        mat = NAMES.index("MAT")
        haven = NAMES.index("HAVEN")
        limits = np.full(n, 1000.0)
        mark = np.full(n, 10.0)
        mark[NAMES.index("ARCH")] = 80.0
        mark[NAMES.index("REBLD")] = 50.0
        severity = np.zeros(n)
        location_full = np.zeros(n)
        severity[aqua] = 100.0
        location_full[aqua] = 130.0
        severity[mat] = 200.0
        location_full[mat] = 240.0
        severity[haven] = 80.0
        location_full[haven] = 105.0
        known_full = location_full.copy()
        known_full[aqua] = 230.0

        def shares(_models, _opened, _mark, damage, _outage, _loss, _fraction):
            if damage == "sev":
                return severity.copy()
            if damage == "loc":
                return location_full.copy()
            if damage == "known":
                return known_full.copy()
            return np.zeros(n)

        class Value:
            def predict(self, *_args):
                return mark.copy()

            def statistical(self, *_args):
                return mark.copy()

        models = type("Models", (), {"value": Value(), "limits": limits})()

        class Season:
            times = [0]
            events = []

        probs = np.zeros(len(CLASSES))
        probs[CLASSES.index("stall")] = 0.1
        probs[CLASSES.index("eyewall")] = 0.1
        state = {
            "fraction": 1.0,
            "mark": mark,
            "open": np.zeros(n),
            "sev": ("sev", 0.0, 0.0),
            "loc": ("loc", 0.0, 0.0),
            "rev": ("rev", 0.0, 0.0),
            "known": ("known", 0.0, 0.0),
            "prelim": False,
            "final": True,
            "location": True,
            "probs": probs,
            "near": True,
            "post_turn": True,
            "round": 1,
        }
        with (
            patch("model.backtest._value_shares", shares),
            patch("model.backtest._scenario_value", lambda *_args: mark.copy()),
        ):
            research_total, research = targets_from_state(dict(state), models, Season(), 0)
            with patch.object(live, "LIVE_BOOK", "full"):
                full_total, full = live.targets_for_live(dict(state), models, Season(), 0)
            with patch.object(live, "LIVE_BOOK", "location"):
                loc_total, loc = live.targets_for_live(dict(state), models, Season(), 0)
            with patch.object(live, "LIVE_BOOK", "no_severity"):
                informed_total, _ = live.targets_for_live(dict(state), models, Season(), 0)
                uninformed = dict(state)
                uninformed.update(location=False, prelim=False, final=False)
                flat_total, _ = live.targets_for_live(uninformed, models, Season(), 0)

        self.assertNotIn("live_book", state)
        # The default live book sends exactly what the holdout scored.
        self.assertTrue(np.allclose(full_total, research_total))
        self.assertAlmostEqual(research["severity"][aqua], 100.0)
        self.assertAlmostEqual(research["location"][aqua], 30.0)
        self.assertAlmostEqual(research["known_damage"][aqua], 35.0)
        # "location" keeps only the residual on the physical names: 130 - 100 on AQUA, nothing on MAT or HAVEN.
        self.assertAlmostEqual(loc_total[aqua], 30.0)
        self.assertEqual(loc_total[mat], 0.0)
        self.assertEqual(loc_total[haven], 0.0)
        self.assertAlmostEqual(loc["location"][aqua], 30.0)
        # "no_severity" is the full book once a view exists, and flat before one does.
        self.assertTrue(np.allclose(informed_total, research_total))
        self.assertTrue(np.allclose(flat_total, 0.0))

    def test_a_resting_order_is_cancelled_after_its_house_half_life(self) -> None:
        half = live.published_half_lives()
        self.assertEqual(half["official_advisory"], 20.0)
        self.assertEqual(half["wire"], 22.0)
        self.assertEqual(half["impact"], 26.0)
        self.assertEqual(half["preliminary"], 4.0)
        self.assertEqual(half["final"], 4.0)
        orders = [{"id": "exit", "symbol": "MAT"}]
        floor = live.MIN_REST_SECONDS
        self.assertEqual(floor, 20.0)
        # Damage-report half-lives (4 s) are floored: a 4-second cancel-and-replace loop on ten
        # names would burn the team's 180 actions a minute.
        final = {"exit": {"placed_at": 100.0, "event_key": "final"}}
        self.assertEqual(live.stale_order_ids(orders, final, 100.0 + floor, half), [])
        self.assertEqual(live.stale_order_ids(orders, final, 100.01 + floor, half), ["exit"])
        official = {"exit": {"placed_at": 100.0, "event_key": "official_advisory"}}
        self.assertEqual(live.stale_order_ids(orders, official, 120.0, half), [])
        self.assertEqual(live.stale_order_ids(orders, official, 120.01, half), ["exit"])
        wire = {"exit": {"placed_at": 100.0, "event_key": "wire"}}
        self.assertEqual(live.stale_order_ids(orders, wire, 122.0, half), [])
        self.assertEqual(live.stale_order_ids(orders, wire, 122.01, half), ["exit"])
        impact = {"exit": {"placed_at": 100.0, "event_key": "impact"}}
        self.assertEqual(live.stale_order_ids(orders, impact, 126.0, half), [])
        self.assertEqual(live.stale_order_ids(orders, impact, 126.01, half), ["exit"])
        preliminary = {"exit": {"placed_at": 100.0, "event_key": "preliminary"}}
        self.assertEqual(live.stale_order_ids(orders, preliminary, 100.0 + floor, half), [])
        self.assertEqual(live.stale_order_ids(orders, preliminary, 100.01 + floor, half), ["exit"])
        untouched = {"exit": {"placed_at": 100.0, "event_key": None}}
        self.assertEqual(live.stale_order_ids(orders, untouched, 1000.0, half), [])

    def test_a_name_does_not_add_or_reverse_after_it_opens(self) -> None:
        plans = [{"symbol": "PORT", "side": "buy", "quantity": 10}]
        with patch.object(live, "LIVE_ONE_ENTRY_PER_ROUND", False):
            self.assertEqual(live.screen_round_orders(plans, {"PORT": 50.0}, {"PORT": 1}, {"PORT": True}), plans)
        with patch.object(live, "LIVE_ONE_ENTRY_PER_ROUND", True):
            self.assertEqual(live.screen_round_orders(plans, {"PORT": 50.0}, {"PORT": 1}, {"PORT": True}), [])
        self.assertTrue(live.allows_round_order(0, False, 0, "buy", 10))
        self.assertTrue(live.allows_round_order(1, False, 0, "buy", 10))
        self.assertFalse(live.allows_round_order(1, False, 0, "sell", 10))
        self.assertFalse(live.allows_round_order(1, True, 50, "buy", 10))
        self.assertTrue(live.allows_round_order(1, True, 50, "sell", 40))
        self.assertFalse(live.allows_round_order(1, True, 50, "sell", 80))
        self.assertFalse(live.allows_round_order(-1, True, 0, "buy", 10))
        self.assertTrue(live.allows_round_order(-1, True, -140, "buy", 140))

    def test_an_empty_quote_still_crosses_the_last_two_sided_touch(self) -> None:
        last = {"ISLE": {"bid": 10.0, "ask": 10.4}, "GRID": {"bid": 19.8, "ask": 20.0}}
        current = {"ISLE": {"bid": None, "ask": None}, "GRID": {}}
        quotes = live.exit_quotes(current, last)
        position = {name: 0.0 for name in NAMES}
        position["ISLE"] = -30.0
        position["GRID"] = 40.0
        plans = {order["symbol"]: order for order in live.flatten_all_orders(position, [], quotes)}
        self.assertEqual(plans["ISLE"]["side"], "buy")
        self.assertEqual(plans["ISLE"]["price"], 10.4)
        self.assertEqual(plans["ISLE"]["quantity"], 30)
        self.assertTrue(plans["ISLE"]["reducing"])
        self.assertEqual(plans["GRID"]["side"], "sell")
        self.assertEqual(plans["GRID"]["price"], 19.8)
        self.assertEqual(plans["GRID"]["quantity"], 40)
        self.assertEqual(live.exit_quotes({"AQUA": {"bid": None, "ask": None}}, {}), {})

    def test_the_name_cap_does_not_wait_for_the_backup_book(self) -> None:
        with patch.object(backup, "NAME_LOSS_CAP", 150.0):
            self.assertEqual(live.names_past_cap({"PORT": -150.0, "ISLE": -149.0}, set()), ["PORT"])
            self.assertEqual(live.names_past_cap({"PORT": -267.67}, {"PORT"}), [])
        # The shipped default has no per-name cap: a 150-dollar bound fires on advisory noise.
        with patch.object(backup, "NAME_LOSS_CAP", None):
            self.assertEqual(live.names_past_cap({"PORT": -5000.0}, set()), [])

    def test_a_round_end_flattens_a_name_the_target_still_wants(self) -> None:
        position = {name: 0.0 for name in NAMES}
        position["HAVEN"] = -140
        quotes = {name: {"bid": 33.5, "ask": 33.82} for name in NAMES}
        plans = live.flatten_all_orders(position, [], quotes)
        haven = [order for order in plans if order["symbol"] == "HAVEN"]
        self.assertEqual(len(haven), 1)
        self.assertEqual(haven[0]["side"], "buy")
        self.assertEqual(haven[0]["quantity"], 140)
        self.assertEqual(haven[0]["price"], 33.82)
        self.assertTrue(haven[0]["reducing"])


class StateAndCatchUpTests(unittest.TestCase):
    def _season(self):
        now = np.datetime64("2026-11-02T09:30:00", "s")
        return live.blank_season(live.packet_open_vector(), now)

    def test_snapshot_fills_earlier_rounds_damage_and_the_latest_advisory(self) -> None:
        season = self._season()
        snap = {
            "type": "snapshot", "round": 2, "storm": "Beryl", "time": "2026-11-03T10:30:00", "phase": "storm",
            "last_seq": 400,
            "obs": {"age": 60, "x": 480.0, "y": 640.0, "vx": 0.0, "vy": -1.8, "radius": 81.0, "intensity": 1.05},
            "track": [[52, 480.0, 654.4, 80.8], [56, 480.0, 647.2, 80.9], [60, 480.0, 640.0, 81.0]],
            "advisory": {"round": 2, "storm": "Beryl", "time": "2026-11-03T10:15:00", "source": "official", "number": 1,
                         "age": 30, "outlook": "direct", "hurricane_conditions": {"cascadia_prime": 0.7}, "cone": []},
            "impacts": [],
            "damage": [
                {"type": "damage_report", "round": 1, "storm": "Arlo", "time": "2026-11-02T14:15:00", "stage": "preliminary",
                 "damage_pct": {"PORT": 30.0}, "industry_loss_musd": 500.0},
                {"type": "damage_report", "round": 1, "storm": "Arlo", "time": "2026-11-02T15:40:00", "stage": "final",
                 "damage_pct": {"PORT": 28.0}, "industry_loss_musd": 480.0},
            ],
            "headlines": [],
        }
        added = live.apply_snapshot(season, snap)
        self.assertEqual(added["damage"], 2)
        self.assertEqual(added["advisory"], 1)
        self.assertEqual(added["obs"], 1)
        self.assertEqual(added["track"], 2)
        self.assertEqual(int(season.rounds[1]), 2)
        finals = [e for e in season.events if e["type"] == "damage_report" and e["stage"] == "final"]
        self.assertEqual(finals[0]["damage_pct"]["PORT"], 28.0)
        storm = season.storms[1]
        self.assertEqual([int(a) for a in storm.obs[:, 0]], [52, 56, 60])
        # Track rows carry the interpolated intensity and a finite-difference velocity.
        self.assertAlmostEqual(float(storm.obs[0, 6]), 1.05)
        self.assertAlmostEqual(float(storm.obs[0, 4]), -1.8, places=6)
        # Replaying the same snapshot adds nothing.
        again = live.apply_snapshot(season, snap)
        self.assertEqual(sum(again.values()), 0)

    def test_track_fills_a_gap_and_the_exposure_counts_the_gap_seconds(self) -> None:
        from model.damage import row_steps
        season = self._season()
        for age in (0, 2, 4):
            live.apply_news(season, {"type": "storm", "round": 1, "storm": "Arlo", "time": f"2026-11-02T10:0{age // 2}:00",
                                     "age": age, "x": 480.0, "y": 740.0 - age, "vx": 0.0, "vy": -1.0, "radius": 80.0, "intensity": 1.0})
        live.apply_news(season, {"type": "storm", "round": 1, "storm": "Arlo", "time": "2026-11-02T10:10:00",
                                 "age": 20, "x": 480.0, "y": 720.0, "vx": 0.0, "vy": -1.0, "radius": 80.0, "intensity": 1.2})
        storm = season.storms[0]
        self.assertEqual([int(a) for a in storm.obs[:, 0]], [0, 2, 4, 20])
        self.assertEqual(row_steps(storm.obs, 2.0).tolist(), [2.0, 2.0, 2.0, 16.0])
        added = live.apply_track(season, {"type": "track", "round": 1, "storm": "Arlo", "time": "2026-11-02T10:10:00", "age": 20,
                                          "points": [[0, 480.0, 740.0, 80.0], [4, 480.0, 736.0, 80.0], [8, 480.0, 732.0, 80.0],
                                                     [12, 480.0, 728.0, 80.0], [16, 480.0, 724.0, 80.0], [20, 480.0, 720.0, 80.0]]})
        self.assertEqual(added, 3)
        self.assertEqual([int(a) for a in storm.obs[:, 0]], [0, 2, 4, 8, 12, 16, 20])
        self.assertEqual(row_steps(storm.obs, 2.0).tolist(), [2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])
        self.assertAlmostEqual(float(storm.obs[4, 6]), 1.1)  # intensity interpolated between 1.0 at age 4 and 1.2 at age 20

    def test_announcement_anchor_is_the_house_move_the_model_did_not_make(self) -> None:
        fair_before = np.zeros(len(NAMES))
        fair_after = np.zeros(len(NAMES))
        gen = NAMES.index("GEN")
        port = NAMES.index("PORT")
        fair_after[gen] = 0.5   # the storm model moved GEN by 0.5 in the window on its own
        fair_after[port] = -1.0
        before = {"GEN": 40.0, "PORT": 72.0, "ARCH": 289.0}
        after = {"GEN": 46.5, "PORT": 71.0, "ARCH": 295.0}
        shift = live.anchor_shift_for(before, after, fair_before, fair_after, ["GEN", "PORT", "ARCH", "ISLE"])
        self.assertAlmostEqual(shift["GEN"], 6.0)      # 6.5 house move, 0.5 explained by the storm
        self.assertAlmostEqual(shift["PORT"], 0.0)     # the house move was entirely the storm
        self.assertNotIn("ARCH", shift)                # ETFs are valued from the basket
        self.assertNotIn("ISLE", shift)                # not quoted on both sides of the window

    def test_state_store_replays_open_anchor_and_news(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            store = live.StateStore(Path(folder) / "c1.jsonl")
            store.append({"kind": "open", "opened": live.packet_open_vector().tolist()})
            store.append({"type": "damage_report", "round": 1, "storm": "Arlo", "time": "2026-11-02T15:40:00", "stage": "final",
                          "damage_pct": {"PORT": 28.0}, "industry_loss_musd": 480.0, "seq": 290})
            store.append({"kind": "anchor", "shift": [0.0] * (len(NAMES) - 1) + [0.0]})
            records = store.load()
            self.assertEqual(len(records), 3)
            season = self._season()
            runner = live.Runner.__new__(live.Runner)
            runner.season = season
            runner.opened = np.zeros(len(NAMES))
            runner.anchor_shift = np.zeros(len(NAMES))
            runner.open_locked = False
            runner.open_from_feed = True
            runner.seen_seq = set()
            runner.justifying_event = None
            runner.announcements = []
            runner.pending_anchor = []
            runner.tracked_round = None
            runner.backup_open = set()
            runner.round_side = {}
            runner.saw_inventory = {}
            runner.state_store = live.StateStore(None)
            runner.exchange = type("X", (), {"key": ""})()
            runner.log = []
            runner.replay(records)
            self.assertTrue(runner.open_locked)
            self.assertAlmostEqual(float(runner.opened[NAMES.index("PORT")]), 72.48)
            self.assertAlmostEqual(float(season.prices[0][NAMES.index("PORT")]), 72.48)
            finals = [e for e in season.events if e["type"] == "damage_report"]
            self.assertEqual(len(finals), 1)
            self.assertIn(290, runner.seen_seq)

    def test_a_rest_still_at_the_touch_is_not_stale_and_the_floor_is_twenty_seconds(self) -> None:
        quote = {"bid": 10.0, "ask": 10.2}
        self.assertTrue(live.still_at_touch({"side": "buy", "price": 10.01}, quote))
        self.assertFalse(live.still_at_touch({"side": "buy", "price": 9.95}, quote))
        self.assertFalse(live.still_at_touch({"side": "sell", "price": 10.01}, quote))
        self.assertFalse(live.still_at_touch({"side": "buy", "price": 10.01}, {"bid": None, "ask": 10.2}))
        half = live.published_half_lives()
        orders = [{"id": "x", "symbol": "MAT"}]
        meta = {"x": {"placed_at": 100.0, "event_key": "final"}}
        # The fitted half-life after a final report is 4 s; the rest still lives MIN_REST_SECONDS.
        self.assertEqual(live.stale_order_ids(orders, meta, 104.1, half), [])
        self.assertEqual(live.stale_order_ids(orders, meta, 100.0 + live.MIN_REST_SECONDS + 0.01, half), ["x"])

    def test_a_crossing_reduction_takes_the_touch_and_rests_the_rest(self) -> None:
        limits = np.full(len(NAMES), 700.0)
        fair = np.full(len(NAMES), 10.1)
        target = np.zeros(len(NAMES))
        mat = NAMES.index("MAT")
        position = {name: 0.0 for name in NAMES}
        position["MAT"] = 100.0           # long 100, target 0: a 100-share sell that would cross
        quotes = {"MAT": {"bid": 10.0, "ask": 10.2, "bid_size": 30, "ask_size": 50}}
        with patch.object(live, "LIVE_CROSS_AT_TOUCH_DEPTH", True):
            legs = [o for o in live.decide_orders(target, fair, position, [], quotes, limits, set()) if o["symbol"] == "MAT"]
        self.assertEqual([(o["quantity"], o["price"], o["reducing"]) for o in legs], [(30, 10.0, True), (70, 10.19, True)])
        # A resting remainder already working at that price is kept, not replaced.
        resting = [{"symbol": "MAT", "side": "sell", "quantity": 70, "price": 10.19, "id": "keep"}]
        with patch.object(live, "LIVE_CROSS_AT_TOUCH_DEPTH", True):
            again = [o for o in live.decide_orders(target, fair, position, resting, quotes, limits, set()) if o["symbol"] == "MAT"]
        self.assertEqual([(o["quantity"], o["price"], o.get("cancel")) for o in again], [(30, 10.0, [])])
        # Without a shown size, or with the flag off, the whole reduction crosses as before.
        with patch.object(live, "LIVE_CROSS_AT_TOUCH_DEPTH", False):
            plain = [o for o in live.decide_orders(target, fair, position, [], quotes, limits, set()) if o["symbol"] == "MAT"]
        self.assertEqual([(o["quantity"], o["price"]) for o in plain], [(100, 10.0)])
        self.assertIsNone(live.touch_depth({"bid": 10.0, "ask": 10.2}, "sell"))

    def test_reconnect_backoff_doubles_and_waits_a_minute_when_bots_are_disabled(self) -> None:
        self.assertEqual(live.next_backoff(0.0, "no close frame received or sent"), 2.0)
        self.assertEqual(live.next_backoff(2.0, "no close frame"), 4.0)
        self.assertEqual(live.next_backoff(40.0, "no close frame"), 60.0)
        self.assertEqual(live.next_backoff(60.0, "no close frame"), 60.0)
        self.assertEqual(live.next_backoff(0.0, "server rejected WebSocket connection: HTTP 403"), 60.0)
        self.assertEqual(live.next_backoff(2.0, "GET /api/v1/competitions/time -> 403 Bots disabled"), 60.0)

    def test_the_per_second_tick_survives_a_network_error(self) -> None:
        class FlakyExchange:
            key = "k"
            def get(self, path):
                raise OSError("connection reset")
        runner = live.Runner.__new__(live.Runner)
        runner.exchange = FlakyExchange()
        runner._clock_error = False
        runner._book_error = False
        runner._halt_state = None
        runner.round_open = False
        runner.book = live.MAIN
        runner.book_log_path = None
        runner.log = []
        notes = []
        runner.note = lambda kind, **f: notes.append((kind, f.get("where")))
        runner._second()          # must not raise
        runner._second()          # and must not log the same error twice
        self.assertEqual(notes, [("error", "clock"), ("error", "book_log")])

    def test_live_size_fraction_scales_the_target_only(self) -> None:
        target = np.full(len(NAMES), 100.0)
        self.assertTrue(np.allclose(live.scale_target(target, 0.5), 50.0))
        self.assertTrue(np.array_equal(live.scale_target(target, None), target))
        self.assertTrue(np.array_equal(live.scale_target(target, 1.0), target))

    def test_first_competition_id_after_a_gap_start_builds_the_season(self) -> None:
        runner = live.Runner.__new__(live.Runner)
        runner.competition_id = None
        runner.fictional_time = None
        runner.clock_state = None
        runner.round_open = False
        runner.flattened_on_exit = True
        built = []
        runner._new_season = lambda clock: built.append(clock.get("competition_id"))
        runner.note = lambda *a, **k: None
        runner._begin_round = lambda: None
        runner.close_round = lambda: None
        runner._apply_clock({"competition_id": "abc", "state": "ready", "fictional_time": "1777-06-11T09:30:00"})
        self.assertEqual(built, ["abc"])
        self.assertEqual(runner.competition_id, "abc")
        runner._apply_clock({"competition_id": "abc", "state": "running", "fictional_time": "1777-06-11T09:31:00"})
        self.assertEqual(built, ["abc"])

    def test_edge_floor_blocks_small_opens_and_turns_a_small_reversal_into_a_flatten(self) -> None:
        port, isle, aqua, grid = (NAMES.index(n) for n in ("PORT", "ISLE", "AQUA", "GRID"))
        current = np.zeros(len(NAMES)); target = np.zeros(len(NAMES)); fair = np.zeros(len(NAMES))
        quotes = {n: {"bid": 50.0, "ask": 50.2} for n in NAMES}          # spread 0.20, floor max(1.0, 0.40) = 1.0
        target[port], fair[port] = 100.0, 50.6                               # open long on a 0.5 edge: blocked
        target[isle], fair[isle] = 100.0, 51.5                               # open long on a 1.4 edge: allowed
        current[aqua], target[aqua], fair[aqua] = 80.0, -60.0, 49.7          # reverse on a 0.4 edge: reduce to flat
        current[grid], target[grid], fair[grid] = 80.0, 120.0, 50.3          # same-side add: never gated
        out = live.apply_edge_floor(target, current, fair, quotes, dollars=1.0, spreads=2.0)
        self.assertEqual(float(out[port]), 0.0)
        self.assertEqual(float(out[isle]), 100.0)
        self.assertEqual(float(out[aqua]), 0.0)
        self.assertEqual(float(out[grid]), 120.0)
        # A wide spread raises the floor: two spreads of 0.80 is 1.60, so a 1.4 edge no longer opens.
        wide = {n: {"bid": 50.0, "ask": 50.8} for n in NAMES}
        self.assertEqual(float(live.apply_edge_floor(target, current, fair, wide, 1.0, 2.0)[isle]), 0.0)
        # An open whose edge sign disagrees with the side does not go out; a reduction is untouched.
        target[port], fair[port] = -100.0, 52.0
        current[isle], target[isle] = 100.0, 40.0
        out = live.apply_edge_floor(target, current, fair, quotes, 1.0, 2.0)
        self.assertEqual(float(out[port]), 0.0)
        self.assertEqual(float(out[isle]), 40.0)
        self.assertIsNone(live.edge_floor(quotes["PORT"], None, None))
        self.assertTrue(np.array_equal(live.apply_edge_floor(target, current, fair, quotes, None, None), target))

    def test_per_name_deadband_holds_small_changes_and_lets_a_dead_name_flatten(self) -> None:
        limits = np.full(len(NAMES), 100.0)
        current = np.zeros(len(NAMES))
        target = np.zeros(len(NAMES))
        port, isle, aqua = NAMES.index("PORT"), NAMES.index("ISLE"), NAMES.index("AQUA")
        current[port], target[port] = 50.0, 60.0      # inside a 0.15 band: hold
        current[isle], target[isle] = 50.0, 80.0      # outside: trade
        current[aqua], target[aqua] = 5.0, 0.0        # dead name: still flattens
        held = live.hold_inside_band(target, current, limits, {"AQUA"}, band=0.15)
        self.assertEqual(float(held[port]), 50.0)
        self.assertEqual(float(held[isle]), 80.0)
        self.assertEqual(float(held[aqua]), 0.0)
        self.assertTrue(np.array_equal(live.hold_inside_band(target, current, limits, set(), band=None), target))

    def test_the_season_open_is_taken_from_quotes_only_at_a_round_one_open(self) -> None:
        season = self._season()
        runner = live.Runner.__new__(live.Runner)
        runner.season = season
        self.assertTrue(runner._at_season_open())
        live.apply_news(season, {"type": "storm_formed", "round": 1, "storm": "Arlo", "time": "2026-11-02T10:00:00", "headline": "formed"})
        self.assertFalse(runner._at_season_open())
        season2 = self._season()
        runner.season = season2
        season2.rounds[1] = 2
        self.assertFalse(runner._at_season_open())


    def test_one_decision_replaces_a_stale_rest_on_every_name(self) -> None:
        # Each cancel ends a pass and re-reads the account. With every name resting at an old
        # price, four passes sent only three names; the pass count must cover all of them.
        class Exchange:
            key = "k"

            def __init__(self) -> None:
                self.orders = {
                    f"o{i}": {"id": f"o{i}", "symbol": name, "side": "sell", "quantity": 20, "price": 9.0}
                    for i, name in enumerate(NAMES)
                }
                self.placed = 0

            def get(self, path: str) -> dict:
                if path.startswith("/api/v1/positions"):
                    return {"positions": [{"symbol": name, "quantity": 0} for name in NAMES]}
                return {"orders": list(self.orders.values())}

            def delete(self, path: str) -> dict:
                self.orders.pop(path.rsplit("/", 1)[1])
                return {}

            def post(self, path: str, payload: dict) -> dict:
                self.placed += 1
                oid = f"n{self.placed}"
                self.orders[oid] = {**payload, "id": oid}
                return {"id": oid}

        runner = object.__new__(live.Runner)
        runner.exchange = Exchange()
        runner.budget = live.ActionBudget()
        runner.order_meta, runner.flatten_order_ids, runner.round_side = {}, set(), {}
        runner.orders_sent, runner.errors, runner.log = [], [], []
        runner.justifying_event = "final"
        runner.note = lambda *args, **kwargs: None
        limits = np.full(len(NAMES), 500.0)
        target = np.full(len(NAMES), -100.0)
        fair = np.full(len(NAMES), 10.0)
        quotes = {name: {"bid": 10.0, "ask": 10.2} for name in NAMES}
        runner.refresh_account()
        for _ in range(live.PLAN_PASSES):
            plans = live.decide_orders(target, fair, runner.position, runner.open_orders, quotes, limits, set())
            if runner._submit_plans(plans):
                break
        fresh = {order["symbol"] for order in runner.exchange.orders.values() if order["id"].startswith("n")}
        self.assertEqual(fresh, set(NAMES))
        self.assertFalse(any(order["id"].startswith("o") for order in runner.exchange.orders.values()))

if __name__ == "__main__":
    unittest.main()
