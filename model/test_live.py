"""Order rules for the live prototype. No network and no key."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model import live
from model.config import NAME_LOSS_CAP, NAMES


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
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.3), 10.0)
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.5), 10.2)
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 9.9), 10.2)
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 9.5), 10.0)

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
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 10.1), 10.2)
        self.assertEqual(live.limit_price("sell", 10.0, 10.2, 10.1, reducing=True), 10.0)
        self.assertEqual(live.limit_price("buy", 10.0, 10.2, 10.1), 10.0)
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
        self.assertEqual(buy[0]["price"], 10.0)

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
            [(100, 10.0, True), (30, 10.2, False)],
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
        self.assertEqual(planned[0]["price"], 10.2)
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

    def test_live_severity_sleeve_is_zero_and_research_is_not(self) -> None:
        from model.backtest import targets_from_state
        from model.config import CLASSES

        n = len(NAMES)
        aqua = NAMES.index("AQUA")
        mat = NAMES.index("MAT")
        gen = NAMES.index("GEN")
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
            "probs": probs,
            "near": True,
            "post_turn": True,
            "round": 1,
        }
        from model.portfolio import disagree_scale as real_disagree

        seen = []

        def spy(shares_in, physics, statistical, mark_in):
            seen.append(float(shares_in[mat]))
            return real_disagree(shares_in, physics, statistical, mark_in)

        with (
            patch("model.backtest._value_shares", shares),
            patch("model.backtest._scenario_value", lambda *_args: mark.copy()),
            patch("model.portfolio.disagree_scale", spy),
        ):
            _total, research = targets_from_state(state, models, Season(), 0)
            pre = dict(state)
            pre["fraction"] = 0.35
            pre["post_turn"] = False
            _pre_total, early = live.targets_for_live(pre, models, Season(), 0)
            _post_total, late = live.targets_for_live(dict(state), models, Season(), 0)

        self.assertNotIn("zero_severity", state)
        self.assertNotIn("zero_severity", pre)
        self.assertNotIn("location_only", state)
        self.assertNotIn("location_only", pre)
        self.assertEqual(seen, [240.0, 0.0, 0.0])
        self.assertAlmostEqual(research["severity"][aqua], 100.0)
        self.assertAlmostEqual(research["location"][aqua], 30.0)
        self.assertAlmostEqual(research["known_damage"][aqua], 35.0)
        self.assertAlmostEqual(research["composition"][mat], 240.0)
        self.assertAlmostEqual(research["composition"][gen], 30.0)
        self.assertAlmostEqual(research["insurer_kink"][haven], 92.5)
        self.assertAlmostEqual(_pre_total[aqua], 30.0)
        self.assertEqual(_pre_total[mat], 0.0)
        self.assertEqual(_pre_total[haven], 0.0)
        for sleeves in (early, late):
            self.assertEqual(sleeves["severity"][aqua], 0.0)
            self.assertAlmostEqual(sleeves["location"][aqua], 30.0)
            self.assertEqual(sleeves["known_damage"][aqua], 0.0)
            self.assertEqual(sleeves["revision"][aqua], 0.0)
            self.assertEqual(sleeves["composition"][mat], 0.0)
            self.assertEqual(sleeves["composition"][gen], 0.0)
            self.assertEqual(sleeves["insurer_kink"][haven], 0.0)
            self.assertEqual(float(sleeves["etf_gap"][aqua]), 0.0)

    def test_a_resting_order_is_cancelled_after_its_house_half_life(self) -> None:
        half = live.published_half_lives()
        self.assertEqual(half["official_advisory"], 20.0)
        self.assertEqual(half["wire"], 22.0)
        self.assertEqual(half["impact"], 26.0)
        self.assertEqual(half["preliminary"], 4.0)
        self.assertEqual(half["final"], 4.0)
        orders = [{"id": "exit", "symbol": "MAT"}]
        final = {"exit": {"placed_at": 100.0, "event_key": "final"}}
        self.assertEqual(live.stale_order_ids(orders, final, 104.0, half), [])
        self.assertEqual(live.stale_order_ids(orders, final, 104.01, half), ["exit"])
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
        self.assertEqual(live.stale_order_ids(orders, preliminary, 104.0, half), [])
        self.assertEqual(live.stale_order_ids(orders, preliminary, 104.01, half), ["exit"])
        untouched = {"exit": {"placed_at": 100.0, "event_key": None}}
        self.assertEqual(live.stale_order_ids(orders, untouched, 1000.0, half), [])

    def test_a_name_does_not_add_or_reverse_after_it_opens(self) -> None:
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
        self.assertEqual(NAME_LOSS_CAP, 150.0)
        self.assertEqual(live.names_past_cap({"PORT": -150.0, "ISLE": -149.0}, set()), ["PORT"])
        self.assertEqual(live.names_past_cap({"PORT": -267.67}, {"PORT"}), [])

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


if __name__ == "__main__":
    unittest.main()
