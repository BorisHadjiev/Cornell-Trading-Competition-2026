"""Order rules for the live prototype. No network and no key."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model import live
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
        self.assertTrue(live.reconnect_after_read_failure("running"))
        self.assertTrue(live.reconnect_after_read_failure(None))
        self.assertFalse(live.reconnect_after_read_failure("ready"))

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

    def test_post_turn_severity_is_capped_at_the_pre_turn_fraction(self) -> None:
        from model.backtest import _rebucket
        from model.config import SLEEVES

        aqua = NAMES.index("AQUA")
        mat = NAMES.index("MAT")
        gen = NAMES.index("GEN")

        def book(severity_aqua: float, severity_mat: float, location_mat: float, tilt_gen: float) -> dict:
            sleeves = {name: np.zeros(len(NAMES)) for name in SLEEVES}
            sleeves["severity"][aqua] = severity_aqua
            sleeves["severity"][mat] = severity_mat
            sleeves["location"][mat] = location_mat
            sleeves["composition"][gen] = tilt_gen
            return sleeves

        post = book(-400.0, 390.0, -10.0, 50.0)
        live.cap_severity_sleeve(post, post_turn=True, fraction=1.0)
        self.assertAlmostEqual(post["severity"][aqua], -140.0)
        self.assertAlmostEqual(post["severity"][mat], 136.5)
        self.assertAlmostEqual(post["location"][mat], -10.0)
        self.assertAlmostEqual(post["composition"][gen], 50.0)
        _rebucket(post)
        self.assertAlmostEqual(post["severity"][aqua], -140.0)
        self.assertEqual(post["severity"][mat], 0.0)
        self.assertAlmostEqual(post["composition"][mat], 126.5)
        self.assertAlmostEqual(post["composition"][gen], 50.0)

        pre = book(-40.0, 9.0, -10.0, 50.0)
        live.cap_severity_sleeve(pre, post_turn=False, fraction=0.35)
        self.assertAlmostEqual(pre["severity"][aqua], -40.0)
        self.assertAlmostEqual(pre["severity"][mat], 9.0)

        already = book(-40.0, 9.0, 0.0, 0.0)
        live.cap_severity_sleeve(already, post_turn=True, fraction=0.35)
        self.assertAlmostEqual(already["severity"][aqua], -40.0)

        def fake(state, models, season, bar):
            sleeves = {name: np.zeros(len(NAMES)) for name in SLEEVES}
            sleeves["severity"][mat] = 200.0
            sleeves["location"][mat] = 40.0
            from model import backtest
            backtest._rebucket(sleeves)
            return sum(sleeves.values()), sleeves

        with patch.object(live, "targets_from_state", fake):
            _total, capped = live.targets_for_live({"post_turn": True, "fraction": 1.0}, None, None, 1)
            _total, unchanged = live.targets_for_live({"post_turn": False, "fraction": 0.35}, None, None, 1)
        self.assertAlmostEqual(capped["composition"][mat], 0.35 * 200.0 + 40.0)
        self.assertEqual(capped["severity"][mat], 0.0)
        self.assertAlmostEqual(unchanged["composition"][mat], 240.0)

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


if __name__ == "__main__":
    unittest.main()
