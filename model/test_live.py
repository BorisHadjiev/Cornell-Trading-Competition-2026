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


if __name__ == "__main__":
    unittest.main()
