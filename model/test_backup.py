"""Backup rule and the 1/2/5/crowded replay. No network and no key."""

from __future__ import annotations

import unittest

import numpy as np

from model import backup
from model.config import NAMES, BOOK_SWITCH_LOSS, NAME_LOSS_CAP, SEASON_STOP_LOSS


def _frame(n: int, port: list[float], rounds: list[int], settlement: float) -> dict:
    prices = np.full((n, len(NAMES)), 10.0)
    prices[:, 0] = np.asarray(port, dtype=float)
    times = np.arange(
        np.datetime64("2025-01-01"),
        np.datetime64("2025-01-01") + np.timedelta64(n, "m"),
        np.timedelta64(1, "m"),
    )
    limits = np.full(len(NAMES), 100.0)
    limits[0] = 200.0
    settle = np.full(len(NAMES), 10.0)
    settle[0] = settlement
    return {
        "prices": prices,
        "rounds": np.asarray(rounds, dtype=int),
        "times": times,
        "limits": limits,
        "settlement": settle,
        "sites": {"PORT": ["cascadia_prime"]},
    }


def _advisory(times: np.ndarray, bar: int, round_id: int) -> dict:
    return {
        "type": "advisory",
        "source": "official",
        "time": times[bar],
        "round": round_id,
        "hurricane_conditions": {"cascadia_prime": 0.8},
    }


class BackupRuleTests(unittest.TestCase):
    def test_caps_are_the_pre_set_bounds_and_do_not_switch_back(self) -> None:
        self.assertEqual(NAME_LOSS_CAP, 150.0)
        self.assertEqual(BOOK_SWITCH_LOSS, 400.0)
        self.assertEqual(SEASON_STOP_LOSS, 800.0)
        sites = {"PORT": ["cascadia_prime"], "ISLE": ["palm_atoll"]}
        advisory = {
            "type": "advisory",
            "source": "official",
            "hurricane_conditions": {"cascadia_prime": 0.8, "palm_atoll": 0.2},
        }
        self.assertEqual(backup.backup_names(advisory, sites, set(), set()), ["PORT"])
        self.assertEqual(backup.backup_names(advisory, sites, {"PORT"}, set()), [])
        self.assertEqual(backup.backup_names(advisory, sites, set(), {"PORT"}), [])
        self.assertEqual(backup.backup_quantity(200), 20)
        self.assertEqual(backup.next_book(backup.MAIN, -399.0), backup.MAIN)
        self.assertEqual(backup.next_book(backup.MAIN, -400.0), backup.BACKUP)
        self.assertEqual(backup.next_book(backup.BACKUP, -50.0), backup.BACKUP)
        self.assertEqual(backup.next_book(backup.BACKUP, -800.0), backup.STOPPED)
        self.assertEqual(backup.next_book(backup.STOPPED, 10_000.0), backup.STOPPED)
        self.assertTrue(backup.name_capped(-150.0))
        self.assertFalse(backup.name_capped(-149.0))

    def test_a_capped_name_is_not_reopened_next_round(self) -> None:
        frame = _frame(
            8,
            [10, 10, 10, 20, 20, 20, 0, 0],
            [1, 1, 1, 1, 1, 2, 2, 2],
            settlement=10.0,
        )
        events = [
            _advisory(frame["times"], 1, 1),
            _advisory(frame["times"], 6, 2),
        ]
        capped = backup.replay_backup(
            frame["prices"], frame["rounds"], frame["times"], events,
            frame["limits"], frame["sites"], frame["settlement"], "1",
        )
        again = backup.replay_backup(
            frame["prices"], frame["rounds"], frame["times"], events,
            frame["limits"], frame["sites"], frame["settlement"], "1", use_caps=False,
        )
        self.assertAlmostEqual(capped, -200.0)
        self.assertAlmostEqual(again, -400.0)

    def test_crowded_island_short_waits_five_marks(self) -> None:
        n = 10
        port = [10.0] * n
        port[6] = 6.0
        frame = _frame(n, port, [1] * n, settlement=6.0)
        events = [_advisory(frame["times"], 1, 1)]
        quick = backup.replay_backup(
            frame["prices"], frame["rounds"], frame["times"], events,
            frame["limits"], frame["sites"], frame["settlement"], "1", use_caps=False,
        )
        slow = backup.replay_backup(
            frame["prices"], frame["rounds"], frame["times"], events,
            frame["limits"], frame["sites"], frame["settlement"], "crowded", use_caps=False,
        )
        self.assertAlmostEqual(quick, 80.0)
        self.assertAlmostEqual(slow, 0.0)
        self.assertEqual(backup.fill_delay("PORT", 0.0, -20.0, "crowded"), 5)
        self.assertEqual(backup.fill_delay("PORT", -20.0, 0.0, "crowded"), 1)
        self.assertEqual(backup.fill_delay("RECON", 0.0, -10.0, "crowded"), 1)

    def test_replay_ignores_a_poison_settlement_until_the_book_is_flat(self) -> None:
        frame = _frame(
            6,
            [10, 10, 10, 8, 8, 8],
            [1, 1, 1, 1, 1, 2],
            settlement=10_000.0,
        )
        events = [_advisory(frame["times"], 1, 1)]
        # Fill at 10, close round 1 at the last mark 8. Flat, so settlement cannot pay.
        pnl = backup.replay_backup(
            frame["prices"], frame["rounds"], frame["times"], events,
            frame["limits"], frame["sites"], frame["settlement"], "1", use_caps=False,
        )
        self.assertAlmostEqual(pnl, 40.0)


if __name__ == "__main__":
    unittest.main()
