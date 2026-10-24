"""Invariant tests that do not need the research pack."""

from __future__ import annotations

import unittest

import numpy as np

from model.announce import record
from model.baseline import symbols_to_sell
from model.config import ARCH_BASKET, NAMES, PHYSICAL
from model.damage import DamageFit, ensemble_damage
from model.data import Site
from model.house import EVENT_TYPES
from model.labels import label_at
from model.paths import PathLibrary
from model.portfolio import deadband_holds, etf_mark_sleeve, fit_inside_limits, shares_from_edge


class LabelTests(unittest.TestCase):
    def test_born_erratic_is_erratic_from_age_zero(self):
        self.assertEqual(label_at("erratic", None, 0), "erratic")
        self.assertEqual(label_at("erratic", None, 100), "erratic")

    def test_flip_before_t_is_erratic(self):
        self.assertEqual(label_at("direct", 50, 50), "erratic")
        self.assertEqual(label_at("direct", 50, 80), "erratic")

    def test_flip_after_t_keeps_formation_class(self):
        self.assertEqual(label_at("direct", 50, 49), "direct")
        self.assertEqual(label_at("chain", None, 200), "chain")


class PathTests(unittest.TestCase):
    def test_samples_carry_intensity_and_the_requested_label(self):
        storm = _Storm("direct", None, _track(x=100, intensity=0.4))
        other = _Storm("erratic", None, _track(x=400, intensity=0.9))
        library = PathLibrary([_Season([storm, other])])
        paths = library.sample("direct", age=0, n=4, rng=np.random.default_rng(0), origin_xy=np.array([10.0, 20.0]))
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(path.shape[1], 7)
            self.assertTrue(np.all(np.isfinite(path[:, 6])))
            self.assertGreater(float(path[0, 6]), 0.2)
            self.assertAlmostEqual(float(path[0, 1]), 10.0)
            self.assertAlmostEqual(float(path[0, 2]), 20.0)

    def test_ensemble_averages_damage_not_the_mean_path(self):
        sites = [Site(name, name, "cascadia_prime", 0.0, 0.0, 1.0) for name in PHYSICAL]
        fit = DamageFit(form="gauss", dt=2.0, intercept=np.zeros(5), slope=np.ones(5), tau=0.2)
        near = _track(x=0, intensity=1.0)
        far = _track(x=500, intensity=1.0)
        averaged = ensemble_damage([near, far], fit, sites)
        mean_path = 0.5 * (near + far)
        damaged_mean = fit.damage_of_obs(mean_path, sites)
        self.assertFalse(np.allclose(averaged, damaged_mean))
        self.assertGreater(float(averaged[0]), float(damaged_mean[0]))


class HouseTests(unittest.TestCase):
    def test_event_types_do_not_include_settlement(self):
        self.assertNotIn("settlement", EVENT_TYPES)
        self.assertNotIn("round1_vs_settlement", EVENT_TYPES)
        self.assertEqual(set(EVENT_TYPES), {"official_advisory", "wire", "impact", "preliminary", "final"})


class BaselineTests(unittest.TestCase):
    def test_sells_only_the_island_business_on_a_high_official_probability(self):
        sites = {"PORT": ["cascadia_prime"], "ISLE": ["coral_key"], "AQUA": [], "GRID": [], "FERRY": []}
        advisory = {
            "type": "advisory",
            "source": "official",
            "hurricane_conditions": {"cascadia_prime": 0.8, "coral_key": 0.1, "azure_isle": 0.0, "palm_atoll": 0.0},
        }
        self.assertEqual(symbols_to_sell(advisory, sites, set()), ["PORT"])
        advisory["source"] = "wire"
        self.assertEqual(symbols_to_sell(advisory, sites, set()), [])


class PortfolioTests(unittest.TestCase):
    def test_arch_basket_holds_haven_and_fundamental_shares_do_not_invent_an_etf_hedge(self):
        self.assertIn("HAVEN", ARCH_BASKET)
        self.assertNotIn("RECON", ARCH_BASKET)
        limits = np.full(len(NAMES), 100.0)
        edge = np.ones(len(NAMES))
        shares = shares_from_edge(edge, limits, fraction=1.0)
        self.assertEqual(shares[NAMES.index("ARCH")], 0.0)
        self.assertEqual(shares[NAMES.index("REBLD")], 0.0)
        self.assertGreater(shares[NAMES.index("GEN")], 0.0)

    def test_etf_sleeve_sells_a_rich_mark_and_stays_inside_limits(self):
        mark = np.zeros(len(NAMES))
        mark[NAMES.index("ARCH")] = 5.0
        limits = np.full(len(NAMES), 100.0)
        limits[NAMES.index("ARCH")] = 50.0
        sleeve = etf_mark_sleeve(mark, limits, fraction=1.0, gap_min=1.0)
        self.assertLess(sleeve[NAMES.index("ARCH")], 0.0)
        self.assertGreater(sleeve[NAMES.index("HAVEN")], 0.0)
        fitted = fit_inside_limits(np.zeros(len(NAMES)), sleeve, limits)
        self.assertTrue(np.all(np.abs(fitted) <= limits + 1e-8))

    def test_deadband_blocks_a_small_move(self):
        limits = np.full(len(NAMES), 100.0)
        current = np.zeros(len(NAMES))
        nudge = np.full(len(NAMES), 1.0)
        self.assertTrue(deadband_holds(current, nudge, limits, fraction=0.08))
        self.assertFalse(deadband_holds(current, np.full(len(NAMES), 20.0), limits, fraction=0.08))


class AnnounceTests(unittest.TestCase):
    def test_records_symbols_and_ignores_other_messages(self):
        log = []
        record({"type": "news", "headline": "a rumor"}, log)
        record({"type": "announcement", "symbols": ["GEN"], "headline": "grid tariff"}, log)
        self.assertEqual(log, [{"symbols": ["GEN"], "headline": "grid tariff", "time": None}])


class _Storm:
    def __init__(self, pattern, flip, obs):
        self.pattern = pattern
        self.flip = flip
        self.obs = obs


class _Season:
    def __init__(self, storms):
        self.storms = storms


def _track(x: float, intensity: float) -> np.ndarray:
    rows = []
    for age in (0, 10, 20):
        rows.append((age, x, 0.0, 0.0, -1.0, 40.0, intensity))
    return np.asarray(rows, dtype=float)


if __name__ == "__main__":
    unittest.main()
