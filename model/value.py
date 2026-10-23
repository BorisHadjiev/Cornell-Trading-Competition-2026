"""Dollars from physical state.

Each term is the company sentence in case.json. Coefficients are estimated
from training seasons. Fair value equals the season open when damage, outage,
and insured loss are at their training means, so an unconditional season is
not a trade against the open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.config import (
    ARCH_BASKET,
    INSURERS,
    NAMES,
    PHYSICAL,
    REBLD_BASKET,
    REBUILDERS,
)


def _ols(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef


def _curve(work: np.ndarray, rate: float) -> np.ndarray:
    return 1.0 - np.exp(-rate * np.maximum(work, 0.0))


@dataclass
class ValueFit:
    physical: dict[str, np.ndarray]  # coef on [damage surprise, outage surprise]
    recon: tuple[float, float]  # rate, coef on demeaned curve. rate estimated
    mat: tuple[float, float]
    gen: tuple[float, float]  # grid-outage hours only
    retention: float
    width: float
    haven_slope: float
    rein_slope: float
    mean_damage: float  # scalar placeholders filled as vectors below
    mean_damage_vec: np.ndarray
    mean_outage: np.ndarray
    mean_recon_curve: float
    mean_mat_curve: float
    mean_gen_curve: float
    mean_haven_pay: float
    mean_rein_pay: float
    mean_season_loss: float
    stat_coef: dict[str, np.ndarray]  # simple linear cross-check, estimated

    def predict(
        self,
        open_prices: np.ndarray,
        cum_damage: np.ndarray,
        outage: np.ndarray,
        season_loss: float,
    ) -> np.ndarray:
        out = open_prices.copy()
        for index, name in enumerate(PHYSICAL):
            dmg = cum_damage[index] - self.mean_damage_vec[index]
            shut = outage[index] - self.mean_outage[index]
            coef = self.physical[name]
            out[index] = open_prices[index] + coef[0] * dmg + coef[1] * shut
        recon_work = cum_damage[0] + cum_damage[3] + cum_damage[4]
        mat_work = float(cum_damage.sum())
        gen_work = outage[PHYSICAL.index("GRID")]
        out[NAMES.index("RECON")] = open_prices[NAMES.index("RECON")] + self.recon[1] * (
            _curve(np.array([recon_work]), self.recon[0])[0] - self.mean_recon_curve
        )
        out[NAMES.index("MAT")] = open_prices[NAMES.index("MAT")] + self.mat[1] * (
            _curve(np.array([mat_work]), self.mat[0])[0] - self.mean_mat_curve
        )
        out[NAMES.index("GEN")] = open_prices[NAMES.index("GEN")] + self.gen[1] * (
            _curve(np.array([gen_work]), self.gen[0])[0] - self.mean_gen_curve
        )
        pay_h = float(np.clip(season_loss - self.retention, 0.0, self.width))
        pay_r = float(max(season_loss - (self.retention + self.width), 0.0))
        out[NAMES.index("HAVEN")] = open_prices[NAMES.index("HAVEN")] + self.haven_slope * (pay_h - self.mean_haven_pay)
        out[NAMES.index("REIN")] = open_prices[NAMES.index("REIN")] + self.rein_slope * (pay_r - self.mean_rein_pay)
        out[NAMES.index("ARCH")] = sum(out[NAMES.index(name)] * qty for name, qty in ARCH_BASKET.items())
        out[NAMES.index("REBLD")] = sum(out[NAMES.index(name)] * qty for name, qty in REBLD_BASKET.items())
        return out

    def statistical(self, open_prices: np.ndarray, cum_damage: np.ndarray, outage: np.ndarray, season_loss: float) -> np.ndarray:
        """A flat linear map. Used only to cut size when it disagrees with predict()."""
        out = open_prices.copy()
        for index, name in enumerate(NAMES):
            coef = self.stat_coef[name]
            dmg = float(cum_damage.sum()) - float(self.mean_damage_vec.sum())
            shut = float(outage.sum()) - float(self.mean_outage.sum())
            out[index] = open_prices[index] + coef[0] * dmg + coef[1] * shut + coef[2] * (season_loss - self.mean_season_loss)
        return out

    def as_dict(self) -> dict:
        return {
            "estimated": True,
            "physical": {name: {"damage": float(coef[0]), "outage": float(coef[1])} for name, coef in self.physical.items()},
            "recon": {"rate": self.recon[0], "coef": self.recon[1], "work": "PORT+GRID+FERRY damage, concave"},
            "mat": {"rate": self.mat[0], "coef": self.mat[1], "work": "all physical damage, concave"},
            "gen": {"rate": self.gen[0], "coef": self.gen[1], "work": "GRID outage hours, no damage term"},
            "insurance": {
                "retention_musd": self.retention,
                "width_musd": self.width,
                "rein_attachment_musd": self.retention + self.width,
                "haven_slope": self.haven_slope,
                "rein_slope": self.rein_slope,
                "shape": "HAVEN pays above retention up to the reinsurance attachment. REIN pays above that limit.",
            },
        }


def _fit_curve(work: np.ndarray, target: np.ndarray, rates: np.ndarray) -> tuple[float, float, float]:
    best = None
    for rate in rates:
        curve = _curve(work, float(rate))
        centered = curve - curve.mean()
        coef = float(_ols(centered[:, None], target)[0]) if np.std(centered) > 1e-8 else 0.0
        pred = coef * centered
        sse = float(np.sum((target - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(rate), coef, float(curve.mean()))
    assert best is not None
    return best[1], best[2], best[3]


def _layer_pay(loss: np.ndarray, retention: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    haven = np.clip(loss - retention, 0.0, width)
    rein = np.maximum(loss - (retention + width), 0.0)
    return haven, rein


def fit_value(
    open_prices: np.ndarray,
    settlement: np.ndarray,
    cum_damage: np.ndarray,
    outage: np.ndarray,
    season_loss: np.ndarray,
) -> ValueFit:
    """open_prices, settlement: (N, 12). cum_damage, outage: (N, 5). season_loss: (N,)."""
    physical: dict[str, np.ndarray] = {}
    mean_damage = cum_damage.mean(axis=0)
    mean_outage = outage.mean(axis=0)
    for index, name in enumerate(PHYSICAL):
        surprise_d = cum_damage[:, index] - mean_damage[index]
        surprise_o = outage[:, index] - mean_outage[index]
        target = settlement[:, index] - open_prices[:, index]
        if np.std(surprise_o) < 1e-8:
            physical[name] = np.array([float(_ols(surprise_d[:, None], target)[0]), 0.0])
        else:
            physical[name] = _ols(np.column_stack([surprise_d, surprise_o]), target)

    recon_work = cum_damage[:, 0] + cum_damage[:, 3] + cum_damage[:, 4]
    mat_work = cum_damage.sum(axis=1)
    gen_work = outage[:, PHYSICAL.index("GRID")]
    rates = np.array([0.01, 0.03, 0.08, 0.15, 0.3, 0.6, 1.2])
    recon_target = settlement[:, NAMES.index("RECON")] - open_prices[:, NAMES.index("RECON")]
    mat_target = settlement[:, NAMES.index("MAT")] - open_prices[:, NAMES.index("MAT")]
    gen_target = settlement[:, NAMES.index("GEN")] - open_prices[:, NAMES.index("GEN")]
    recon = _fit_curve(recon_work, recon_target, rates)
    mat = _fit_curve(mat_work, mat_target, rates)
    gen = _fit_curve(gen_work, gen_target, rates)

    # Grid the kink over the training loss range. Nothing here is a typed-in attachment.
    lo, hi = float(np.min(season_loss)), float(np.max(season_loss))
    grid = np.linspace(lo, hi, 24)
    best_ins = None
    haven_y = settlement[:, NAMES.index("HAVEN")] - open_prices[:, NAMES.index("HAVEN")]
    rein_y = settlement[:, NAMES.index("REIN")] - open_prices[:, NAMES.index("REIN")]
    for retention in grid:
        for width in grid:
            width = float(width)
            if width <= 1.0:
                continue
            if retention + width > hi + 1.0 and retention > hi:
                continue
            pay_h, pay_r = _layer_pay(season_loss, float(retention), width)
            if np.std(pay_h) < 1e-6 and np.std(pay_r) < 1e-6:
                continue
            h_coef = float(_ols((pay_h - pay_h.mean())[:, None], haven_y)[0]) if np.std(pay_h) > 1e-6 else 0.0
            r_coef = float(_ols((pay_r - pay_r.mean())[:, None], rein_y)[0]) if np.std(pay_r) > 1e-6 else 0.0
            pred_h = h_coef * (pay_h - pay_h.mean())
            pred_r = r_coef * (pay_r - pay_r.mean())
            sse = float(np.sum((haven_y - pred_h) ** 2) + np.sum((rein_y - pred_r) ** 2))
            if best_ins is None or sse < best_ins[0]:
                best_ins = (sse, float(retention), width, h_coef, r_coef, float(pay_h.mean()), float(pay_r.mean()))
    assert best_ins is not None

    stat_coef: dict[str, np.ndarray] = {}
    dmg_sum = cum_damage.sum(axis=1)
    out_sum = outage.sum(axis=1)
    for index, name in enumerate(NAMES):
        if name in ("ARCH", "REBLD"):
            stat_coef[name] = np.zeros(3)
            continue
        design = np.column_stack([
            dmg_sum - dmg_sum.mean(),
            out_sum - out_sum.mean(),
            season_loss - season_loss.mean(),
        ])
        stat_coef[name] = _ols(design, settlement[:, index] - open_prices[:, index])

    fit = ValueFit(
        physical=physical,
        recon=(recon[0], recon[1]),
        mat=(mat[0], mat[1]),
        gen=(gen[0], gen[1]),
        retention=best_ins[1],
        width=best_ins[2],
        haven_slope=best_ins[3],
        rein_slope=best_ins[4],
        mean_damage=float(mean_damage.mean()),
        mean_damage_vec=mean_damage,
        mean_outage=mean_outage,
        mean_recon_curve=recon[2],
        mean_mat_curve=mat[2],
        mean_gen_curve=gen[2],
        mean_haven_pay=best_ins[5],
        mean_rein_pay=best_ins[6],
        mean_season_loss=float(season_loss.mean()),
        stat_coef=stat_coef,
    )
    return fit


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))
