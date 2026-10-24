"""Wind kernel and outage clock.

Company damage is the weight-sum of site exposure. Intensity is an input.
Two falloff forms are fit; the caller keeps the one with lower held-out error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.config import PHYSICAL
from model.data import Site, StormObs


def _falloff(dist: np.ndarray, radius: np.ndarray, form: str) -> np.ndarray:
    scaled = dist / np.maximum(radius, 1.0)
    if form == "exp":
        return np.exp(-scaled)
    if form == "gauss":
        return np.exp(-(scaled ** 2))
    raise ValueError(form)


def site_wind(x, y, radius, intensity, sites: list[Site], form: str) -> np.ndarray:
    """Instantaneous wind at each site. x, y, radius, intensity are (T,)."""
    sx = np.array([site.x for site in sites], dtype=float)
    sy = np.array([site.y for site in sites], dtype=float)
    dist = np.hypot(x[:, None] - sx, y[:, None] - sy)
    fall = _falloff(dist, radius[:, None], form)
    return intensity[:, None] * fall


def row_steps(obs: np.ndarray, dt: float) -> np.ndarray:
    """Seconds each observation stands for: the age gap to the previous row, `dt` for the first.

    Research observations are every 2 s, so this equals `dt` everywhere and the fit is unchanged.
    A live feed can drop frames (a reconnect, a track catch-up every 4 s), and a fixed step would
    then under-count the wind the storm actually delivered.
    """
    if len(obs) == 0:
        return np.zeros(0, dtype=float)
    steps = np.full(len(obs), float(dt), dtype=float)
    if len(obs) > 1:
        gaps = np.diff(obs[:, 0])
        steps[1:] = np.where(gaps > 1e-9, gaps, float(dt))
    return steps


def integrate_exposure(obs: np.ndarray, sites: list[Site], form: str, dt: float) -> np.ndarray:
    """Weight-summed time integral of wind for each physical company. obs columns x,y,radius,intensity at 1,2,5,6."""
    wind = site_wind(obs[:, 1], obs[:, 2], obs[:, 5], obs[:, 6], sites, form)
    weights = np.array([site.weight for site in sites], dtype=float)
    companies = [site.company for site in sites]
    integral = (wind * row_steps(obs, dt)[:, None]).sum(axis=0)
    out = np.zeros(len(PHYSICAL), dtype=float)
    for index, company in enumerate(companies):
        out[PHYSICAL.index(company)] += weights[index] * integral[index]
    return out


def site_exposure(obs: np.ndarray, sites: list[Site], form: str, dt: float) -> np.ndarray:
    wind = site_wind(obs[:, 1], obs[:, 2], obs[:, 5], obs[:, 6], sites, form)
    return (wind * row_steps(obs, dt)[:, None]).sum(axis=0)


def outage_hours(obs: np.ndarray, sites: list[Site], form: str, dt: float, tau: float) -> np.ndarray:
    """Company-weighted hours with wind above tau. No all-clear is required; the clock is the path."""
    wind = site_wind(obs[:, 1], obs[:, 2], obs[:, 5], obs[:, 6], sites, form)
    weights = np.array([site.weight for site in sites], dtype=float)
    companies = [site.company for site in sites]
    hours = ((wind > tau) * row_steps(obs, dt)[:, None]).sum(axis=0) / 3600.0
    out = np.zeros(len(PHYSICAL), dtype=float)
    for index, company in enumerate(companies):
        out[PHYSICAL.index(company)] += weights[index] * hours[index]
    return out


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


@dataclass
class DamageFit:
    form: str
    dt: float
    intercept: np.ndarray  # (5,) estimated
    slope: np.ndarray  # (5,) estimated
    tau: float  # estimated outage threshold, in wind units

    def predict_damage(self, exposure: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * exposure

    def damage_of_obs(self, obs: np.ndarray, sites: list[Site]) -> np.ndarray:
        return self.predict_damage(integrate_exposure(obs, sites, self.form, self.dt))

    def as_dict(self) -> dict:
        return {
            "estimated": True,
            "form": self.form,
            "dt_seconds": self.dt,
            "intercept": {name: float(self.intercept[i]) for i, name in enumerate(PHYSICAL)},
            "slope": {name: float(self.slope[i]) for i, name in enumerate(PHYSICAL)},
            "outage_threshold": self.tau,
        }


def fit_form(storms: list[StormObs], sites: list[Site], form: str, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exposure = np.vstack([integrate_exposure(storm.obs, sites, form, dt) for storm in storms])
    actual = np.vstack([
        np.array([storm.final_damage[name] for name in PHYSICAL], dtype=float) for storm in storms
    ])
    intercept = np.zeros(len(PHYSICAL))
    slope = np.zeros(len(PHYSICAL))
    for index in range(len(PHYSICAL)):
        intercept[index], slope[index] = _ols(exposure[:, index], actual[:, index])
    return intercept, slope, exposure


def mae_by_company(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = np.abs(predicted - actual)
    return {name: float(err[:, index].mean()) for index, name in enumerate(PHYSICAL)}


def predict_matrix(intercept: np.ndarray, slope: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    return intercept + slope * exposure


def training_mean_predictor(train_actual: np.ndarray, n_rows: int) -> np.ndarray:
    mean = train_actual.mean(axis=0)
    return np.repeat(mean[None, :], n_rows, axis=0)


def choose_tau(
    storms: list[StormObs],
    sites: list[Site],
    form: str,
    dt: float,
    candidates: np.ndarray,
) -> float:
    """Pick the wind threshold whose outage hours best track GRID settlement is done outside.

    Here we only return the candidate grid. The value fit selects tau. This helper
    scores how well a threshold separates high and low GRID final damage, which is
    the observable the clock has to explain before dollars are attached.
    """
    best_tau = float(candidates[0])
    best_score = -1.0
    grid_damage = np.array([storm.final_damage["GRID"] for storm in storms], dtype=float)
    if np.std(grid_damage) < 1e-8:
        return best_tau
    for tau in candidates:
        hours = np.array([
            outage_hours(storm.obs, sites, form, dt, float(tau))[PHYSICAL.index("GRID")]
            for storm in storms
        ])
        if np.std(hours) < 1e-8:
            continue
        corr = abs(float(np.corrcoef(hours, grid_damage)[0, 1]))
        if corr > best_score:
            best_score = corr
            best_tau = float(tau)
    return best_tau


def ensemble_damage(path_obs: list[np.ndarray], fit: DamageFit, sites: list[Site]) -> np.ndarray:
    """Average damage across paths. Do not damage the mean path."""
    if not path_obs:
        raise ValueError("ensemble_damage requires at least one path")
    stacked = [fit.damage_of_obs(obs, sites) for obs in path_obs]
    return np.mean(np.vstack(stacked), axis=0)
