"""Preliminary print to final print, by company. Damage training stays on the final print."""

from __future__ import annotations

import numpy as np

from model.config import PHYSICAL


def fit_revision(prelim: np.ndarray, final: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """prelim, final: (N, 5). Returns intercept, slope, each estimated."""
    intercept = np.zeros(prelim.shape[1])
    slope = np.zeros(prelim.shape[1])
    for index in range(prelim.shape[1]):
        design = np.column_stack([np.ones(len(prelim)), prelim[:, index]])
        coef, *_ = np.linalg.lstsq(design, final[:, index], rcond=None)
        intercept[index] = float(coef[0])
        slope[index] = float(coef[1])
    return intercept, slope


def apply_revision(intercept: np.ndarray, slope: np.ndarray, prelim: np.ndarray) -> np.ndarray:
    return intercept + slope * prelim


def mae_by_company(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = np.abs(predicted - actual)
    return {name: float(err[:, i].mean()) for i, name in enumerate(PHYSICAL)}
