"""Platt scaling per archetype family + final submission formatting.

Platt: p' = sigmoid(a * logit(p) + b), 2 parameters per family, fit by
logistic MLE on backtest settlements. a < 1 shrinks overconfidence.
Families with too few settlements fall back to identity.

submit_int: clip to [0.01, 0.99] and round to the API's integer percent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from jumpprob.models.market_anchor import inv_logit, logit

MIN_SETTLEMENTS = 150


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.array([logit(v) for v in p])

    def nll(ab):
        z = ab[0] * x + ab[1]
        # log-likelihood of Bernoulli with logits z
        return float(np.sum(np.logaddexp(0, z)) - np.sum(y * z))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def fit_calibration_table(backtests: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """backtests: rows with archetype, p_model, outcome."""
    table: dict[str, tuple[float, float]] = {}
    for arch, g in backtests.groupby("archetype"):
        if len(g) < MIN_SETTLEMENTS:
            table[arch] = (1.0, 0.0)
            continue
        table[arch] = fit_platt(g["p_model"].to_numpy(), g["outcome"].to_numpy())
    return table


def apply_calibration(p: float, arch: str, table: dict[str, tuple[float, float]]) -> float:
    a, b = table.get(arch, (1.0, 0.0))
    return float(inv_logit(a * logit(p) + b))


def submit_int(p: float) -> int:
    """API accepts integers 1..99 (0 and 100 rejected)."""
    return int(np.clip(round(p * 100), 1, 99))
