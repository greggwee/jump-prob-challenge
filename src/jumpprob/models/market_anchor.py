"""Market-implied goal intensities + logit-space blending.

implied_lambdas: find (lam_a, lam_b) whose Dixon-Coles distribution best
reproduces de-vigged market probabilities (overdetermined -> weighted least
squares, rho held at the model's fitted value).

blend: combine probability sources in log-odds space:
    logit(p) = sum_i w_i logit(p_i),  sum w_i = 1
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from jumpprob.models.goal_model import dc_grid
from jumpprob.simulate.markets import p_draw, p_total_ge, p_win


def implied_lambdas(
    p_home: float,
    p_draw_mkt: float,
    p_away: float,
    p_over25: float | None = None,
    rho: float = -0.05,
) -> tuple[float, float]:
    """Invert de-vigged 1X2 (+ optional over-2.5) to (lam_home, lam_away)."""

    targets = [("home", p_home), ("draw", p_draw_mkt), ("away", p_away)]
    if p_over25 is not None:
        targets.append(("over25", p_over25))

    def loss(x) -> float:
        la, lb = np.exp(x)
        grid = dc_grid(la, lb, rho)
        preds = {
            "home": p_win(grid, "a"),
            "draw": p_draw(grid),
            "away": p_win(grid, "b"),
            "over25": p_total_ge(grid, 3),
        }
        return sum((preds[k] - v) ** 2 for k, v in targets)

    res = minimize(loss, x0=np.log([1.3, 1.1]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
    la, lb = np.exp(res.x)
    return float(la), float(lb)


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def blend(probs: list[float], weights: list[float]) -> float:
    """Log-odds weighted average. Weights must sum to 1."""
    w = np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("blend weights must sum to 1")
    return float(inv_logit(np.dot(w, [logit(p) for p in probs])))
