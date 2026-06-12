"""Elo-anchored goal intensity model + Dixon-Coles scoreline distribution.

Intensity (per side s in a match):
    log lambda_s = b0 + b1 * (elo_s - elo_opp)/100 + b2 * is_home_ground_s

is_home_ground is 1 only for the designated home team at a non-neutral venue
(World Cup group matches are neutral for almost everyone). Fit by weighted
Poisson MLE; match weights decay exponentially with age:
    w = 0.5 ** (age_days / half_life_days)

Scoreline distribution: independent Poissons with the Dixon-Coles tau
correction on the {0,1}x{0,1} cells, parameter rho_dc fit by weighted MLE on
the same training window. tau (Dixon & Coles 1997):
    (0,0): 1 - lh*la*rho   (1,0): 1 + la*rho
    (0,1): 1 + lh*rho      (1,1): 1 - rho
rho < 0 raises the probability of draws/low scores relative to independence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

MAX_GOALS = 12  # grid truncation; P(>12) is negligible for internationals


@dataclass
class GoalModel:
    b0: float
    b1: float
    b2: float
    rho_dc: float
    half_life_days: float

    def intensities(self, elo_a: float, elo_b: float, a_is_home_ground: bool = False) -> tuple[float, float]:
        """Expected goals (lambda_a, lambda_b) for team A vs team B."""
        d = (elo_a - elo_b) / 100.0
        la = np.exp(self.b0 + self.b1 * d + self.b2 * float(a_is_home_ground))
        lb = np.exp(self.b0 - self.b1 * d)
        return float(la), float(lb)

    def scoreline_grid(self, lam_a: float, lam_b: float) -> np.ndarray:
        """(MAX_GOALS+1)^2 joint pmf with DC correction, renormalized."""
        return dc_grid(lam_a, lam_b, self.rho_dc)


def dc_grid(lam_a: float, lam_b: float, rho: float) -> np.ndarray:
    ks = np.arange(MAX_GOALS + 1)
    pa = np.exp(-lam_a) * lam_a**ks / _factorials()
    pb = np.exp(-lam_b) * lam_b**ks / _factorials()
    grid = np.outer(pa, pb)
    grid[0, 0] *= 1.0 - lam_a * lam_b * rho
    grid[1, 0] *= 1.0 + lam_b * rho
    grid[0, 1] *= 1.0 + lam_a * rho
    grid[1, 1] *= 1.0 - rho
    grid = np.clip(grid, 0.0, None)
    return grid / grid.sum()


_FACT_CACHE: np.ndarray | None = None


def _factorials() -> np.ndarray:
    global _FACT_CACHE
    if _FACT_CACHE is None:
        from math import factorial

        _FACT_CACHE = np.array([factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)
    return _FACT_CACHE


def _training_rows(results_elo: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float, window_years: float = 16.0):
    """Two rows per match (one per side). Returns design arrays + weights."""
    df = results_elo[
        (results_elo["date"] < as_of) & (results_elo["date"] >= as_of - pd.Timedelta(days=365.25 * window_years))
    ]
    age = (as_of - df["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (age / half_life_days)

    d = ((df["elo_home"] - df["elo_away"]) / 100.0).to_numpy()
    home_ground = (~df["neutral"].to_numpy()).astype(float)

    # side 1: home team; side 2: away team (never the ground host)
    elo_diff = np.concatenate([d, -d])
    is_home = np.concatenate([home_ground, np.zeros(len(df))])
    goals = np.concatenate([df["home_score"].to_numpy(), df["away_score"].to_numpy()]).astype(float)
    weights = np.concatenate([w, w])
    return elo_diff, is_home, goals, weights, df


def fit_goal_model(
    results_elo: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_days: float = 730.0,
) -> GoalModel:
    """Fit (b0, b1, b2) by weighted Poisson MLE, then rho_dc by weighted MLE."""
    elo_diff, is_home, goals, w, df = _training_rows(results_elo, as_of, half_life_days)

    def nll(beta):
        log_lam = beta[0] + beta[1] * elo_diff + beta[2] * is_home
        lam = np.exp(np.clip(log_lam, -8, 4))
        return -np.sum(w * (goals * log_lam - lam))

    res = minimize(nll, x0=np.array([0.2, 0.3, 0.2]), method="BFGS")
    if not res.success and res.status != 2:  # status 2 = precision loss, still usable
        raise RuntimeError(f"goal model fit failed: {res.message}")
    b0, b1, b2 = res.x

    # rho_dc: weighted MLE over the four low-score cells, lambdas held fixed
    d_h = (df["elo_home"].to_numpy() - df["elo_away"].to_numpy()) / 100.0
    hg = (~df["neutral"].to_numpy()).astype(float)
    lam_h = np.exp(b0 + b1 * d_h + b2 * hg)
    lam_a = np.exp(b0 - b1 * d_h)
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    w_match = 0.5 ** ((as_of - df["date"]).dt.days.to_numpy(dtype=float) / half_life_days)

    low = (hs <= 1) & (as_ <= 1)
    lh, la, x, y, wm = lam_h[low], lam_a[low], hs[low], as_[low], w_match[low]

    def nll_rho(rho):
        tau = np.ones_like(lh)
        tau = np.where((x == 0) & (y == 0), 1 - lh * la * rho, tau)
        tau = np.where((x == 1) & (y == 0), 1 + la * rho, tau)
        tau = np.where((x == 0) & (y == 1), 1 + lh * rho, tau)
        tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
        return -np.sum(wm * np.log(np.clip(tau, 1e-10, None)))

    res_rho = minimize_scalar(nll_rho, bounds=(-0.3, 0.3), method="bounded")
    return GoalModel(float(b0), float(b1), float(b2), float(res_rho.x), half_life_days)
