"""Parameter uncertainty for the goal model under decay-weighted Poisson MLE.

The decay-weighted objective is a pseudo-likelihood, so the correct covariance
is the sandwich  V = H^-1 M H^-1  with
    H = sum_i w_i lambda_i x_i x_i^T          (expected information)
    M = sum_i w_i^2 (y_i - lambda_i)^2 x_i x_i^T   (score variance)
sample_betas draws B ~ N(beta_hat, V) for posterior-predictive simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jumpprob.models.goal_model import GoalModel, _training_rows


def beta_covariance(model: GoalModel, results_elo: pd.DataFrame, as_of: pd.Timestamp) -> np.ndarray:
    elo_diff, is_home, goals, w, _ = _training_rows(results_elo, as_of, model.half_life_days)
    X = np.column_stack([np.ones_like(elo_diff), elo_diff, is_home])
    lam = np.exp(np.clip(model.b0 + model.b1 * elo_diff + model.b2 * is_home, -8, 4))
    H = (X * (w * lam)[:, None]).T @ X
    resid2 = (goals - lam) ** 2
    M = (X * (w**2 * resid2)[:, None]).T @ X
    Hinv = np.linalg.inv(H)
    return Hinv @ M @ Hinv


def sample_betas(model: GoalModel, cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    mean = np.array([model.b0, model.b1, model.b2])
    return rng.multivariate_normal(mean, cov, size=n)
