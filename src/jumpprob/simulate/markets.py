"""Answer binary question archetypes from a goal model.

Full-match questions are computed exactly from the Dixon-Coles scoreline grid
(no Monte Carlo needed at this phase — the grid IS the joint distribution).
Half-specific goal questions use Poisson thinning of the match intensity:
second-half intensity = (1 - rho_1h) * lambda. The DC correction applies at
full-match level only; treating half marginals as Poisson is an approximation,
validated in the backtest.

SECOND_HALF_SHARE measured from goalscorers.csv (47k goals, non-own-goal):
0.556 (assumption A2, estimated 2026-06-11 — recompute via scripts/refit).
"""

from __future__ import annotations

import numpy as np

from jumpprob.models.goal_model import GoalModel, MAX_GOALS

SECOND_HALF_SHARE = 0.556


def grid_and_lams(model: GoalModel, elo_a: float, elo_b: float, a_home_ground: bool = False):
    la, lb = model.intensities(elo_a, elo_b, a_home_ground)
    return model.scoreline_grid(la, lb), la, lb


def p_win(grid: np.ndarray, side: str = "a") -> float:
    a, b = np.tril_indices_from(grid, k=-1)  # rows (a goals) > cols (b goals)
    p_a = grid[a, b].sum()
    if side == "a":
        return float(p_a)
    return float(grid[np.triu_indices_from(grid, k=1)].sum())


def p_draw(grid: np.ndarray) -> float:
    return float(np.trace(grid))


def p_total_ge(grid: np.ndarray, k: int) -> float:
    totals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    return float(grid[totals >= k].sum())


def p_btts(grid: np.ndarray) -> float:
    return float(grid[1:, 1:].sum())


def p_btts_and_total_ge(grid: np.ndarray, k: int) -> float:
    totals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    mask = np.zeros_like(grid, dtype=bool)
    mask[1:, 1:] = True
    return float(grid[mask & (totals >= k)].sum())


def p_scores_in_half(lam_team: float, half: int) -> float:
    share = (1.0 - SECOND_HALF_SHARE) if half == 1 else SECOND_HALF_SHARE
    return float(1.0 - np.exp(-share * lam_team))


def answer_all(model: GoalModel, elo_a: float, elo_b: float, a_home_ground: bool = False) -> dict[str, float]:
    """All Phase 2 archetypes for one fixture, keyed by archetype id."""
    grid, la, lb = grid_and_lams(model, elo_a, elo_b, a_home_ground)
    return {
        "win_a": p_win(grid, "a"),
        "win_b": p_win(grid, "b"),
        "draw": p_draw(grid),
        "total_ge_2": p_total_ge(grid, 2),
        "total_ge_3": p_total_ge(grid, 3),
        "btts": p_btts(grid),
        "btts_and_3plus": p_btts_and_total_ge(grid, 3),
        "a_scores_2h": p_scores_in_half(la, 2),
        "b_scores_2h": p_scores_in_half(lb, 2),
        "a_scores_1h": p_scores_in_half(la, 1),
    }
