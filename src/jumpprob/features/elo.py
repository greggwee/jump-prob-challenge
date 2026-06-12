"""World-Football-Elo-style ratings computed from the results corpus.

We compute Elo ourselves rather than scraping eloratings.net: identical formula
family, but point-in-time correct by construction (the rating used for a match
is the rating the instant before kickoff) and fully reproducible. Sanity check
against eloratings.net current values lives in the test suite.

Formula (following eloratings.net's published system):
  expected W_e = 1 / (1 + 10^(-dr/400)),  dr = R_home - R_away + HOME_ADV*(not neutral)
  R' = R + K * G * (W - W_e)
  K by match importance (see K_BY_CATEGORY), G goal-difference multiplier:
    margin <= 1 -> 1.0 ; margin == 2 -> 1.5 ; margin >= 3 -> (11 + margin) / 8
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOME_ADV = 100.0
INITIAL_RATING = 1500.0

K_BY_CATEGORY = {
    "world_cup": 60.0,
    "continental_final": 50.0,
    "qualifier_major": 40.0,
    "minor_tournament": 30.0,
    "friendly": 20.0,
}

# tournament-string fragments -> category, checked in order (first hit wins)
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("FIFA World Cup qualification", "qualifier_major"),
    ("FIFA World Cup", "world_cup"),
    ("UEFA Euro qualification", "qualifier_major"),
    ("UEFA Euro", "continental_final"),
    ("Copa América", "continental_final"),
    ("African Cup of Nations qualification", "qualifier_major"),
    ("African Cup of Nations", "continental_final"),
    ("AFC Asian Cup qualification", "qualifier_major"),
    ("AFC Asian Cup", "continental_final"),
    ("CONCACAF Championship qualification", "qualifier_major"),
    ("CONCACAF Championship", "continental_final"),
    ("Gold Cup", "continental_final"),
    ("Confederations Cup", "continental_final"),
    ("UEFA Nations League", "qualifier_major"),
    ("CONCACAF Nations League", "qualifier_major"),
    ("Friendly", "friendly"),
]


def k_factor(tournament: str) -> float:
    for fragment, category in _CATEGORY_RULES:
        if fragment in tournament:
            return K_BY_CATEGORY[category]
    return K_BY_CATEGORY["minor_tournament"]


def goal_multiplier(margin: int) -> float:
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def compute_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Run Elo over the full results frame (must be date-sorted).

    Returns the input frame plus columns elo_home, elo_away — the ratings
    *before* the match (point-in-time: usable as a pre-match feature).
    """
    out, _ = _run_elo(results)
    return out


def current_ratings(results: pd.DataFrame) -> pd.Series:
    """Final ratings after the last match in the frame, sorted descending."""
    _, ratings = _run_elo(results)
    return pd.Series(ratings).sort_values(ascending=False)


def _run_elo(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ratings: dict[str, float] = {}
    elo_home = np.empty(len(results))
    elo_away = np.empty(len(results))

    home_teams = results["home_team"].to_numpy()
    away_teams = results["away_team"].to_numpy()
    home_scores = results["home_score"].to_numpy()
    away_scores = results["away_score"].to_numpy()
    neutrals = results["neutral"].to_numpy()
    ks = results["tournament"].map(k_factor).to_numpy()

    for i in range(len(results)):
        rh = ratings.get(home_teams[i], INITIAL_RATING)
        ra = ratings.get(away_teams[i], INITIAL_RATING)
        elo_home[i], elo_away[i] = rh, ra

        dr = rh - ra + (0.0 if neutrals[i] else HOME_ADV)
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        margin = abs(int(home_scores[i]) - int(away_scores[i]))
        if home_scores[i] > away_scores[i]:
            w = 1.0
        elif home_scores[i] < away_scores[i]:
            w = 0.0
        else:
            w = 0.5
        delta = ks[i] * goal_multiplier(margin) * (w - we)
        ratings[home_teams[i]] = rh + delta
        ratings[away_teams[i]] = ra - delta

    out = results.copy()
    out["elo_home"] = elo_home
    out["elo_away"] = elo_away
    return out, ratings
