"""Squad-composition features for team intensities.

Squad attacking quality (point-in-time): for a team entering tournament t, the
minutes-weighted mean of its squad players' shrunk SoT/90 rates computed from
tournaments BEFORE t (player_match.parquet). Players without history contribute
the population prior (so a squad of unknowns scores exactly average).

The squad proxy uses players who appear in tournament t as the roster. Rosters
are public weeks before kickoff, so this is knowable-in-principle; the residual
selection bias (players who actually played vs named) is registered in PLAN §9.

Goal-model integration is a one-parameter multiplicative adjustment fit on
prior tournaments only (Elo term untouched — measures what squads add BEYOND Elo):
    log E[goals] = log(lambda_v1) + b_sq * (sq_team - sq_opp)
where sq is standardized within the training window.

Lineup-day hook: scale lambda by the XI's share of squad attacking output,
exponent gamma = 1 (assumption A12 — not estimable without historical lineup
panels; revisit after the group stage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

M0 = 270.0  # pseudo-minutes, same shrinkage as the player prop model


def squad_rates(players: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Per-player shrunk SoT/90 from matches strictly before as_of."""
    pre = players[players["date"] < as_of]
    prior = pre["sot"].sum() / max(pre["minutes"].sum(), 1.0)
    tot = pre.groupby("player")[["minutes", "sot"]].sum()
    tot["rate"] = (tot["sot"] + prior * M0) / (tot["minutes"] + M0)
    return tot, float(prior)


def squad_quality(players: pd.DataFrame, tournament: str, as_of: pd.Timestamp) -> pd.Series:
    """Squad attacking quality per team for one tournament (point-in-time)."""
    rates, prior = squad_rates(players, as_of)
    roster = players[players["tournament"] == tournament]
    out = {}
    if prior <= 0:  # no history at all before as_of: every squad is average
        return pd.Series({t: 1.0 for t in roster["team"].unique()}, name="squad_quality")
    for team, g in roster.groupby("team"):
        names = g["player"].unique()
        pre_min = rates.reindex(names)["minutes"].fillna(0.0)
        rate = rates.reindex(names)["rate"].fillna(prior)
        w = pre_min + M0  # confidence weight: history + prior mass
        out[team] = float((rate * w).sum() / w.sum() / prior)  # normalized: 1.0 = average
    return pd.Series(out, name="squad_quality")


def fit_squad_coef(lam_v1: np.ndarray, goals: np.ndarray, sq_diff: np.ndarray,
                   tau: float = 0.1) -> float:
    """One-parameter Poisson MAP for the squad adjustment on training rows.

    Ridge prior b ~ N(0, tau^2): with small training windows (48 matches for
    the first evaluation) an unregularized fit pegs at the search bound —
    classic overfit. tau=0.1 says 'a one-sigma squad gap moves lambda by
    <= ~10%' a priori; the data must earn anything larger."""
    z = (sq_diff - sq_diff.mean()) / max(sq_diff.std(), 1e-9)

    def nll(b):
        lam = lam_v1 * np.exp(b * z)
        return float(np.sum(lam - goals * np.log(lam)) + 0.5 * b**2 / tau**2)

    res = minimize_scalar(nll, bounds=(-1.0, 1.0), method="bounded")
    return float(res.x)


def adjust_lambda_for_xi(lam: float, squad_player_rates: pd.Series, xi_names: list[str],
                         gamma: float = 1.0) -> float:
    """Lineup-day hook: scale lambda by the announced XI's share of expected
    attacking output vs the squad's best XI. gamma=1 is assumption A12."""
    best11 = squad_player_rates.nlargest(11).sum()
    xi = squad_player_rates.reindex(xi_names).fillna(squad_player_rates.median()).nlargest(11).sum()
    if best11 <= 0:
        return lam
    return float(lam * (xi / best11) ** gamma)
