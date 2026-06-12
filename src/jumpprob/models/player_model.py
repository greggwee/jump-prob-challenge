"""Player prop rates with hierarchical shrinkage, point-in-time.

Rate estimate for player i before match m:
    rate_i = (events_before + prior_rate * M0) / (minutes_before + M0)
a Gamma-Poisson posterior mean: M0 pseudo-minutes of the population prior.
P(1+ event | plays M minutes) = 1 - exp(-rate * M)   (Poisson thinning).

In live use, M is mixed over a minutes distribution (Jensen — see PLAN §3.1);
in backtest we condition on actual minutes to isolate rate quality.

Population prior rates are computed from the TRAINING window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_rates(players: pd.DataFrame, m0: float, prior_sot: float, prior_goal: float) -> pd.DataFrame:
    """Add shrunk pre-match rate columns (point-in-time: shifted cumsums)."""
    df = players.sort_values(["player", "date"]).copy()
    g = df.groupby("player")
    for col in ("minutes", "sot", "goals"):
        df[f"{col}_pre"] = g[col].cumsum() - df[col]
    df["rate_sot"] = (df["sot_pre"] + prior_sot * m0) / (df["minutes_pre"] + m0)
    df["rate_goal"] = (df["goals_pre"] + prior_goal * m0) / (df["minutes_pre"] + m0)
    return df


def p_at_least_one(rate_per_min: pd.Series | float, minutes: pd.Series | float):
    return 1.0 - np.exp(-np.asarray(rate_per_min, dtype=float) * np.asarray(minutes, dtype=float))
