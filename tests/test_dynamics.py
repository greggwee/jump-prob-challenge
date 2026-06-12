"""Synthetic recovery test: generate events from known alpha/beta along
simulated state paths; the conditional-multinomial estimator must recover them."""

import numpy as np
import pandas as pd
import pytest

from jumpprob.models.dynamics import BUCKET, fit_dynamics, margin_state, state_paths

ALPHA_TRUE = {"goal": [0.0, 0.1, 0.2, 0.25, 0.3, 0.5], "corner": [0.0, 0.0, 0.1, 0.2, 0.4, 0.7]}
BETA_TRUE = {"goal": [0.3, 0.15, 0.0, -0.1, -0.25], "corner": [0.5, 0.3, 0.0, -0.2, -0.4]}


def _simulate(n_matches=4000, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    base = {"goal": 1.3 / 90, "corner": 5.0 / 90}
    for mid in range(n_matches):
        margin = 0
        for slot in range(90):
            b = BUCKET[slot]
            for team, sgn in (("H", 1), ("A", -1)):
                s = margin_state(sgn * margin)
                for fam in ("goal", "corner"):
                    h = base[fam] * np.exp(ALPHA_TRUE[fam][b] + BETA_TRUE[fam][s])
                    k = rng.poisson(h)
                    for _ in range(k):
                        rows.append({"match_id": mid, "team": team, "period": 1 if slot < 45 else 2,
                                     "slot": slot, "family": fam,
                                     "home_team": "H", "away_team": "A"})
                        if fam == "goal":
                            margin += sgn
    return pd.DataFrame(rows)


def test_recovery():
    stream = _simulate()
    dyn = fit_dynamics(pd.concat([stream]))
    for fam in ("goal", "corner"):
        np.testing.assert_allclose(dyn.alpha[fam], ALPHA_TRUE[fam], atol=0.08)
        np.testing.assert_allclose(dyn.beta[fam], BETA_TRUE[fam], atol=0.08)


def test_state_paths_track_goals():
    stream = pd.DataFrame([
        {"match_id": 1, "team": "H", "period": 1, "slot": 10, "family": "goal",
         "home_team": "H", "away_team": "A"},
        {"match_id": 1, "team": "A", "period": 2, "slot": 70, "family": "goal",
         "home_team": "H", "away_team": "A"},
    ])
    p = state_paths(stream)[1]
    assert p["H"][10] == margin_state(0)      # state at START of scoring slot
    assert p["H"][11] == margin_state(1)      # leading after
    assert p["A"][11] == margin_state(-1)
    assert p["H"][71] == margin_state(0)      # equalizer restores level
