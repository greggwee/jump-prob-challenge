"""Squad feature tests: synthetic recovery of b_sq, quality normalization,
and the lineup hook's direction."""

import numpy as np
import pandas as pd
import pytest

from jumpprob.features.squad import adjust_lambda_for_xi, fit_squad_coef, squad_quality


def test_fit_squad_coef_recovery():
    rng = np.random.default_rng(2)
    n = 30_000
    lam = rng.uniform(0.8, 2.2, n)
    sqd = rng.normal(0, 1, n)
    b_true = 0.12
    goals = rng.poisson(lam * np.exp(b_true * sqd))
    assert fit_squad_coef(lam, goals, sqd) == pytest.approx(b_true, abs=0.02)


def test_squad_quality_unknowns_are_average():
    players = pd.DataFrame({
        "player": ["x", "y"], "team": ["NewLand", "NewLand"],
        "tournament": ["t2"] * 2, "date": pd.to_datetime(["2024-06-01"] * 2),
        "minutes": [90, 90], "sot": [1, 0],
    })
    # no history before as_of -> every roster scores exactly 1.0 (average)
    q = squad_quality(players, "t2", pd.Timestamp("2024-01-01"))
    assert q["NewLand"] == pytest.approx(1.0)


def test_xi_hook_direction():
    rates = pd.Series({f"p{i}": r for i, r in enumerate(np.linspace(0.05, 0.30, 16))})
    best_xi = list(rates.nlargest(11).index)
    weak_xi = list(rates.nsmallest(11).index)
    lam = 1.5
    assert adjust_lambda_for_xi(lam, rates, best_xi) == pytest.approx(lam)
    assert adjust_lambda_for_xi(lam, rates, weak_xi) < lam
