"""Unit tests for goal model math and market answers — synthetic, no data files."""

import numpy as np
import pandas as pd
import pytest

from jumpprob.models.goal_model import GoalModel, dc_grid, fit_goal_model
from jumpprob.simulate import markets


def make_model(rho=0.0):
    return GoalModel(b0=0.25, b1=0.4, b2=0.25, rho_dc=rho, half_life_days=730)


def test_grid_is_distribution():
    g = dc_grid(1.4, 1.1, -0.05)
    assert g.sum() == pytest.approx(1.0)
    assert (g >= 0).all()


def test_dc_rho_negative_raises_low_scores():
    indep = dc_grid(1.4, 1.1, 0.0)
    dc = dc_grid(1.4, 1.1, -0.1)
    assert dc[0, 0] > indep[0, 0]
    assert dc[1, 1] > indep[1, 1]
    assert dc[1, 0] < indep[1, 0]


def test_intensities_symmetry():
    m = make_model()
    la, lb = m.intensities(1800, 1800, a_is_home_ground=False)
    assert la == pytest.approx(lb)
    la2, lb2 = m.intensities(1900, 1800)
    assert la2 > la and lb2 < lb


def test_market_identities():
    """Coherence: win+draw+loss = 1; BTTS&3+ <= BTTS and <= 3+; monotone totals."""
    m = make_model(rho=-0.05)
    grid, la, lb = markets.grid_and_lams(m, 1900, 1750)
    assert markets.p_win(grid, "a") + markets.p_win(grid, "b") + markets.p_draw(grid) == pytest.approx(1.0)
    assert markets.p_btts_and_total_ge(grid, 3) <= min(markets.p_btts(grid), markets.p_total_ge(grid, 3))
    assert markets.p_total_ge(grid, 2) >= markets.p_total_ge(grid, 3)
    assert 0 < markets.p_scores_in_half(la, 2) < 1 - np.exp(-la) + 1e-9


def test_fit_recovers_known_parameters():
    """Generate synthetic matches from known betas; MLE must recover them."""
    rng = np.random.default_rng(7)
    n = 20_000
    b0, b1, b2 = 0.2, 0.35, 0.3
    elo_h = rng.normal(1700, 150, n)
    elo_a = rng.normal(1700, 150, n)
    neutral = rng.random(n) < 0.3
    d = (elo_h - elo_a) / 100
    lam_h = np.exp(b0 + b1 * d + b2 * (~neutral))
    lam_a = np.exp(b0 - b1 * d)
    df = pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
            "home_team": "H",
            "away_team": "A",
            "home_score": rng.poisson(lam_h),
            "away_score": rng.poisson(lam_a),
            "tournament": "Friendly",
            "neutral": neutral,
            "elo_home": elo_h,
            "elo_away": elo_a,
        }
    ).sort_values("date")
    model = fit_goal_model(df, as_of=pd.Timestamp("2024-03-01"), half_life_days=100_000)
    assert model.b0 == pytest.approx(b0, abs=0.03)
    assert model.b1 == pytest.approx(b1, abs=0.03)
    assert model.b2 == pytest.approx(b2, abs=0.04)
    assert abs(model.rho_dc) < 0.05  # data generated independent -> rho ~ 0
