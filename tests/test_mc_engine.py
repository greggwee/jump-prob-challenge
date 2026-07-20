"""MC engine verification: coherence properties, reduction to the analytic
grid in the uncoupled limit, and seed convergence."""

import numpy as np
import pytest

from jumpprob.models.count_model import CountModel
from jumpprob.models.dynamics import Dynamics
from jumpprob.models.goal_model import GoalModel, dc_grid
from jumpprob.simulate import markets as mk
from jumpprob.simulate.mc_engine import MCEngine, answer

GM = GoalModel(b0=0.22, b1=0.35, b2=0.25, rho_dc=-0.05, half_life_days=730)


def make_count_model():
    cm = CountModel()
    for fam, (c1h, c2h) in {"corners": (0.7, 0.8), "cards": (-0.3, 0.3), "fouls": (1.8, 1.9),
                            "shots_on_target": (0.5, 0.7), "offsides": (-0.4, -0.3)}.items():
        cm.params[fam] = {"c0_1": c1h, "c0_2": c2h, "c1": 0.3, "c2": -0.1, "k": 10.0}
    return cm


def make_dynamics(flat=False):
    d = Dynamics()
    for fam in ("goal", "corner", "card", "foul", "sot", "offside"):
        d.alpha[fam] = [0.0] * 6 if flat else [0.0, 0.05, 0.2, 0.25, 0.3, 0.6]
        d.beta[fam] = [0.0] * 5 if flat else [0.25, 0.15, 0.0, -0.1, -0.15]
    return d


@pytest.fixture(scope="module")
def engine():
    return MCEngine(GM, make_count_model(), make_dynamics(), n_draws=60_000, seed=1)


def test_reduction_to_independent_grid():
    """Coupling off, frailty off, flat profile -> marginals match dc_grid(rho=0)."""
    eng = MCEngine(GM, None, make_dynamics(), n_draws=400_000, coupled=False,
                   frailty=False, seed=3)
    elo_a, elo_b = 1900.0, 1760.0
    la, lb = GM.intensities(elo_a, elo_b)
    res = eng.simulate(elo_a, elo_b)
    grid = dc_grid(la, lb, 0.0)
    assert answer(res, "win", "a") == pytest.approx(mk.p_win(grid, "a"), abs=0.004)
    assert answer(res, "draw") == pytest.approx(mk.p_draw(grid), abs=0.004)
    assert answer(res, "total_ge", threshold=3) == pytest.approx(mk.p_total_ge(grid, 3), abs=0.004)
    assert answer(res, "btts") == pytest.approx(mk.p_btts(grid), abs=0.004)
    assert res.score_a.mean() == pytest.approx(la, abs=0.01)
    assert res.score_b.mean() == pytest.approx(lb, abs=0.01)


def test_lambda_override_matches_analytic_grid():
    """simulate(lam=...) bypasses Elo and uses the given intensities exactly —
    uncoupled marginals must match dc_grid(la, lb, 0) (used for odds-anchored
    timing markets)."""
    eng = MCEngine(GM, None, make_dynamics(), n_draws=400_000, coupled=False,
                   frailty=False, seed=8)
    la, lb = 1.85, 0.95
    res = eng.simulate(0.0, 0.0, lam=(la, lb))
    assert res.score_a.mean() == pytest.approx(la, abs=0.01)
    assert res.score_b.mean() == pytest.approx(lb, abs=0.01)
    grid = dc_grid(la, lb, 0.0)
    assert answer(res, "win", "a") == pytest.approx(mk.p_win(grid, "a"), abs=0.004)
    assert answer(res, "ht_tied") == pytest.approx(
        float(np.trace(dc_grid((1 - mk.SECOND_HALF_SHARE) * la,
                               (1 - mk.SECOND_HALF_SHARE) * lb, 0.0))), abs=0.02)


def test_coherence_properties(engine):
    rng = np.random.default_rng(9)
    for _ in range(20):
        ea, eb = rng.uniform(1500, 2100, 2)
        res = engine.simulate(ea, eb)
        p_win_a = answer(res, "win", "a")
        p_win_b = answer(res, "win", "b")
        p_draw = answer(res, "draw")
        assert p_win_a + p_win_b + p_draw == pytest.approx(1.0, abs=1e-9)
        # joints bounded by marginals
        p_btts = answer(res, "btts")
        p_t3 = answer(res, "total_ge", threshold=3)
        p_joint = answer(res, "btts_and_total_ge", threshold=3)
        assert p_joint <= min(p_btts, p_t3) + 1e-9
        p_fgw = answer(res, "first_goal_and_win", "a")
        assert p_fgw <= p_win_a + 1e-9
        # halves below full match
        p_score_2h = answer(res, "team_scores_half", "a", half=2)
        p_score_any = answer(res, "team_goals_ge", "a", threshold=1)
        assert p_score_2h <= p_score_any + 1e-9
        assert answer(res, "half_total_ge", threshold=2, half=2) <= answer(res, "total_ge", threshold=2) + 1e-9
        # monotone thresholds
        assert answer(res, "team_corners_ge", "a", threshold=3) >= answer(res, "team_corners_ge", "a", threshold=5)
        # complements
        p_le2 = answer(res, "total_le", threshold=2)
        assert p_le2 + p_t3 == pytest.approx(1.0, abs=1e-9)
        # ht components partition
        assert (answer(res, "ht_tied") + answer(res, "ht_leading", "a")
                + answer(res, "ht_leading", "b")) == pytest.approx(1.0, abs=1e-9)


def test_state_coupling_direction(engine):
    """Trailing boost must lift comeback probabilities vs the uncoupled engine."""
    unc = MCEngine(GM, make_count_model(), make_dynamics(), n_draws=60_000,
                   coupled=False, seed=1)
    ea, eb = 1700.0, 1700.0
    p_cpl = answer(engine.simulate(ea, eb), "draw")
    p_unc = answer(unc.simulate(ea, eb), "draw")
    # trailing teams scoring more -> more equalizers -> more draws
    assert p_cpl > p_unc


def test_seed_convergence():
    dyn = make_dynamics()
    archs = [("win", "a", None, None), ("draw", None, None, None),
             ("btts", None, None, None), ("ht_tied", None, None, None),
             ("team_corners_ge", "a", 5, None), ("cards_ge", None, 2, 2)]
    answers = []
    for seed in (101, 202):
        eng = MCEngine(GM, make_count_model(), dyn, n_draws=100_000, seed=seed)
        res = eng.simulate(1850.0, 1750.0)
        answers.append([answer(res, a, s, threshold=t, half=h) for a, s, t, h in archs])
    np.testing.assert_allclose(answers[0], answers[1], atol=0.005)


def test_posterior_predictive_widens(engine):
    """With beta uncertainty, extreme probabilities move toward the middle."""
    cov = np.diag([0.002, 0.004, 0.001])  # synthetic but realistic-scale variance
    pp = MCEngine(GM, make_count_model(), make_dynamics(), n_draws=100_000,
                  beta_cov=cov, seed=5)
    plain = MCEngine(GM, make_count_model(), make_dynamics(), n_draws=100_000, seed=5)
    ea, eb = 2100.0, 1500.0  # heavy favourite -> extreme p
    p_pp = answer(pp.simulate(ea, eb), "win", "a")
    p_plain = answer(plain.simulate(ea, eb), "win", "a")
    assert p_pp < p_plain  # pulled toward the middle
    assert p_pp > 0.5


def test_either_team_half_goals_ge_counts_cells():
    """Endgame kind (2026-07-17): the same-half team-goals lambda must fire on any
    (team, half) cell reaching the threshold — verified on a hand-built SimResult."""
    from jumpprob.simulate.mc_engine import SimResult
    res = SimResult(
        score_a=np.array([2, 0, 3, 0]),     # draw 1: 2 in H1(a); draw 3: 2 in H2(a)
        score_b=np.array([0, 2, 0, 0]),     # draw 2: 1+1 split -> no cell reaches 2
        ht_a=np.array([2, 0, 1, 0]),
        ht_b=np.array([0, 1, 0, 0]),
        first_team=np.zeros(4), first_team_h2=np.zeros(4), counts={},
    )
    assert answer(res, "either_team_half_goals_ge", threshold=2) == pytest.approx(0.5)
    assert answer(res, "either_team_half_goals_ge", threshold=3) == pytest.approx(0.0)
