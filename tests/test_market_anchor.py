"""Tests for de-vigging and lambda inversion — pure math, no data files."""

import numpy as np
import pytest

from jumpprob.models.devig import devig_basic, devig_power, devig_shin
from jumpprob.models.goal_model import dc_grid
from jumpprob.models.market_anchor import blend, implied_lambdas
from jumpprob.simulate.markets import p_draw, p_total_ge, p_win


def fair_odds(probs, vig=0.05):
    """Build odds with proportional overround from true probabilities."""
    probs = np.asarray(probs)
    return 1.0 / (probs * (1 + vig))


class TestDevig:
    def test_all_methods_sum_to_one(self):
        odds = fair_odds([0.55, 0.27, 0.18])
        for fn in (devig_basic, devig_power, devig_shin):
            assert fn(odds).sum() == pytest.approx(1.0, abs=1e-9)

    def test_basic_roundtrip_proportional_vig(self):
        true = np.array([0.55, 0.27, 0.18])
        assert devig_basic(fair_odds(true)) == pytest.approx(true, abs=1e-12)

    def test_shin_reduces_longshot_more_than_basic(self):
        """Shin should shave more probability off the longshot than basic
        normalization does (FLB correction direction)."""
        true = np.array([0.70, 0.22, 0.08])
        # bookmaker applies FLB: inflates longshot price more
        pi_biased = true + np.array([0.01, 0.015, 0.035])
        odds = 1.0 / pi_biased
        p_shin = devig_shin(odds)
        p_basic = devig_basic(odds)
        assert p_shin[2] < p_basic[2]          # longshot shaved harder
        assert p_shin[0] > p_basic[0]          # favourite restored
        assert abs(p_shin[0] - true[0]) < abs(p_basic[0] - true[0])

    def test_power_behaves_like_shin_directionally(self):
        true = np.array([0.70, 0.22, 0.08])
        pi_biased = true + np.array([0.01, 0.015, 0.035])
        p_pow = devig_power(1.0 / pi_biased)
        p_basic = devig_basic(1.0 / pi_biased)
        assert p_pow[2] < p_basic[2]

    def test_rejects_invalid_odds(self):
        with pytest.raises(ValueError):
            devig_basic([0.9, 2.0, 3.0])


class TestLambdaInversion:
    def test_roundtrip(self):
        """Generate market probs from known lambdas; inversion must recover them."""
        la_true, lb_true, rho = 1.7, 0.9, -0.05
        grid = dc_grid(la_true, lb_true, rho)
        la, lb = implied_lambdas(
            p_win(grid, "a"), p_draw(grid), p_win(grid, "b"), p_total_ge(grid, 3), rho=rho
        )
        assert la == pytest.approx(la_true, abs=0.01)
        assert lb == pytest.approx(lb_true, abs=0.01)

    def test_roundtrip_without_totals(self):
        la_true, lb_true, rho = 1.2, 1.2, -0.05
        grid = dc_grid(la_true, lb_true, rho)
        la, lb = implied_lambdas(p_win(grid, "a"), p_draw(grid), p_win(grid, "b"), rho=rho)
        assert la == pytest.approx(la_true, abs=0.05)
        assert lb == pytest.approx(lb_true, abs=0.05)


class TestBlend:
    def test_blend_endpoints_and_midpoint(self):
        assert blend([0.6, 0.6], [0.5, 0.5]) == pytest.approx(0.6)
        assert blend([0.8, 0.2], [1.0, 0.0]) == pytest.approx(0.8)
        mid = blend([0.8, 0.2], [0.5, 0.5])
        assert mid == pytest.approx(0.5)  # symmetric logits cancel

    def test_blend_weights_validated(self):
        with pytest.raises(ValueError):
            blend([0.5, 0.5], [0.7, 0.7])
