"""Tests for de-vigging and lambda inversion — pure math, no data files."""

import numpy as np
import pytest

from jumpprob.models.devig import consensus, devig_basic, devig_power, devig_shin, overround
from jumpprob.models.goal_model import dc_grid
from jumpprob.models.market_anchor import blend, blend_lambdas, implied_lambdas
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


class TestConsensus:
    def test_identical_books_reduce_to_single_devig(self):
        odds = fair_odds([0.55, 0.27, 0.18])
        books = [("a", odds), ("b", odds), ("c", odds)]
        np.testing.assert_allclose(consensus(books), devig_shin(odds), atol=1e-9)

    def test_sums_to_one(self):
        books = [("soft", fair_odds([0.5, 0.3, 0.2], vig=0.08)),
                 ("sharp", fair_odds([0.5, 0.3, 0.2], vig=0.02))]
        assert consensus(books).sum() == pytest.approx(1.0, abs=1e-9)

    def test_sharper_book_dominates(self):
        """A tight-margin book pulls the consensus toward its view more than a
        soft book does (inverse-overround weighting)."""
        sharp_true = np.array([0.62, 0.24, 0.14])
        soft_true = np.array([0.50, 0.28, 0.22])
        books = [
            ("sharpbook", fair_odds(sharp_true, vig=0.02)),  # 2% margin
            ("softbook", fair_odds(soft_true, vig=0.10)),    # 10% margin
        ]
        # no named multipliers -> weighting is purely margin-driven
        c = consensus(books, sharp_weights={"sharpbook": 1.0, "softbook": 1.0})
        # consensus should sit closer to the sharp book's home prob than a plain
        # average (0.56) would
        assert c[0] > 0.575
        assert abs(c[0] - sharp_true[0]) < abs(c[0] - soft_true[0])

    def test_prediction_market_pseudo_book(self):
        """A zero-vig prediction-market mid-price (odds = 1/p) folds in and,
        via ~0 overround, earns strong weight."""
        kalshi_p = np.array([0.58, 0.26, 0.16])
        kalshi_odds = 1.0 / kalshi_p  # overround ~ 0
        soft = fair_odds([0.45, 0.30, 0.25], vig=0.08)
        books = [("kalshi", list(kalshi_odds)), ("softbook", soft)]
        c = consensus(books)
        assert overround(kalshi_odds) < 1e-9
        assert abs(c[0] - kalshi_p[0]) < abs(c[0] - devig_shin(soft)[0])

    def test_overround_measurement(self):
        assert overround(fair_odds([0.5, 0.3, 0.2], vig=0.05)) == pytest.approx(0.05, abs=1e-9)


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


class TestBlendLambdas:
    def test_endpoints(self):
        model, market = (1.2, 1.0), (1.8, 0.7)
        assert blend_lambdas(model, market, 1.0) == pytest.approx(market)
        assert blend_lambdas(model, market, 0.0) == pytest.approx(model)

    def test_geometric_midpoint(self):
        la, lb = blend_lambdas((1.0, 1.0), (4.0, 0.25), 0.5)
        assert la == pytest.approx(2.0)   # sqrt(1*4)
        assert lb == pytest.approx(0.5)   # sqrt(1*0.25)

    def test_market_pulls_toward_it(self):
        """A market that rates team A much stronger pulls blended lambda_a up."""
        model = (1.0, 1.3)   # model: A weaker
        market = (1.9, 0.9)  # market: A stronger (the USA-vs-PAR situation)
        la, lb = blend_lambdas(model, market, 0.85)
        assert la > 1.6 and lb < 1.0
        # win prob from the blended grid should move decisively toward A
        from jumpprob.models.goal_model import dc_grid
        from jumpprob.simulate.markets import p_win
        p_a_model = p_win(dc_grid(*model, -0.05), "a")
        p_a_blend = p_win(dc_grid(la, lb, -0.05), "a")
        assert p_a_blend > p_a_model + 0.15
