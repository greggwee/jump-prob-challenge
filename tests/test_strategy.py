"""Tests for evidence-gated extremizing (rec #2)."""

import pytest

from jumpprob.models.strategy import extremize, field_proxy, k_for


class TestExtremize:
    def test_k1_is_identity(self):
        for belief, field in [(0.7, 0.5), (0.3, 0.55), (0.9, 0.8)]:
            assert extremize(belief, field, 1.0) == pytest.approx(belief, abs=1e-9)

    def test_belief_equals_field_no_gamble(self):
        # if we agree with the field, no k changes the submission
        for k in (1.0, 1.5, 3.0):
            assert extremize(0.62, 0.62, k) == pytest.approx(0.62, abs=1e-9)

    def test_k_amplifies_deviation(self):
        # belief above field -> extremized further above; below -> further below
        assert extremize(0.70, 0.50, 2.0) > 0.70
        assert extremize(0.30, 0.50, 2.0) < 0.30
        # symmetric in logit space
        hi = extremize(0.70, 0.50, 2.0)
        lo = extremize(0.30, 0.50, 2.0)
        assert hi + lo == pytest.approx(1.0, abs=1e-9)

    def test_extremize_stays_in_bounds(self):
        assert 0 < extremize(0.95, 0.5, 3.0) < 1


class TestFieldProxy:
    def test_market_quote_lightly_dulled(self):
        # a direct market quote -> field near the market, slightly toward 50
        f = field_proxy("win", 0.8, {}, market=0.8)
        assert 0.5 < f < 0.8

    def test_no_line_regresses_belief_toward_base(self):
        base = {"cards_ge": 0.64}
        # our belief 0.40, base 0.64 -> field sits between (crowd moves halfway)
        f = field_proxy("cards_ge", 0.40, base, market=None, rho=0.5)
        assert 0.40 < f < 0.64

    def test_no_line_belief_equals_base(self):
        base = {"x": 0.30}
        assert field_proxy("x", 0.30, base) == pytest.approx(0.30, abs=1e-9)


class TestKFor:
    def test_default_honest(self):
        assert k_for("count", {}, 1.0, 1.0) == 1.0
        assert k_for("count", {"count": 1.0}, 1.0, 1.0) == 1.0

    def test_confirmed_edge_grants_leverage(self):
        assert k_for("count", {"count": 1.3}, 1.0, 1.0) == pytest.approx(1.3)

    def test_stage_multiplier_scales_excess(self):
        # knockout (2x) gains 1.5x on the excess; final (3x) gains 2x
        assert k_for("count", {"count": 1.3}, 2.0, 1.0) == pytest.approx(1.0 + 0.3 * 1.5)
        assert k_for("count", {"count": 1.3}, 3.0, 1.0) == pytest.approx(1.0 + 0.3 * 2.0)

    def test_aggression_scales_excess(self):
        assert k_for("count", {"count": 1.4}, 1.0, 0.5) == pytest.approx(1.0 + 0.4 * 0.5)
