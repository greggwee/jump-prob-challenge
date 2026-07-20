"""Tests for evidence-gated extremizing."""

import pytest

import numpy as np

from jumpprob.models.market_anchor import inv_logit, logit
from jumpprob.models.strategy import (
    calibrate_field_sharpness, extremize, field_model, field_proxy, k_for)


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


class TestFieldModel:
    def test_endpoints(self):
        # lambda=1 -> field is the belief; lambda=0 -> field is 50
        assert field_model(0.8, 1.0) == pytest.approx(0.8)
        assert field_model(0.8, 0.0) == pytest.approx(0.5)
        assert field_model(0.2, 0.5) < 0.5  # belief below 50 stays below, shrunk

    def test_self_modulating_extremizer(self):
        """The composition extremize(belief, field_model(belief, lam), k) must
        equal inv_logit([lam + k(1-lam)] logit belief) — the elegant identity."""
        belief, lam, k = 0.72, 0.4, 1.6
        field = field_model(belief, lam)
        got = extremize(belief, field, k)
        coef = lam + k * (1 - lam)
        want = inv_logit(coef * logit(belief))
        assert got == pytest.approx(want, abs=1e-9)

    def test_clueless_field_gives_full_extremizing(self):
        # lam=0 (clueless field) -> coefficient = k (full leverage)
        belief, k = 0.7, 2.0
        got = extremize(belief, field_model(belief, 0.0), k)
        assert got == pytest.approx(inv_logit(k * logit(belief)), abs=1e-9)

    def test_sharp_field_gives_no_extremizing(self):
        # lam=1 (sharp field) -> no leverage regardless of k
        belief, k = 0.7, 3.0
        assert extremize(belief, field_model(belief, 1.0), k) == pytest.approx(belief, abs=1e-9)


class TestCalibrateFieldSharpness:
    def test_recovers_global_scale(self):
        """Generate outcomes from a known field sharpness; calibration recovers
        a multiplier that reproduces the observed field Brier."""
        rng = np.random.default_rng(3)
        n = 4000
        beliefs = rng.uniform(0.15, 0.85, n)
        families = rng.choice(["count", "player", "timing"], n)
        # true field at half the prior sharpness
        lam_true = np.array({"count": 0.40, "player": 0.45, "timing": 0.35}[f] * 0.5
                            for f in families) if False else \
                   np.array([{"count": 0.40, "player": 0.45, "timing": 0.35}[f] * 0.5 for f in families])
        f_true = inv_logit(lam_true * np.log(beliefs / (1 - beliefs)))
        y = (rng.random(n) < f_true).astype(float)
        observed = float(np.mean((f_true - y) ** 2))
        table, g = calibrate_field_sharpness(beliefs, y, families, observed)
        assert 0.3 < g < 0.8  # true scale was 0.5
        assert all(0 <= v <= 1 for v in table.values())


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


class TestRisk:
    GATE = {"offsides_ge": 0.5}
    BASE = {"offsides_ge": 0.525}

    def test_shrinks_gated_toward_base(self):
        from jumpprob.models.strategy import apply_risk
        # an over-confident low offsides is pulled toward the 0.525 base
        out = apply_risk(0.16, "offsides_ge", self.GATE, self.BASE)
        assert 0.16 < out < 0.525
        # symmetric: an over-confident high one is pulled down toward base
        hi = apply_risk(0.80, "offsides_ge", self.GATE, self.BASE)
        assert 0.525 < hi < 0.80

    def test_noop_for_ungated_archetype(self):
        from jumpprob.models.strategy import apply_risk
        # player props / edge archetypes are NOT in the gate -> untouched
        assert apply_risk(0.10, "player_sot_ge", self.GATE, self.BASE) == pytest.approx(0.10)
        assert apply_risk(0.90, "corners_gt", self.GATE, self.BASE) == pytest.approx(0.90)

    def test_base_rate_aware_keeps_justified_extreme(self):
        from jumpprob.models.strategy import apply_risk
        # if the model already sits at the base rate, nothing moves (no unjustified deviation)
        assert apply_risk(0.525, "offsides_ge", self.GATE, self.BASE) == pytest.approx(0.525)

    def test_risk_enabled_default_off(self, monkeypatch):
        import jumpprob.models.strategy as S
        monkeypatch.setattr(S, "_risk_env_loaded", True, raising=False)
        monkeypatch.setenv("RISK_GATE_ENABLE", "0")
        assert S.risk_enabled() is False
        monkeypatch.setenv("RISK_GATE_ENABLE", "1")
        assert S.risk_enabled() is True

    def test_both_teams_sot_shrinks_toward_base(self):
        # both-teams SoT is no-edge noise: confident 88/24 splits, Brier 0.33 vs
        # 0.21 for the base rate (the -89 RBP leak found 2026-06-22). The LIVE gate
        # must shrink it toward its empirical ~0.70 base so the over-confident calls
        # don't go out raw. Single base (n=10 too thin to split full/half reliably).
        from jumpprob.models.strategy import apply_risk, load_risk_gate, load_base_rates
        gate, base = load_risk_gate(), load_base_rates()
        assert "both_teams_sot_ge" in gate
        b = base["both_teams_sot_ge"]
        assert 0.6 < b < 0.8                                   # base lifted off the bogus 0.525
        hi = apply_risk(0.93, "both_teams_sot_ge", gate, base)  # over-confident call shrinks down
        assert b < hi < 0.93
        lo = apply_risk(0.10, "both_teams_sot_ge", gate, base)  # under-confident call shrinks up
        assert 0.10 < lo < b
