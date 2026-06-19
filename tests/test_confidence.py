"""Phase 1 asymmetric confidence calibration (pure)."""

import pytest

from jumpprob.models.strategy import apply_confidence

GATE = {"player_ga": {"a": 1.0, "b": 0.0, "f": 0.15},
        "player_sot_ge": {"a": 0.999, "b": 0.008, "f": 0.15}}


def test_floor_raises_confident_low_bet():
    # the catastrophic case: we bet 7% on a score-or-assist that the crowd hedges higher
    assert apply_confidence(0.07, "player_ga", GATE) == pytest.approx(0.15)
    assert apply_confidence(0.11, "player_ga", GATE) == pytest.approx(0.15)


def test_floor_is_one_sided_keeps_high_confidence():
    # a correct high-confidence call must NOT be capped (only the low tail is floored)
    assert apply_confidence(0.92, "player_sot_ge", GATE) > 0.85
    assert apply_confidence(0.60, "player_ga", GATE) == pytest.approx(0.60, abs=1e-9)


def test_noop_for_ungated_archetype():
    # archetypes we have edge on (e.g. player_scores) are untouched
    assert apply_confidence(0.03, "player_scores", GATE) == pytest.approx(0.03)
    assert apply_confidence(0.05, "offsides_ge", GATE) == pytest.approx(0.05)


def test_empty_gate_is_inert():
    assert apply_confidence(0.04, "player_ga", {}) == pytest.approx(0.04)


def test_platt_then_floor_order():
    # a<1 compresses toward 0.5 first, then the floor binds
    g = {"x": {"a": 0.5, "b": 0.0, "f": 0.0}}
    # logit(0.1)=-2.197; a=0.5 -> -1.0986 -> sigmoid=0.25
    assert apply_confidence(0.1, "x", g) == pytest.approx(0.25, abs=1e-3)
