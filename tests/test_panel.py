"""Phase L0: routing, robust aggregation, cache I/O, and the gated blend.

The load-bearing invariant under test: at the default env (all role weights 0),
apply_expert returns the model probability UNCHANGED for every action — so live
submissions are byte-for-byte identical until a weight is deliberately flipped.
"""

import json
import time

import pytest

from jumpprob.agents import panel as P


# --- routing ---

def test_market_priced_excluded():
    for arch in ("win", "total_ge", "total_le", "btts"):
        assert P.route(arch) == ("tactical", P.EXCLUDE)


def test_proven_edge_shadowed():
    for arch in ("corners_gt", "sot_gt", "team_sot_ge", "player_scores"):
        role, action = P.route(arch)
        assert action == P.SHADOW


def test_flagged_player_props_routing():
    assert P.route("player_ga") == ("lineup", P.BLEND)
    assert P.route("player_sot_ge") == ("lineup", P.BLEND_CAPPED)


def test_weak_counts_blend():
    for arch in ("offsides_ge", "fouls_gt", "cards_ge", "btts_and_total_ge"):
        role, action = P.route(arch)
        assert action == P.BLEND


def test_unknown_archetype_excluded():
    assert P.route("some_new_market_2027") == P.DEFAULT_ROUTE
    assert P.DEFAULT_ROUTE[1] == P.EXCLUDE


# --- master switch + gate ---

def test_expert_switch_semantics(monkeypatch):
    # set explicit values (these win over .env, so the test is hermetic).
    for v in ("0", "false", "no", ""):
        monkeypatch.setenv("AI_EXPERT_ENABLE", v)
        assert P.expert_enabled() is False
    for v in ("1", "true", "yes"):
        monkeypatch.setenv("AI_EXPERT_ENABLE", v)
        assert P.expert_enabled() is True


def test_load_gate_missing_is_empty():
    assert P.load_gate("does/not/exist.json") == {}


def test_load_gate_roundtrip(tmp_path):
    path = tmp_path / "expert_gate.json"
    path.write_text('{"player_ga": {"mode": "blend", "w": 0.5}}')
    g = P.load_gate(path)
    assert g["player_ga"]["w"] == 0.5


# --- ensemble-informed one-sided floor (the player_sot_ge tool) ---

def test_ensemble_floor_lifts_deep_low():
    # model egregiously below the expert -> lifted to f*expert
    assert P.ensemble_floor(0.09, 0.55, f=0.5) == pytest.approx(0.275)


def test_ensemble_floor_leaves_moderate_untouched():
    # model already above f*expert -> unchanged (body not disturbed)
    assert P.ensemble_floor(0.15, 0.25, f=0.5) == pytest.approx(0.15)


def test_ensemble_floor_never_lowers():
    assert P.ensemble_floor(0.8, 0.2, f=0.5) == pytest.approx(0.8)
    assert P.ensemble_floor(0.3, None, f=0.5) == pytest.approx(0.3)


# --- robust aggregation (trim-median in log-odds) ---

def test_combine_empty_is_none():
    assert P.combine_logodds([]) is None
    assert P.combine_logodds([None, None]) is None


def test_combine_single_and_pair():
    assert P.combine_logodds([0.5]) == pytest.approx(0.5)
    assert P.combine_logodds([0.4, 0.6]) == pytest.approx(0.5, abs=1e-9)


def test_combine_trims_rogue_model():
    # three sane panelists + one rogue extreme: the rogue must be trimmed away.
    sane = [0.50, 0.52, 0.48]
    with_rogue = sane + [0.02]
    out = P.combine_logodds(with_rogue)
    assert out == pytest.approx(0.50, abs=0.03)  # ~unchanged, rogue trimmed


def test_combine_median_of_three():
    assert P.combine_logodds([0.2, 0.5, 0.9]) == pytest.approx(0.5)


# --- blend math ---

def test_blend_weight_zero_is_identity():
    assert P.blend_logodds(0.11, 0.44, w=0.0) == pytest.approx(0.11)


def test_blend_weight_one_is_expert():
    assert P.blend_logodds(0.11, 0.44, w=1.0) == pytest.approx(0.44, abs=1e-6)


def test_blend_cap_limits_pull():
    # uncapped would move a lot; cap bounds the logit-space move.
    capped = P.blend_logodds(0.11, 0.90, w=1.0, cap=0.5)
    uncapped = P.blend_logodds(0.11, 0.90, w=1.0, cap=None)
    assert capped < uncapped
    assert capped > 0.11  # still pulled up, just not all the way


def test_blend_none_expert_is_identity():
    assert P.blend_logodds(0.3, None, w=0.5) == pytest.approx(0.3)


# --- cache I/O ---

def test_write_load_roundtrip(tmp_path):
    path = tmp_path / "ai_panel.json"
    entries = {"abc123": {"p_model": 0.11, "p_ensemble": 0.44, "generated_at": time.time(),
                          "action": "blend_capped"}}
    P.write_panel(entries, path)
    loaded = P.load_panel(path)
    assert "abc123" in loaded
    assert loaded["abc123"]["p_ensemble"] == pytest.approx(0.44)


def test_load_missing_file_is_empty():
    assert P.load_panel("does/not/exist.json") == {}


def test_stale_entry_filtered(tmp_path, monkeypatch):
    path = tmp_path / "ai_panel.json"
    old = time.time() - 100 * 3600  # 100h old
    P.write_panel({"old": {"p_ensemble": 0.5, "generated_at": old}}, path)
    monkeypatch.setenv("AI_PANEL_MAX_AGE_H", "36")
    assert P.load_panel(path) == {}                 # filtered by freshness
    assert "old" in P.load_panel(path, check_age=False)  # but present when unchecked


# --- the daemon-side apply: shadow-by-default invariants ---

def test_apply_exclude_is_pure_model():
    p, info = P.apply_expert("m1", "win", 0.6, {"m1": {"p_ensemble": 0.9}})
    assert p == 0.6
    assert info["action"] == P.EXCLUDE
    assert info["expert"] is None  # excluded markets never even read the panel


def test_apply_cache_miss_is_pure_model():
    p, info = P.apply_expert("m1", "player_ga", 0.2, {})
    assert p == 0.2
    assert info.get("miss") is True


def test_apply_master_switch_off_is_pure_model(monkeypatch):
    # the load-bearing safety invariant: even WITH a gate, nothing moves unless armed.
    monkeypatch.setenv("AI_EXPERT_ENABLE", "0")
    gate = {"player_ga": {"mode": "blend", "w": 1.0}}
    p, info = P.apply_expert("m1", "player_ga", 0.2, {"m1": {"p_ensemble": 0.6}}, gate)
    assert p == pytest.approx(0.2)                     # disabled -> unchanged
    assert info["expert"] == pytest.approx(0.6)        # but logged


def test_apply_shadow_never_blends_even_armed(monkeypatch):
    monkeypatch.setenv("AI_EXPERT_ENABLE", "1")
    gate = {"corners_gt": {"mode": "blend", "w": 1.0}}  # even with a gate entry,
    p, info = P.apply_expert("m1", "corners_gt", 0.3, {"m1": {"p_ensemble": 0.8}}, gate)
    assert p == 0.3                                    # shadow route is protected
    assert info["expert"] == pytest.approx(0.8)


def test_apply_no_gate_entry_is_pure_model(monkeypatch):
    monkeypatch.setenv("AI_EXPERT_ENABLE", "1")
    p, _ = P.apply_expert("m1", "player_ga", 0.2, {"m1": {"p_ensemble": 0.6}}, {})
    assert p == pytest.approx(0.2)                     # armed but no gate -> unchanged


def test_apply_blend_moves_when_armed(monkeypatch):
    monkeypatch.setenv("AI_EXPERT_ENABLE", "1")
    gate = {"player_ga": {"mode": "blend", "w": 1.0}}
    p, info = P.apply_expert("m1", "player_ga", 0.2, {"m1": {"p_ensemble": 0.6}}, gate)
    assert p == pytest.approx(0.6, abs=1e-6)
    assert info["mode"] == "blend"


def test_apply_asym_floor_when_armed(monkeypatch):
    monkeypatch.setenv("AI_EXPERT_ENABLE", "1")
    gate = {"player_sot_ge": {"mode": "asym_floor", "f": 0.5}}
    # deep-low gets floored to 0.5*ensemble; nothing else
    p, info = P.apply_expert("m1", "player_sot_ge", 0.09, {"m1": {"p_ensemble": 0.55}}, gate)
    assert p == pytest.approx(0.275)
    assert info["mode"] == "asym_floor"
