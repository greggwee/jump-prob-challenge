"""AI expert panel — routing, cache I/O, and the daemon-side gated blend.

This module is what the LIVE runner imports. It does NOT import anthropic and
never calls an LLM: the panel is produced OUT OF BAND
(scripts/ai_panel_prep.py -> scripts/ai_panel_write.py -> data/processed/ai_panel.json)
and this module only reads that cache and applies a gated, per-archetype blend.

Invariants (mirror docs/LLM-EXPERT-PLAN.md §3 — these never bend):
  * EXCLUDE archetypes (market-priced / sharp) never consult the panel.
  * SHADOW archetypes (proven edge) are logged but NEVER blended, even after a
    role weight is flipped on — they are protected.
  * BLEND / BLEND_CAPPED archetypes use the per-role weight (env, default 0).
  * Cache miss or stale entry -> pure model (fallback-safe).
  * Default state (all weights 0) -> submissions are byte-for-byte unchanged.

Routing uses the REAL parser archetype strings (verified 2026-06-20 against
jumpprob.submit.parser); friendly names live in the plan doc, not here.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

PANEL_PATH = Path("data/processed/ai_panel.json")

EXCLUDE, SHADOW, BLEND, BLEND_CAPPED = "exclude", "shadow", "blend", "blend_capped"

# archetype -> (role, action). See docs/LLM-EXPERT-PLAN.md §4.
#   exclude       : market-priced / sharp — the LLM must never move it.
#   shadow        : we have validated OOS edge — log the expert, never blend.
#   blend         : weak / no-edge / model-blind — blend at the role weight.
#   blend_capped  : like blend, but cap how far the expert can pull (player_sot_ge tail).
ROUTING: dict[str, tuple[str, str]] = {
    # --- market-priced goal markets: sharp, never touch ---
    "win": ("tactical", EXCLUDE),
    "draw": ("tactical", EXCLUDE),
    "btts": ("tactical", EXCLUDE),
    "total_ge": ("tactical", EXCLUDE),
    "total_le": ("tactical", EXCLUDE),
    # --- proven-edge counts (corners +0.045, SoT +0.036): shadow, protect our wins ---
    "corners_gt": ("tactical", SHADOW),
    "corners_total_ge": ("tactical", SHADOW),
    "team_corners_ge": ("tactical", SHADOW),
    "sot_gt": ("tactical", SHADOW),
    "total_sot_ge": ("tactical", SHADOW),
    "total_sot_half_ge": ("tactical", SHADOW),
    "team_sot_ge": ("tactical", SHADOW),
    "both_teams_sot_ge": ("tactical", SHADOW),
    # --- goal-model-derived team/half-state (we have goal-backbone edge): shadow ---
    "team_goals_ge": ("tactical", SHADOW),
    "ht_leading": ("tactical", SHADOW),
    "ht_tied": ("tactical", SHADOW),
    # --- rare discrete events (near-random): shadow, log only ---
    "pen_or_red": ("tactical", SHADOW),
    "penalty_awarded": ("tactical", SHADOW),
    # --- weak / no-edge counts: blend (low weight, set in L2) ---
    "offsides_ge": ("tactical", BLEND),
    "fouls_gt": ("tactical", BLEND),
    "cards_ge": ("tactical", BLEND),
    "cards_gt": ("tactical", BLEND),
    "team_cards_ge": ("tactical", BLEND),
    # --- half-specific scoring (thinning approx): blend ---
    "team_scores_half": ("tactical", BLEND),
    "half_total_ge": ("tactical", BLEND),
    "goals_gt_half": ("tactical", BLEND),
    "first_goal_half": ("tactical", BLEND),
    # --- compounds (independence assumption breaks): blend ---
    "btts_and_total_ge": ("tactical", BLEND),
    "first_goal_and_win": ("tactical", BLEND),
    "first_goal_and_opp_2h": ("tactical", BLEND),
    # --- player props (the user's flagged cases): lineup pass ---
    "player_ga": ("lineup", BLEND),            # assists are model-invisible; history helps
    "player_sot_ge": ("lineup", BLEND_CAPPED), # catastrophic-low tail — rescue but cap the pull
    "player_scores": ("lineup", SHADOW),       # we have edge; raw history HURTS — protect it
}
DEFAULT_ROUTE = ("tactical", EXCLUDE)          # unknown archetype -> pure model

GATE_PATH = Path("data/processed/expert_gate.json")
DEFAULT_CAP = 1.5            # logit-space pull cap for BLEND_CAPPED blends
DEFAULT_MAX_AGE_H = 36.0
CLIP_LO, CLIP_HI = 0.03, 0.97


def route(archetype: str) -> tuple[str, str]:
    return ROUTING.get(archetype, DEFAULT_ROUTE)


def expert_enabled() -> bool:
    """Master kill-switch. The whole expert blend is OFF (pure shadow) unless
    AI_EXPERT_ENABLE is set truthy in the live env. This is the single arming
    control: with it unset, no role weight or gate file can change a submission."""
    return os.environ.get("AI_EXPERT_ENABLE", "0").strip().lower() not in ("", "0", "false", "no")


def load_gate(path: Path = GATE_PATH) -> dict:
    """Per-archetype blend config fitted in Phase L2 (scripts/fit_expert_gate.py):
        {archetype: {"mode": "blend", "w": 0.5}}                 symmetric log-odds blend
        {archetype: {"mode": "asym_floor", "f": 0.5}}           one-sided ensemble floor
    Missing file -> {} -> every blendable archetype stays shadow (no-op)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _max_age_h() -> float:
    try:
        return float(os.environ.get("AI_PANEL_MAX_AGE_H", DEFAULT_MAX_AGE_H))
    except ValueError:
        return DEFAULT_MAX_AGE_H


# --- pure math (shared with expert.py; no LLM, fully unit-testable) ---

def clip01(p: float, lo: float = CLIP_LO, hi: float = CLIP_HI) -> float:
    return min(hi, max(lo, float(p)))


def _logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def combine_logodds(probs: list[float]) -> float | None:
    """Robust ensemble: trim-median in log-odds (parkjpd's load-bearing choice).
    n>=4: drop the lowest and highest, median the rest; n==3: median; n<=2: mean.
    A single rogue model cannot set the price."""
    xs = sorted(_logit(clip01(p)) for p in probs if p is not None)
    n = len(xs)
    if n == 0:
        return None
    if n >= 4:
        xs = xs[1:-1]
    if n == 3 or n >= 4:
        mid = len(xs) // 2
        med = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0
        return _sigmoid(med)
    return _sigmoid(sum(xs) / len(xs))  # n <= 2


def blend_logodds(p_model: float, p_expert: float | None, w: float, cap: float | None = None) -> float:
    """Log-odds blend of model and expert at weight w, with an optional cap on
    how far (in logit space) the result may move from the model — the 'rescue but
    don't overshoot' control for player_sot_ge. w<=0 or no expert -> pure model."""
    if p_expert is None or w <= 0:
        return float(p_model)
    lm = _logit(p_model)
    target = (1.0 - w) * lm + w * _logit(clip01(p_expert))
    if cap is not None:
        delta = target - lm
        if abs(delta) > cap:
            target = lm + math.copysign(cap, delta)
    return _sigmoid(target)


def ensemble_floor(p_model: float, p_expert: float | None, f: float) -> float:
    """One-sided ensemble-informed floor: lift the submission to f * expert ONLY
    when the model sits egregiously below the expert (the catastrophic-low tail);
    moderate bets are untouched and nothing is ever lowered. This is the surgical
    'rescue but don't disturb the body' tool for player_sot_ge, where a UNIFORM
    blend hurt (the ensemble over-predicts SoT) but the deep-low tail needs lifting.
    Generalises the Phase-1 flat floor to a per-player, evidence-set floor."""
    if p_expert is None or f <= 0:
        return float(p_model)
    return max(float(p_model), f * clip01(p_expert))


# --- cache I/O ---

def write_panel(entries: dict, path: Path = PANEL_PATH, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": time.time(), "meta": meta or {}, "entries": entries}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_panel(path: Path = PANEL_PATH, check_age: bool = True) -> dict:
    """{market_id(str): entry} for fresh entries. Missing/corrupt file -> {} (no-op).
    Entries carry generated_at; stale ones are dropped (guard 6)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    entries = payload.get("entries", {}) or {}
    if not check_age:
        return entries
    max_age = _max_age_h() * 3600.0
    now = time.time()
    out = {}
    for mid, e in entries.items():
        ts = e.get("generated_at", payload.get("generated_at", 0.0))
        if now - ts <= max_age:
            out[mid] = e
    return out


def apply_expert(market_id, archetype: str, p_model: float, panel: dict,
                 gate: dict | None = None) -> tuple[float, dict]:
    """Return (p_final, info). info is always logged (shadow forward-testing).

    Safety ladder (each rung is a pure-model fallback):
      * EXCLUDE archetype (sharp)            -> never reads the panel.
      * cache miss / no ensemble             -> pure model.
      * SHADOW archetype (proven edge)       -> logged, NEVER blended.
      * master switch off (AI_EXPERT_ENABLE) -> logged, NEVER blended.
      * no gate entry / zero param           -> pure model.
    Only when ALL pass does the per-archetype gate apply its mode:
      blend       -> symmetric log-odds blend at weight w (cap for blend_capped).
      asym_floor  -> one-sided ensemble floor at fraction f (player_sot_ge).
    """
    role, action = route(archetype)
    info = {"role": role, "action": action, "expert": None}
    if action == EXCLUDE:
        return float(p_model), info
    entry = panel.get(str(market_id)) if panel else None
    p_expert = entry.get("p_ensemble") if entry else None
    if p_expert is None:
        info["miss"] = True
        return float(p_model), info
    info["expert"] = round(float(p_expert), 4)
    if action == SHADOW or not expert_enabled():
        return float(p_model), info          # logged, never blended
    gate = gate if gate is not None else load_gate()
    g = gate.get(archetype)
    if not g:
        return float(p_model), info
    mode = g.get("mode", "blend")
    if mode == "asym_floor":
        f = float(g.get("f", 0.0))
        info["mode"], info["f"] = "asym_floor", f
        return ensemble_floor(p_model, p_expert, f), info
    w = float(g.get("w", 0.0))
    cap = g.get("cap", DEFAULT_CAP if action == BLEND_CAPPED else None)
    info["mode"], info["w"] = "blend", w
    return blend_logodds(p_model, p_expert, w, cap), info
