"""Evidence-gated extremizing for P(finish #1) under relative-Brier scoring.

The objective is rank #1 (winner-take-most), not expected points. Lichtendahl &
Winkler (2007): under relative-rank scoring the optimum is to EXTREMIZE — but
only where you have genuine edge over the field, else you just add un-edged
variance and raise your own Brier.

We extremize the DEVIATION FROM THE FIELD, not the raw probability:
    logit(submit) = logit(field) + k * (logit(belief) - logit(field))
- belief == field  -> submit == field  (no gamble; calibrated floor)
- belief deviates   -> deviation amplified by k  (variance on our edge)

k is per-family, gated on confirmed live edge and scaled by stage multiplier and
an aggression dial. k=1 reproduces honest reporting (today's default until the
edge gate fills in). Anchored direct-market questions are NOT extremized — there
we ARE the field, so it is pure floor.

Field proxy (we cannot see the crowd; rec L1): for a no-line market the crowd
clusters near the archetype base rate and makes a weaker version of our
adjustment, so field = base_rate moved a fraction rho toward our belief. Our
edge is the part of the adjustment the crowd does not make.
"""

from __future__ import annotations

import json
from pathlib import Path

from jumpprob.models.market_anchor import inv_logit, logit

PROC = Path("data/processed")
CROWD_RHO = 0.5          # crowd moves halfway from base rate toward the right answer
STAGE_K_GAIN = {1.0: 1.0, 2.0: 1.5, 3.0: 2.0}  # extra leverage by stage multiplier

# Phase C structural field model. Field sharpness lambda per family: how much of
# the truth-direction signal the naive contestant field captures.
#   lambda = 1 -> field = truth (sharp, no edge for us);  lambda = 0 -> field = 50
#   (clueless crowd, maximum edge for us).
# Priors from reasoning + the Phase B finding (deep markets are efficient, but the
# SportsPredict field is a NAIVE crowd, not an efficient market, so on no-line
# markets lambda is low). The ONLY ground-truth calibration is aggregate RBP
# (calibrate_field_sharpness); these priors hold until that signal arrives.
#   goal   : field reads the odds (the brief recommends them) -> high
#   player : some lean on star names, lots of base-rate guessing -> mid
#   count  : obscure, near base-rate / 50 clustering -> low
#   timing : very obscure (first-goal combos, h2>h1) -> lowest
FIELD_SHARPNESS = {"goal": 0.85, "player": 0.45, "count": 0.40, "timing": 0.35}


def _load(name: str, default: dict) -> dict:
    p = PROC / name
    return json.loads(p.read_text()) if p.exists() else default


def load_base_rates() -> dict:
    return _load("base_rates.json", {})


def load_edge_gate() -> dict:
    return _load("edge_gate.json", {})


def load_field_sharpness() -> dict:
    """Per-family field sharpness, calibrated values overriding the priors."""
    return {**FIELD_SHARPNESS, **_load("field_model.json", {})}


def field_model(belief: float, lam: float) -> float:
    """Structural field estimate (Phase C): our belief shrunk toward 50 by the
    field's sharpness. A logit-scaling — lam=1 keeps the belief, lam=0 -> 0.5.

    Composed with extremize(belief, field, k), the submission becomes
        inv_logit( [lam + k(1-lam)] * logit(belief) ),
    so the field sharpness self-modulates the leverage: a clueless field (lam->0)
    yields full extremizing (coefficient -> k); a sharp field (lam->1) yields none
    (coefficient -> 1). The single lam per family IS our edge over the field."""
    return float(inv_logit(lam * logit(belief)))


def field_proxy(archetype: str, belief: float, base_rates: dict,
                market: float | None = None, rho: float = CROWD_RHO) -> float:
    """Estimated field probability for one market."""
    if market is not None:
        # direct market quote: the crowd ~ the sharp line, lightly dulled to 50
        return float(inv_logit(0.9 * logit(market)))
    base = base_rates.get(archetype, 0.5)
    return float(inv_logit(logit(base) + rho * (logit(belief) - logit(base))))


def extremize(belief: float, field: float, k: float) -> float:
    """Amplify belief's deviation from the field by factor k (>=1)."""
    return float(inv_logit(logit(field) + k * (logit(belief) - logit(field))))


def calibrate_field_sharpness(beliefs, outcomes, families, observed_field_brier):
    """Tune one global multiplier on the sharpness priors so the field model's
    predicted aggregate Brier matches the observed field Brier (derived from
    aggregate RBP: field_Brier = our_Brier + RBP/(100·n) at the group stage).

    Only a global scalar is identifiable from one aggregate number — the API
    exposes no per-family field data (Phase A) — so this is a coarse level nudge
    over the family priors, not a per-family fit. Returns ({family: lambda}, g)."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    p = np.clip(np.asarray(beliefs, float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcomes, float)
    lam0 = np.array([FIELD_SHARPNESS.get(f, 0.5) for f in families])
    lp = np.log(p / (1 - p))

    def pred(g):
        f = 1.0 / (1.0 + np.exp(-np.clip(g * lam0, 0.0, 1.0) * lp))
        return float(np.mean((f - y) ** 2))

    g = float(minimize_scalar(lambda g: (pred(g) - observed_field_brier) ** 2,
                              bounds=(0.2, 3.0), method="bounded").x)
    table = {f: round(min(g * FIELD_SHARPNESS.get(f, 0.5), 1.0), 3) for f in set(families)}
    return table, g


def k_for(family: str, gate: dict, stage_mult: float = 1.0, aggression: float = 1.0) -> float:
    """Per-family k: 1 + (gate_excess) * stage_gain * aggression. 1.0 = honest."""
    base_k = gate.get(family, 1.0)
    if base_k <= 1.0:
        return 1.0
    return 1.0 + (base_k - 1.0) * STAGE_K_GAIN.get(stage_mult, 1.0) * aggression
