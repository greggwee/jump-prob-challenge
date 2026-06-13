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


def _load(name: str, default: dict) -> dict:
    p = PROC / name
    return json.loads(p.read_text()) if p.exists() else default


def load_base_rates() -> dict:
    return _load("base_rates.json", {})


def load_edge_gate() -> dict:
    return _load("edge_gate.json", {})


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


def k_for(family: str, gate: dict, stage_mult: float = 1.0, aggression: float = 1.0) -> float:
    """Per-family k: 1 + (gate_excess) * stage_gain * aggression. 1.0 = honest."""
    base_k = gate.get(family, 1.0)
    if base_k <= 1.0:
        return 1.0
    return 1.0 + (base_k - 1.0) * STAGE_K_GAIN.get(stage_mult, 1.0) * aggression
