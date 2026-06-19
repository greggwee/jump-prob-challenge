"""History grounding for the Phase-3 experts.

The whole point of the expert layer is to reason from a player's and team's TRACK
RECORD — the signal both our shrinkage-heavy model and the obscure-player crowd
under-use. This assembles that record from data we already have (club per-90 rates
from Understat + API-Football, the archetype base rate) and turns it into a
history-implied probability the expert can weigh against our model's prior. Pure
data, no LLM — testable on its own.

SoT note: Understat has no shots-on-target, so for those players we estimate SoT
from total shots at the league-typical conversion; API-Football players carry direct
SoT/90. The estimate is flagged so the expert knows its provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.normalize.players import build_index, resolve

RATES = Path("data/processed/player_rates_combined.parquet")
BASE_RATES = Path("data/processed/base_rates.json")
SOT_PER_SHOT = 0.37   # league-typical shots-on-target / total-shots, for Understat players
_cache: dict = {}


def _load():
    if "df" not in _cache:
        df = pd.read_parquet(RATES)
        _cache["df"] = df
        _cache["idx"] = build_index(df["player_name"])
        _cache["base"] = json.loads(BASE_RATES.read_text()) if BASE_RATES.exists() else {}
    return _cache


def _r(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)


def player_history(name: str) -> dict:
    """Resolve a market player name to their club track record. Returns sot_p90
    (direct or estimated-from-shots), volume/quality rates, and minutes (confidence)."""
    c = _load()
    match, method = resolve(name, *c["idx"])
    if match is None:
        return {"query": name, "resolved": None, "method": "none"}
    row = c["df"][c["df"]["player_name"] == match].iloc[0]
    sot, sot_src = row.get("sot_p90"), "direct"
    if sot is None or (isinstance(sot, float) and np.isnan(sot)):
        sot = float(row.get("shots_p90") or 0.0) * SOT_PER_SHOT
        sot_src = "est_from_shots"
    return {
        "query": name, "resolved": match, "method": method,
        "source": row.get("source"), "league": row.get("league"),
        "minutes": int(row.get("minutes") or 0),
        "sot_p90": round(float(sot), 2), "sot_source": sot_src,
        "shots_p90": _r(row.get("shots_p90")), "npg_p90": _r(row.get("npg_p90")),
        "goals_p90": _r(row.get("goals_p90")), "assists_p90": _r(row.get("assists_p90")),
        "ga_p90": _r((row.get("npg_p90") or 0) + (row.get("assists_p90") or 0)),
        "xG_p90": _r(row.get("xG_p90")), "xA_p90": _r(row.get("xA_p90")),
        "key_passes_p90": _r(row.get("key_passes_p90")),
    }


def poisson_ge1(rate_per90: float, minutes: float = 90.0, share: float = 1.0) -> float:
    """P(>=1 event) from a per-90 rate over `minutes`, optionally a half/window `share`
    (Poisson thinning — the same math the model uses). The history-implied probability."""
    lam = max(rate_per90, 0.0) * (minutes / 90.0) * share
    return float(1.0 - np.exp(-lam))


def history_implied(name: str, archetype: str, minutes: float = 70.0,
                    half: bool = False) -> dict:
    """A history-only probability for a player prop, to anchor the expert. `half`=True
    restricts to one half (uses the measured 2H goal/shot share ~0.52)."""
    h = player_history(name)
    if h["resolved"] is None:
        base = _load()["base"].get(archetype)
        return {**h, "implied": base, "note": "no club join -> base rate"}
    share = 0.52 if half else 1.0
    if archetype == "player_sot_ge":
        implied = poisson_ge1(h["sot_p90"], minutes, share)
    elif archetype == "player_scores":
        implied = poisson_ge1(h["npg_p90"], minutes, share)
    elif archetype == "player_ga":
        implied = poisson_ge1(h["ga_p90"], minutes, share)
    else:
        implied = None
    return {**h, "minutes_assumed": minutes, "half": half, "implied": _r(implied)}


if __name__ == "__main__":
    # the players we mispriced (METHODS §13): show the history-implied SoT/GA vs what we bet
    print("player                | hist sot/90 | min  | hist-implied (≥1 SoT, 2H, 70')  [we bet]")
    for nm, bet in [("Viktor Gyökeres", "11% (G+A)"), ("Takefusa Kubo", "7% (G+A)"),
                    ("Luis Díaz", "10% (SoT)"), ("Salem Al-Dawsari", "—"),
                    ("Scott McTominay", "15% (SoT)")]:
        hi = history_implied(nm, "player_sot_ge", minutes=70, half=True)
        if hi["resolved"]:
            print(f"  {nm:20s} | {hi['sot_p90']:>5} {hi['sot_source'][:4]:>5} | {hi['minutes']:>5} | "
                  f"{hi['implied']:.2f}   [{bet}]")
        else:
            print(f"  {nm:20s} | (no club join)")
