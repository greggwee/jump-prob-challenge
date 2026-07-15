"""Compute base rates + the confirmed-edge gate that drives extremizing (rec #2).

Base rates per archetype (the field proxy for no-line markets) come from the
backtests. The edge gate asks, per archetype FAMILY, on SETTLED LIVE questions:
is our realized Brier beating the proxy-field's Brier, with enough sample and a
real margin? If yes -> set k>1 for that family (we have earned leverage). If not
-> k=1 (stay honest). This is the evidence gate the user chose.

Run after each matchday settles:
  python scripts/update_edge_gate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.strategy import field_model, load_field_sharpness
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import COUNT_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

PROC = Path("data/processed")
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
MIN_SETTLED = 20          # need this many settled in a family before trusting edge
MARGIN = 0.005            # our Brier must beat field proxy by this to earn k>1
K_ON_CONFIRMED = 1.3      # leverage granted to a confirmed-edge family

FAMILY = {**{a: "count" for a in COUNT_ARCHS},
          **{a: "player" for a in PLAYER_ARCHS},
          **{a: "timing" for a in TIMING_ARCHS}}


def archetype_base_rates() -> dict:
    """Pool historical outcomes per archetype from the backtests."""
    rates: dict[str, float] = {}
    counts = PROC / "backtest_counts.csv"
    if counts.exists():
        c = pd.read_csv(counts)
        c = c[c["src"] == "model"]
        rates.update(c.groupby("archetype")["outcome"].mean().to_dict())
    # map the backtest count names to the live parser families' typical base rate
    # (coarse: use the pooled count base rate for all count archetypes)
    if counts.exists():
        pooled_count = float(c["outcome"].mean())
        for a in COUNT_ARCHS:
            rates.setdefault(a, pooled_count)
    # both-teams SoT is a no-edge count with a genuinely HIGH base (~0.70 over the
    # live settled set); the pooled count rate (~0.525) under-states it, which let the
    # model's confident 88/24 noise through (the -89 RBP leak, 2026-06-22). Pin it so
    # the daily regen keeps this risk-gate base aligned with risk_gate.json's shrink.
    rates["both_teams_sot_ge"] = 0.70
    # P2 (2026-07-02 review): our per-match variation on fouls/offsides is ANTI-signal — a
    # FLAT value beats both our submissions and the crowd on all settled-with-crowd data
    # (fouls: flat .245 vs crowd .263 vs us .277, n=21; offsides: flat@.40 .233 vs crowd
    # .253 vs us .280, n=22). Pin the levels the risk gate shrinks toward (risk_gate.json
    # holds the matching strong shrink s=0.15):
    #   fouls_gt 0.50   = the SYMMETRIC coin-flip. fouls_gt compares exchangeable teams, so it
    #     must sit at/below 0.50 (P(A>B)=P(B>A)<0.5 once tie mass is removed). The realized rate
    #     is sample-unstable (0.57 n=21 @2026-07-02 vs 0.44 n=16 @2026-07-08) with no structural
    #     reason for a first-named-team foul bias, so 0.50 is the robust hug (was 0.55, which the
    #     2026-07-07 settled-match review flagged as over-priced: our .58 vs realized .44);
    #   offsides_ge 0.40 = crowd 0.49 shrunk toward realized 0.36 (the StatsBomb->live definition
    #     gap makes the crowd itself too HIGH; 0.40 tracks realized .39 — 2026-07-07 review VERIFIED
    #     we're well-calibrated here, do NOT lift toward the crowd).
    rates["fouls_gt"] = 0.50
    rates["offsides_ge"] = 0.40
    praw = PROC / "backtest_players_raw.csv"
    if praw.exists():
        p = pd.read_csv(praw)
        by = p.groupby("archetype")["outcome"].mean().to_dict()
        rates["player_sot_ge"] = by.get("sot_ge_1", 0.3)
        rates["player_scores"] = by.get("scores", 0.12)
        rates["player_ga"] = by.get("scores", 0.12) * 1.6  # goal-or-assist ~ 1.6x scoring
    bt2 = PROC / "backtest_v2_go.csv"
    if bt2.exists():
        v = pd.read_csv(bt2)
        rates.update(v.groupby("archetype")["outcome"].mean().to_dict())
    return {k: round(float(v), 4) for k, v in rates.items()}


def confirmed_edge_gate(base_rates: dict) -> dict:
    """Per-family k from settled live results vs the proxy field."""
    client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    settled = pd.DataFrame(client.results(LOBBY))
    settled = settled.dropna(subset=["brier_score"])
    md = PROC / "matchday_predictions.csv"
    if settled.empty or not md.exists():
        return {}
    settled = settled.merge(pd.read_csv(md)[["market_id", "archetype"]], on="market_id", how="left")
    settled["p"] = settled["probability_submitted"] / 100.0
    settled["outcome"] = np.where(
        np.isclose((settled["p"] - 1) ** 2, settled["brier_score"], atol=1e-6), 1.0, 0.0)
    settled["family"] = settled["archetype"].map(FAMILY)

    sharp = load_field_sharpness()
    gate = {}
    for fam, g in settled.dropna(subset=["family"]).groupby("family"):
        our = float(((g["p"] - g["outcome"]) ** 2).mean())
        lam = sharp.get(fam, 0.5)
        fld = float(np.mean([(field_model(p, lam) - y) ** 2
                             for p, y in zip(g["p"], g["outcome"])]))
        confirmed = len(g) >= MIN_SETTLED and our < fld - MARGIN
        gate[fam] = K_ON_CONFIRMED if confirmed else 1.0
        print(f"  {fam:7} n={len(g):3d}  our Brier {our:.4f} vs field {fld:.4f}  "
              f"-> k={gate[fam]}{'  CONFIRMED' if confirmed else ''}")
    return gate


base = archetype_base_rates()
(PROC / "base_rates.json").write_text(json.dumps(base, indent=1))
print(f"base rates written for {len(base)} archetypes")
print("edge gate (per family):")
gate = confirmed_edge_gate(base)
(PROC / "edge_gate.json").write_text(json.dumps(gate, indent=1))
print(f"\nwrote edge_gate.json: {gate}")
print("k=1 everywhere until a family confirms edge over >=20 settled questions.")
