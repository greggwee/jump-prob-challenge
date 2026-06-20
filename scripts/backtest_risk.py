"""Backtest the Phase-R de-confidence shrink on settled matches (RBP).

On the no-edge archetypes in risk_gate.json (offsides/fouls/cards — ~0 edge in the
6-tournament backtest), our model deviates from the base rate and loses RBP. This
applies the gate to the settled submissions and reports the Brier/RBP change, in
the contest metric (RBP = (field_brier - our_brier)*100, realized crowd Brier as a
conservative field proxy). Validates the gate before arming RISK_GATE_ENABLE.

  .venv/Scripts/python.exe scripts/backtest_risk.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jumpprob.models import strategy

SETTLED = "data/processed/settled_with_crowd.parquet"


def _brier(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def main() -> None:
    gate = strategy.load_risk_gate()
    base = strategy.load_base_rates()
    df = pd.read_parquet(SETTLED)
    df = df[~df["known_bug"]].copy()
    df["risk_p"] = [strategy.apply_risk(r.our_p, r.archetype, gate, base) for r in df.itertuples()]

    inset = df["archetype"].isin(gate)
    print(f"risk gate: {gate}")
    print(f"settled (ex-bug): {len(df)} | in no-edge gate: {int(inset.sum())} "
          f"({sorted(set(df.loc[inset, 'archetype']))})\n")

    b0, b1 = _brier(df["our_p"], df["outcome"]), _brier(df["risk_p"], df["outcome"])
    dc = df[df["crowd_p"].notna()]
    r0 = ((dc["crowd_brier"] - _brier(dc["our_p"], dc["outcome"])) * 100).sum()
    r1 = ((dc["crowd_brier"] - _brier(dc["risk_p"], dc["outcome"])) * 100).sum()
    print("=== overall (all settled; gate is a no-op outside the no-edge set) ===")
    print(f"  Brier  {b0.mean():.4f} -> {b1.mean():.4f}   ({b0.mean()-b1.mean():+.4f})")
    print(f"  RBP    {r0:+.1f} -> {r1:+.1f}   -> de-confidence adds {r1-r0:+.1f} RBP "
          f"(vs consensus, conservative)")

    print("\n=== per no-edge archetype (where crowd available) ===")
    rows = []
    for a, g in dc[inset.loc[dc.index]].groupby("archetype"):
        rb0 = ((g["crowd_brier"] - _brier(g["our_p"], g["outcome"])) * 100).sum()
        rb1 = ((g["crowd_brier"] - _brier(g["risk_p"], g["outcome"])) * 100).sum()
        rows.append({"archetype": a, "n": len(g), "s": gate[a], "RBP_before": rb0,
                     "RBP_after": rb1, "dRBP": rb1 - rb0})
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
