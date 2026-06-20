"""Backtest the OFFLINE expert layer on settled matches — in RBP terms.

The web-search layer can't be backtested (settled matches leak the result via
post-match articles), so this measures the part that IS backtestable: the
rich-grounding gate blend on player props, exactly as armed. Frames it in the
contest metric — relative Brier points, RBP = (field_brier - our_brier) * 100,
using the realized crowd Brier as the field proxy — so the answer is "how many
points would the expert layer have added/lost across these settled questions".

  .venv/Scripts/python.exe scripts/backtest_expert.py --panel data/processed/ai_panel_backtest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.agents import panel as P

WEIGHTS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75]
TAIL_LOW = 0.20


def _brier_vec(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def _armed_blend(p_model, p_exp, archetype, gate):
    """Apply the gate exactly as the live daemon would (current weights)."""
    role, action = P.route(archetype)
    if action == P.SHADOW or action == P.EXCLUDE:
        return float(p_model)
    g = gate.get(archetype)
    if not g:
        return float(p_model)
    if g.get("mode") == "asym_floor":
        return P.ensemble_floor(p_model, p_exp, float(g.get("f", 0.0)))
    cap = P.DEFAULT_CAP if action == P.BLEND_CAPPED else None
    return P.blend_logodds(p_model, p_exp, float(g.get("w", 0.0)), cap)


def _loo_match(g: pd.DataFrame, cap):
    """Honest LOO-match optimal weight (for the re-tune) from a grid incl. 0."""
    y = g["outcome"].values.astype(float)
    pm, pe, m = g["our_p"].values, g["p_exp"].values, g["match"].values
    pre = {w: np.array([P.blend_logodds(a, b, w, cap) for a, b in zip(pm, pe)]) for w in WEIGHTS}
    sq, chosen = [], []
    for held in np.unique(m):
        tr, he = m != held, m == held
        bw = min(WEIGHTS, key=lambda w: np.mean((pre[w][tr] - y[tr]) ** 2)) if tr.sum() else 0.0
        chosen.append(bw)
        sq.extend(((pre[bw][he] - y[he]) ** 2).tolist())
    modal = max(set(chosen), key=chosen.count)
    return float(np.mean(sq)), modal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/ai_panel_backtest.json")
    ap.add_argument("--settled", default="data/processed/settled_with_crowd.parquet")
    args = ap.parse_args()

    ent = json.loads(Path(args.panel).read_text())["entries"]
    pan = pd.DataFrame([{"market_id": str(k), "p_exp": v["p_ensemble"]} for k, v in ent.items()])
    pan = pan[pan["p_exp"].notna()]
    s = pd.read_parquet(args.settled)
    s["market_id"] = s["market_id"].astype(str)
    s = s[~s["known_bug"]].copy()
    df = s.merge(pan, on="market_id", how="inner").dropna(subset=["p_exp", "outcome", "our_p"])
    gate = P.load_gate()
    print(f"backtest set: {len(df)} settled player props ({df['match'].nunique()} matches)")
    print(f"live gate: {gate}\n")

    df["b_model"] = _brier_vec(df["our_p"], df["outcome"])
    df["b_crowd"] = df["crowd_brier"]
    df["p_armed"] = [_armed_blend(r.our_p, r.p_exp, r.archetype, gate) for r in df.itertuples()]
    df["b_armed"] = _brier_vec(df["p_armed"], df["outcome"])
    # RBP per question = (field_brier - our_brier)*100, field := realized crowd Brier
    df["rbp_model"] = (df["b_crowd"] - df["b_model"]) * 100
    df["rbp_armed"] = (df["b_crowd"] - df["b_armed"]) * 100

    rows = []
    for a, gg in df.groupby("archetype"):
        role, action = P.route(a)
        cap = P.DEFAULT_CAP if action == P.BLEND_CAPPED else None
        oos, cvw = _loo_match(gg, cap) if action in (P.BLEND, P.BLEND_CAPPED) else (gg["b_model"].mean(), 0.0)
        rows.append({
            "archetype": a, "route": action, "n": len(gg), "n_crowd": int(gg["b_crowd"].notna().sum()),
            "b_model": gg["b_model"].mean(), "b_armed": gg["b_armed"].mean(),
            "b_crowd": gg["b_crowd"].mean(),
            "RBP_model": gg["rbp_model"].sum(), "RBP_armed": gg["rbp_armed"].sum(),
            "dRBP": gg["rbp_armed"].sum() - gg["rbp_model"].sum(),
            "loo_oos": oos, "loo_w": cvw,
        })
    rep = pd.DataFrame(rows).sort_values("dRBP", ascending=False)
    pd.set_option("display.width", 220)
    print(rep.round(4).to_string(index=False))

    nc = df["b_crowd"].notna()                       # RBP needs the harvested crowd consensus
    tot_d = df.loc[nc, "rbp_armed"].sum() - df.loc[nc, "rbp_model"].sum()
    print(f"\n=== AS ARMED (current gate) ===")
    print(f"  Brier — ALL {len(df)} settled player props:  model={df['b_model'].mean():.4f} -> "
          f"armed={df['b_armed'].mean():.4f}  ({df['b_model'].mean()-df['b_armed'].mean():+.4f}/q)")
    print(f"  RBP   — the {int(nc.sum())} with crowd data:   model={df.loc[nc,'rbp_model'].sum():+.1f} -> "
          f"armed={df.loc[nc,'rbp_armed'].sum():+.1f}  -> expert layer adds {tot_d:+.1f} RBP "
          f"(vs consensus, a conservative field proxy)")

    tail = df[(df["our_p"] < TAIL_LOW) & (df["outcome"] == 1)]
    if len(tail):
        tnc = tail["b_crowd"].notna()
        print(f"\n  catastrophic-low tail (bet <{TAIL_LOW}, happened): n={len(tail)}  "
              f"Brier {tail['b_model'].mean():.3f} -> {tail['b_armed'].mean():.3f}"
              + (f"  RBP {(tail.loc[tnc,'b_crowd']-tail.loc[tnc,'b_model']).sum()*100:+.1f} -> "
                 f"{(tail.loc[tnc,'b_crowd']-tail.loc[tnc,'b_armed']).sum()*100:+.1f} (n={int(tnc.sum())})"
                 if tnc.any() else ""))


if __name__ == "__main__":
    main()
