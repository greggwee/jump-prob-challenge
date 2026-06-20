"""Phase L1 — the forward-test gate for the LLM expert blend.

Joins the out-of-band panel (data/processed/ai_panel.json) to the settled,
crowd-graded dataset (settled_with_crowd.parquet) and asks, per archetype:
does blending the model toward the expert ensemble BEAT the pure model
out-of-sample? The weight is chosen honestly by leave-one-MATCH-out CV from a
grid that INCLUDES w=0 — so if blending never helps, CV picks w=0 and the blend
Brier equals the model Brier (no free lunch).

A (role, archetype) pair is promotable iff its honest blend Brier < its model
Brier with no regression. The crowd Brier is shown as the target we're chasing.
Run after scripts/ai_panel_write.py produces the cache.

  .venv/Scripts/python.exe scripts/forward_test_llm.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.agents import panel as P

WEIGHTS = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
TAIL_LOW = 0.20   # "catastrophic-low" = we bet < this but the event happened


def _blend_col(p_model, p_exp, w, cap):
    return np.array([P.blend_logodds(float(pm), float(pe), w, cap)
                     for pm, pe in zip(p_model, p_exp)])


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _loo_match(df: pd.DataFrame, cap: float | None) -> tuple[float, float]:
    """Honest leave-one-match-out: for each held-out match, pick the grid weight
    that minimises Brier on the OTHER matches, then score the held-out rows at it.
    Returns (oos_brier, modal_weight). Grid includes 0.0 -> 'don't blend' is allowed."""
    matches = df["match"].unique()
    # precompute blended prob per weight (vectorised)
    bw = {w: _blend_col(df["our_p"].values, df["p_exp"].values, w, cap) for w in WEIGHTS}
    y = df["outcome"].values.astype(float)
    idx_by_match = {m: np.where(df["match"].values == m)[0] for m in matches}
    sq, chosen = [], []
    for held in matches:
        held_idx = idx_by_match[held]
        train_mask = np.ones(len(df), bool)
        train_mask[held_idx] = False
        if train_mask.sum() == 0:
            best_w = 0.0
        else:
            best_w = min(WEIGHTS, key=lambda w: np.mean((bw[w][train_mask] - y[train_mask]) ** 2))
        chosen.append(best_w)
        sq.extend(((bw[best_w][held_idx] - y[held_idx]) ** 2).tolist())
    modal = max(set(chosen), key=chosen.count) if chosen else 0.0
    return float(np.mean(sq)), float(modal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(P.PANEL_PATH))
    ap.add_argument("--settled", default="data/processed/settled_with_crowd.parquet")
    args = ap.parse_args()

    payload = json.loads(Path(args.panel).read_text())
    ent = payload.get("entries", {})
    panel = pd.DataFrame([
        {"market_id": str(k), "p_exp": v.get("p_ensemble"), "n_models": v.get("n"),
         "action": v.get("action")}
        for k, v in ent.items()
    ])
    panel = panel[panel["p_exp"].notna()]

    s = pd.read_parquet(args.settled)
    s["market_id"] = s["market_id"].astype(str)
    s = s[~s["known_bug"]].copy()
    df = s.merge(panel, on="market_id", how="inner", suffixes=("", "_p"))
    print(f"joined {len(df)} settled questions to panel "
          f"({df['archetype'].nunique()} archetypes, {df['match'].nunique()} matches)\n")

    rows = []
    for arch, g in df.groupby("archetype"):
        g = g.dropna(subset=["p_exp", "outcome", "our_p"])
        if len(g) < 3:
            continue
        role, action = P.route(arch)
        cap = P.DEFAULT_CAP if action == P.BLEND_CAPPED else None
        b_model = _brier(g["our_p"], g["outcome"])
        b_crowd = float(g["crowd_brier"].mean())
        oos_blend, cv_w = _loo_match(g, cap)
        # catastrophic-low tail: we bet < TAIL_LOW but it happened
        tail = g[(g["our_p"] < TAIL_LOW) & (g["outcome"] == 1)]
        tail_model = _brier(tail["our_p"], tail["outcome"]) if len(tail) else np.nan
        tail_blend = (_brier(_blend_col(tail["our_p"].values, tail["p_exp"].values, cv_w, cap),
                             tail["outcome"]) if len(tail) else np.nan)
        rows.append({
            "archetype": arch, "action": action, "n": len(g), "m": g["match"].nunique(),
            "base": float(g["outcome"].mean()),
            "b_model": b_model, "b_crowd": b_crowd, "b_blend": oos_blend,
            "cv_w": cv_w, "improve": b_model - oos_blend,
            "tail_n": len(tail), "tail_model": tail_model, "tail_blend": tail_blend,
        })

    rep = pd.DataFrame(rows).sort_values("improve", ascending=False)
    pd.set_option("display.width", 200)
    show = rep[["archetype", "action", "n", "m", "base", "b_model", "b_crowd",
                "b_blend", "cv_w", "improve"]].copy()
    print(show.round(4).to_string(index=False))

    print("\n--- catastrophic-low tail (we bet <0.20 but it happened) ---")
    t = rep[rep["tail_n"] > 0][["archetype", "tail_n", "tail_model", "tail_blend"]]
    print(t.round(4).to_string(index=False) if len(t) else "  (none)")

    blendable = rep[rep["action"].isin([P.BLEND, P.BLEND_CAPPED])]
    promote = blendable[(blendable["improve"] > 0) & (blendable["cv_w"] > 0)]
    print("\n=== L1 GATE ===")
    print("PROMOTE (blend beats model OOS, CV picked w>0):")
    for _, r in promote.iterrows():
        print(f"  + {r['archetype']:16s} w={r['cv_w']:.2f}  "
              f"Brier {r['b_model']:.4f} -> {r['b_blend']:.4f} ({r['improve']:+.4f})")
    if promote.empty:
        print("  (none yet)")
    shadow_help = rep[(rep["action"] == P.SHADOW) & (rep["improve"] > 0) & (rep["cv_w"] > 0)]
    if len(shadow_help):
        print("NOTE — shadow archetypes the blend WOULD help (reconsider only with more data):")
        for _, r in shadow_help.iterrows():
            print(f"  ~ {r['archetype']:16s} would w={r['cv_w']:.2f}  "
                  f"Brier {r['b_model']:.4f} -> {r['b_blend']:.4f}")

    # pooled over blendable
    bd = df[df["archetype"].isin(blendable["archetype"])]
    if len(bd):
        pooled_model = _brier(bd["our_p"], bd["outcome"])
        print(f"\npooled blendable: n={len(bd)}  model Brier={pooled_model:.4f}  "
              f"crowd Brier={bd['crowd_brier'].mean():.4f}")


if __name__ == "__main__":
    main()
