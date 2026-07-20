"""Fit the per-archetype expert gate -> data/processed/expert_gate.json.

For each blendable archetype, pick the blend parameter by honest leave-one-MATCH-out
CV with a PARSIMONY rule (the smallest param within GATE_EPS of the best OOS Brier --
this regularises toward conservative weights, so we never default to full replacement)
and an improvement gate (emit only if the OOS blend beats the pure model).

  * symmetric blend  : player_ga, fouls_gt, offsides_ge, ... (weight capped at MAX_W)
  * asym_floor       : player_sot_ge -- a uniform blend HURT it in L1 (the ensemble
                       over-predicts SoT), so we only LIFT the catastrophic-low tail
                       toward f*ensemble. Emitted if it beats the model on mean Brier,
                       OR cuts the catastrophic-low tail while mean stays within TAIL_TOL
                       (the user's explicit risk-management objective).

The gate is INERT until AI_EXPERT_ENABLE is set in the live env. Run after
scripts/ai_panel_write.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.agents import panel as P

BLEND_W = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75]   # MAX_W = 0.75: never full replacement
FLOOR_F = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
GATE_EPS = 1e-3      # parsimony tolerance: smallest param within this of best
MARGIN = 5e-4        # min OOS improvement to ship a blend
TAIL_TOL = 3e-3      # allowed mean-Brier give-up for a tail (risk-mgmt) rescue
TAIL_LOW = 0.20


def _load(panel_path: str) -> pd.DataFrame:
    ent = json.loads(Path(panel_path).read_text())["entries"]
    panel = pd.DataFrame([{"market_id": str(k), "p_exp": v["p_ensemble"]} for k, v in ent.items()])
    panel = panel[panel["p_exp"].notna()]
    s = pd.read_parquet("data/processed/settled_with_crowd.parquet")
    s["market_id"] = s["market_id"].astype(str)
    s = s[~s["known_bug"]].copy()
    return s.merge(panel, on="market_id", how="inner")


def _apply(mode, pm, pe, param, cap):
    if mode == "asym_floor":
        return np.array([P.ensemble_floor(m, e, param) for m, e in zip(pm, pe)])
    return np.array([P.blend_logodds(m, e, param, cap) for m, e in zip(pm, pe)])


def _parsimonious(briers: dict, grid: list) -> float:
    best = min(briers.values())
    return min(p for p in grid if briers[p] <= best + GATE_EPS)


def _fit(g: pd.DataFrame, mode: str, grid: list, cap):
    y = g["outcome"].values.astype(float)
    pm, pe = g["our_p"].values, g["p_exp"].values
    matches = g["match"].values
    pre = {p: _apply(mode, pm, pe, p, cap) for p in grid}
    # honest LOO-match: per held-out match, parsimonious param chosen on the rest
    sq = []
    for held in np.unique(matches):
        tr, he = matches != held, matches == held
        if tr.sum() == 0:
            param = 0.0
        else:
            param = _parsimonious({p: np.mean((pre[p][tr] - y[tr]) ** 2) for p in grid}, grid)
        sq.extend(((pre[param][he] - y[he]) ** 2).tolist())
    oos = float(np.mean(sq))
    gate_param = _parsimonious({p: float(np.mean((pre[p] - y) ** 2)) for p in grid}, grid)
    b_model = float(np.mean((pm - y) ** 2))
    # catastrophic-low tail
    tail = (pm < TAIL_LOW) & (y == 1)
    tail_model = float(np.mean((pm[tail] - y[tail]) ** 2)) if tail.any() else np.nan
    tail_gate = float(np.mean((pre[gate_param][tail] - y[tail]) ** 2)) if tail.any() else np.nan
    return dict(oos=oos, gate_param=gate_param, b_model=b_model, n=len(g),
                tail_n=int(tail.sum()), tail_model=tail_model, tail_gate=tail_gate)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(P.PANEL_PATH))
    args = ap.parse_args()

    df = _load(args.panel)
    gate, report = {}, []
    for arch, g in df.groupby("archetype"):
        g = g.dropna(subset=["p_exp", "outcome", "our_p"])
        role, action = P.route(arch)
        if action not in (P.BLEND, P.BLEND_CAPPED) or len(g) < 4:
            continue
        if action == P.BLEND_CAPPED:
            # player_sot_ge: try a CAPPED symmetric blend (works once the rich
            # ensemble is calibrated) AND the asymmetric floor; keep whichever is
            # better OOS. apply_expert auto-caps blend_capped routes.
            rb, rf = _fit(g, "blend", BLEND_W, P.DEFAULT_CAP), _fit(g, "asym_floor", FLOOR_F, None)
            if rb["oos"] <= rf["oos"]:
                r, mode, key, param = rb, "blend", "w", rb["gate_param"]
            else:
                r, mode, key, param = rf, "asym_floor", "f", rf["gate_param"]
            beats_mean = r["oos"] < r["b_model"] - MARGIN
            tail_help = (mode == "asym_floor" and r["tail_n"] > 0 and r["tail_gate"] < r["tail_model"]
                         and r["oos"] <= r["b_model"] + TAIL_TOL)
            emit = param > 0 and (beats_mean or tail_help)
            reason = "beats-mean" if beats_mean else ("tail-rescue" if tail_help else "no-support")
        else:                               # symmetric blend
            r = _fit(g, "blend", BLEND_W, None)
            mode, key, param = "blend", "w", r["gate_param"]
            emit = param > 0 and r["oos"] < r["b_model"] - MARGIN
            reason = "beats-mean" if emit else "no-support"
        if emit:
            gate[arch] = {"mode": mode, key: round(float(param), 3)}
        report.append({"archetype": arch, "n": r["n"], "mode": mode, key: param,
                       "b_model": r["b_model"], "oos": r["oos"],
                       "improve": r["b_model"] - r["oos"], "emit": emit, "why": reason,
                       "tail_n": r["tail_n"], "tail_model": r["tail_model"], "tail_gate": r["tail_gate"]})

    rep = pd.DataFrame(report).sort_values("improve", ascending=False)
    pd.set_option("display.width", 200)
    print(rep.round(4).to_string(index=False))

    Path(P.GATE_PATH).write_text(json.dumps(gate, indent=2))
    print(f"\nwrote gate ({len(gate)} archetypes) -> {P.GATE_PATH}")
    for a, gdef in gate.items():
        print(f"  {a}: {gdef}")
    print("\nINERT until AI_EXPERT_ENABLE=1 in the live env.")


if __name__ == "__main__":
    main()
