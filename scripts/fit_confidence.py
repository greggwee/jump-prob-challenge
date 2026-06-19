"""Phase 1: fit asymmetric confidence calibration on the authoritative settled set.

Per the quant review, this is deliberately conservative:
  * ship-now mechanism is observable-independent: per-family **regularized Platt**
    (bias/slope, ridge-shrunk to identity) + a **tail floor** clip(p, f, 1-f).
    (Shrink-toward-the-live-crowd-consensus is Phase 2 — it needs the consensus
    observable at submit time, which Platt+floor do not.)
  * every config is selected and gated by **leave-one-MATCH-out CV** over the 26
    matches (CV unit = match, so single outlier matches cannot be memorised);
  * a family/archetype only ships a non-identity calibration if it does NOT regress
    OOS (no-regression gate). Otherwise it stays identity.
  * the known fixed-bug (both-teams-SoT) rows are excluded.

Writes data/processed/confidence_gate.json = {family_or_arch: {a, b, f}} and prints a
before/after table incl. the worst-decile (tail) Brier, the quantity we most want to
shrink.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROC = "data/processed"
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _apply(p, a, b, f):
    """Platt (a,b) then a ONE-SIDED lower floor: raise low bets (the catastrophic
    confident-low misses) without capping our correct high-confidence calls. f is
    capped at the break-even region (~0.15); above that we'd be discarding real edge."""
    z = a * _logit(p) + b
    q = 1.0 / (1.0 + np.exp(-z))
    return np.maximum(q, f)


def fit_platt_reg(p, y, la=8.0, lb=1.5):
    """Ridge-regularized Platt by Brier loss. Strong prior a->1 (la), lighter b->0 (lb),
    so it only moves where the data clearly supports it (squad-quality lesson)."""
    def obj(t):
        a, b = t
        q = _apply(p, a, b, 0.0)
        return np.mean((q - y) ** 2) + la * (a - 1.0) ** 2 + lb * b ** 2
    r = minimize(obj, [1.0, 0.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def loo_match_brier(df, fit_a_b, floor):
    """Honest leave-one-match-out CV Brier of the config (a,b refit on each training fold)."""
    tot, n = 0.0, 0
    for m in df["match"].unique():
        tr, te = df[df["match"] != m], df[df["match"] == m]
        if fit_a_b:
            a, b = fit_platt_reg(tr["our_p"].values, tr["outcome"].values)
        else:
            a, b = 1.0, 0.0
        q = _apply(te["our_p"].values, a, b, floor)
        tot += float(np.sum((q - te["outcome"].values) ** 2)); n += len(te)
    return tot / n


def tail_brier(df, a, b, f, frac=0.1):
    """Mean Brier of the worst (highest-loss) decile under a config — the tail we hedge."""
    q = _apply(df["our_p"].values, a, b, f)
    b2 = (q - df["outcome"].values) ** 2
    k = max(1, int(len(b2) * frac))
    return float(np.sort(b2)[-k:].mean())


# capped at ~0.15: the break-even floor only helps where the true rate exceeds it;
# above ~0.15 we'd be asserting every rare event is >15% likely, discarding real edge.
FLOORS = [0.0, 0.06, 0.08, 0.10, 0.12, 0.15]
GATE_EPS = 5e-4  # require a MEANINGFUL CV gain; below this is noise -> stay identity


def fit_group(df, label):
    """Return (gate_entry|None, report dict). Ships only if LOO-CV improves on identity."""
    base = loo_match_brier(df, fit_a_b=False, floor=0.0)
    # candidate A: Platt only; candidate B: Platt + best floor (floor chosen by CV)
    platt_cv = loo_match_brier(df, fit_a_b=True, floor=0.0)
    best = {"a": 1.0, "b": 0.0, "f": 0.0, "cv": base, "kind": "identity"}
    if platt_cv < best["cv"] - GATE_EPS:
        best = {"a": None, "b": None, "f": 0.0, "cv": platt_cv, "kind": "platt"}
    for f in FLOORS[1:]:
        cv = loo_match_brier(df, fit_a_b=True, floor=f)
        if cv < best["cv"] - GATE_EPS:
            best = {"a": None, "b": None, "f": f, "cv": cv, "kind": f"platt+floor{f}"}
        cv0 = loo_match_brier(df, fit_a_b=False, floor=f)   # floor-only (no Platt)
        if cv0 < best["cv"] - GATE_EPS:
            best = {"a": 1.0, "b": 0.0, "f": f, "cv": cv0, "kind": f"floor{f}"}
    # final shipped (a,b) refit on the full group for the chosen kind
    if best["kind"] != "identity" and best["a"] is None:
        a, b = fit_platt_reg(df["our_p"].values, df["outcome"].values)
    else:
        a, b = best["a"] or 1.0, best["b"] or 0.0
    f = best["f"]
    rep = {"label": label, "n": len(df), "kind": best["kind"], "a": round(a, 3),
           "b": round(b, 3), "f": f, "cv_id": round(base, 4), "cv_new": round(best["cv"], 4),
           "tail_id": round(tail_brier(df, 1, 0, 0), 3), "tail_new": round(tail_brier(df, a, b, f), 3)}
    entry = None if best["kind"] == "identity" else {"a": round(a, 3), "b": round(b, 3), "f": f}
    return entry, rep


def main():
    df = pd.read_parquet(f"{PROC}/settled_with_crowd.parquet")
    df = df[~df["known_bug"]].copy()
    print(f"fitting on {len(df)} settled questions (bug rows excluded), {df['match'].nunique()} matches\n")

    # Family table is DIAGNOSTIC only (the heterogeneity within families — e.g. player_ga
    # bad vs player_scores +38 — makes family the wrong granularity for the floor).
    rows = []
    for fam in ["offsides", "player", "count", "goal", "timing", "other"]:
        g = df[df["family"] == fam]
        if len(g) >= 8:
            _, rep = fit_group(g, f"fam:{fam}")
            rows.append(rep)
    print("=== PER-FAMILY (diagnostic) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # SHIPPED gate is per-archetype: targets the floor at the archetypes with the
    # bet-too-low failure, and naturally EXEMPTS player_scores (n=4 < cutoff -> identity,
    # preserving our +38 edge there). Each ships only on a meaningful no-regression CV gain.
    gate, rows2 = {}, []
    for arch in ["offsides_ge", "player_ga", "fouls_gt", "win", "player_sot_ge", "player_scores", "sot_gt"]:
        g = df[df["archetype"] == arch]
        if len(g) < 8:
            continue
        entry, rep = fit_group(g, f"arch:{arch}")
        rows2.append(rep)
        if entry:
            gate[arch] = entry
    print("\n=== PER-ARCHETYPE (the shipped granularity) ===")
    print(pd.DataFrame(rows2).to_string(index=False))

    with open(f"{PROC}/confidence_gate.json", "w") as fh:
        json.dump(gate, fh, indent=1)
    print(f"\nwrote confidence_gate.json: {gate}")


if __name__ == "__main__":
    main()
