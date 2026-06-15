"""Improvement #2: backtest our no-line edge over the field, offline.

The live edge gate needs 20+ settled questions per family vs an UNOBSERVABLE
field — it may never fire (the conservatism trap). Instead, measure our
per-family edge over a realistic field on the six historical tournaments, and let
the extremizer fire on this BACKTESTED edge.

Field proxies (per settled backtest question, given our model belief p):
  sharp field   field_model(p, 0.70)  -- a fairly sharp crowd (CONSERVATIVE gate)
  prior field   field_model(p, prior) -- the Phase C prior sharpness
  base field    the archetype base rate regressed 50% toward 50 (lower bound)

A family earns a backtest gate (k=1.3) iff our Brier beats the SHARP field with
a paired t-test p<0.05 — i.e. we beat even a fairly sharp crowd, so we certainly
beat the naive contest field. This is the offline analog of "confirmed edge".

NOTE this measures EXPECTED edge (where our honest belief beats the field), which
is the correct gate for WHERE to take rank-variance — it does not (and cannot,
by the proper-scoring-rule theorem) show extremizing lowers Brier.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from jumpprob.models.strategy import FIELD_SHARPNESS, field_model

PROC = Path("data/processed")
K_ON_EDGE = 1.3
SHARP_LAM = 0.70


def load_family_predictions() -> dict[str, pd.DataFrame]:
    """Per no-line family: a frame of (p_model, outcome) from the backtests."""
    out = {}
    c = pd.read_csv(PROC / "backtest_counts.csv")
    out["count"] = c[c["src"] == "model"][["p", "outcome"]].rename(columns={"p": "p_model"})
    p = pd.read_csv(PROC / "backtest_players_raw.csv")
    out["player"] = p[["p_model", "outcome"]]
    v = pd.read_csv(PROC / "backtest_v2_go.csv")
    tim = v[v["archetype"].isin(["ht_tied", "h2_more_goals_than_h1"])]
    out["timing"] = tim[["p_v2", "outcome"]].rename(columns={"p_v2": "p_model"})
    return out


gate, report = {}, []
for fam, df in load_family_predictions().items():
    p = df["p_model"].clip(1e-6, 1 - 1e-6).to_numpy()
    y = df["outcome"].to_numpy()
    our_b = (p - y) ** 2
    sharp_b = (np.array([field_model(pi, SHARP_LAM) for pi in p]) - y) ** 2
    prior_b = (np.array([field_model(pi, FIELD_SHARPNESS[fam]) for pi in p]) - y) ** 2
    # paired t-test: is our per-question Brier lower than the sharp field's?
    diff = sharp_b - our_b   # positive = we beat the sharp field
    t, pval = stats.ttest_rel(sharp_b, our_b)
    confirmed = diff.mean() > 0 and pval < 0.05
    # SURVIVORSHIP GUARD: the player backtest is biased — its players HAD
    # international history (that's why they're in the data), but live WC squads
    # mostly don't, so our live player fair value is data-starved (e.g. Kubo: 90
    # min, predicted 7% score-or-assist, he did it -> Brier 0.865). Extremizing
    # there amplifies a broken model. Keep player at k=1 until the club-rate
    # model lands (PLAYER-DATA-PLAN.md #2), regardless of the backtest.
    if fam == "player":
        confirmed = False
    gate[fam] = K_ON_EDGE if confirmed else 1.0
    report.append({"family": fam, "n": len(df), "our_brier": our_b.mean(),
                   "sharp_field": sharp_b.mean(), "prior_field": prior_b.mean(),
                   "edge_vs_sharp": diff.mean(), "t": t, "p_value": pval,
                   "k": gate[fam], "confirmed": confirmed})

rep = pd.DataFrame(report)
print(rep.round(4).to_string(index=False))
(PROC / "backtest_gate.json").write_text(json.dumps(gate, indent=1))
print(f"\nwrote backtest_gate.json: {gate}")
print("Families with confirmed backtest edge get k=1.3 (extremizer fires); the "
      "live edge gate still augments this as live data accrues.")
