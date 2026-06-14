"""Statistician's review of gigabot1 (Bot A) settled predictions.

Pulls settled results, recovers outcomes, and analyses accuracy overall, by
archetype, by match, calibration (reliability), and the worst/best calls.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
pd.set_option("display.width", 200)

c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
res = pd.DataFrame(c.results(LOBBY))
# cache a fresh snapshot for the dashboard
preds = c.my_predictions(LOBBY)
Path("data/processed/api_snapshot.json").write_text(json.dumps(
    {"fetched_at": datetime.now(timezone.utc).isoformat(), "predictions": preds, "results": c.results(LOBBY)}))

res = res.dropna(subset=["brier_score"]).copy()
res["p"] = res["probability_submitted"] / 100.0
amb = np.isclose(res["p"], 0.5)
res["outcome"] = np.where(np.isclose((res["p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0, 0.0)
res.loc[amb, "outcome"] = np.nan

md = pd.read_csv("data/processed/matchday_predictions.csv")[["market_id", "match", "archetype"]]
res = res.merge(md, on="market_id", how="left")

n = len(res)
print(f"=== gigabot1 (Bot A): {n} settled questions ===")
print(f"mean Brier {res.brier_score.mean():.4f}  (coin-flip 0.2500; lower better)")
print(f"vs coin-flip: {0.25 - res.brier_score.mean():+.4f} per question")
print(f"YES-outcome rate {res.outcome.mean():.0%}")
print(f"questions we 'won' (Brier<0.25): {(res.brier_score<0.25).mean():.0%}\n")

print("=== BY ARCHETYPE (where we're strong/weak + calibration) ===")
by = res.groupby("archetype").agg(n=("brier_score","size"), brier=("brier_score","mean"),
        mean_pred=("p","mean"), realized=("outcome","mean")).sort_values("brier")
by["miscal"] = (by["mean_pred"] - by["realized"]).round(3)
print(by.round(3).to_string())

print("\n=== BY MATCH ===")
bm = res.groupby("match").agg(n=("brier_score","size"), brier=("brier_score","mean")).sort_values("brier")
print(bm.round(3).to_string())

print("\n=== CALIBRATION (reliability) ===")
res["bucket"] = pd.cut(res.p, np.linspace(0,1,6), include_lowest=True)
rel = res.groupby("bucket", observed=True).agg(n=("p","size"), mean_pred=("p","mean"), realized=("outcome","mean"))
rel["gap"] = (rel["realized"] - rel["mean_pred"]).round(3)
print(rel.round(3).to_string())

print("\n=== 8 WORST CALLS (highest Brier) ===")
worst = res.sort_values("brier_score", ascending=False).head(8)
for r in worst.itertuples():
    oc = "YES" if r.outcome==1 else "NO" if r.outcome==0 else "?"
    print(f"  Brier {r.brier_score:.3f}  pred {int(r.p*100):>2} outcome {oc:>3}  [{r.match}] {r.question[:48]}")

print("\n=== 8 BEST CALLS (lowest Brier) ===")
best = res.sort_values("brier_score").head(8)
for r in best.itertuples():
    oc = "YES" if r.outcome==1 else "NO" if r.outcome==0 else "?"
    print(f"  Brier {r.brier_score:.3f}  pred {int(r.p*100):>2} outcome {oc:>3}  [{r.match}] {r.question[:48]}")
