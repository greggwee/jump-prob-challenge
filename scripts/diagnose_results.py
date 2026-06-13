"""Performance + reconciliation: how the two matches scored, field comparison,
and where the 700 -> 697 lobby count went."""

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
preds = pd.DataFrame(c.my_predictions(LOBBY))
results = c.results(LOBBY)
res = pd.DataFrame(results)

md = pd.read_csv("data/processed/matchday_predictions.csv")
q2match = dict(zip(md["question"], md["match"]))
preds["match"] = preds["question"].map(q2match)

# reconciliation: submitted vs present
print(f"submitted (matchday batch): {len(md)}  |  in lobby now: {len(preds)}")
submitted_q = set(md["question"])
present_q = set(preds["question"])
print(f"questions submitted but NOT in lobby now: {len(submitted_q - present_q)}")
for q in sorted(submitted_q - present_q):
    print(f"    dropped: {q[:70]}")

# per-match scoring
settled = preds.dropna(subset=["brier_score"])
print(f"\nsettled overall: {len(settled)}  mean Brier {settled.brier_score.mean():.4f}")
for m in ("USA vs PAR", "CAN vs BIH"):
    s = settled[settled["match"] == m]
    if len(s):
        print(f"\n  {m}: {len(s)} settled, mean Brier {s.brier_score.mean():.4f} "
              f"(0.25 = coin flip)")
        s2 = s.assign(outcome=np.isclose((s.probability/100 - 1)**2, s.brier_score, atol=1e-6).astype(int))
        for r in s2.sort_values("brier_score", ascending=False).itertuples():
            print(f"    p={r.probability:>2} outcome={r.outcome} brier={r.brier_score:.3f}  {r.question[:46]}")

# field comparison if the results endpoint carries it
if res is not None and "field_average_brier" in res.columns:
    rr = res.dropna(subset=["field_average_brier"])
    print(f"\nfield avg Brier (settled): {rr.field_average_brier.mean():.4f} "
          f"| ours: {settled.brier_score.mean():.4f}")
else:
    print("\n(results endpoint columns:", list(res.columns) if res is not None else None, ")")
