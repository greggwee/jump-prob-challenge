"""Diagnose whether our predictions are registered for specific matches.

The /markets?match_id= endpoint only returns OPEN markets, so closed matches
look empty there. We instead reconcile our stored per-match question lists
against the live /predictions records (which carry question text + brier).
"""

import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

# questions we generated per match (from the matchday dry-run)
md = pd.read_csv("data/processed/matchday_predictions.csv")
match_questions = {m: set(g["question"]) for m, g in md.groupby("match")}

for key in ("SPORTSPREDICT_KEY_A", "SPORTSPREDICT_KEY_B"):
    c = SportsPredictClient(key, dry_run=True)
    if not c.key:
        print(f"{key}: NO KEY SET")
        continue
    preds = c.my_predictions(LOBBY)
    pred_q = {p["question"]: p for p in preds}
    print(f"\n===== {key}: {len(preds)} predictions in lobby =====")
    for match in ("USA vs PAR", "CAN vs BIH"):
        qs = match_questions.get(match, set())
        present = [q for q in qs if q in pred_q]
        missing = [q for q in qs if q not in pred_q]
        print(f"\n  {match}: {len(present)}/{len(qs)} of our questions have a prediction record")
        for q in present:
            p = pred_q[q]
            b = p.get("brier_score")
            print(f"    OK  prob={p['probability']:>2}  {'brier='+str(b) if b is not None else 'open':14}  {q[:48]}")
        for q in missing:
            print(f"    -- NO RECORD: {q[:60]}")
