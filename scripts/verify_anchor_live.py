"""Confirm the odds-anchored PATCH took on the live API for a few key markets."""

import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
preds = c.my_predictions(LOBBY)
live = {p["market_id"]: p for p in preds}  # market_id is the unique key
sheet = pd.read_csv("data/processed/odds_anchor_run.csv")

checks = sheet.reindex(sheet["delta"].abs().sort_values(ascending=False).index).head(12)
print(f"{'live':>4} {'want':>4} {'pure':>4} {'status':>7}  match / question")
ok = closed = 0
for r in checks.itertuples():
    p = live.get(r.market_id)
    lv = p["probability"] if p else None
    status = p["market_status"] if p else "missing"
    if status != "open":
        closed += 1
        mark = "closed"
    elif lv == r.sub_a:
        ok += 1
        mark = "OK"
    else:
        mark = "??"
    print(f"{str(lv):>4} {r.sub_a:>4} {r.sub_pure:>4} {status:>7}  {mark} [{r.match}] {r.question[:38]}")
print(f"\n{ok} confirmed at anchored value, {closed} already closed (kickoff passed), "
      f"of {len(checks)} checked")
