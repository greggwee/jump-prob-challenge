"""Full diagnostic of recent point losses (settled markets).

Splits losses into the categories that matter for a RELATIVE-Brier contest:
  - did we lose where we DEVIATED from the closing line and were wrong? (our fault)
  - or where we MATCHED the market and it was an upset? (the whole field lost too,
    so little RELATIVE damage)
  - or on the NO-LINE markets (props/counts) where relative position is decided?
Uses the CLV ledger (pure/market/blend per market) where available.
"""

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import COUNT_ARCHS, GOAL_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
pd.set_option("display.width", 200)

c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
res = pd.DataFrame(c.results(LOBBY)).dropna(subset=["brier_score"]).copy()
res["p"] = res["probability_submitted"] / 100.0
res["outcome"] = np.where(np.isclose((res["p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0, 0.0)
res.loc[np.isclose(res["p"], 0.5), "outcome"] = np.nan

md = pd.read_csv("data/processed/matchday_predictions.csv")[["market_id", "match", "archetype"]]
res = res.merge(md, on="market_id", how="left")

# order matches by kickoff (opening_time) for recency
ko = {m["name"]: m["opening_time"] for m in c.matches(EVENT, LOBBY)}
res["kickoff"] = res["match"].map(ko)
res["excess"] = res["brier_score"] - 0.25  # >0 = worse than a coin flip

fam = {**{a: "goal" for a in GOAL_ARCHS}, **{a: "count" for a in COUNT_ARCHS},
       **{a: "player" for a in PLAYER_ARCHS}}  # timing is inside GOAL_ARCHS
res["family"] = res["archetype"].map(fam).fillna("other")

print(f"=== {len(res)} settled questions across {res['match'].nunique()} matches ===\n")

bm = res.dropna(subset=["match"]).groupby("match").agg(
    n=("brier_score", "size"), brier=("brier_score", "mean"),
    excess_sum=("excess", "sum"), kickoff=("kickoff", "first")).sort_values("kickoff")
print("BY MATCH (chronological; excess_sum = total Brier above coin-flip = points bled):")
print(bm.round(3).to_string())

recent = bm.tail(6).index.tolist()
rec = res[res["match"].isin(recent)]
print(f"\n=== LAST {len(recent)} MATCHES: where the damage came from ===")
print("\nby family (mean Brier, excess vs 0.25, n):")
fb = rec.groupby("family").agg(n=("brier_score", "size"), brier=("brier_score", "mean"),
                               excess=("excess", "sum")).sort_values("excess", ascending=False)
print(fb.round(3).to_string())

print("\nworst 12 individual calls in the last matches:")
for r in rec.sort_values("brier_score", ascending=False).head(12).itertuples():
    oc = "YES" if r.outcome == 1 else "NO" if r.outcome == 0 else "?"
    print(f"  Brier {r.brier_score:.3f}  pred {int(r.p*100):>2} -> {oc:<3}  [{r.match}] {r.question[:50]}")

# CLV split: did our deviation from the closing line help or hurt on goal markets?
try:
    led = pd.read_csv("data/processed/clv_ledger.csv")
    led["ts"] = pd.to_datetime(led["ts"])
    closing = led.sort_values("ts").groupby("market_id").last().reset_index()
    j = rec.merge(closing[["market_id", "sub_pure", "sub_market", "sub_a"]], on="market_id").dropna(subset=["outcome"])
    if len(j):
        for s in ("sub_pure", "sub_market", "sub_a"):
            j[f"b_{s}"] = (j[s] / 100.0 - j["outcome"]) ** 2
        print(f"\n=== CLV on {len(j)} recent goal/anchored markets in the ledger ===")
        print(f"  pure-model Brier {j.b_sub_pure.mean():.4f}")
        print(f"  closing-line     {j.b_sub_market.mean():.4f}")
        print(f"  our blend        {j.b_sub_a.mean():.4f}")
        print(f"  deviation value (closing - blend): {j.b_sub_market.mean()-j.b_sub_a.mean():+.4f}  "
              f"(negative = our 15% model deviation HURT vs just taking the line)")
    else:
        print("\n(CLV ledger has no recent settled goal markets yet — accumulating.)")
except FileNotFoundError:
    print("\n(no CLV ledger yet)")
