"""Improvement #1: closing-line value (CLV) of gigabot1.

For each settled market, compares the realized Brier of three probability
sources we logged at submission (the last CLV-ledger entry before kickoff):
  pure   model only (no odds)
  market the de-vigged closing consensus (w=1)
  blend  what we submitted (85% market / 15% model)

CLV of the blend = Brier(market) - Brier(blend): positive means our 15% model
deviation BEAT the closing line (genuine edge); negative means it was noise that
cost us. Reported overall, on book-priced MAIN markets (real CLV), and by family.
"""

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import COUNT_ARCHS, GOAL_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
MAIN = {"win", "draw", "total_ge", "total_le", "btts", "btts_and_total_ge"}  # book-priced-ish

led = pd.read_csv("data/processed/clv_ledger.csv")
# closing entry = last ledger row per market before it settled
led["ts"] = pd.to_datetime(led["ts"])
closing = led.sort_values("ts").groupby("market_id").last().reset_index()

c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
res = pd.DataFrame(c.results(LOBBY)).dropna(subset=["brier_score"])
res["p"] = res["probability_submitted"] / 100.0
res["outcome"] = np.where(np.isclose((res["p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0, 0.0)
res.loc[np.isclose(res["p"], 0.5), "outcome"] = np.nan

m = closing.merge(res[["market_id", "outcome"]], on="market_id").dropna(subset=["outcome"])
if m.empty:
    print("No settled markets are in the CLV ledger yet — accumulates as anchored "
          "markets (logged from now on) settle. Run after a matchday completes.")
    raise SystemExit

for src in ("sub_pure", "sub_market", "sub_a"):
    m[f"b_{src}"] = (m[src] / 100.0 - m["outcome"]) ** 2

def report(df, label):
    if df.empty:
        print(f"\n{label}: no settled markets yet"); return
    bp, bm, bb = df.b_sub_pure.mean(), df.b_sub_market.mean(), df.b_sub_a.mean()
    print(f"\n{label} (n={len(df)}):")
    print(f"  pure-model Brier  {bp:.4f}")
    print(f"  closing-line Brier {bm:.4f}")
    print(f"  our blend Brier   {bb:.4f}")
    print(f"  anchor value (market - pure): {bp - bm:+.4f}  (positive = odds beat pure model)")
    print(f"  CLV of blend (market - blend): {bm - bb:+.4f}  "
          f"(positive = our 15% model BEAT the close)")

report(m, "ALL settled ledger markets")
report(m[m.archetype.isin(MAIN)], "MAIN markets (real book CLV)")

# improvement #5: right-size the odds-blend weight from goal-market CLV.
GOAL = MAIN | {"team_scores_half", "half_total_ge", "team_goals_ge", "goals_gt_half",
               "ht_tied", "ht_leading", "h2_more_goals_than_h1"}
gm = m[m.archetype.isin(GOAL)]
print("\n=== #5 weight recommendation (goal-market CLV) ===")
if len(gm) < 30:
    print(f"  data-gated: only {len(gm)} settled goal markets in the ledger "
          f"(need ~30+). Hold w_market=0.85; recommendation fires as data accrues.")
else:
    bp, bm_, bb = gm.b_sub_pure.mean(), gm.b_sub_market.mean(), gm.b_sub_a.mean()
    print(f"  n={len(gm)}  Brier pure {bp:.4f} | blend(0.85) {bb:.4f} | market(1.0) {bm_:.4f}")
    if bm_ < bb - 0.0005:
        print(f"  -> 15% model has NEGATIVE CLV ({bm_-bb:+.4f}); RAISE w_market toward 0.95")
    elif bb <= bm_ and bb <= bp:
        print("  -> blend is best; HOLD w_market at 0.85")
    elif bp < bb:
        print("  -> market is hurting; LOWER w_market")
    else:
        print("  -> blend ~ market; mild raise toward 0.90 reasonable")
print("\nby family:")
fam = {**{a:"goal" for a in GOAL_ARCHS}, **{a:"count" for a in COUNT_ARCHS},
       **{a:"player" for a in PLAYER_ARCHS}}
m["family"] = m.archetype.map(fam).fillna("other")
g = m.groupby("family").agg(n=("outcome","size"), pure=("b_sub_pure","mean"),
        close=("b_sub_market","mean"), blend=("b_sub_a","mean"))
g["clv"] = (g["close"] - g["blend"]).round(4)
print(g.round(4).to_string())
m.to_csv("data/processed/clv_analysis.csv", index=False)
