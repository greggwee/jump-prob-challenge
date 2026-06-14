"""Phase B measurement: the CROWD BIAS = Polymarket crowd price vs the sharp-book
consensus (truth). This is what we exploit in relative scoring, and the prior the
Phase C structural field model inherits.

For each match where we have BOTH a sharp consensus and a Polymarket price:
  bias_outcome = crowd_prob - sharp_prob
Aggregated, this reveals favourite-longshot (crowd overprices longshots,
underprices favourites) and draw bias.
"""

import numpy as np
import pandas as pd

from jumpprob.ingest import odds_api
from jumpprob.ingest import prediction_markets as pm

print("fetching Polymarket crowd prices...")
crowd = pm.fetch_match_crowd()
print(f"  {len(crowd)} matches with a full 3-way crowd price")

print("fetching sharp-book consensus...")
events = odds_api.fetch_events()
anchors = odds_api.build_anchors(events)
print(f"  {len(anchors)} matches with sharp consensus")

rows = []
for (ha, aa), anc in anchors.items():
    mc = pm.crowd_for(crowd, ha, aa)
    if mc is None:
        continue
    rows.append({
        "match": f"{ha} v {aa}",
        "sharp_home": anc.p_home, "crowd_home": mc.p_home,
        "sharp_draw": anc.p_draw, "crowd_draw": mc.p_draw,
        "sharp_away": anc.p_away, "crowd_away": mc.p_away,
        "fav_sharp": max(anc.p_home, anc.p_away),
        "fav_crowd": mc.p_home if anc.p_home >= anc.p_away else mc.p_away,
        "dog_sharp": min(anc.p_home, anc.p_away),
        "dog_crowd": mc.p_away if anc.p_home >= anc.p_away else mc.p_home,
        "vol": mc.volume, "spread": mc.spread,
    })

df = pd.DataFrame(rows)
if df.empty:
    print("\nno overlap between Polymarket and sharp consensus — cannot measure bias yet")
    raise SystemExit
df.to_csv("data/processed/phase_b_crowd_bias.csv", index=False)

print(f"\n{len(df)} matches matched on both. Per-match (sharp -> crowd):")
for r in df.itertuples():
    print(f"  {r.match:28} fav {r.fav_sharp:.0%}->{r.fav_crowd:.0%}  "
          f"draw {r.sharp_draw:.0%}->{r.crowd_draw:.0%}  dog {r.dog_sharp:.0%}->{r.dog_crowd:.0%}"
          f"  (vol ${r.vol:,.0f})")

print("\n=== CROWD BIAS (crowd - sharp), volume-weighted where it matters ===")
w = df["vol"].clip(lower=1)
def wavg(col_c, col_s):
    return float(np.average(df[col_c] - df[col_s], weights=w))
print(f"  favourite: {wavg('fav_crowd','fav_sharp'):+.3f}  "
      f"(negative = crowd UNDERprices favourites = favourite-longshot bias)")
print(f"  underdog:  {wavg('dog_crowd','dog_sharp'):+.3f}  "
      f"(positive = crowd OVERprices longshots)")
print(f"  draw:      {wavg('crowd_draw','sharp_draw'):+.3f}")
print(f"\nmean |crowd-sharp| on favourite: {float(np.mean(np.abs(df.fav_crowd-df.fav_sharp))):.3f}")
print("This bias vector is the empirical prior for the Phase C field model.")
