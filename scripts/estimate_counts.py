"""Build event aggregate tables and estimate assumptions A3-A6 from data.

Prints: card half-split (A3), referee strictness spread (A4), corners-goals
correlation (A5), penalty base rate (A6), offside convention reconciliation,
and overdispersion diagnostics (variance/mean) per count family.
"""

import pandas as pd

from jumpprob.ingest.aggregate_events import build_tables

print("aggregating events (one pass over ~314 matches)...")
team_half, players = build_tables()
print(f"team-half rows: {len(team_half)}, player rows: {len(players)}\n")

per_match = team_half.groupby(["match_id", "tournament", "referee"], dropna=False).sum(numeric_only=True).reset_index()
per_team_match = team_half.groupby(["match_id", "team"]).sum(numeric_only=True).reset_index()

cards_by_half = team_half.assign(cards=team_half.yellows + team_half.reds).groupby("period")["cards"].sum()
print(f"A3 cards by half: 1H={cards_by_half[1]}, 2H={cards_by_half[2]}, ratio={cards_by_half[2]/cards_by_half[1]:.2f}")

ref = per_match.assign(cards=per_match.yellows + per_match.reds).groupby("referee")["cards"].agg(["mean", "count"])
ref5 = ref[ref["count"] >= 5]
print(f"A4 referee cards/match (refs with 5+ matches, n={len(ref5)}): "
      f"min={ref5['mean'].min():.2f}, median={ref5['mean'].median():.2f}, max={ref5['mean'].max():.2f}, "
      f"spread={ref5['mean'].max()/ref5['mean'].min():.2f}x")

corr = per_team_match[["corners", "goals", "shots"]].corr()
print(f"A5 corners-goals corr={corr.loc['corners','goals']:.3f}, corners-shots corr={corr.loc['corners','shots']:.3f}")

pens = per_match["penalties_awarded"]
print(f"A6 penalties: mean/match={pens.mean():.3f}, P(>=1)={(pens >= 1).mean():.3f}")

print(f"offside convention: dedicated events total={per_match['offside_events'].sum()}, "
      f"pass-offsides total={per_match['pass_offsides'].sum()}")

print("\noverdispersion (var/mean, per team-half):")
for col in ("corners", "yellows", "offside_events", "pass_offsides", "shots", "shots_on_target"):
    s = team_half[col]
    print(f"  {col}: mean={s.mean():.2f} var/mean={s.var()/s.mean():.2f}")

print("\nper-match means by tournament:")
print(per_match.groupby("tournament")[["corners", "yellows", "reds", "pass_offsides", "shots_on_target", "penalties_awarded"]].mean().round(2).to_string())
