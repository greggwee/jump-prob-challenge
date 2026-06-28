"""Build base rates for the event markets our minute data can't reach (stoppage-time
goals, substitutions) from api-football fixture events.

The free api-football plan only serves seasons 2022-2024, so we use WORLD CUP 2022 as
the most representative international-tournament base (same context as WC26). Stoppage
and substitution rates are stable year-to-year, so a 2022 base transfers.

Outputs data/processed/event_baselines.json:
  - goal_share_1h_stoppage / _2h_stoppage : fraction of all goals scored in 45+x / 90+x.
      These feed the SAME window model as hydration breaks: lambda_W = lambda_tot * share,
      P(>=1) = 1 - e^-lambda_W  (so stoppage goals inherit the match's scoring rate).
  - sub_before_ht  : P(>=1 in-play substitution before halftime) -- an ~injury base rate.
  - substitute_scores : P(a player who came on as a sub scores).  (in/out flagged uncertain)

  python scripts/build_event_baselines.py            # pull (cached, throttled) + write
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jumpprob.ingest import apifootball as af
from jumpprob.submit.env import load_dotenv

SOURCES = [(1, 2022)]  # (league_id, season): World Cup 2022. Free tier = 2022-2024 only.
OUT = Path("data/processed/event_baselines.json")


def main():
    load_dotenv()
    rows, goals_by_min = [], {"1h_stop": 0, "2h_stop": 0, "total": 0}
    for league, season in SOURCES:
        fixtures = af.fetch_fixtures(league, season)
        done = [f for f in fixtures if f["fixture"]["status"]["short"] == "FT"]
        print(f"league {league} {season}: {len(done)} finished fixtures")
        for i, f in enumerate(done):
            fid = f["fixture"]["id"]
            try:
                ev = af.fetch_fixture_events(fid)
            except Exception as e:  # quota/rate -> stop, keep what we cached
                print(f"  stopped at {i}/{len(done)}: {str(e)[:80]}")
                break
            goals = [e for e in ev if e["type"] == "Goal"]
            subs = [e for e in ev if e["type"] == "subst"]
            g1 = sum(1 for g in goals if g["time"]["elapsed"] == 45 and g["time"].get("extra"))
            g2 = sum(1 for g in goals if g["time"]["elapsed"] == 90 and g["time"].get("extra"))
            goals_by_min["1h_stop"] += g1
            goals_by_min["2h_stop"] += g2
            goals_by_min["total"] += len(goals)
            sub_pre = any((s["time"]["elapsed"] or 0) < 45 for s in subs)
            # api-football subst event: player = the one going OFF, assist = the one coming ON.
            # The substitute is `assist` — reading `player` counted starters and 2x-inflated this.
            came_on = {(s.get("assist") or {}).get("id") for s in subs}
            sub_scored = any((g.get("player") or {}).get("id") in came_on for g in goals)
            rows.append({"fid": fid, "goals": len(goals), "g1": g1, "g2": g2,
                         "sub_pre_ht": sub_pre, "sub_scored": sub_scored})

    df = pd.DataFrame(rows)
    tot = max(goals_by_min["total"], 1)
    base = {
        "source": "WorldCup2022 (api-football, free-tier 2022-2024)",
        "n_matches": int(len(df)),
        "n_goals": int(goals_by_min["total"]),
        # goal SHARES -> feed lambda_W = lambda_tot * share (same as hydration windows)
        "goal_share_1h_stoppage": goals_by_min["1h_stop"] / tot,
        "goal_share_2h_stoppage": goals_by_min["2h_stop"] / tot,
        # substitution base rates (per match)
        "sub_before_ht": float(df["sub_pre_ht"].mean()) if len(df) else None,
        "substitute_scores": float(df["sub_scored"].mean()) if len(df) else None,
        # raw per-match P(>=1) for stoppage, for sanity vs the lambda-scaled model
        "p_1h_stoppage_goal_raw": float((df["g1"] > 0).mean()) if len(df) else None,
        "p_2h_stoppage_goal_raw": float((df["g2"] > 0).mean()) if len(df) else None,
    }
    OUT.write_text(json.dumps(base, indent=1))
    print(json.dumps(base, indent=1))


if __name__ == "__main__":
    main()
