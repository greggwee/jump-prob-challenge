"""Matchday runner.

  python scripts/run_matchday.py             # dry-run: build + log everything, no POST
  python scripts/run_matchday.py --a1-test   # LIVE: both bots, ONE market (assumption A1)
  python scripts/run_matchday.py --live      # LIVE: batch submit all open markets, both bots
"""

import sys

import pandas as pd

from jumpprob.pipeline import Pipeline
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import Runner

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

live = "--live" in sys.argv
a1_test = "--a1-test" in sys.argv

bot_a = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=not (live or a1_test))
bot_b = SportsPredictClient("SPORTSPREDICT_KEY_B", dry_run=not (live or a1_test))

print("building models...")
pipe = Pipeline()
players = pd.read_parquet("data/processed/player_match.parquet")
runner = Runner(pipe, players)

print("fetching all open markets...")
all_markets = bot_a.markets(LOBBY)
open_markets = [m for m in all_markets if m.get("status") == "open"]
print(f"{len(open_markets)} open markets")

payloads, skipped = runner.predictions_for(open_markets)
df = pd.DataFrame(payloads)
df.to_csv("data/processed/matchday_predictions.csv", index=False)
print(f"answered {len(df)} markets; skipped {len(skipped)}")
if skipped:
    print(pd.DataFrame(skipped).groupby("reason").size().to_string())
    pd.DataFrame(skipped).to_csv("data/processed/matchday_skipped.csv", index=False)

print("\nper-archetype submission summary (bot B):")
if len(df):
    print(df.groupby("archetype")["p_bot_b"].agg(["count", "mean", "min", "max"]).round(3).to_string())

if a1_test:
    row = df.iloc[0]
    print(f"\nA1 TEST on: {row['match']} | {row['question']}")
    ra = bot_a.submit_batch([{"market_id": row["market_id"], "lobby_id": row["lobby_id"], "probability": int(row["sub_a"])}])
    print("bot A response:", ra)
    rb = bot_b.submit_batch([{"market_id": row["market_id"], "lobby_id": row["lobby_id"], "probability": int(row["sub_b"])}])
    print("bot B response:", rb)
    ok_b = rb and rb[0].get("succeeded", 0) == 1
    print("A1 RESULT:", "both bots accepted on same market" if ok_b else "REJECTED — one prediction per market per ACCOUNT; submit via Bot A only")
elif live:
    # DEPRECATED live path. The production submitter is scripts/run_matchday_odds.py:
    # it anchors Bot A goal intensities to the betting line (W_MARKET) and applies the
    # Phase-1 confidence floor, then submits `sub_a`. THIS script builds UN-anchored
    # predictions (no odds fetch), so submitting `sub_b`/`sub_a` from here would clobber
    # the live entry with the wrong configuration. Refuse and redirect so there is a
    # single live path (the divergence was a real footgun — see CALIBRATION-AGENTS-PLAN).
    print("\nLIVE submission is DISABLED in this script. Use the odds-anchored path:")
    print('    python scripts/run_matchday_odds.py --live [--matches "Team A vs Team B"]')
    print("(run_matchday.py is dry-build only: it writes matchday_predictions.csv for the join tables.)")
    sys.exit(2)
else:
    print("\nDRY RUN complete — payloads logged to data/submissions/, predictions to data/processed/matchday_predictions.csv")
