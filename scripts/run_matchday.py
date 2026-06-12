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
    # A1 test (2026-06-12) showed predictions are one-per-market-per-ACCOUNT:
    # the second key gets 'already exists'. All submissions go through Bot A
    # as a single consolidated leaderboard entry; KEY_B is backup/ops only.
    # Submit the pure calibrated model (sub_b) — the configuration the
    # backtests validated; the 25% base-rate blend variant was a two-bot
    # decorrelation device that no longer applies with a single entry.
    pa = [{"market_id": r["market_id"], "lobby_id": r["lobby_id"], "probability": int(r["sub_b"])} for r in payloads]
    print(f"\nLIVE: submitting {len(pa)} predictions via Bot A...")
    out_a = bot_a.submit_batch(pa)
    total = sum(b.get("total", 0) for b in out_a)
    ok = sum(b.get("succeeded", 0) for b in out_a)
    failed = sum(b.get("failed", 0) for b in out_a)
    print(f"batches: {len(out_a)} | total {total} succeeded {ok} failed {failed}")
    errs = {}
    for b in out_a:
        for r in b.get("results", []):
            if not r.get("success"):
                errs[r.get("error", "?")] = errs.get(r.get("error", "?"), 0) + 1
    if errs:
        print("failure reasons:", errs)
    print("done — verify with /predictions")
else:
    print("\nDRY RUN complete — payloads logged to data/submissions/, predictions to data/processed/matchday_predictions.csv")
