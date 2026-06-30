"""Phase 7 live run: odds-anchored Bot A for all open markets.

  python scripts/run_matchday_odds.py            # dry-run: fetch, anchor, show before/after
  python scripts/run_matchday_odds.py --live     # PATCH Bot A to anchored values

Pure-model and odds-anchored values are both logged to
data/processed/odds_anchor_run.csv so closing-line value is measurable later.
"""

import sys

import numpy as np
import pandas as pd

from pathlib import Path

from jumpprob.ingest import odds_api
from jumpprob.models.calibration import submit_int
from jumpprob.models.dynamics import Dynamics
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import Runner
from jumpprob.submit.parser import parse_question
from jumpprob.normalize.teams import canonical
from jumpprob.submit.audit import book_vs_model_divergence

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
live = "--live" in sys.argv

client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=not live)

print("fetching live odds...")
events = odds_api.fetch_events()
anchors = odds_api.build_anchors(events)
print(f"{len(anchors)} fixtures with book consensus")

print("loading models...")
pipe = Pipeline()
players = pd.read_parquet("data/processed/player_match.parquet")
# v2 MC engine for timing markets (rec #3); count_model=None — timing needs only
# goals. Calibration (rec #1) auto-loads from data/processed/live_calibration.json.
dyn = Dynamics.from_json(Path("data/processed/minute_dynamics.json"))
mc = MCEngine(pipe.goal_model, None, dyn, n_draws=50_000, coupled={"goal"}, seed=20260613)
runner = Runner(pipe, players, mc=mc)
print(f"calibration loaded: {runner.calib}")

# optional targeted submission: --matches "Name A||Name B" limits to those
# matches (used by the near-kickoff scheduler so a run touches only ~10 markets).
matches_filter = None
if "--matches" in sys.argv:
    matches_filter = set(sys.argv[sys.argv.index("--matches") + 1].split("||"))

all_markets = [m for m in client.markets(LOBBY)
               if m.get("status") == "open"
               and (matches_filter is None or m["match"]["name"] in matches_filter)]
print(f"{len(all_markets)} open markets" + (f" (filtered to {matches_filter})" if matches_filter else ""))
market_rows = [{**m, "match": m["match"]} for m in all_markets]

# player-prop market anchor (verified live: SoT / scorer / assist). Fetch on a full run;
# reuse the cached snapshot on a targeted scheduler tick (per-event prop credits add up).
try:
    player_props = odds_api.fetch_player_props() if matches_filter is None else odds_api.load_latest_props()
    print(f"player props: {len(player_props)} fixtures")
except Exception as e:
    print(f"player props unavailable ({str(e)[:50]}) -> model-only")
    player_props = {}

# three passes for CLV (improvement #1): pure model (w=0, no anchors), the
# odds-anchored blend we submit (w=0.85), and the pure closing line (w=1).
pure, _ = runner.predictions_for(market_rows)
anchored, skipped = runner.predictions_for(market_rows, anchors=anchors, player_props=player_props)
blend_w = runner.w_market
runner.w_market = 1.0
market, _ = runner.predictions_for(market_rows, anchors=anchors, player_props=player_props)
runner.w_market = blend_w  # restore the blend weight
pure_by_mkt = {r["market_id"]: r for r in pure}
mkt_by_mkt = {r["market_id"]: r for r in market}

rows = []
for r in anchored:
    p = pure_by_mkt.get(r["market_id"], {})
    m = mkt_by_mkt.get(r["market_id"], {})
    rows.append({**r, "sub_pure": p.get("sub_a", r["sub_a"]),
                 "sub_market": m.get("sub_a", r["sub_a"])})
df = pd.DataFrame(rows)
df["delta"] = df["sub_a"] - df["sub_pure"]
df.to_csv("data/processed/odds_anchor_run.csv", index=False)

# append to the persistent CLV ledger (pure / market-close / blend per market)
ledger = Path("data/processed/clv_ledger.csv")
stamp = pd.Timestamp.now(tz="UTC").isoformat()
led = df[["market_id", "match", "question", "archetype", "sub_pure", "sub_market", "sub_a"]].copy()
led.insert(0, "ts", stamp)
led.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)

# which matches got anchored?
anchored_matches = sorted({r["match"] for r in anchored
                           if (runner.resolve_teams(r["match"], [r["question"]]) or (None, None)) and True})
moved = df[df["delta"].abs() >= 3]
print(f"\nanswered {len(df)} markets; {len(moved)} moved >=3 pts by the odds anchor")
print("\nbiggest moves (pure Bot A -> odds-anchored Bot A):")
big = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(20)
for r in big.itertuples():
    print(f"  {r.sub_pure:>2} -> {r.sub_a:<2} ({r.delta:+d})  [{r.match}] {r.question[:46]}")

# Name-join audit (robust future-proofing): a player prop whose subject matches NO book key
# despite the match having book data is a likely new spelling-miss class — this is exactly how
# the Haaland (middle name) and Sørloth/Ødegaard (ø-dropped) drops were found. Surface them
# every run so the next one is caught immediately instead of post-settlement.
_PLAYER_ARCH = {"player_scores", "player_ga", "player_sot_ge"}
anchor_misses = []
for r in anchored:
    if r["archetype"] not in _PLAYER_ARCH:
        continue
    teams = runner.resolve_teams(r["match"], [r["question"]])
    pp = (player_props.get(teams) or player_props.get((teams[1], teams[0])) or {}) if teams else {}
    if not pp:
        continue                       # no book data for this match -> uncovered, not a miss
    q = parse_question(r["question"], teams={canonical(t) for t in teams})
    subj = q.subject if q else None
    if subj and not odds_api.resolve_player(subj, pp):
        anchor_misses.append((r["match"], subj, r["archetype"]))
if anchor_misses:
    print(f"\n!! {len(anchor_misses)} player props have book data but NO name match "
          f"(possible new spelling miss -> extend norm_player/_TRANSLIT or resolve_player):")
    for mt, subj, a in anchor_misses[:15]:
        print(f"    [{mt}] {subj!r}  ({a})")

# Book-vs-model divergence audit: a large gap between the pure model and the de-vigged book
# is either a model DATA-GAP (Haaland 7 vs 52: rate not loaded) or a book market that doesn't
# match the question (scope/threshold/semantics). At w_market=0.85 a bad anchor is costly, so
# surface them every run + log a trend ledger. Reporting only — submissions are unchanged.
div = book_vs_model_divergence(df, min_pts=20)
if len(div):
    print(f"\n!! {len(div)} markets where the BOOK diverges >=20pts from the model "
          f"(data-gap or possible market mismatch — review, esp. player props):")
    for r in div.head(20).itertuples():
        print(f"    model {r.sub_pure:>2} -> book {r.sub_market:<3} final {r.sub_a:<3} "
              f"({r.book_gap:+d}, lg {r.logit_gap}) [{r.match}] {r.archetype}: {r.question[:36]}")
    _dl = Path("data/processed/divergence_audit.csv")
    led = div[["market_id", "match", "archetype", "question",
               "sub_pure", "sub_market", "sub_a", "book_gap", "logit_gap"]].copy()
    led.insert(0, "ts", stamp)
    led.to_csv(_dl, mode="a", header=not _dl.exists(), index=False)

if not live:
    print("\nDRY RUN — rerun with --live to PATCH Bot A. Sheet: data/processed/odds_anchor_run.csv")
else:
    preds = client.my_predictions(LOBBY)
    by_market = {p["market_id"]: p for p in preds}
    n_patch = n_new = n_fail = 0
    failures = []
    for r in df.itertuples():
        existing = by_market.get(r.market_id)
        new = int(r.sub_a)
        try:
            if existing is None:
                client.submit_batch([{"market_id": r.market_id, "lobby_id": LOBBY, "probability": new}])
                n_new += 1
            elif int(existing["probability"]) != new:
                client.update_prediction(existing["id"], new)
                n_patch += 1
        except Exception as e:  # one market's failure must not abort the batch
            n_fail += 1
            failures.append((r.market_id, str(e)[:80]))
    print(f"\nLIVE: PATCHed {n_patch}, submitted {n_new} new, {n_fail} failed, "
          f"{len(df) - n_patch - n_new - n_fail} already current.")
    if failures:
        print("failures (re-run to retry these):")
        for mid, err in failures[:10]:
            print(f"  {mid}: {err}")
