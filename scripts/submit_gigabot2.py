"""Submit the aggressive v2 rank-chaser to gigabot2 (KEY_B) — a separate sanctioned entry.

Bot A (KEY_A / gigabot1) stays the steady calibrated entry; gigabot2 is the high-variance
rank play, sized by the reliability-gated v2 logic: extremize ONLY where the field is clueless
AND we're reliable AND we're confident, otherwise leave near the calibrated belief.

  python scripts/submit_gigabot2.py                       # dry-run: compute + summary, NO submit
  python scripts/submit_gigabot2.py --live                # submit ALL open markets to gigabot2
  python scripts/submit_gigabot2.py --live --matches "A vs B||C vs D"   # targeted (scheduler)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.dynamics import Dynamics
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import COUNT_ARCHS, PLAYER_ARCHS, TIMING_ARCHS, Runner

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

# --- v2 reliability-gated sizing (mirror of scripts/gigalint_size.py) -----------------
SHARP = {"goal": 0.85, "player": 0.45, "count": 0.40, "timing": 0.35}
TIER_A = {"team_sot_ge", "sot_gt", "team_corners_ge", "pen_or_red", "ht_leading",
          "team_scores_half", "goals_gt_half", "first_goal_half"}
TIER_C = {"offsides_ge", "fouls_gt", "cards_ge", "cards_gt", "team_cards_ge",
          "both_teams_sot_ge", "corners_gt", "total_sot_ge", "total_sot_half_ge",
          "corners_total_ge", "player_ga", "player_sot_ge", "player_scores",
          "btts_and_total_ge"}
KCAP = {"A": (3.0, 0.03, 0.97), "B": (1.8, 0.10, 0.90), "C": (1.0, 0.12, 0.88)}


def _logit(p): return float(np.log(p / (1 - p)))
def _sig(x): return float(1 / (1 + np.exp(-x)))
def tier(a): return "A" if a in TIER_A else ("C" if a in TIER_C else "B")


def size_v2(q: float, a: str) -> float:
    q = float(np.clip(q, 1e-3, 1 - 1e-3)); t = tier(a); k, lo, hi = KCAP[t]
    if abs(q - 0.5) < 0.05:                 # coin-flip: no signal, don't gamble
        return q
    if t == "C":                            # no-edge / miscalibrated: keep near calibrated
        return float(np.clip(q, lo, hi))
    return float(np.clip(_sig(k * _logit(q)), lo, hi))   # reliable: extremize


def main() -> None:
    live = "--live" in sys.argv
    mfilter = None
    if "--matches" in sys.argv:
        mfilter = set(sys.argv[sys.argv.index("--matches") + 1].split("||"))
    client = SportsPredictClient("SPORTSPREDICT_KEY_B", dry_run=not live)

    anchors = {}
    try:
        from jumpprob.ingest import odds_api
        anchors = odds_api.build_anchors(odds_api.fetch_events())
    except Exception as e:
        print(f"odds unavailable ({str(e)[:50]}) -> model-only goal markets", file=sys.stderr)

    pipe = Pipeline()
    players = pd.read_parquet("data/processed/player_match.parquet")
    dyn = Dynamics.from_json(Path("data/processed/minute_dynamics.json"))
    mc = MCEngine(pipe.goal_model, None, dyn, n_draws=20_000, coupled={"goal"}, seed=20260613)
    runner = Runner(pipe, players, mc=mc)

    mkts = [m for m in client.markets(LOBBY) if m.get("status") == "open"
            and (mfilter is None or m["match"]["name"] in mfilter)]
    preds, _ = runner.predictions_for([{**m, "match": m["match"]} for m in mkts], anchors=anchors)
    plan = [{"market_id": r["market_id"], "lobby_id": r["lobby_id"], "match": r["match"],
             "question": r["question"], "archetype": r["archetype"], "tier": tier(r["archetype"]),
             "submit": int(np.clip(round(size_v2(r["p_bot_a"], r["archetype"]) * 100), 1, 99))}
            for r in preds]
    df = pd.DataFrame(plan)
    df.to_csv("data/processed/gigabot2_plan.csv", index=False)
    extremes = int(((df.submit <= 5) | (df.submit >= 95)).sum())
    print(f"computed {len(df)} markets | tiers {df.tier.value_counts().to_dict()} | extreme bets: {extremes}")
    print("sample:", [(r.match[:14], r.submit) for r in df.head(6).itertuples()])

    if not live:
        print("DRY RUN — rerun with --live to submit to gigabot2. Plan: data/processed/gigabot2_plan.csv")
        return

    existing = {p["market_id"]: p for p in client.my_predictions(LOBBY)}
    new = [{"market_id": r.market_id, "lobby_id": r.lobby_id, "probability": int(r.submit)}
           for r in df.itertuples() if r.market_id not in existing]
    n_new = n_patch = n_fail = 0
    failures = []
    if new:
        try:
            res = client.submit_batch(new)
            n_new = sum(b.get("succeeded", 0) for b in res)
            n_fail += sum(b.get("failed", 0) for b in res)
        except Exception as e:
            failures.append(("batch", str(e)[:90]))
    for r in df.itertuples():                      # update any that already existed
        ex = existing.get(r.market_id)
        if ex and int(ex["probability"]) != int(r.submit):
            try:
                client.update_prediction(ex["id"], int(r.submit)); n_patch += 1
            except Exception as e:
                n_fail += 1; failures.append((r.market_id[:8], str(e)[:60]))
    print(f"\nLIVE gigabot2: submitted {n_new} new, PATCHed {n_patch}, {n_fail} failed "
          f"({len(df)} total markets)")
    for mid, err in failures[:8]:
        print("  fail", mid, err)


if __name__ == "__main__":
    main()
