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

# cp1252 stdout pipe under the scheduler: a printed player name outside cp1252 ('ć') must
# never raise and abort a submission (it killed Bot A's USA near-KO tick).
for _s in (sys.stdout, sys.stderr):
    try:
        if _s is not None:
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.dynamics import Dynamics
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.constants import COUNT_ARCHS, PLAYER_ARCHS, TIMING_ARCHS
from jumpprob.submit.runner import Runner

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

# --- v3 edge-gated sizing (re-tiered 2026-06-30 on Bot A's validated edge map) --------
# Top-match-RBP = amplify variance ONLY where we are directionally RIGHT vs the noisy
# field (sharp market anchors + validated-skill markets); stay flat where we are a
# coin-flip (no-edge counts/timing/base-rate markets) -- extremizing those is pure
# variance with no directional edge and just bleeds expected RBP.
TIER_A = {  # sharpest edge -> extremize hardest (k=3.0, cap 3/97)
    # market-anchored goal markets: matching the sharp line beats the field; push the favourite
    "win", "draw", "btts", "total_ge", "total_le", "team_goals_ge",
    # knockout recoveries with validated out-of-sample skill (+.03-.05 Brier)
    "first_goal_match", "win_margin", "clean_sheet", "score_both_halves",
    # half-state goal markets off the (anchored) grid + proven-edge SoT
    "ht_leading", "team_scores_half", "goals_gt_half", "first_goal_half",
    "first_goal_and_win", "team_sot_ge", "sot_gt",
}
TIER_C = {  # NO directional edge -> keep near the calibrated value (do NOT gamble)
    "offsides_ge", "offsides_total_ge", "fouls_gt", "cards_ge", "cards_gt",
    "team_cards_ge", "both_teams_sot_ge", "both_teams_cards", "total_shots_ge",
    "btts_and_total_ge", "pen_or_red", "penalty_awarded", "red_card",
    "goal_pre_hb1", "goal_post_hb2", "cards_post_hb2", "corners_pre_hb1",
    "offsides_pre_hb1", "goal_1h_stoppage", "goal_2h_stoppage",
    "sub_before_ht", "substitute_scores", "any_player_sot_ge", "any_player_goals_ge",
}
# TIER_B (default): book-anchored PLAYER PROPS (sharp but mid base-rate -> moderate push)
#   player_sot_ge/_scores/_ga, the moderate-edge counts (corners, total SoT), and the
#   forward-validated knockout markets (advance, ht_tied, h2_more_goals, half_total_ge).
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

    anchors, player_props = {}, {}
    try:
        from jumpprob.ingest import odds_api
        anchors = odds_api.build_anchors(odds_api.fetch_events())
        player_props = odds_api.load_latest_props()   # reuse the full-run snapshot (no extra credits)
    except Exception as e:
        print(f"odds unavailable ({str(e)[:50]}) -> model-only goal markets", file=sys.stderr)

    pipe = Pipeline()
    players = pd.read_parquet("data/processed/player_match.parquet")
    dyn = Dynamics.from_json(Path("data/processed/minute_dynamics.json"))
    mc = MCEngine(pipe.goal_model, None, dyn, n_draws=20_000, coupled={"goal"}, seed=20260613)
    runner = Runner(pipe, players, mc=mc)

    mkts = [m for m in client.markets(LOBBY) if m.get("status") == "open"
            and (mfilter is None or m["match"]["name"] in mfilter)]
    preds, _ = runner.predictions_for([{**m, "match": m["match"]} for m in mkts],
                                      anchors=anchors, player_props=player_props)
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
