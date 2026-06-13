"""Prepare Bot B (v2) manual-entry sheet for all currently-open markets.

Fetches live open markets per match, computes Bot A (v1 analytic) and Bot B
(v2 MC simulator) probabilities, and writes a timing-rich file the dashboard
renders for manual entry:
  data/processed/bot_b_manual.csv
    [match, kickoff, question, archetype, p_bot_a, sub_a, p_bot_b, sub_b, diff]

Heavy compute (one Pipeline + one MCEngine, one simulation per match) lives
here so the Streamlit app stays fast and just reads the CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.calibration import submit_int
from jumpprob.models.dynamics import Dynamics
from jumpprob.models.uncertainty import beta_covariance
from jumpprob.normalize.teams import canonical
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine, answer
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.parser import parse_question
from jumpprob.submit.runner import Runner

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
matches = client.matches(EVENT, LOBBY)
open_matches = [m for m in matches if m.get("open_market_count", 0) > 0]
print(f"{len(open_matches)} matches with open markets")

print("loading models...")
pipe = Pipeline()
players = pd.read_parquet("data/processed/player_match.parquet")
runner = Runner(pipe, players)
dyn = Dynamics.from_json(Path("data/processed/minute_dynamics.json"))
cov = beta_covariance(pipe.goal_model, pipe.results_elo,
                      pipe.results_elo["date"].max() + pd.Timedelta(days=1))
eng = MCEngine(pipe.goal_model, pipe.count_model, dyn, n_draws=100_000,
               coupled={"goal"}, beta_cov=cov, seed=20260613)

rows = []
for m in open_matches:
    mkts = [k for k in client.markets(LOBBY, m["id"]) if k.get("status") == "open"]
    if not mkts:
        continue
    questions = [k["question"] for k in mkts]
    teams = runner.resolve_teams(m["name"], questions)
    if teams is None:
        print(f"  ! could not resolve teams for {m['name']} — skipping")
        continue
    team_a, team_b = teams
    ctx = runner.context(team_a, team_b)
    res = eng.simulate(pipe.ratings[team_a], pipe.ratings[team_b])
    for k in mkts:
        q = parse_question(k["question"])
        if q is None:
            continue
        side = None
        if q.subject:
            cn = canonical(q.subject)
            side = "a" if cn == team_a else "b" if cn == team_b else None
        p1 = runner.answer(q, ctx)
        p2 = answer(res, q.archetype, side, threshold=q.threshold, half=q.half)
        if p1 is None:
            continue
        if p2 is None:
            p2 = p1  # player/penalty archetypes: v2 == v1
        p1 = float(np.clip(p1, 0.01, 0.99))
        p2 = float(np.clip(p2, 0.01, 0.99))
        rows.append({
            "match": m["name"], "kickoff": m["closing_time"],
            "market_id": k["id"], "question": k["question"], "archetype": q.archetype,
            "p_bot_a": round(p1, 4), "sub_a": submit_int(p1),
            "p_bot_b": round(p2, 4), "sub_b": submit_int(p2),
            "diff": submit_int(p2) - submit_int(p1),
        })

df = pd.DataFrame(rows)
out = Path("data/processed/bot_b_manual.csv")
df.to_csv(out, index=False)
print(f"\nwrote {len(df)} markets across {df['match'].nunique()} matches -> {out}")
print(f"markets where Bot B differs from Bot A by >=3 pts: {(df['diff'].abs() >= 3).sum()}")
