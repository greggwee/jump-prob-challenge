"""v2 shadow mode: produce v2 predictions for every open market WITHOUT
submitting (one-prediction-per-account, A1) and log them next to v1's.

Run right after run_matchday.py. Output: data/processed/shadow_v2.csv with
(market_id, question, archetype, p_v1_submitted, p_v2). Settled comparison
happens in scripts/score_shadow.py once results land.

v2 covers team-level archetypes via the MC engine (one simulation per match,
100k draws — convergence-checked +/-0.005); player and penalty archetypes
reuse the v1 values (not simulated).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.dynamics import Dynamics
from jumpprob.models.uncertainty import beta_covariance
from jumpprob.normalize.teams import canonical
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine, answer
from jumpprob.submit.parser import parse_question
from jumpprob.submit.runner import Runner

DYN_PATH = Path("data/processed/minute_dynamics.json")
PRED_PATH = Path("data/processed/matchday_predictions.csv")

pipe = Pipeline()
players = pd.read_parquet("data/processed/player_match.parquet")
runner = Runner(pipe, players)
dyn = Dynamics.from_json(DYN_PATH)
cov = beta_covariance(pipe.goal_model, pipe.results_elo,
                      pipe.results_elo["date"].max() + pd.Timedelta(days=1))
# production config: goal-only coupling (ablation-decided, see V2-FINDINGS.md)
eng = MCEngine(pipe.goal_model, pipe.count_model, dyn, n_draws=100_000,
               coupled={"goal"}, beta_cov=cov, seed=20260612)

v1 = pd.read_csv(PRED_PATH)
rows = []
for mname, g in v1.groupby("match"):
    teams = runner.resolve_teams(mname, g["question"].tolist())
    if teams is None:
        continue
    team_a, team_b = teams
    ra, rb = pipe.ratings[team_a], pipe.ratings[team_b]
    res = eng.simulate(ra, rb)
    for r in g.itertuples():
        q = parse_question(r.question)
        if q is None:
            continue
        side = None
        if q.subject:
            c = canonical(q.subject)
            side = "a" if c == team_a else "b" if c == team_b else None
        p2 = answer(res, q.archetype, side, threshold=q.threshold, half=q.half)
        if p2 is None and q.archetype == "win" and side is None:
            pass
        rows.append({"market_id": r.market_id, "match": mname, "question": r.question,
                     "archetype": q.archetype, "p_v1": r.p_bot_b,
                     "p_v2": round(p2, 4) if p2 is not None else r.p_bot_b,
                     "v2_simulated": p2 is not None})

out = pd.DataFrame(rows)
out.to_csv("data/processed/shadow_v2.csv", index=False)
n_sim = int(out.v2_simulated.sum())
print(f"shadow v2: {len(out)} markets, {n_sim} simulated by v2, "
      f"{len(out) - n_sim} fall back to v1 (player/penalty archetypes)")
print(f"mean |p_v2 - p_v1| on simulated markets: "
      f"{(out[out.v2_simulated].p_v2 - out[out.v2_simulated].p_v1).abs().mean():.4f}")
