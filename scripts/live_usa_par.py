"""Live v1+v2 run for USA vs Paraguay (closes 2026-06-13T03:30Z).

v1 (analytic engine) -> Bot A: PATCH the existing account predictions to fresh
values computed on refreshed data.
v2 (MC simulator, goal-only coupling) -> Bot B: ATTEMPT batch submission on the
same markets. A1 says one-prediction-per-account, so this doubles as the
re-verification the user asked for; per-market API responses are reported.
"""

import numpy as np
import pandas as pd

from jumpprob.models.dynamics import Dynamics
from jumpprob.models.uncertainty import beta_covariance
from jumpprob.models.calibration import submit_int
from jumpprob.normalize.teams import canonical
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine, answer
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.parser import parse_question
from jumpprob.submit.runner import Runner

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
MATCH_ID = "f3e6d1f5-dfb2-4cab-8673-906307cc0b11"  # USA vs PAR

bot_a = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=False)
bot_b = SportsPredictClient("SPORTSPREDICT_KEY_B", dry_run=False)

print("models (on refreshed data)...")
pipe = Pipeline()
players = pd.read_parquet("data/processed/player_match.parquet")
runner = Runner(pipe, players)
dyn = Dynamics.from_json(__import__("pathlib").Path("data/processed/minute_dynamics.json"))
cov = beta_covariance(pipe.goal_model, pipe.results_elo,
                      pipe.results_elo["date"].max() + pd.Timedelta(days=1))
eng = MCEngine(pipe.goal_model, pipe.count_model, dyn, n_draws=100_000,
               coupled={"goal"}, beta_cov=cov, seed=13)

mkts = bot_a.markets(LOBBY, MATCH_ID)
open_mkts = [m for m in mkts if m.get("status") == "open"]
print(f"USA vs PAR: {len(mkts)} markets, {len(open_mkts)} open")

team_a, team_b = "United States", "Paraguay"
ra, rb = pipe.ratings[team_a], pipe.ratings[team_b]
ctx = runner.context(team_a, team_b)
res = eng.simulate(ra, rb)

rows = []
for m in open_mkts:
    q = parse_question(m["question"])
    if q is None:
        rows.append({"market_id": m["id"], "question": m["question"], "p1": None, "p2": None})
        continue
    side = None
    if q.subject:
        c = canonical(q.subject)
        side = "a" if c == team_a else "b" if c == team_b else None
    p1 = runner.answer(q, ctx)
    p2 = answer(res, q.archetype, side, threshold=q.threshold, half=q.half,)
    if p2 is None:
        p2 = p1  # player/penalty archetypes: v2 reuses v1
    rows.append({"market_id": m["id"], "question": m["question"],
                 "p1": float(np.clip(p1, 0.01, 0.99)) if p1 is not None else None,
                 "p2": float(np.clip(p2, 0.01, 0.99)) if p2 is not None else None})

df = pd.DataFrame(rows)
print("\nforecasts:")
for r in df.itertuples():
    print(f"  v1={submit_int(r.p1) if r.p1 else '--':>3} v2={submit_int(r.p2) if r.p2 else '--':>3}  {r.question}")

# --- Bot A: PATCH existing predictions to fresh v1 ---
preds = bot_a.my_predictions(LOBBY)
by_market = {p["market_id"]: p for p in preds}
print("\nBot A (v1 via PATCH):")
for r in df.itertuples():
    if r.p1 is None:
        continue
    new = submit_int(r.p1)
    existing = by_market.get(r.market_id)
    if existing is None:
        out = bot_a.submit_batch([{"market_id": r.market_id, "lobby_id": LOBBY, "probability": new}])
        print(f"  {new:>3} NEW    {r.question[:60]}")
    elif int(existing["probability"]) != new:
        bot_a.update_prediction(existing["id"], new)
        print(f"  {int(existing['probability']):>3} -> {new:<3} PATCH  {r.question[:60]}")
    else:
        print(f"  {new:>3} unchanged     {r.question[:60]}")

# --- Bot B: attempt v2 submissions (A1 re-verification) ---
print("\nBot B (v2 submission attempt):")
payload = [{"market_id": r.market_id, "lobby_id": LOBBY, "probability": submit_int(r.p2)}
           for r in df.itertuples() if r.p2 is not None]
out = bot_b.submit_batch(payload)
for batch in out:
    for res_item in batch.get("results", []):
        q = df[df.market_id == res_item["market_id"]]["question"].iloc[0][:60]
        status = "ACCEPTED" if res_item.get("success") else f"REJECTED: {res_item.get('error')}"
        print(f"  {status}  {q}")

df.to_csv("data/processed/live_usa_par.csv", index=False)
