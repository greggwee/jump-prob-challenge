"""GigaLint rank-aggressive submission sheet — COMPUTE ONLY, never submits.

Layers 1-4 of the plan: Bot A's calibrated belief q  ->  rank-extremize transform
    p = sigmoid(K * logit(q))            # K>1 amplifies CONVICTION, self-scaling:
                                         #   a coin-flip (q~0.5) stays moderate,
                                         #   a confident call is pushed toward the cap
Because the prize is best-SINGLE-match rank, downside variance is free, so we lean
aggressive. This script writes data/processed/gigalint_sheet.csv and prints it.
It does NOT touch the website or any account.

  python scripts/gigalint_sheet.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.dynamics import Dynamics
from jumpprob.pipeline import Pipeline
from jumpprob.simulate.mc_engine import MCEngine
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import Runner

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
K = 2.5                  # rank-aggression (1 = honest, higher = more extreme)
LO, HI = 0.02, 0.98      # submission cap (mathematically reasonable; widen to .01/.99 for max variance)


def _logit(p): return float(np.log(p / (1 - p)))
def _sig(x): return float(1 / (1 + np.exp(-x)))


def main() -> None:
    client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    now = datetime.now(timezone.utc)
    ko = {m["name"]: (datetime.fromisoformat(m["opening_time"].replace("Z", "+00:00")) - now).total_seconds() / 3600
          for m in client.matches(EVENT, LOBBY)}

    anchors = {}
    try:                                            # graceful: odds API may be out of credits
        from jumpprob.ingest import odds_api
        anchors = odds_api.build_anchors(odds_api.fetch_events())
        print(f"odds anchors: {len(anchors)} fixtures", file=sys.stderr)
    except Exception as e:
        print(f"ODDS UNAVAILABLE ({str(e)[:60]}) -> goal markets use model-only", file=sys.stderr)

    pipe = Pipeline()
    players = pd.read_parquet("data/processed/player_match.parquet")
    dyn = Dynamics.from_json(Path("data/processed/minute_dynamics.json"))
    mc = MCEngine(pipe.goal_model, None, dyn, n_draws=20_000, coupled={"goal"}, seed=20260613)
    runner = Runner(pipe, players, mc=mc)

    markets = [m for m in client.markets(LOBBY) if m.get("status") == "open"]
    preds, _ = runner.predictions_for([{**m, "match": m["match"]} for m in markets], anchors=anchors)

    out = []
    for r in preds:
        q = float(np.clip(r["p_bot_a"], 1e-4, 1 - 1e-4))     # Bot A calibrated belief
        p = float(np.clip(_sig(K * _logit(q)), LO, HI))      # rank-aggressive submission
        conv = abs(q - 0.5)
        out.append({"kickoff_h": round(ko.get(r["match"], 99), 1), "match": r["match"],
                    "question": r["question"], "archetype": r["archetype"],
                    "bot_a": round(q * 100), "aggressive": round(p * 100),
                    "dir": "YES" if q >= 0.5 else "NO", "conviction": round(conv, 2),
                    "fade_candidate": "*" if conv >= 0.30 else ""})
    df = pd.DataFrame(out).sort_values(["kickoff_h", "match", "conviction"], ascending=[True, True, False])
    df.to_csv("data/processed/gigalint_sheet.csv", index=False)
    print(f"wrote gigalint_sheet.csv: {len(df)} markets / {df['match'].nunique()} matches | K={K} cap[{LO},{HI}]")


if __name__ == "__main__":
    main()
