"""Fit the per-archetype CROWD PRIOR from harvested post-lock crowd data (P1).

The crowd is only revealed after lock, but on base-rate/no-edge archetypes its value
clusters tightly per archetype (pen_or_red 37.7±3.0, cards_ge 48.7±5.1, substitute
31±1.7 ...), so the archetype mean is a usable PRE-LOCK proxy for where the crowd will
sit. The submit-time clamp (jumpprob.submit.audit.crowd_clamp) uses this to stop
catastrophic unbacked deviations (the measured risk curve: |our-crowd| 25+ pts was
-0.095/mkt, all bug-class) while matchup-dependent archetypes (win ±17.6, sot_gt ±19.3)
are excluded by the std gate and stay with the book/model.

Run after new crowd rows are harvested (daily_learn step). Append-only inputs:
data/processed/crowd_consensus_raw.csv (id8, crowd, out) joined to our settled results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jumpprob.submit.api_client import SportsPredictClient  # noqa: E402
from jumpprob.submit.parser import parse_question  # noqa: E402

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
OUT = ROOT / "data" / "processed" / "crowd_prior.json"


def main() -> None:
    cr = pd.read_csv(ROOT / "data" / "processed" / "crowd_consensus_raw.csv")
    cr["id8"] = cr.id8.astype(str)
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    r = pd.DataFrame(c.results(LOBBY))
    r = r[r.brier_score.notna()].copy()
    r["id8"] = r.market_id.str[:8]
    # archetype from the CLV ledger (parsed live WITH team context) — a bare re-parse here
    # mis-labels team-SoT questions as player_sot_ge (the player->team reroute needs the
    # team set), polluting the player prior with team questions. Fall back to a bare parse
    # only for markets that predate the ledger.
    led = pd.read_csv(ROOT / "data" / "processed" / "clv_ledger.csv")
    led = led.sort_values("ts").groupby("market_id").last().reset_index()[["market_id", "archetype"]]
    r = r.merge(led, on="market_id", how="left")
    bare = r["archetype"].isna()
    r.loc[bare, "archetype"] = r.loc[bare, "question"].apply(
        lambda q: (lambda x: x.archetype if x else None)(parse_question(q)))
    r["arch"] = r["archetype"]
    j = r.merge(cr[["id8", "crowd"]], on="id8", how="inner").dropna(subset=["arch", "crowd"])

    g = j.groupby("arch")["crowd"].agg(["count", "mean", "std"])
    prior = {a: {"n": int(row["count"]), "mean": round(float(row["mean"]), 1),
                 "std": round(float(row["std"]), 1) if np.isfinite(row["std"]) else None}
             for a, row in g.iterrows()}
    OUT.write_text(json.dumps(prior, indent=1))
    usable = {a for a, v in prior.items()
              if v["n"] >= 5 and v["std"] is not None and v["std"] <= 12.0}
    print(f"crowd_prior: {len(prior)} archetypes from {len(j)} crowd rows -> {OUT}")
    print(f"  usable as clamp proxy (n>=5, std<=12): {len(usable)}: {sorted(usable)}")


if __name__ == "__main__":
    main()
