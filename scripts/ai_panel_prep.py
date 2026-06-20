"""Phase L0 — assemble the out-of-band panel INPUT for a slate.

Reads a source of (market_id, question, archetype, our_p) rows, attaches the
history grounding for player props, and writes data/processed/ai_panel_prep.json.
EXCLUDE archetypes (market-priced / sharp) are skipped — no point spending tokens
on markets the LLM must never move.

Sources:
  --source settled  (default)  data/processed/settled_with_crowd.parquet  [forward-test]
  --source live                data/processed/odds_anchor_run.csv          [a live slate]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from jumpprob.agents import panel as P
from jumpprob.agents.grounding import history_implied
from jumpprob.submit.parser import parse_question

PREP = Path("data/processed/ai_panel_prep.json")
PLAYER = {"player_sot_ge", "player_ga", "player_scores"}
_GKEYS = ("resolved", "league", "minutes", "sot_p90", "shots_p90", "goals_p90",
          "ga_p90", "assists_p90", "minutes_assumed", "half", "implied")


def _grounding(question: str, archetype: str) -> dict | None:
    if archetype not in PLAYER:
        return None
    q = parse_question(question)
    if q is None or not q.subject:
        return None
    hi = history_implied(q.subject, archetype, minutes=70, half=(q.half == 2))
    if hi.get("resolved") is None:
        return {"resolved": None, "note": "no club-history join -> base rate"}
    return {k: hi.get(k) for k in _GKEYS}


def _rows(source: str):
    if source == "live":
        df = pd.read_csv("data/processed/odds_anchor_run.csv")
        return [(r.market_id, r.question, r.archetype, float(r.p_bot_a)) for r in df.itertuples()]
    df = pd.read_parquet("data/processed/settled_with_crowd.parquet")
    return [(r.market_id, r.question, r.archetype, float(r.our_p)) for r in df.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["settled", "live"], default="settled")
    ap.add_argument("--out", default=str(PREP))
    args = ap.parse_args()

    items = []
    for mid, question, archetype, our_p in _rows(args.source):
        if our_p is None or (isinstance(our_p, float) and math.isnan(our_p)):
            continue
        role, action = P.route(archetype)
        if action == P.EXCLUDE:
            continue
        items.append({
            "market_id": str(mid), "question": question, "archetype": archetype,
            "role": role, "action": action, "model_pct": our_p,
            "grounding": _grounding(question, archetype),
        })

    Path(args.out).write_text(json.dumps(items, indent=2))
    by_action = Counter(i["action"] for i in items)
    grounded = sum(1 for i in items if i["grounding"] and i["grounding"].get("resolved"))
    print(f"prep: {len(items)} items ({args.source}) -> {args.out}")
    print(f"  by action: {dict(by_action)}")
    print(f"  player props with resolved history: {grounded}")


if __name__ == "__main__":
    main()
