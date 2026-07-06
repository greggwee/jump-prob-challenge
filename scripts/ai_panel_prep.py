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
BASE_RATES = json.loads(Path("data/processed/base_rates.json").read_text())
PLAYER = {"player_sot_ge", "player_ga", "player_scores"}

# real model data for the LLM: each match's expected goals + team Elo, so a panelist
# reasons from our numbers (tempo, who's stronger) rather than the question alone.
_PIPE = None


def _match_data(match: str) -> dict | None:
    global _PIPE
    try:
        from jumpprob.normalize.teams import canonical
        from jumpprob.pipeline import Pipeline
        from jumpprob.submit.runner import CODE_MAP
        if _PIPE is None:
            _PIPE = Pipeline()
        a, b = [canonical(CODE_MAP.get(t.strip(), t.strip())) for t in match.split(" vs ")]
        la, lb = _PIPE.goal_model.intensities(_PIPE.ratings[a], _PIPE.ratings[b])
        return {"exp_goals": f"{a} {la:.2f} vs {b} {lb:.2f} (total {la + lb:.2f})",
                "elo": f"{a} {_PIPE.ratings[a]:.0f} vs {b} {_PIPE.ratings[b]:.0f}"}
    except Exception:
        return None
# Tier-1 rich grounding: full club profile + matchup + the archetype base rate, so
# the panel reasons from the player's whole track record and the opponent (inferred
# from the matchup) and stays anchored to the empirical base rate (curbs over-pricing).
_GKEYS = ("resolved", "position", "league", "minutes", "sot_p90", "shots_p90",
          "goals_p90", "npg_p90", "assists_p90", "ga_p90", "xG_p90", "xA_p90",
          "key_passes_p90", "minutes_assumed", "half", "implied")


def _grounding(question: str, archetype: str, match: str) -> dict | None:
    if archetype not in PLAYER:
        # tactical/team questions: who is playing, the base rate, AND our real model
        # numbers (expected goals + Elo) so the panel reasons from the same evidence.
        g = {"matchup": match, "base_rate": BASE_RATES.get(archetype)}
        md = _match_data(match)
        if md:
            g.update(md)
        return g
    q = parse_question(question)
    if q is None or not q.subject:
        return None
    hi = history_implied(q.subject, archetype, minutes=70, half=(q.half == 2))
    base = BASE_RATES.get(archetype)
    if hi.get("resolved") is None:
        return {"resolved": None, "matchup": match, "base_rate": base,
                "note": "no club-history join -> lean on base rate + matchup"}
    g = {k: hi.get(k) for k in _GKEYS}
    g["matchup"] = match            # both teams; opponent inferred from the player's side
    g["base_rate"] = base           # empirical prior for this exact market type
    return g


def _rows(source: str):
    if source == "live":
        df = pd.read_csv("data/processed/odds_anchor_run.csv")
        return [(r.market_id, r.question, r.archetype, float(r.p_bot_a), r.match) for r in df.itertuples()]
    df = pd.read_parquet("data/processed/settled_with_crowd.parquet")
    return [(r.market_id, r.question, r.archetype, float(r.our_p), r.match) for r in df.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["settled", "live"], default="settled")
    ap.add_argument("--out", default=str(PREP))
    ap.add_argument("--gate-only", action="store_true",
                    help="only archetypes in expert_gate.json — the live-budget path "
                         "(no spend on shadow archetypes that can't move a submission)")
    args = ap.parse_args()

    gate = set(P.load_gate()) if args.gate_only else None
    items = []
    for mid, question, archetype, our_p, match in _rows(args.source):
        if our_p is None or (isinstance(our_p, float) and math.isnan(our_p)):
            continue
        role, action = P.route(archetype)
        if action == P.EXCLUDE:
            continue
        if gate is not None and archetype not in gate:
            continue
        items.append({
            "market_id": str(mid), "question": question, "archetype": archetype,
            "role": role, "action": action, "model_pct": our_p, "match": match,
            "grounding": _grounding(question, archetype, match),
        })

    Path(args.out).write_text(json.dumps(items, indent=2))
    by_action = Counter(i["action"] for i in items)
    grounded = sum(1 for i in items if i["grounding"] and i["grounding"].get("resolved"))
    print(f"prep: {len(items)} items ({args.source}) -> {args.out}")
    print(f"  by action: {dict(by_action)}")
    print(f"  player props with resolved history: {grounded}")


if __name__ == "__main__":
    main()
