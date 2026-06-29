"""LLM plausibility review of our hand-set / data-derived base rates.

A sanity net that runs OFF the live path: it asks a calibrated football-analytics
LLM whether each base rate is plausible and flags any that disagree by >10pp. This
is the cheap check that would have caught substitute_scores=0.695 before it shipped
(the LLM independently puts it near 0.30).

  python scripts/check_base_rates.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from jumpprob.agents.expert import client

# what each rate MEANS (per typical international men's match), so the LLM reviews the
# right quantity. Only keys listed here are reviewed.
DESC = {
    "substitute_scores": "a substitute (a player NOT in the starting XI) scores a goal",
    "sub_before_ht": "at least one substitution is made before halftime",
    "goal_1h_stoppage": "a goal is scored in first-half stoppage (added) time",
    "goal_2h_stoppage": "a goal is scored in second-half stoppage time",
    "any_player_sot_ge": "at least one player records 2 or more shots on target",
    "any_player_goals_ge": "at least one player scores 2 or more goals (a brace)",
    "cards_post_hb2": "a card is shown after the 67th minute (incl. stoppage)",
    "both_teams_cards": "both teams receive at least one card",
    "total_shots_ge": "there are ~20+ total shots (on and off target)",
    "offsides_total_ge": "there are ~3+ total offside calls",
    "goal_pre_hb1": "a goal is scored before the 22nd minute",
    "goal_post_hb2": "a goal is scored after the 67th minute",
    "corners_pre_hb1": "2+ corners are taken before the 22nd minute",
    "offsides_pre_hb1": "either team is offside before the 22nd minute",
}
TOL = 0.10


class Review(BaseModel):
    plausible: bool = Field(description="is our base rate plausible (within ~10pp)?")
    llm_estimate: float = Field(ge=0.0, le=1.0, description="your own best base rate, 0-1")
    reason: str = Field(default="", description="one sentence")


def main() -> None:
    c = client()
    if c is None:
        raise SystemExit("no ANTHROPIC_API_KEY in env/.env — cannot run the review")
    rates: dict = {}
    for f in ("data/processed/event_baselines.json", "data/processed/base_rates.json"):
        rates.update(json.loads(Path(f).read_text()))
    rates = {k: float(v) for k, v in rates.items() if k in DESC and isinstance(v, (int, float))}

    flags = []
    for k, v in sorted(rates.items()):
        prompt = (
            f"Our forecasting model uses a per-match base rate of {v:.1%} for this event: "
            f"\"{DESC[k]}\", in a typical men's international football match. Is that plausible? "
            "Give your own calibrated best estimate (0-1) and set plausible=false if ours is off "
            "by more than ~10 percentage points."
        )
        try:
            r = c.messages.parse(
                model="claude-opus-4-8", max_tokens=1500, thinking={"type": "adaptive"},
                system="You are a calibrated football-analytics expert. Base rates are per-match probabilities.",
                messages=[{"role": "user", "content": prompt}], output_format=Review)
            rev = r.parsed_output
        except Exception as e:
            print(f"  ! {k}: review failed ({str(e)[:60]})")
            continue
        off = (not rev.plausible) or abs(rev.llm_estimate - v) > TOL
        print(f"  [{'FLAG' if off else 'ok'}] {k:20} ours={v:.2f} llm={rev.llm_estimate:.2f}  {rev.reason[:74]}")
        if off:
            flags.append((k, v, rev.llm_estimate, rev.reason))

    print()
    if flags:
        print(f"=== {len(flags)} base rate(s) FLAGGED — review before shipping ===")
        for k, v, e, reason in flags:
            print(f"  {k}: ours {v:.2f} vs LLM {e:.2f} — {reason}")
    else:
        print("=== all base rates plausible (within 10pp of the LLM's estimate) ===")


if __name__ == "__main__":
    main()
