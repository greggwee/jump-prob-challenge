"""Phase 1b: shrink the coverage hole with API-Football broad-league rates.

Takes the players Understat could not cover,
fetches their club season via API-Football (~1 request each — budget-safe), merges
with the Understat rates into one player_rates table, and re-runs the census to
measure the NEW hole. Writes player_rates_combined.parquet (the table the player
model joins against) and prints which players were rescued and which remain.

Requires API_FOOTBALL_KEY in .env. Run after the Understat ingest.
"""

import pandas as pd

from jumpprob.ingest.apifootball import build_hole_rates
from jumpprob.ingest.understat import build_player_rates
from jumpprob.normalize.players import build_index, resolve
from jumpprob.submit.parser import parse_question

PLAYER_ARCHS = {"player_scores", "player_sot_ge", "player_ga"}
# subjects that parse as player-props but are not real players (team/country tokens)
NON_PLAYERS = {"haiti"}


def prop_player_names() -> list[str]:
    md = pd.read_csv("data/processed/matchday_predictions.csv")
    names = set()
    for q in md["question"]:
        p = parse_question(q)
        if p and p.archetype in PLAYER_ARCHS and p.subject:
            names.add(p.subject.strip())
    return sorted(names)


def main():
    names = prop_player_names()
    print(f"{len(names)} distinct prop players")

    us = build_player_rates()
    us["source"] = "understat"
    ex, tok = build_index(us["player_name"])
    hole = [n for n in names if resolve(n, ex, tok)[0] is None
            and n.lower() not in NON_PLAYERS]
    print(f"Understat hole: {len(hole)} players -> fetching via API-Football")

    af = build_hole_rates(hole)
    print(f"API-Football returned rates for {len(af)} players")
    if not af.empty:
        cols = [c for c in ["player_name", "league", "minutes", "shots_p90", "sot_p90", "npg_p90"] if c in af]
        print(af[cols].to_string(index=False))

    combined = pd.concat([us, af], ignore_index=True)
    # one row per folded name; prefer the source with more minutes (richer signal)
    combined = (combined.sort_values("minutes", ascending=False)
                .drop_duplicates(subset="folded", keep="first").reset_index(drop=True))
    combined.to_parquet("data/processed/player_rates_combined.parquet", index=False)

    ex2, tok2 = build_index(combined["player_name"])
    resolved = [n for n in names if resolve(n, ex2, tok2)[0] is not None]
    still = [n for n in names if resolve(n, ex2, tok2)[0] is None and n.lower() not in NON_PLAYERS]
    cov = len(resolved) / max(len(names), 1)
    print(f"\n=== COVERAGE after API-Football: {len(resolved)}/{len(names)} = {cov:.0%} "
          f"| HOLE = {1 - cov:.0%} ({len(still)} still uncovered) ===")
    print("still uncovered:", ", ".join(still) if still else "(none)")
    pd.DataFrame({"player": still}).to_csv("data/processed/coverage_unresolved.csv", index=False)
    print(f"\nwrote player_rates_combined.parquet ({len(combined)} players)")


if __name__ == "__main__":
    main()
