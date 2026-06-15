"""Phase 1 deliverable: the COVERAGE CENSUS.

What fraction of the players the contest actually asks us about have a club-season
join? That hole rate is the load-bearing number (review H1): it gates the
re-backtest and whether player extremizing can ever be re-enabled. Coverage-hole
players fall back to the data-starved prior (the Kubo failure mode).

The prediction population = the player names appearing in player-prop markets, not
abstract 23-man squads — exactly the players whose props we must price.
"""

import pandas as pd

from jumpprob.ingest.understat import build_player_rates
from jumpprob.normalize.players import build_index, resolve
from jumpprob.submit.parser import parse_question

PLAYER_ARCHS = {"player_scores", "player_sot_ge", "player_ga"}

md = pd.read_csv("data/processed/matchday_predictions.csv")
names = set()
for q in md["question"]:
    p = parse_question(q)
    if p and p.archetype in PLAYER_ARCHS and p.subject:
        names.add(p.subject.strip())
print(f"distinct players in prop markets: {len(names)}")

print("building Understat club rates (top-5 + RFPL x 2 seasons)...")
rates = build_player_rates()
print(f"  {len(rates)} club players cached")
exact, tok = build_index(rates["player_name"])

resolved, unresolved, fuzzy = [], [], []
for nm in sorted(names):
    match, method = resolve(nm, exact, tok)
    if match is None:
        unresolved.append(nm)
    else:
        resolved.append(nm)
        if method != "exact":
            fuzzy.append((nm, match))

# separate genuine league-holes from API name corruption (the U+FFFD replacement
# char the API serves for some accented names — unrecoverable at the source)
corrupted = [nm for nm in unresolved if "�" in nm]
genuine = [nm for nm in unresolved if "�" not in nm]

cov = len(resolved) / max(len(names), 1)
print(f"\n=== COVERAGE: {len(resolved)}/{len(names)} = {cov:.0%} resolved | "
      f"HOLE = {1 - cov:.0%} ({len(unresolved)} on the data-starved prior) ===")
print(f"    of the hole: {len(genuine)} genuine league-gaps, "
      f"{len(corrupted)} API-corrupted names (note: most are also uncovered-league)")
if fuzzy:
    print(f"\n{len(fuzzy)} token-set (non-exact) joins — spot-check for mis-resolution (L3):")
    for nm, mt in fuzzy[:15]:
        print(f"  '{nm}'  ->  '{mt}'")
print(f"\nunresolved players (which leagues are we missing?):")
for nm in unresolved[:50]:
    print(f"  {nm}")

pd.DataFrame({"player": sorted(unresolved)}).to_csv("data/processed/coverage_unresolved.csv", index=False)
rates.to_parquet("data/processed/understat_rates.parquet", index=False)
print(f"\nwrote understat_rates.parquet ({len(rates)} players) + coverage_unresolved.csv")
