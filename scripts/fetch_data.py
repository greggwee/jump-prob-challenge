"""Fetch all Phase 1 raw data. Idempotent; pass --force to re-download."""

import sys

from jumpprob.ingest import results, statsbomb

force = "--force" in sys.argv

print("fetching martj42 results/goalscorers/shootouts...")
results.fetch(force=force)
print("fetching StatsBomb match metadata for 6 tournaments...")
statsbomb.fetch_matches(force=force)
print("done.")
