"""Score v2 shadow predictions against v1 on SETTLED live markets.

Pulls settled results from the API (Brier per question is returned for our
submitted v1 predictions; outcomes are recovered from our own Brier:
outcome = 1 if (p_v1 - brier_implied...) — no: outcome is recoverable exactly
since brier = (p - y)^2 with y in {0,1}: y = 1 iff brier < (p - 0)^2 ... i.e.
y = 1 if |sqrt(brier) - p| vs ... simpler: y = round(p - signed_sqrt) — we
compute y = 1 when (p - 1)^2 == brier (within fp tolerance) else 0.

Prints cumulative Brier v1 vs v2 on simulated markets + the promotion tracker
(threshold per V2-FINDINGS: 150+ settlements).
"""

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

shadow = pd.read_csv("data/processed/shadow_v2.csv")
client = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
results = client.results(LOBBY)
settled = pd.DataFrame([r for r in results if r.get("brier_score") is not None])
if settled.empty:
    print("no settled markets yet")
    raise SystemExit

settled["p"] = settled["probability"] / 100.0
# recover binary outcome from our own brier: y=1 iff (p-1)^2 == brier.
# At p=0.50 both outcomes give brier 0.25 — unrecoverable, drop (don't guess).
ambiguous = np.isclose(settled.p, 0.5)
if ambiguous.any():
    print(f"dropping {int(ambiguous.sum())} p=0.50 settlements (outcome unrecoverable from brier)")
settled = settled[~ambiguous].copy()
settled["outcome"] = (np.isclose((settled.p - 1) ** 2, settled.brier_score, atol=1e-6)).astype(float)

m = shadow.merge(settled[["market_id", "outcome"]], on="market_id")
sim = m[m.v2_simulated].copy()
sim["sq1"] = (sim.p_v1 - sim.outcome) ** 2
sim["sq2"] = (sim.p_v2 - sim.outcome) ** 2
n = len(sim)
print(f"settled simulated markets: {n}")
print(f"v1 Brier {sim.sq1.mean():.4f} | v2 Brier {sim.sq2.mean():.4f} | "
      f"v2 edge {sim.sq1.mean() - sim.sq2.mean():+.4f}")
print(f"promotion tracker: {n}/150 settlements"
      + (" — THRESHOLD REACHED, evaluate promotion" if n >= 150 else ""))
print("\nby archetype:")
print(sim.groupby("archetype")[["sq1", "sq2"]].mean().round(4).to_string())
