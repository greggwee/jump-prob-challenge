"""Verify the offsides over-pricing (#2): is it structural, and is the pooled
count calibration making it worse (it shifts counts UP; offsides needs DOWN)?"""

import numpy as np
import pandas as pd

from jumpprob.models.calibration import fit_platt
from jumpprob.models.market_anchor import inv_logit, logit
from jumpprob.submit.api_client import SportsPredictClient

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"

# 1. backtest offsides: is the model calibrated there? (large sample)
c = pd.read_csv("data/processed/backtest_counts.csv")
off = c[(c["src"] == "model") & (c["archetype"] == "offsides_a_ge_2")]
print(f"BACKTEST offsides (n={len(off)}): mean pred {off.p.mean():.3f}, "
      f"realized {off.outcome.mean():.3f}, Brier {((off.p-off.outcome)**2).mean():.4f}")
a, b = fit_platt(off["p"].to_numpy(), off["outcome"].to_numpy())
cal = np.array([inv_logit(a * logit(p) + b) for p in off["p"]])
print(f"  offsides-specific Platt: a={a:.3f} b={b:+.3f}  "
      f"-> Brier {((cal-off.outcome)**2).mean():.4f}")
pooled = pd.read_csv("data/processed/backtest_counts.csv")
pooled = pooled[pooled["src"] == "model"]
pa, pb = fit_platt(pooled["p"].to_numpy(), pooled["outcome"].to_numpy())
print(f"  POOLED count Platt (what we currently apply): a={pa:.3f} b={pb:+.3f} "
      f"(b>0 shifts offsides UP — wrong direction)")

# 2. live settled offsides
res = pd.DataFrame(SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True).results(LOBBY))
res = res.dropna(subset=["brier_score"])
res["p"] = res.probability_submitted / 100
res["outcome"] = np.where(np.isclose((res.p - 1) ** 2, res.brier_score, atol=1e-6), 1.0, 0.0)
md = pd.read_csv("data/processed/matchday_predictions.csv")[["market_id", "archetype"]]
res = res.merge(md, on="market_id", how="left")
lo = res[res.archetype == "offsides_ge"]
print(f"\nLIVE settled offsides (n={len(lo)}): mean pred {lo.p.mean():.3f}, "
      f"realized {lo.outcome.mean():.3f}")
print("  (small live n — backtest is the robust signal for the fix)")
