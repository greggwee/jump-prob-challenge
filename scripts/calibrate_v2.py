"""Verification item 4: reliability table of v2 backtest probabilities and a
fresh Platt calibration per archetype fit on v2 outputs (NOT v1's table).

Writes data/processed/v2_calibration.json {archetype: [a, b]} and prints the
reliability table (predicted-bucket mean vs realized frequency).
"""

import json

import numpy as np
import pandas as pd

from jumpprob.models.calibration import fit_platt, MIN_SETTLEMENTS

bt = pd.read_csv("data/processed/backtest_v2_go.csv")  # production config

print("reliability (all archetypes pooled):")
bins = np.linspace(0, 1, 11)
bt["bucket"] = pd.cut(bt.p_v2, bins, include_lowest=True)
rel = bt.groupby("bucket", observed=True).agg(
    n=("outcome", "size"), p_mean=("p_v2", "mean"), realized=("outcome", "mean"))
rel["gap"] = rel.realized - rel.p_mean
print(rel.round(3).to_string())

def _intercept_only(p, y):
    """Platt with slope fixed at 1 — for archetypes whose predictions barely
    vary, the slope is unidentifiable and free fits go wild (negative a)."""
    from scipy.optimize import minimize_scalar
    from jumpprob.models.market_anchor import logit
    x = np.array([logit(v) for v in p])

    def nll(b):
        z = x + b
        return float(np.sum(np.logaddexp(0, z)) - np.sum(y * z))

    return 1.0, float(minimize_scalar(nll, bounds=(-2, 2), method="bounded").x)


table = {}
for arch, g in bt.groupby("archetype"):
    if len(g) < MIN_SETTLEMENTS:
        table[arch] = (1.0, 0.0)
        continue
    a, b = fit_platt(g.p_v2.to_numpy(), g.outcome.to_numpy())
    if a < 0.2:  # slope unidentifiable on narrow support -> intercept-only
        a, b = _intercept_only(g.p_v2.to_numpy(), g.outcome.to_numpy())
    table[arch] = (round(a, 4), round(b, 4))

with open("data/processed/v2_calibration.json", "w") as f:
    json.dump(table, f, indent=1)

print("\nPlatt per archetype (a < 1 = was overconfident):")
for k, v in sorted(table.items()):
    print(f"  {k}: a={v[0]:.3f} b={v[1]:+.3f}")

# effect of calibration on Brier
from jumpprob.models.calibration import apply_calibration
bt["p_cal"] = [apply_calibration(p, a, {k: tuple(v) for k, v in table.items()})
               for p, a in zip(bt.p_v2, bt.archetype)]
print(f"\nBrier raw {((bt.p_v2 - bt.outcome) ** 2).mean():.4f} -> "
      f"calibrated (in-sample) {((bt.p_cal - bt.outcome) ** 2).mean():.4f}")
