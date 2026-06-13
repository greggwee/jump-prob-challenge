"""Fit per-family Platt calibration for the LIVE submission path (rec #1).

Calibration is a model-family property (count models share an overconfidence
profile, players another), and per-archetype samples are thin — so we POOL by
family and fit one Platt (a, b) each:
  goal   <- backtest_phase2_hl730.csv  (applied only to UNANCHORED goal markets)
  count  <- backtest_counts.csv (src==model)
  player <- backtest_players_raw.csv

a < 1 means the family was overconfident (Platt shrinks it toward 0.5).
Writes data/processed/live_calibration.json = {family: [a, b]}.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.calibration import fit_platt
from jumpprob.models.market_anchor import inv_logit, logit


def _intercept_only(p, y):
    """Slope fixed at 1 — for families whose probs barely vary (slope unidentifiable)."""
    from scipy.optimize import minimize_scalar
    x = np.array([logit(v) for v in p])
    b = minimize_scalar(lambda b: float(np.sum(np.logaddexp(0, x + b)) - np.sum(y * (x + b))),
                        bounds=(-2, 2), method="bounded").x
    return 1.0, float(b)


def fit(p, y, name):
    a, b = fit_platt(np.asarray(p), np.asarray(y))
    if a < 0.2:  # narrow support -> slope unidentifiable
        a, b = _intercept_only(np.asarray(p), np.asarray(y))
    raw = float(np.mean((np.asarray(p) - y) ** 2))
    cal = np.array([inv_logit(a * logit(v) + b) for v in p])
    cal_b = float(np.mean((cal - y) ** 2))
    print(f"  {name:7} n={len(p):5d}  a={a:.3f} b={b:+.3f}  Brier {raw:.4f} -> {cal_b:.4f} "
          f"({'overconfident' if a < 0.95 else 'ok'})")
    return [round(a, 4), round(b, 4)]


table = {}

g = pd.read_csv("data/processed/backtest_phase2_hl730.csv")
table["goal"] = fit(g["p_model"], g["outcome"].to_numpy(), "goal")

c = pd.read_csv("data/processed/backtest_counts.csv")
c = c[c["src"] == "model"]
table["count"] = fit(c["p"], c["outcome"].to_numpy(), "count")

praw = Path("data/processed/backtest_players_raw.csv")
if praw.exists():
    p = pd.read_csv(praw)
    table["player"] = fit(p["p_model"], p["outcome"].to_numpy(), "player")

Path("data/processed/live_calibration.json").write_text(json.dumps(table, indent=1))
print("\nwrote data/processed/live_calibration.json:", table)
