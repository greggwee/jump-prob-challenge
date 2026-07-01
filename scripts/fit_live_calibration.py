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

# PLAYER: identity. The backtest Platt fit ([0.96, -0.26]) does NOT transfer to the
# LIVE settled distribution -- 5-fold CV on the live settled player props: the fit
# scores 0.2092 OOS vs 0.2058 for neutral (the -0.26 intercept pushed already-low,
# now BOOK-ANCHORED props further DOWN, compounding under-prediction). Player props
# are calibrated by the de-vigged book anchor now; shipping a non-transferring Platt
# shift just adds harm. Re-enable only if a fit beats neutral on LIVE settled data.
table["player"] = [1.0, 0.0]

# OFFSIDES (#2): a StatsBomb->SportsPredict definition gap over-prices offsides
# LIVE (backtest is calibrated; live n=8 had pred 0.50 vs realized 0.125). Fit a
# regularized intercept-only shift from LIVE settled offsides (slope=1; one
# parameter, heavily shrunk toward 0 so a tiny sample can't over-correct), and
# keep offsides OUT of the pooled count Platt (which shifts the wrong way).
def fit_offsides_live(tau: float = 0.8):
    from jumpprob.submit.api_client import SportsPredictClient
    LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
    try:
        res = pd.DataFrame(SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
                           .results(LOBBY)).dropna(subset=["brier_score"])
    except Exception as e:
        print(f"  offsides: live fetch failed ({e}); identity"); return [1.0, 0.0]
    res["p"] = res["probability_submitted"] / 100.0
    res["outcome"] = np.where(np.isclose((res["p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0, 0.0)
    md = pd.read_csv("data/processed/matchday_predictions.csv")[["market_id", "archetype"]]
    res = res.merge(md, on="market_id", how="left")
    o = res[res["archetype"] == "offsides_ge"]
    if len(o) < 5:
        print(f"  offsides: only {len(o)} live settled; identity"); return [1.0, 0.0]
    from scipy.optimize import minimize_scalar
    x = np.array([logit(v) for v in o["p"].clip(0.01, 0.99)])
    y = o["outcome"].to_numpy()
    # intercept-only logistic MLE + L2(b)/tau^2 shrinkage toward 0
    b = minimize_scalar(lambda b: float(np.sum(np.logaddexp(0, x + b)) - np.sum(y * (x + b))
                                        + 0.5 * b * b / tau ** 2), bounds=(-3, 3), method="bounded").x
    print(f"  offsides n={len(o)} live: pred {o['p'].mean():.3f} realized {y.mean():.3f} "
          f"-> shrunk intercept b={b:+.3f} (down-shift)")
    return [1.0, round(float(b), 4)]

table["offsides"] = fit_offsides_live()

Path("data/processed/live_calibration.json").write_text(json.dumps(table, indent=1))
print("\nwrote data/processed/live_calibration.json:", table)
