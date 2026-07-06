"""Forward shadow-scorer for the two open quant-review hypotheses.

Validates on FRESH settled data with FIXED, pre-registered parameters (no tuning on
the test set -> no overfitting), via an append-only ledger that grows as matches
settle. Run it after each settlement batch (wired into daily_learn); the verdict
only firms up once enough fresh knockout questions accumulate.

  H1  de-extremize: is Bot A over-confident on the tails (calibration slope ~0.78)?
      Score sigmoid(g * logit(our_p)) for g < 1 vs the live submission (g = 1).
  H2  A>B over-prediction: do the comparison (_gt) markets over-predict (independence
      ignores tie mass + positive count correlation)? -> the _gt subgroup of H1, plus
      a calibration read (mean predicted vs realized).

PRE-REGISTERED de-extremize factor: g = 0.78 (the measured calibration slope). The
grid 0.7-1.0 is shown for context, but the verdict is read off g = 0.78 so we are not
cherry-picking the best-fitting g on the test set.

  python scripts/shadow_decalib.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.parser import parse_question
from jumpprob.submit.runner import COUNT_ARCHS, GOAL_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
LEDGER = Path("data/processed/shadow_decalib_ledger.csv")
GT = {"sot_gt", "corners_gt", "cards_gt", "fouls_gt"}      # the A>B comparison markets
GAMMA_REG = 0.78                                            # pre-registered de-extremize factor
GRID = [1.0, 0.9, 0.85, 0.8, 0.78, 0.7]


def _family(a: str) -> str:
    a = str(a)
    if "offside" in a:
        return "offsides"
    if a in TIMING_ARCHS:
        return "timing"
    if a in PLAYER_ARCHS:
        return "player"
    if a in COUNT_ARCHS:
        return "count"
    if a in GOAL_ARCHS:
        return "goal"
    return "other"


def _logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def update_ledger() -> pd.DataFrame:
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    res = pd.DataFrame(c.results(LOBBY)).dropna(subset=["brier_score"]).copy()
    res["our_p"] = res["probability_submitted"] / 100.0
    res["outcome"] = np.where(np.isclose((res["our_p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0,
                       np.where(np.isclose(res["our_p"] ** 2, res["brier_score"], atol=1e-6), 0.0, np.nan))
    res = res.dropna(subset=["outcome"])
    res["archetype"] = res["question"].map(lambda q: (parse_question(q).archetype if parse_question(q) else "unknown"))
    res["family"] = res["archetype"].map(_family)
    res = res[["market_id", "archetype", "family", "our_p", "outcome", "created_date"]]

    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        new = res[~res["market_id"].isin(set(old["market_id"]))]
        ledger = pd.concat([old, new], ignore_index=True)
        print(f"ledger: {len(old)} existing + {len(new)} new settlements -> {len(ledger)}")
    else:
        ledger = res
        print(f"ledger: seeded with {len(ledger)} settlements")
    ledger.to_csv(LEDGER, index=False)
    return ledger


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score(ledger: pd.DataFrame) -> None:
    y = ledger["outcome"].to_numpy()
    x = _logit(ledger["our_p"].to_numpy())
    live = _brier(_sig(x), y)
    print(f"\n=== H1 de-extremize (n={len(ledger)}; live Brier {live:.4f}) ===")
    print(f"  {'gamma':>6} {'Brier':>8} {'vs live':>9}")
    for g in GRID:
        b = _brier(_sig(g * x), y)
        tag = "  <- pre-registered" if g == GAMMA_REG else ("  (live)" if g == 1.0 else "")
        print(f"  {g:>6} {b:>8.4f} {b - live:>+9.4f}{tag}")
    print("  verdict: read gamma=0.78; <0 vs live = de-extremizing helps. Need ~hundreds of")
    print("  fresh settlements to resolve a 0.005 edge -- treat small-n as directional only.")

    # per family at the pre-registered gamma
    print(f"\n  by family @ gamma={GAMMA_REG} (negative = de-extremize helps that family):")
    for fam, g_ in ledger.groupby("family"):
        if len(g_) < 8:
            continue
        xf = _logit(g_["our_p"].to_numpy()); yf = g_["outcome"].to_numpy()
        d = _brier(_sig(GAMMA_REG * xf), yf) - _brier(_sig(xf), yf)
        print(f"    {fam:9} n={len(g_):4d}  dBrier {d:+.4f}")

    # H2: comparison markets
    gt = ledger[ledger["archetype"].isin(GT)]
    print(f"\n=== H2 A>B comparison markets (_gt; n={len(gt)}) ===")
    if len(gt):
        xg = _logit(gt["our_p"].to_numpy()); yg = gt["outcome"].to_numpy()
        d = _brier(_sig(GAMMA_REG * xg), yg) - _brier(_sig(xg), yg)
        print(f"  mean predicted {gt['our_p'].mean():.3f}  vs realized {yg.mean():.3f}  "
              f"(over-predict by {gt['our_p'].mean() - yg.mean():+.3f})")
        print(f"  de-extremize @0.78 dBrier {d:+.4f}  -> if predicted>realized AND dBrier<0,")
        print("  the independence/tie-mass fix is warranted; ship to shadow before live.")
    else:
        print("  none settled yet.")


def main() -> None:
    score(update_ledger())


if __name__ == "__main__":
    main()
