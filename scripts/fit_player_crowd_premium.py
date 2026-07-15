"""Fit the PLAYER-PROP crowd premium (P4b, 2026-07-06).

Discovery: the Jump platform crowd we are SCORED AGAINST prices player props
systematically ABOVE the book we anchor to (crowd Brier .223 vs book .258 over 29 settled
book-anchored props; +12 pts on SoT, +6 scorer/GA), and it is BETTER calibrated — the crowd
refuses to price a name attacker as low as a global book does (root cause: a book LOW-SIDE
miscalibration, book prices in [0,.34) realize .44). Anchoring 0.97 to the book therefore
leaves us BELOW our own benchmark on player props — MEX vs ENG cost -111 RBP on five
player/SoT questions the crowd had higher. Quant audit 2026-07-06 confirmed the edge is real
(survives dropping any match; clean outcome reconstruction) but the shift MAGNITUDE is not
data-identified (LOO curve monotone, raw edge p~.065) — so the runner under-shifts on purpose.

We cannot see the crowd pre-lock, but the per-archetype premium (crowd - book) is stable
enough to predict. The live runner lifts book-anchored player props by PLAYER_CROWD_LIFT
(0.5 — a deliberate conservative fraction, NOT a fitted optimum) times this premium, capped.
Refit here as crowd data accumulates (wired into daily_learn).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jumpprob.submit.api_client import SportsPredictClient  # noqa: E402

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
OUT = ROOT / "data" / "processed" / "player_crowd_premium.json"
PLAYER_ARCHS = {"player_scores", "player_sot_ge", "player_ga"}


def main() -> None:
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    r = pd.DataFrame(c.results(LOBBY))
    r = r[r.brier_score.notna()].copy()
    r["id8"] = r.market_id.str[:8]
    led = pd.read_csv(ROOT / "data" / "processed" / "clv_ledger.csv")
    led = led.sort_values("ts").groupby("market_id").last().reset_index()
    r = r.merge(led[["market_id", "archetype", "sub_pure", "sub_market"]], on="market_id", how="left")
    cr = pd.read_csv(ROOT / "data" / "processed" / "crowd_consensus_raw.csv")
    cr["id8"] = cr.id8.astype(str)
    j = r.merge(cr[["id8", "crowd"]], on="id8", how="inner").dropna(subset=["crowd", "sub_market"])
    # Fit on book-anchored player props only (|book - pure_model| >= 2: a real book line moved the
    # value). NOTE (quant audit 2026-07-06) this is a CONSERVATISM device, not an edge-maximiser:
    # the crowd premium is actually LARGER on the unfiltered/model-agreed props (+.156 over all 53
    # player props vs +.089 here), and shrinks as the filter tightens — so fitting here deliberately
    # UNDER-estimates. Paired with the damped runner shift, the live lift is conservative twice over.
    pp = j[j.archetype.isin(PLAYER_ARCHS) & ((j.sub_market - j.sub_pure).abs() >= 2)].copy()
    pp["prem"] = (pp.crowd - pp.sub_market) / 100.0

    prem = {}
    for a in PLAYER_ARCHS:
        s = pp[pp.archetype == a]
        # shrink a thin per-archetype estimate toward the pooled premium (Beta-ish: the
        # pooled mean carries weight of ~5 pseudo-obs) so one archetype's small n can't swing it
        pooled = float(pp.prem.mean()) if len(pp) else 0.0
        n = len(s)
        est = (n * float(s.prem.mean()) + 5 * pooled) / (n + 5) if n else pooled
        prem[a] = round(max(0.0, est), 3)
    OUT.write_text(json.dumps({"premium": prem, "n": int(len(pp)),
                               "pooled": round(float(pp.prem.mean()), 3) if len(pp) else 0.0}, indent=1))
    print(f"player_crowd_premium: n={len(pp)} book-anchored props | per-archetype (prob): {prem} -> {OUT}")


if __name__ == "__main__":
    main()
