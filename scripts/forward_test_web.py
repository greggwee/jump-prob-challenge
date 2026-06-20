"""Phase L2 Tier-3 — forward-test the web-search blend once matches settle.

Joins the near-kickoff web ledger (data/processed/ai_panel_web_ledger.csv) to the
settled, crowd-graded dataset and asks, per web-panelled player prop: did blending
the model toward the WEB ensemble (at the live gate weight) beat the pure model,
and how close to the crowd did it land? Web search can't be tested offline (settled
matches leak the result), so this is the only honest read — run it after settlements
accumulate (rebuild settled_with_crowd.parquet first).

  .venv/Scripts/python.exe scripts/forward_test_web.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jumpprob.agents import panel as P

LEDGER = "data/processed/ai_panel_web_ledger.csv"
SETTLED = "data/processed/settled_with_crowd.parquet"


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def main() -> None:
    try:
        led = pd.read_csv(LEDGER)
    except FileNotFoundError:
        print(f"no web ledger yet ({LEDGER}) — run scripts/ai_panel_web.py near a kickoff first.")
        return
    led = led.dropna(subset=["web_prob"]).drop_duplicates("market_id", keep="last")
    led["market_id"] = led["market_id"].astype(str)

    s = pd.read_parquet(SETTLED)
    s["market_id"] = s["market_id"].astype(str)
    df = led.merge(s[["market_id", "outcome", "our_p", "crowd_brier"]], on="market_id", how="inner")
    print(f"web-panelled props logged: {len(led)} | settled so far: {len(df)}")
    if not len(df):
        print("none settled yet — re-run after these matches finish (rebuild settled_with_crowd.parquet).")
        return

    gate = P.load_gate()
    rows = []
    for r in df.itertuples():
        g = gate.get(r.archetype, {})
        w = float(g.get("w", 0.0))
        cap = P.DEFAULT_CAP if P.route(r.archetype)[1] == P.BLEND_CAPPED else None
        blended = P.blend_logodds(r.our_p, r.web_prob, w, cap)
        rows.append({"archetype": r.archetype, "y": r.outcome, "model": r.our_p,
                     "web": r.web_prob, "blend": blended, "crowd_brier": r.crowd_brier})
    d = pd.DataFrame(rows)
    print(f"\noverall  n={len(d)}  base={d.y.mean():.2f}")
    print(f"  Brier  model={_brier(d.model, d.y):.4f}  web={_brier(d.web, d.y):.4f}  "
          f"blend={_brier(d.blend, d.y):.4f}  crowd={d.crowd_brier.mean():.4f}")
    print("\nper-archetype (blend vs model):")
    for a, gg in d.groupby("archetype"):
        bm, bb = _brier(gg.model, gg.y), _brier(gg.blend, gg.y)
        print(f"  {a:14s} n={len(gg):>2}  model={bm:.4f} -> blend={bb:.4f}  ({bm-bb:+.4f})")


if __name__ == "__main__":
    main()
