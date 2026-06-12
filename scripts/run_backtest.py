"""Run the Phase 2 backtest across decay half-lives; save results + print summary."""

from pathlib import Path

import pandas as pd

from jumpprob.backtest.harness import run_backtest, summarize

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

for hl in (365.0, 730.0, 1460.0):
    bt = run_backtest(half_life_days=hl)
    bt.to_csv(OUT / f"backtest_phase2_hl{int(hl)}.csv", index=False)
    s = summarize(bt)
    overall_model = ((bt["p_model"] - bt["outcome"]) ** 2).mean()
    overall_base = ((bt["p_base"] - bt["outcome"]) ** 2).mean()
    print(f"\n=== half-life {int(hl)}d | n={len(bt)} settlements | "
          f"Brier model={overall_model:.4f} base={overall_base:.4f} edge={overall_base - overall_model:+.4f} ===")
    print(s.round(4).to_string(index=False))
