"""Verification item 5: per-archetype Brier for full v2 vs each ablation."""

import pandas as pd

configs = {"full": "backtest_v2.csv", "no_coupling": "backtest_v2_nc.csv",
           "no_pp": "backtest_v2_npp.csv", "no_frailty": "backtest_v2_nf.csv"}
frames = {}
for name, f in configs.items():
    df = pd.read_csv(f"data/processed/{f}")
    df["sq"] = (df.p_v2 - df.outcome) ** 2
    frames[name] = df.groupby("archetype")["sq"].mean()

cmp = pd.DataFrame(frames)
for c in ("no_coupling", "no_pp", "no_frailty"):
    cmp[f"d_{c}"] = (cmp[c] - cmp["full"]).round(4)  # >0 means component HELPS
print(cmp.round(4).to_string())
print("\noverall:")
print({k: round(float(v.mean()), 4) for k, v in frames.items()})
