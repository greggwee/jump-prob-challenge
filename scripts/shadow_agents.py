"""Phase 3 shadow eval (no LLM yet): does the player/team HISTORY signal beat our
live model on settled player props?

This is the gate the quant review demanded BEFORE building the LLM expert layer: if
even the raw history-implied probability (Poisson thinning of the club per-90 rate,
no opponent/lineup/context adjustment) already beats our model out-of-sample on
player props, the approach is validated and worth the expert build. If it doesn't,
the LLM layer is unlikely to either. Pure data — runnable without an Anthropic key.
"""

import numpy as np
import pandas as pd

from jumpprob.agents.grounding import history_implied
from jumpprob.submit.parser import parse_question

PLAYER_ARCHS = {"player_sot_ge", "player_ga", "player_scores"}


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main():
    df = pd.read_parquet("data/processed/settled_with_crowd.parquet")
    df = df[df["archetype"].isin(PLAYER_ARCHS) & ~df["known_bug"]].copy()
    rows = []
    for r in df.itertuples():
        q = parse_question(r.question)
        if not q or not q.subject:
            continue
        half = "second half" in str(r.question).lower() or "first half" in str(r.question).lower()
        hi = history_implied(q.subject, r.archetype, minutes=70, half=half)
        imp = hi.get("implied")
        if imp is None:
            continue  # no club join -> history can't speak; left to the model
        rows.append({"archetype": r.archetype, "player": q.subject, "resolved": hi["resolved"],
                     "our_p": r.our_p, "hist": imp, "crowd_p": r.crowd_p, "outcome": r.outcome,
                     "match": r.match, "q": r.question})
    s = pd.DataFrame(rows)
    print(f"settled player props with a club-history join: {len(s)} of {len(df)} "
          f"({len(s)/max(len(df),1):.0%} coverage)\n")

    # a simple, fixed blend the expert/model could ship: lean toward history when the
    # model is far below it (the catastrophic-low case), else keep the model.
    s["blend"] = np.where(s["hist"] > s["our_p"], 0.5 * s["our_p"] + 0.5 * s["hist"], s["our_p"])

    print("=== Brier (lower better) on settled player props ===")
    for col in ["our_p", "hist", "crowd_p", "blend"]:
        print(f"  {col:8s} {brier(s[col], s['outcome']):.4f}")
    print("\n=== by archetype: our model vs history (Brier) ===")
    g = s.groupby("archetype").apply(lambda d: pd.Series({
        "n": len(d), "our": brier(d["our_p"], d["outcome"]), "hist": brier(d["hist"], d["outcome"]),
        "blend": brier(d["blend"], d["outcome"]), "our_mean": d["our_p"].mean(),
        "hist_mean": d["hist"].mean(), "real": d["outcome"].mean()}), include_groups=False)
    print(g.round(3).to_string())

    print("\n=== the catastrophic-low cases (we bet <<history, it happened) ===")
    cat = s[(s["outcome"] == 1) & (s["our_p"] < s["hist"] - 0.15)].sort_values("our_p")
    for r in cat.itertuples():
        print(f"  us {r.our_p:.0%} hist {r.hist:.0%} crowd {r.crowd_p:.0%} -> YES  "
              f"[{r.resolved}] {str(r.q)[:42]}")


if __name__ == "__main__":
    main()
