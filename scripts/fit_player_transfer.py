"""Debug + fit the club->international transfer discount tau for the player model.

Wires the combined club rates in as a player-specific prior (runner.player_rate) and
checks, on the settled player props, that it (a) lifts our systematically-low attacker
predictions toward the realized rate and (b) does not regress player_scores (where we
already have edge). Sweeps tau; scale fixed at the league average (1.0) to isolate the
RATE change (the live model applies the real match scale on top, common to old & new).
"""

import numpy as np
import pandas as pd

from jumpprob.pipeline import Pipeline
from jumpprob.submit.parser import parse_question
from jumpprob.submit.runner import H1, H2, MINUTE_SCENARIOS, Runner

ARCHS = ["player_sot_ge", "player_ga", "player_scores"]


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def main():
    print("loading models...")
    runner = Runner(Pipeline(), pd.read_parquet("data/processed/player_match.parquet"))
    sw = pd.read_parquet("data/processed/settled_with_crowd.parquet")
    sw = sw[sw["archetype"].isin(ARCHS) & ~sw["known_bug"]].copy()

    def predict(subj, arch, half, tau, use_club):
        runner.player_transfer = tau
        saved = runner.club_idx
        if not use_club:
            runner.club_idx = None
        rs, rg, rga = runner.player_rate(subj)
        runner.club_idx = saved
        rate = {"player_sot_ge": rs, "player_scores": rg, "player_ga": rga}[arch]
        share = H2 if half == 2 else H1 if half == 1 else 1.0
        return float(sum(w * (1 - np.exp(-rate * share * m)) for w, m in MINUTE_SCENARIOS))

    rows = []
    for r in sw.itertuples():
        q = parse_question(r.question)
        if not q or not q.subject:
            continue
        rows.append({"arch": r.archetype, "subj": q.subject, "half": q.half,
                     "out": r.outcome, "our_p": r.our_p})
    d = pd.DataFrame(rows)
    print(f"settled player props: {len(d)} | realized rate {d['out'].mean():.0%}\n")

    d["old"] = [predict(x.subj, x.arch, x.half, 1.0, False) for x in d.itertuples()]
    print(f"submitted (live our_p) Brier {brier(d['our_p'], d['out']):.4f}")
    print(f"OLD reconstructed (pop prior, scale=1) Brier {brier(d['old'], d['out']):.4f} "
          f"| mean {d['old'].mean():.3f}\n")
    print("tau    NEW Brier   mean    | by archetype (Brier)")
    for tau in [0.55, 0.65, 0.72, 0.80, 0.90, 1.0]:
        d["new"] = [predict(x.subj, x.arch, x.half, tau, True) for x in d.itertuples()]
        by = d.groupby("arch").apply(lambda g: brier(g["new"], g["out"]), include_groups=False)
        tag = "  ".join(f"{a.split('_')[1]}:{by[a]:.3f}" for a in ARCHS if a in by)
        print(f"{tau:.2f}   {brier(d['new'], d['out']):.4f}    {d['new'].mean():.3f}   | {tag}")


if __name__ == "__main__":
    main()
