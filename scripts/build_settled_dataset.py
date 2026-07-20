"""Phase 0: the authoritative settled dataset (calibration foundation).

Joins, per settled question:
  - our submitted prediction + recovered outcome (live API results),
  - the AUTHORITATIVE archetype + family (matchday_predictions, not a keyword guess),
  - the real crowd consensus harvested from /api/probability/match-stats
    (data/processed/crowd_consensus_raw.csv, keyed by the 8-char market-id prefix,
    verified unique across the 257 settled markets).

Writes data/processed/settled_with_crowd.parquet and prints the regenerated
per-family / per-archetype edge-vs-consensus table that the quant review required
(the v1 table used a keyword classifier + proxy field and is superseded).

Reflexivity note: the leave-one-out consensus c_-i = (N*c - p_us)/(N-1) differs from
c by O(1/N); with N~2818 forecasters that is <0.0004 in probability, so we use c
directly and flag the approximation rather than guess per-question N.
"""

import numpy as np
import pandas as pd

from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.constants import COUNT_ARCHS, GOAL_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
pd.set_option("display.width", 200)


def family_of(arch: str) -> str:
    """Mirror runner._family: offsides is its own family; else goal/count/player/timing."""
    a = str(arch)
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


def build() -> pd.DataFrame:
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    res = pd.DataFrame(c.results(LOBBY)).dropna(subset=["brier_score"]).copy()
    res["our_p"] = res["probability_submitted"] / 100.0
    # recover outcome from our Brier (cross-checked below against the crowd file's `out`)
    res["out_brier"] = np.where(np.isclose((res["our_p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0,
                        np.where(np.isclose(res["our_p"] ** 2, res["brier_score"], atol=1e-6), 0.0, np.nan))

    md = pd.read_csv("data/processed/matchday_predictions.csv")[["market_id", "match", "archetype"]]
    df = res.merge(md, on="market_id", how="left")  # res already carries `question`

    crowd = pd.read_csv("data/processed/crowd_consensus_raw.csv")  # id8, crowd(%), out
    crowd["crowd_p"] = crowd["crowd"] / 100.0
    df["id8"] = df["market_id"].str[:8]
    # LEFT join: keep EVERY settled question; crowd_p is present only for the
    # manually-harvested matches (26 of 30). A prior INNER join silently dropped
    # ~40 settled questions (incl. 12 player props) just because their match's
    # crowd consensus was never harvested — they still have our_p + outcome and
    # belong in the backtest (RBP-vs-crowd is simply unavailable for them).
    df = df.merge(crowd[["id8", "crowd_p", "out"]], on="id8", how="left")

    # outcome: crowd `out` where harvested (resolves the p=0.5 ambiguity), else the
    # value recovered from our own Brier; drop the rare row where neither resolves.
    df["outcome"] = df["out"].fillna(df["out_brier"])
    df = df.dropna(subset=["outcome"]).copy()
    df["outcome"] = df["outcome"].astype(float)
    chk = df.dropna(subset=["out_brier", "out"])
    mism = int((chk["out_brier"] != chk["out"]).sum())
    print(f"outcome cross-check (crowd subset): {mism} mismatches of {len(chk)} (expect 0)")

    df["family"] = df["archetype"].map(family_of)
    df["our_brier"] = (df["our_p"] - df["outcome"]) ** 2
    df["crowd_brier"] = (df["crowd_p"] - df["outcome"]) ** 2
    df["edge"] = df["crowd_brier"] - df["our_brier"]          # >0 = we beat the consensus-of-mean
    # the fixed both-teams-SoT parser bug mis-labeled those questions as player_sot_ge and
    # priced them ~9% instead of ~70% (METHODS §13); exclude from the calibration fit so we
    # don't over-correct player props for a bug that can't recur.
    q = df["question"].fillna("").str.lower()
    df["known_bug"] = q.str.contains("both teams") & q.str.contains("shot on target")
    return df[["market_id", "match", "archetype", "family", "question", "known_bug", "our_p",
               "crowd_p", "outcome", "our_brier", "crowd_brier", "edge"]]


def main():
    df = build()
    print(f"\n=== settled_with_crowd: {len(df)} questions, {df['archetype'].nunique()} archetypes ===")
    print(f"our mean Brier {df.our_brier.mean():.4f} | consensus-of-mean {df.crowd_brier.mean():.4f} "
          f"| edge total {df.edge.sum()*100:+.1f} pts (vs consensus-of-mean)")

    print("\n=== BY FAMILY (edge vs consensus-of-mean) ===")
    fam = df.groupby("family").agg(n=("edge", "size"), ourB=("our_brier", "mean"),
            crowdB=("crowd_brier", "mean"), edge=("edge", lambda s: s.sum()*100)).sort_values("edge")
    print(fam.round(3).to_string())

    print("\n=== BY ARCHETYPE (authoritative; sorted by edge) ===")
    arch = df.groupby(["family", "archetype"]).agg(n=("edge", "size"), ourB=("our_brier", "mean"),
            crowdB=("crowd_brier", "mean"), our_p=("our_p", "mean"), real=("outcome", "mean"),
            edge=("edge", lambda s: s.sum()*100)).reset_index().sort_values("edge")
    print(arch.round(3).to_string(index=False))

    bug = df[df["known_bug"]]
    print(f"\n=== known fixed-bug cases (both-teams-SoT, excluded from fit): {len(bug)} ===")
    for r in bug.itertuples():
        print(f"  our {r.our_p:.0%} crowd {r.crowd_p:.0%} -> {int(r.outcome)}  edge {r.edge*100:+.1f}  "
              f"[{r.archetype}] {str(r.question)[:46]}")
    psot = df[df["archetype"] == "player_sot_ge"]
    psot_clean = psot[~psot["known_bug"]]
    print(f"\nplayer_sot_ge: edge {psot.edge.sum()*100:+.1f} all (n={len(psot)})  ->  "
          f"{psot_clean.edge.sum()*100:+.1f} ex-bug (n={len(psot_clean)})")

    df.to_parquet("data/processed/settled_with_crowd.parquet", index=False)
    print(f"\nwrote settled_with_crowd.parquet ({len(df)} rows; {int(df.known_bug.sum())} flagged bug)")


if __name__ == "__main__":
    main()
