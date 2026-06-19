"""Statistical review of settled Bot A predictions through the RELATIVE-scoring lens.

The contest scores points = field_Brier - our_Brier per question. The API exposes
no field data (audit_field_api.py: every leaderboard/stats endpoint 500s), so we
estimate the field with the SAME proxy the live extremizer uses (strategy.field_model:
our belief shrunk toward 50 by a per-family sharpness lambda). This lets us ask the
questions that actually matter for rank:

  1. ABSOLUTE skill  - per archetype, is our Brier below the base-rate / coin-flip
     baseline? (Are the models any good at all?)
  2. RELATIVE points - proxy RBP = field_Brier - our_Brier, by family/archetype.
     Where do we EARN points (beat the crowd) vs BLEED them?
  3. The ASYMMETRY   - per-question RBP is bounded above by field_Brier but reaches
     -(0.98 - field_Brier) below: confident-and-wrong is catastrophic, confident-
     and-right is only modestly rewarded. We quantify the skew and show where our
     risk-taking (deviation from the field) paid off vs backfired.

RBP is reported in Brier units; the contest multiplies by 100 for display points.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models import strategy
from jumpprob.submit.api_client import SportsPredictClient
from jumpprob.submit.runner import COUNT_ARCHS, GOAL_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
pd.set_option("display.width", 220)


def family_of(arch: str) -> str:
    if arch in TIMING_ARCHS:
        return "timing"
    if arch in GOAL_ARCHS or str(arch).startswith(("win", "draw", "btts", "total", "half_total", "ht_", "h2_")):
        return "goal"
    if arch in PLAYER_ARCHS or str(arch).startswith("player"):
        return "player"
    if arch in COUNT_ARCHS or any(k in str(arch) for k in ("corner", "sot", "foul", "card", "offside")):
        return "count"
    return "other"


def load_settled() -> pd.DataFrame:
    c = SportsPredictClient("SPORTSPREDICT_KEY_A", dry_run=True)
    res = pd.DataFrame(c.results(LOBBY)).dropna(subset=["brier_score"]).copy()
    Path("data/processed/api_snapshot.json").write_text(json.dumps(
        {"fetched_at": datetime.now(timezone.utc).isoformat(),
         "predictions": c.my_predictions(LOBBY), "results": c.results(LOBBY)}))
    res["p"] = res["probability_submitted"] / 100.0
    # recover outcome from our Brier: brier == (p-1)^2 -> YES, (p-0)^2 -> NO
    res["outcome"] = np.where(np.isclose((res["p"] - 1) ** 2, res["brier_score"], atol=1e-6), 1.0, 0.0)
    res.loc[np.isclose(res["p"], 0.5), "outcome"] = np.nan  # 0.5 is sign-ambiguous
    md = pd.read_csv("data/processed/matchday_predictions.csv")[["market_id", "match", "archetype"]]
    res = res.merge(md, on="market_id", how="left")
    res["family"] = res["archetype"].map(family_of).fillna("other")
    return res.dropna(subset=["outcome"]).copy()


def add_field(res: pd.DataFrame) -> pd.DataFrame:
    """Field proxy + RBP. field = belief shrunk toward 50 by family sharpness lambda
    (the deployed strategy.field_model). RBP_field uses that; RBP_base uses the
    archetype base rate as the field (assumption-light, for no-line markets)."""
    lam = strategy.load_field_sharpness()
    base = strategy.load_base_rates()
    res = res.copy()
    res["lam"] = res["family"].map(lambda f: lam.get(f, 0.5))
    res["field_p"] = [strategy.field_model(p, l) for p, l in zip(res["p"], res["lam"])]
    res["field_brier"] = (res["field_p"] - res["outcome"]) ** 2
    res["rbp"] = res["field_brier"] - res["brier_score"]          # proxy relative points
    res["base_p"] = res["archetype"].map(lambda a: base.get(a, np.nan))
    res["base_brier"] = (res["base_p"] - res["outcome"]) ** 2
    res["rbp_base"] = res["base_brier"] - res["brier_score"]      # vs naive base-rate crowd
    res["conf"] = (res["p"] - 0.5).abs()                          # how far we leaned from 50
    return res


def main():
    res = add_field(load_settled())
    n = len(res)
    print(f"=== Bot A settled review: {n} signed questions, {res['match'].nunique()} matches ===")
    print(f"mean Brier {res.brier_score.mean():.4f}  (coin-flip 0.25)  | YES-rate {res.outcome.mean():.0%}")
    print(f"proxy RBP total {res.rbp.sum()*100:+.1f} pts  ({res.rbp.mean()*100:+.3f}/q)  "
          f"[field = belief shrunk by family sharpness]\n")

    print("=== 1. BY FAMILY: skill (Brier) and proxy relative points ===")
    fam = res.groupby("family").agg(
        n=("brier_score", "size"), brier=("brier_score", "mean"),
        win_rate=("outcome", lambda s: ((res.loc[s.index, "p"] > 0.5) == (s > 0.5)).mean()),
        rbp_total=("rbp", lambda s: s.sum() * 100), rbp_per_q=("rbp", lambda s: s.mean() * 100),
    ).sort_values("rbp_total", ascending=False)
    print(fam.round(3).to_string(), "\n")

    print("=== 2. BY ARCHETYPE: where we earn vs bleed (sorted by total proxy RBP) ===")
    arch = res.groupby(["family", "archetype"]).agg(
        n=("brier_score", "size"), brier=("brier_score", "mean"),
        mean_p=("p", "mean"), realized=("outcome", "mean"),
        rbp=("rbp", lambda s: s.sum() * 100)).reset_index()
    arch["miscal"] = (arch["mean_p"] - arch["realized"]).round(3)
    arch = arch.sort_values("rbp", ascending=False)
    print(arch.round(3).to_string(index=False), "\n")

    print("=== 3. THE ASYMMETRY: per-question RBP distribution ===")
    r = res["rbp"] * 100
    print(f"  mean {r.mean():+.3f}  median {r.median():+.3f}  skew {r.skew():+.2f} (left = fat loss tail)")
    print(f"  best single +{r.max():.1f}   worst single {r.min():.1f}   "
          f"ratio worst/best {abs(r.min())/max(r.max(),1e-9):.1f}x")
    print(f"  upside (sum of +RBP) {r[r > 0].sum():+.1f}  from {(r > 0).sum()} q   |  "
          f"downside (sum of -RBP) {r[r < 0].sum():+.1f} from {(r < 0).sum()} q")
    print("\n  by confidence bucket (|p-0.5|): how leaning harder pays or punishes")
    res["cbucket"] = pd.cut(res["conf"], [0, 0.1, 0.2, 0.35, 0.5], include_lowest=True)
    cb = res.groupby("cbucket", observed=True).agg(
        n=("rbp", "size"), hit_rate=("outcome", lambda s: ((res.loc[s.index, "p"] > 0.5) == (s > 0.5)).mean()),
        brier=("brier_score", "mean"), rbp_per_q=("rbp", lambda s: s.mean() * 100),
        rbp_total=("rbp", lambda s: s.sum() * 100))
    print(cb.round(3).to_string())

    print("\n=== 4. RISK-TAKING: our 10 biggest deviations from the field proxy ===")
    res["dev"] = (res["p"] - res["field_p"]).abs()
    for x in res.sort_values("dev", ascending=False).head(10).itertuples():
        oc = "YES" if x.outcome == 1 else "NO"
        print(f"  dev {x.dev:.2f}  p {int(x.p*100):>2} field {int(x.field_p*100):>2} -> {oc:<3} "
              f"RBP {x.rbp*100:+5.1f}  [{x.family}] {str(x.question)[:46]}")

    print("\n=== 5. WORST 8 point bleeds (most negative RBP) ===")
    for x in res.sort_values("rbp").head(8).itertuples():
        oc = "YES" if x.outcome == 1 else "NO"
        print(f"  RBP {x.rbp*100:+6.1f}  Brier {x.brier_score:.3f}  p {int(x.p*100):>2} -> {oc:<3} "
              f"[{x.match}] {str(x.question)[:44]}")
    print("\n=== 6. BEST 8 point gains (most positive RBP) ===")
    for x in res.sort_values("rbp", ascending=False).head(8).itertuples():
        oc = "YES" if x.outcome == 1 else "NO"
        print(f"  RBP {x.rbp*100:+6.1f}  Brier {x.brier_score:.3f}  p {int(x.p*100):>2} -> {oc:<3} "
              f"[{x.match}] {str(x.question)[:44]}")


if __name__ == "__main__":
    main()
