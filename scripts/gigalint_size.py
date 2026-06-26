"""Refined rank-aggressive sizing for the GigaLint sheet — COMPUTE ONLY.

Aggression is gated, not flat. We only spend an extreme swing where all three hold:
  1. the FIELD is clueless on the archetype (low field-sharpness -> big RBP to harvest),
  2. WE are reliable there (calibrated / positive edge in the settled data), and
  3. WE are confident on this question (conviction = |q-0.5| is real).
Otherwise we leave the bet near Bot A's calibrated value. This stops wasteful 2%/98%
swings on miscalibrated player props, no-edge counts, and coin-flips.

Reads data/processed/gigalint_sheet.csv (Bot A belief q per market) and writes
gigalint_sheet_v2.csv, plus a per-match projected-RBP table vs the 160-180 target.
No submission anywhere.
"""
import numpy as np
import pandas as pd

from jumpprob.submit.runner import COUNT_ARCHS, PLAYER_ARCHS, TIMING_ARCHS

# field sharpness per family (strategy.py priors): how well the naive crowd reads it.
# low => the field is lost there => extremizing a reliable read harvests big RBP.
SHARP = {"goal": 0.85, "player": 0.45, "count": 0.40, "timing": 0.35}

# reliability tiers from the settled analysis (edge vs consensus + calibration):
#  A = obscure AND we beat the field there  -> spend the variance here
#  B = anchored goal/win  -> reliable direction but the field is sharp too -> modest
#  C = no edge (offsides/fouls/cards/both_teams/corners) OR miscalibrated (player props)
#      -> DO NOT extremize; leave near the calibrated number
TIER_A = {"team_sot_ge", "sot_gt", "team_corners_ge", "pen_or_red", "ht_leading",
          "team_scores_half", "goals_gt_half", "first_goal_half"}
TIER_C = {"offsides_ge", "fouls_gt", "cards_ge", "cards_gt", "team_cards_ge",
          "both_teams_sot_ge", "corners_gt", "total_sot_ge", "total_sot_half_ge",
          "corners_total_ge", "player_ga", "player_sot_ge", "player_scores",
          "btts_and_total_ge"}
KCAP = {"A": (3.0, 0.03, 0.97), "B": (1.8, 0.10, 0.90), "C": (1.0, 0.12, 0.88)}
JENSEN = 0.025          # field's irreducible cross-forecaster Brier penalty (observed)


def fam(a):
    if a in PLAYER_ARCHS: return "player"
    if a in TIMING_ARCHS: return "timing"
    if a in COUNT_ARCHS or a == "offsides_ge": return "count"
    return "goal"


def tier(a): return "A" if a in TIER_A else ("C" if a in TIER_C else "B")
def _logit(p): return float(np.log(p / (1 - p)))
def _sig(x): return float(1 / (1 + np.exp(-x)))


def main():
    df = pd.read_csv("data/processed/gigalint_sheet.csv")
    out = []
    for r in df.itertuples():
        q = float(np.clip(r.bot_a / 100, 1e-3, 1 - 1e-3))
        a = r.archetype; t = tier(a); k, lo, hi = KCAP[t]; conv = abs(q - 0.5)
        if conv < 0.05:                    # coin-flip: no directional signal, don't gamble
            p = q
        elif t == "C":                     # no-edge / miscalibrated: keep near calibrated
            p = float(np.clip(q, lo, hi))
        else:                              # reliable read: extremize (self-scales by conviction)
            p = float(np.clip(_sig(k * _logit(q)), lo, hi))
        s = SHARP[fam(a)]
        fb = _sig(s * _logit(q))           # field's belief = ours dulled by their sharpness
        up = 1 if q >= 0.5 else 0          # model-conforming outcome
        rbp_up = ((fb - up) ** 2 + JENSEN - (p - up) ** 2) * 100
        e_you = q * (1 - p) ** 2 + (1 - q) * p ** 2
        e_fld = q * (fb - 1) ** 2 + (1 - q) * fb ** 2 + JENSEN
        out.append({"kickoff_h": r.kickoff_h, "match": r.match, "question": r.question,
                    "archetype": a, "tier": t, "bot_a": r.bot_a, "agg_v1": r.aggressive,
                    "submit_v2": round(p * 100), "rbp_upside": round(rbp_up, 1),
                    "rbp_exp": round(e_fld * 100 - e_you * 100, 1)})
    v = pd.DataFrame(out)
    v.to_csv("data/processed/gigalint_sheet_v2.csv", index=False)
    g = v.groupby(["kickoff_h", "match"]).agg(
        proj_upside=("rbp_upside", "sum"), proj_expected=("rbp_exp", "sum"),
        extremes_v1=("agg_v1", lambda x: int(((x <= 5) | (x >= 95)).sum())),
        extremes_v2=("submit_v2", lambda x: int(((x <= 5) | (x >= 95)).sum())),
    ).reset_index().sort_values("kickoff_h")
    g["hits_target"] = np.where(g.proj_upside >= 160, "YES", "")
    print("per-match projection (upside = match goes to form; target 160-180):")
    print(g.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
