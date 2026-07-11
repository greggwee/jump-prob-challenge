"""Phase-2 endgame policy ladder: how much variance (if any) to buy, given the deficit.

From the 2026-07-10 endgame MC (block-bootstrap on real settled match blocks): honest play
nets +481 +/- 379 over the closing slate with P(lose ground) ~10%; blanket extremizing adds
ZERO mean (multipliers scale losses too) at ~2x the spread. So variance is bought (a) only as
far as the deficit demands, (b) only on the structural-edge kinds (runner.ENDGAME_KINDS), and
(c) as late as possible — the 3x final is the most efficient place for it.

The leaderboard API needs the web JWT we don't hold, so the operator reads the current
standing off the site (Profile & History -> My Bots) and runs:

    .venv/Scripts/python scripts/endgame_policy.py --score 2724 --cutoff 4064 \
        --matches-2x 2 --final 1 [--honest-rate 2.0] [--cutoff-rate 3.4]

It prints the needed per-question rate vs our honest rate and the recommended tier:
    HONEST                 (needed <= honest + 0.5)
    ENDGAME_EXTREMIZE=1.25 (needed <= honest + 2.0)
    ENDGAME_EXTREMIZE=1.5  (needed <= honest + 4.0)
    ENDGAME_EXTREMIZE=1.75 (beyond — and a warning: the target is probably luck-dependent;
                            buying more variance LOWERS expected rank)
Set the env var in .env (the scheduler inherits it); REMOVE it once back on track — the
ladder de-escalates. The dial only touches ENDGAME_KINDS on the live pass; absent env =
byte-identical submissions.
"""

from __future__ import annotations

import argparse

QPM = 15  # questions per knockout match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, required=True, help="our current RBP (bot page)")
    ap.add_argument("--cutoff", type=float, required=True, help="current score at the target rank (e.g. #10)")
    ap.add_argument("--matches-2x", type=int, default=2, help="remaining 2x matches we can score (SF/3rd)")
    ap.add_argument("--final", type=int, default=1, help="1 if the 3x final remains, else 0")
    ap.add_argument("--honest-rate", type=float, default=2.0,
                    help="our measured honest raw RBP per question at 1x (conservative KO-era default)")
    ap.add_argument("--cutoff-rate", type=float, default=3.4,
                    help="cutoff holder's raw RBP/q pace (their score / ~1x-eq questions so far)")
    args = ap.parse_args()

    w_rem = (2.0 * args.matches_2x + 3.0 * args.final) * QPM       # multiplier-weighted questions
    cutoff_proj = args.cutoff + args.cutoff_rate * w_rem * 0.88    # persistence-regressed pace
    need_total = cutoff_proj - args.score
    need_rate = need_total / w_rem if w_rem else float("inf")

    print(f"remaining 1x-eq questions: {w_rem:.0f}  (SF/3rd x{args.matches_2x}, final x{args.final})")
    print(f"cutoff now {args.cutoff:.0f} -> projected {cutoff_proj:.0f} at close (pace {args.cutoff_rate}/q x0.88)")
    print(f"we need {need_total:+.0f} over {w_rem:.0f} 1x-eq q = {need_rate:+.2f}/q  vs honest {args.honest_rate:+.2f}/q")

    h = args.honest_rate
    if need_rate <= h + 0.5:
        rec = "HONEST — do not set ENDGAME_EXTREMIZE (or remove it). The edge alone gets there."
    elif need_rate <= h + 2.0:
        rec = "ENDGAME_EXTREMIZE=1.25 — mild tilt on the structural-edge kinds only."
    elif need_rate <= h + 4.0:
        rec = "ENDGAME_EXTREMIZE=1.5 — meaningful tilt; concentrate on the 3x final if possible."
    else:
        rec = ("ENDGAME_EXTREMIZE=1.75 — the target needs a tail outcome; understand that buying "
               "this variance LOWERS expected rank. Consider whether the prize structure justifies it.")
    print(f"\nRECOMMENDATION: {rec}")
    print("(De-escalate the moment a good match puts the target back in honest reach.)")


if __name__ == "__main__":
    main()
