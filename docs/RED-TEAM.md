# Red-team review — Probability Cup forecasting system

*Adversarial quant/statistician review of Bot A (live) and Bot B (shadow).
Written inline by the system author wearing a skeptic's hat (the spawned
verifier agent hit a session limit before producing output). Findings are
ordered by impact; each tagged BUG (correctness) or STRATEGY (suboptimality
under the relative-to-field objective).*

## Executive summary

The system is statistically sound in its **absolute** forecasting — the goal
model, de-vig, NegBin counts, and the new odds anchor are all defensible. The
problem is that **the contest does not reward absolute accuracy; it rewards
beating the field, and the prize is winner-take-most.** Three of the system's
foundational choices are optimized for the wrong objective:

1. It anchors 85% to sharp-book odds — which makes it converge toward a field
   that also reads odds, driving its edge on main markets toward **zero**.
2. It reports honest, calibrated probabilities — provably **suboptimal** when
   the objective is P(finish #1) rather than expected points.
3. It treats the contest as point-accumulation, ignoring that only ranks ~1–10
   pay and #1 pays ~100×, which calls for **variance-seeking**, not calibration.

The single biggest reframe: **calibration is a defensive floor; the edge must
come from markets the crowd cannot price, played with deliberate variance.**

---

## HIGH severity

### H1 — STRATEGY — The odds-anchoring / field-correlation paradox
Relative points per question = `(field_Brier − your_Brier) × mult`. The brief
itself lists betting markets (Pinnacle, Betfair, OddsPortal) as a resource, so
a large share of the field anchors to the same odds we now anchor 85% to. On
main markets (win/draw/totals), therefore:

> your_prediction ≈ market ≈ field ⇒ (field_Brier − your_Brier) ≈ 0.

We spent Phase 7 improving **absolute** calibration on exactly the markets where
**relative** edge is structurally near zero. This is not wasted — it stops the
USA-vs-Paraguay class of blowup where a *sharper* field beats a wrong model, so
it removes a source of **negative** points. But it cannot generate **positive**
points. Anchoring is a defensive move sold as an offensive one.

**Where edge actually lives:** markets the crowd *can't* anchor because no liquid
line exists — team corners/cards/fouls/offsides thresholds, shots-on-target
comparisons, half-specific goals, player props, first-goal combos. There, the
crowd is guessing near base rates and our models genuinely deviate. **That is
where points are won or lost, and it is exactly where the system is weakest**
(no calibration on the live path — see H4 — and independence bugs — see M1).

**Recommendation:** treat main markets as a defensive floor (anchor, accept
≈0 edge) and pour modelling + calibration + controlled variance into the
no-line markets. Measure our realized edge there as settlements arrive.

### H2 — STRATEGY — Honest calibration is the wrong objective for P(rank #1)
The whole system rests on Brier being a strictly proper scoring rule ⇒ report
your true `p`. The proof (METHODS §1) is correct but proves the wrong thing: it
shows honesty minimizes **expected** Brier, hence maximizes **expected
cumulative points**, hence **expected rank** (~median). But the payout is
winner-take-most: #1 = the fellowship, #2–5 a tablet, #6–10 a gift card, the
rest nothing. The rational objective is **P(finish #1)** (with a secondary top-10
floor), which is a **variance-seeking** objective.

Lichtendahl & Winkler (2007) prove exactly this: when forecasters are scored on
*relative rank*, proper scoring rules stop being proper and the equilibrium is
to **extremize** — report more extreme than your true belief. Intuition: with
many entrants, the winner is essentially the player whose cumulative edge landed
in the far-right tail. Let your per-question edge over the field have mean μ and
SD σ. To be the *max* of ~K players' cumulative edges, you want your distribution
to reach furthest right — which favors **higher σ even at the cost of some μ**.
Honest calibration deliberately *minimizes* σ-vs-field (that's what "matching
the crowd where you agree" does), so it is close to the variance-MINIMIZING
choice — the opposite of what winning needs.

**Recommendation:** apply controlled extremizing on the markets where we hold a
genuine edge (H1's no-line markets), sized to leaderboard position and stage
multiplier — small when we already lead, larger when trailing and late. This is
NOT dishonesty for its own sake (which would just raise expected Brier); it is
the rank-optimal transform of an honest belief under a winner-take-most rule.

### H3 — STRATEGY — The one-account constraint orphans the entire v2 engine
Only one prediction per market per account, so Bot B (the MC simulator) **cannot
score** — it is shadow-only. Yet v2 is *coherent across all markets*, handles
the **timing markets natively** (first-goal-and-win, ht-tied, h2>h1, first-goal-
half) that the live v1 path only crudely approximates, and produces posterior-
predictive shrinkage. Meanwhile the live submitter (v1) approximates those very
markets and applies no calibration.

So the better engine is benched on the markets where it is clearly better.

**Recommendation:** the single live submission should be a *blend of engines by
market type*: odds-anchored grid for main markets; **v2 simulator for the timing
markets** (it is strictly more correct there); calibrated count/prop models for
the no-line markets. The odds anchor feeds all three. The "Bot A vs Bot B" split
is a fiction left over from the pre-constraint design; collapse it into one
best-of-breed submitter. (Separately: the spare API key is only useful on a
*second account*, which the rules may forbid — RECON flags this as unverified
and a disqualification risk. Do not pursue without written confirmation.)

---

## MEDIUM severity

### M1 — BUG — Conditional independence in "more X than" comparisons
`p_a_gt_b` computes P(team A corners > team B corners) treating the two counts as
**independent** given intensities. Within a match they are correlated: a team
dominating territory takes more corners *and* concedes fewer (negative
correlation in shares), while an end-to-end game lifts both (positive). The
independent convolution mis-sizes the tails of the difference distribution, so
every "more corners/SoT/fouls/cards than" market is biased — and these are
no-line markets where our edge is supposed to live (H1). v2's shared game-state
partially corrects this; the live v1 path does not.
**Fix:** model the difference with a correlation term, or submit these from v2.

### M2 — BUG — The live path applies no calibration
`submit/runner.py` never calls the Platt layer (`models/calibration.py`); it
exists but is only wired into `pipeline.predict_fixture`, which the live path
doesn't use. For anchored markets that is fine (the market is calibrated). For
**count/player markets it is a real defect** — those models were measurably
overconfident in backtest, and they are exactly the edge markets (H1). We are
submitting known-overconfident numbers on the markets that matter most.
**Fix:** apply per-archetype Platt to the model-only markets in the live path.

### M3 — STRATEGY — Multiplier structure is ignored
Knockout 2×, final 3×, and multipliers **apply to losses too**. Optimal play is
stage-dependent: accumulate a buffer in the group stage (1×, low variance), then
take **more** variance in knockouts/final where the same edge is worth 2–3×.
The system plays every stage identically. A −5 blunder in the final costs −15;
a +5 coup pays +15. This asymmetry around our *current rank* should drive
aggressiveness, and currently doesn't.

### M4 — STRATEGY — "Answer everything" is unexamined under relative scoring
Answering all 677 markets is correct for expected points but not obviously for
P(rank #1). A market where we equal the field is pure variance with zero mean
edge — it adds noise to our cumulative score without expected benefit. Under a
rank objective, noise we don't control is bad. There's a case for **skipping**
markets where we have no edge over the field (most anchored main markets) to
reduce un-edged variance, and concentrating on edge markets. This needs the
field model (L1) to action well, but the principle stands and is currently
ignored.

### M5 — BUG/STRATEGY — Blend weight and λ-inversion fit on the wrong anchor
The 0.85 market weight was fit against FiveThirtyEight forecasts (Phase 3), not
real sharp-book consensus. Sharp books are *sharper* than 538 was, which argues
for even more market weight on main markets — but that deepens H1 (more
convergence to the field). The "right" weight depends on the field-correlation
analysis, not on absolute-Brier backtest. The number is borrowed and the
objective it was fit for is the wrong one.

---

## LOW severity

### L1 — STRATEGY — No field model
The system cannot see crowd predictions and has no model of the field beyond
"the field resembles the market." For a relative-scoring contest this is the
central missing piece: you cannot deliberately deviate from the crowd if you
can't estimate the crowd. The API exposes only aggregate RBP (profile page), not
per-question field Brier. **Recommendation:** build a proxy field —
`de-vigged odds regressed ~20% toward 50 on main markets; base-rate ± noise on
no-line markets` — and target deviations against *that*, not against the market.
Refine it from aggregate RBP-by-stage as settlements accumulate. Even a coarse
field model converts H1/H2/M4 from hand-waving into sizing decisions.

### L2 — STRATEGY — Base-rate 0.25 shrink pulls toward the field
Unanchored markets shrink 0.25 toward a league-average base context. But the
crowd also knows base rates, so shrinking toward base pulls us *toward* the field
— reducing edge on the exact markets where we need it. The shrink is also ad hoc
(chosen, not fit). It is defensible as variance reduction for a poorly-estimated
model, but it is in tension with the rank objective.

### L3 — BUG — Posterior-predictive only on goal betas
v2 perturbs only the goal-model betas, not rho_dc, the dynamics parameters, or
count-model dispersion. Thin-data fixtures get *some* widening but the uncertainty
is understated. Low impact (it's shadow-only), but worth noting if v2 is promoted
(H3).

---

## Top recommendations, ranked by EV

1. **Reframe the edge (fixes H1+M2+M4).** Concede main markets to the anchor as a
   defensive floor; concentrate modelling, **calibration**, and variance on the
   no-line markets (counts/props/timing) where the crowd guesses. Turn on the
   Platt layer in the live path immediately — cheapest concrete win.
2. **Adopt rank-optimal extremizing (H2+M3).** Transform honest edge into
   variance-seeking submissions, sized by leaderboard position and stage
   multiplier (gentle early/ahead, aggressive late/behind). This is the change
   most aligned with the actual prize structure.
3. **Collapse Bot A/Bot B into one best-of-breed submitter (H3).** Submit timing
   markets from v2 (correct), main markets from the anchored grid (calibrated
   floor), no-line markets from calibrated count/prop models — all fed by odds.
4. **Build a proxy field model (L1).** Without it, every "deviate where the crowd
   is wrong" instruction is guesswork. With it, H1/H2/M4 become quantified sizing.
5. **Fix the count-comparison independence bug (M1)** or route those markets
   through v2's correlated simulator.

## What is genuinely fine (don't touch)
The de-vig (Shin) and consensus weighting are correct and well-tested. The goal
model and its MLE are sound. The odds anchor correctly fixed a real failure mode
(it removes negative-EV blowups). The decision to reject the squad feature and
GBM on OOS evidence was disciplined and right. The issue is not the machinery —
it is that the machinery is aimed at absolute accuracy when the prize is a
relative, winner-take-most race.
