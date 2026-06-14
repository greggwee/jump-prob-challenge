# Sports-betting-theory review of the gigabot1 approach

*A sharp bettor's evaluation of the algorithm. Written inline by the system
author in a skeptic's seat. The lens is professional sports betting: fair value,
closing-line value (CLV), market efficiency, bet sizing, and bankroll
discipline — translated to a relative-Brier contest.*

## 0. The right frame: you are betting into the field as a soft book

A relative-Brier contest is, in betting terms, **betting into the field as if it
were a soft bookmaker**. Your per-question profit is `field_Brier − your_Brier` —
which is positive exactly when your probability is closer to the outcome than the
field's implied probability. That is the *same structure* as finding value
against a soft line: profit comes from the gap between your estimate and the
counterparty's price. Everything below follows from taking that frame seriously.

Three consequences a sharp bettor would state immediately:
- **Your counterparty is the field, not the bookmaker.** The sharp book is your
  *fair-value reference*, not your opponent.
- **You only have edge where your probability differs from the field's** — and
  are right. Matching the field is a push (0 points), exactly like betting at the
  same number the soft book offers.
- **You cannot see your counterparty's line.** This is the defining handicap
  (§4) and it has no analog in real betting, where the line is always visible.

## 1. What is sound (betting-theoretically correct)

- **Sharp closing line as fair value.** Anchoring 85% to a de-vigged sharp-book
  consensus is textbook: the closing line is the most accurate public estimate
  of true probability, and decades of CLV research say a forecaster who beats the
  closing line long-term has genuine edge, while one who matches it is at fair
  value. Using it as the truth anchor is correct.
- **Shin de-vig.** Removing the overround with Shin (insider-trading model)
  rather than proportional normalization is the right call — it corrects the
  favourite-longshot distortion in the quoted prices. Correct.
- **Closing-line capture via the near-kickoff scheduler.** Submitting as late as
  safely possible is exactly how sharps bet: the closing line is sharpest and
  carries the most information (lineups, late money). Correct and well-executed.
- **Extremizing as bet sizing.** In a winner-take-most race, deviating harder
  from the field where you have edge is the scoring analog of sizing a bet to
  your edge. The Lichtendahl–Winkler basis is sound.
- **Evidence-gating as bankroll discipline.** Refusing to "size up" (extremize)
  until edge is confirmed is the staking-discipline instinct — don't lever an
  unproven edge. Correct in spirit (but see §5 for how it backfires here).
- **Rejecting the squad feature and GBM on OOS evidence** is the betting
  equivalent of not betting a model you can't backtest into profit. Disciplined.

## 2. The CLV problem: on main markets you ARE the line

Here is the first hard critique. Bot A is 85% the sharp closing line. So on
win/draw/totals, **your submission ≈ the closing line ≈ fair value**. In betting
terms your **CLV is ≈ 0 by construction** — you are not beating the close, you
*are* the close. Against the *field* (the soft book) this is a defensive floor:
you won't get destroyed the way a wrong model would (the USA-vs-Paraguay
blow-up). But you also **cannot win points on these markets** — you're laying the
same number the field can read off the same odds.

The corollary cuts at the 15% model weight: **the only way the 15% model earns
anything on main markets is if it has positive CLV** — if it systematically beats
the closing line. We have *never demonstrated that.* The Phase 3 blend was fit on
FiveThirtyEight, not on real closing-line outcomes. So the 15% is, on current
evidence, as likely to be CLV-negative noise as edge. A sharp would say: **prove
the 15% beats the close, or drop it to ~5% and stop pretending.**

## 3. Market efficiency confirms the edge is not on main markets

Phase B measured Polymarket ≈ sharp books to 0.3%. In betting terms: the deep,
liquid markets are *efficient and mutually arbitraged*. You cannot beat them
without genuine private information, which we do not have. This is not a flaw —
it is a correct empirical finding that **forecloses main markets as a source of
edge** and points all the way to the soft, low-limit corners of the board:
team/player props, half-specific lines, obscure counts. In betting, that is
exactly where the edge lives — soft books post lazy prop lines with low limits —
*and* where the retail field is worst. The thesis is sound. The problem is it is
**unvalidated** (§5).

## 4. The defining handicap: you cannot see the line you are betting into

A sharp bettor *always* knows the price they are taking. We do not. Phase A
confirmed the API exposes **no field data** — not per question, not per archetype,
only an aggregate RBP scalar on a web page. So we are betting into a soft book
**whose prices we cannot observe**, estimating them with a structural prior (the
field-sharpness λ). No competent bettor would accept this blind, yet the contest
forces it. The implication: **every "edge" claim is an estimate against an
estimated line.** The field model (Phase C) is the best we can do, but it is a
prior dressed as a price. This is the single biggest epistemic limitation of the
whole system, and it is structural, not fixable by better modelling.

## 5. The conservatism trap: the gate may never fire

This is the deepest strategic critique, and it is a betting-theory point about
*tournament* play. To win a winner-take-most contest you must take variance —
like a short-stacked poker player who must gamble, not fold to the blinds. Our
extremizer is gated to `k=1` (no leverage) until a family shows *confirmed* edge
over ≥20 settled questions. But:
- We get ~10 no-line settlements per matchday across many families → **20+ per
  family takes many matchdays**, by which point the group stage may be over.
- The "confirmed edge" test compares our Brier to an **unobservable field**
  (§4) proxied by a prior — so "confirmation" is itself a guess.

Net: **the gate may stay at k=1 for the entire contest.** We will be
well-calibrated, beat coin-flip, and **finish mid-pack — never taking the
variance a winner-take-most race requires.** A sharp tournament player would call
this *folding into the money bubble and busting short of first*. The evidence
gate, prudent in a cash game, is miscalibrated for a winner-take-all tournament
where the prize is rank #1 only.

## 6. Iterative improvements, prioritized by EV

1. **Measure CLV directly (cheap, do first).** We already log pure vs anchored
   per market. Extend it: log our submission vs the *final closing* consensus,
   and track realised Brier of (a) pure model, (b) anchor, (c) 15%-blend.
   After ~100 settlements this answers "does the 15% model beat the close?"
   empirically. If no, cut it. This converts the §2 hand-wave into a number.
2. **Validate the no-line edge offline, don't wait for live confirmation.**
   The §5 trap exists because we wait for unmeasurable live edge. Instead,
   backtest our prop/count models against a *retail-field proxy* (base-rate-
   toward-50, or bookmaker prop lines where they exist — Phase D) on the six
   historical tournaments. That gives a **backtested** per-family edge estimate
   to seed the extremizer, so it can fire on proven-in-backtest edge rather than
   never-confirmed live edge. This is the highest-EV change.
3. **Adopt a tournament-stage variance policy (judgment call).** Given the gate
   may never fire and the prize is rank #1, take a *small, principled* extremize
   (k≈1.2–1.3) on the no-line markets where backtest (#2) shows edge — from the
   start, scaled up in knockouts — accepting the leverage risk because the rank
   objective demands variance. This is the "you must gamble to win the
   tournament" correction. Size it to leaderboard position once RBP is known.
4. **Ingest bookmaker prop lines (Phase D).** Where books price corners/cards/
   player props on big matches, they (a) anchor our prop *truth* estimate and
   (b) give a real *field-line* proxy on those props (the field anchors to them).
   This is the one place market data sharpens the no-line edge — and it lets us
   measure CLV on props, not just main markets.
5. **Right-size the main-market weight.** Pending #1: if the 15% model shows no
   CLV, move main markets to ~95% market (or skip scoring them and pour the
   effort/variance budget into props). Stop spending variance where edge is zero.

## 7. The meta-verdict

The algorithm is **built to not lose** — it is well-calibrated, anchored to fair
value, disciplined about unproven features, and correctly identifies that edge
lives in the soft corners of the board. By the standards of *cash-game* betting
it is sound. But the contest is a **winner-take-most tournament**, and there the
setup is structurally underpowered: it is CLV-neutral where it is most confident
(main markets), unvalidated where it claims edge (props), blind to the line it is
betting into (the field), and gated so conservatively it may never take the
variance required to finish first. The fixes are not new models — they are
**(a) measure CLV, (b) backtest the prop edge so the variance switch can turn on,
(c) accept tournament variance.** Calibration got us a seat; only proven,
deliberately-levered edge on the soft markets wins the tournament.
