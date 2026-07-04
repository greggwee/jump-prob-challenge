# Adversarial Review: R16 Readiness (2026-07-05)

Reviewer: adversarial senior quant pass (session 3 — prior two runs lost reports; this file is written incrementally).

Scope: mechanism-level diagnosis of the five worst categories, an in-tournament count-drift fix design, R16-specific critique (regulation scoping, ET/pens knobs, the 6 new templates), new-edge proposals, and a ranked SHIP/VALIDATE/REJECT list.

Status: COMPLETE.

## Executive summary

1. **The worst category is a live parser bug, not a model error.** "At halftime, will
   both teams have at least 1 shot on target?" parses with `half=None` (verified against
   the current parser) and is priced as the FULL-MATCH probability (.78–.92) when the
   scope is the first half (truth ~.55–.65, crowd ~.61). One-line fix + a base_rates
   update (the live risk gate's stale 0.70 base would otherwise eat the fix — §3.6).
2. **WC26 count levels have drifted and every safety layer encodes the OLD level.**
   Cards realized .364/.300/.200 by date tercile vs ~.52 predicted; total SoT 7/7 YES at
   .52 predicted. The count Platt (+0.18 up-shift), Bot-A base blend, risk-gate bases,
   and crowd prior all pin counts at historical levels (§1.6). Fix: one shrunk
   multiplicative level factor per family (estimator fully specified and prototyped:
   cards s=.935, SoT s=1.29 at tau=.25, 14-day half-life), Platt→identity, bases
   regenerated (§2).
3. **`leads_at_any_point` is systematically 4–12 pts low** (validated on 9,559
   goal-matches + 200k-draw MC): the +0.03 comeback fudge should be ~+0.05 average and
   strength-dependent, and the question scope INCLUDES extra time. MC ever-led tracking
   is a 5-line change (§3.3.1).
4. **New R16 templates aren't registered in GOAL_ARCHS** — anchored grid answers get
   diluted 25% toward league-average; the crowd clamp is blind to all six (§3.4).
5. **Two innovations survive scrutiny**: harvest actual WC26 counts via the existing
   api-football module (fits levels on counts, not binaries, and detects settlement-
   definition mismatches), and a crowd-gap drift tripwire (the crowd is measurably
   WC26-adapted; use it as a sensor, not a target). Rest-day models, ET book anchors,
   per-ref cards: killed (§4.3). Full ranked ship list in Task 5.

Method note on data: all numbers below were recomputed 2026-07-05 from a fresh read-only
`/results` pull (927 settled rows), outcome recovered from our own Brier and disambiguated
with the crowd file's `out` column; 23 rows submitted at exactly p=50 with no crowd
harvest were dropped as irrecoverable. Archetypes were re-classified from question TEXT
via `parse_question` (the matchday_predictions.csv join only covers recent runs — see
Finding 0). Historical structure comes from `data/processed/team_half_counts.parquet`
(628 team-halves, 6 tournaments).

---

## Finding 0 (meta): the team's own "worst category" table is computed on a biased subset

`settled_with_crowd.parquet` gets its archetype from a join to `matchday_predictions.csv`,
which only contains the markets of the most recent run(s). Older settled markets have
archetype=NaN and silently fall out of every per-archetype tally. Recomputing archetypes
from question text over ALL 904 usable settled rows changes several headline numbers:

- `cards_ge` thr=4 full-match: n=13 with crowd/text classification, realized .462 vs our .485
  — nearly calibrated over the WHOLE tournament; the alarming .200 realized is the RECENT
  (knockout-era) slice only. Cards are not uniformly over-priced; they have DRIFTED down
  late (see Finding 2). A level fix must be time-weighted, not a flat tournament constant.
- `both_teams_sot_ge` exists in the text-classified set (n=17) but is INVISIBLE in the
  parquet-based per-archetype table. The single worst category in the platform's own
  breakdown was not even reachable by the team's main calibration artifact.

Action: `scripts/build_settled_dataset.py` should classify archetype from text (fallback
to the join), and keep a `created_date`-based era column so drift is measurable.

---

## Task 1 — Mechanism diagnosis of the five worst categories

### 1.1 Both teams shots on target (platform RBP −7.8, n=17) — SCOPE/DEFINITION BUG, live TODAY

Decomposition of the 17 settled `both_teams_sot_ge` rows:

| slice | n | our_p | realized | crowd_p | verdict |
|---|---|---|---|---|---|
| "At halftime, will both teams have ≥1 SoT?" (parsed half=None) | 8 | .78–.92 (mean .84 ex-bug) | .500 | .59–.64 | **LIVE PARSER BUG** |
| "…in the second half?" (parsed half=2) | 9 | ~.70 ex-bug | .667 | .67–.76 | roughly calibrated |
| old known-bug rows (swallowed by player pattern; our .09/.24) | 2 | — | 1.0 | .59–.76 | fixed already |

The mechanism is NOT a WC26 level shift and NOT dispersion. The phrase
**"At halftime, will both teams have at least 1 shot on target?"** falls through to the
no-half pattern `will both teams have at least (\d+) shots? on target` in
`src/jumpprob/submit/parser.py` and is priced as the FULL-MATCH probability (.77–.92)
when the true scope is the FIRST HALF (historical P(both ≥1 SoT in a half) = .643;
realized here .500; crowd .61). We are submitting ~+25 pts over truth on every such
question, and the pattern recurs every matchday. The parser already special-cases
"At halftime, …" for corners_gt and ht_tied — both-teams SoT was simply missed.

Fix (one line, SHIP-NOW): add before the generic both-teams patterns:
```python
(re.compile(r"at halftime, will both teams have at least (\d+) shots? on target", re.I),
 lambda m: ParsedQuestion("both_teams_sot_ge", threshold=int(m[1]), half=1)),
```
(and the "(\d+) or more" variant with the same prefix). Audit: grep all OPEN questions for
`^At halftime,` and verify every archetype it prefixes has an explicit half=1 route.

Secondary (minor): the handler's independence assumption `P(both) = pa*pb` overstates by
2–4% relative at thr=1–2 (measured on 628 historical match-halves: empirical/product ratio
.978 at ≥1, .958 at ≥2; within-half SoT corr between teams = −0.104). Worth a 0.97
multiplier only after the scope bug is dead; it is second-order today.

### 1.2 Total shots on target (RBP −1.7) — WC26 LEVEL SHIFT UP, overwhelming signal

- `total_sot_ge` thr=8 full match: **7/7 YES** at our mean .516 (crowd .623). Historical
  P(≥8 SoT) = .471. If the true probability were still .47, P(7/7) ≈ 0.5%; even at .65 it
  is ~5%. WC26 SoT is genuinely elevated.
- `total_sot_half_ge` thr=4 second half: 9/12 YES (.75) at our .617, crowd .628.
- Corroboration from team-level: `team_sot_ge` realized > pred at low thresholds in H2.
- The static count model was fit on 6 tournaments averaging 1.85–2.0 SoT/team-half
  (7.4/match); WC26 outcomes are consistent with ~8.5–9.5/match (+15–30%).

Mechanism: LEVEL error (μ too low), possibly amplified by a platform definition of SoT
that is more generous than StatsBomb's (StatsBomb excludes blocked shots from SoT; many
data providers count shots saved+goals; if the platform's settlement source (likely
Opta/API-Football "Shots on Goal") counts differently the gap is structural, not just
conditions). Either way the correction is the same: scale μ_sot up. See Task 2.

### 1.3 Total cards (RBP −0.9) — LEVEL DRIFT DOWN, time-varying, plus a gate anchored to a stale base

- Whole tournament, thr=4 full match: realized .462 vs our .485 — fine.
- Recent (knockout-era) slice: realized .200 vs pred .522 (the prompt's recompute, n=15).
- Second-half cards thr=2: realized .25 vs our .625 (n=4); `team_cards_ge` ≥1 in 2H:
  realized .167 vs our .64 (n=6). The late-match card spike in the historical minute
  profile (α card share .347 post-67') is NOT materializing in WC26.
- The `risk_gate` (LIVE, s=0.5) shrinks cards toward `base_rates.json` = .525 — a base
  computed from historical data. When the truth drifts to ~.2–.3, the "safety" gate
  actively pins us at the wrong level. The pooled count Platt in `live_calibration.json`
  is `[0.972, +0.181]` — an UP-shift fit on the historical backtest — pushing cards the
  WRONG way (a .45 model becomes ~.49 submitted).

Mechanism: LEVEL (μ_cards too high for WC26 conditions, worsening over time — ref
directives / knockout leniency), compounded by two stale correction layers that both
point up. This is the clearest case for the Task-2 shrunk level recalibration, with
time-decay weighting so June-12 group matches don't out-vote July knockouts.

### 1.4 Player to score (RBP −0.7, n=18) — HISTORICAL ERA EFFECT, mostly self-healed

Text-classified: `player_scores` n=30, our .245, realized .300. The under-pricing
(model priced unknown-to-us scorers ~8–15%) predates the book-anchor + Phase-1 floor;
the crowd-matched rows where we lose are concentrated in the early harvested matches.
Post-anchor rows are book-driven at w≈.97. No new mechanism to fix beyond keeping the
anchor fed; residual risk is book-line COVERAGE near kickoff (quota resilience), not
the model. Verdict: leave alone; monitor via the CLV ledger.

### 1.5 Team corner comparisons (RBP −0.1, n=29) — TIE-MASS + NOISE, no fixable signal

`corners_gt` full-match: n=14, our .461, realized .286, crowd .410 (edge −11.8 pts);
half-scoped slices go the other way (H2: realized .875 vs our .538, +1.2 pts; H1 0/6 at
our .285, crowd .390). `team_corners_ge` (n=21) is a WINNER: +27.0 pts vs crowd.
Realized .286 vs a fair ~.42 on 14 comparisons is p≈.2 binomial noise; the family's
aggregate RBP (−0.1) is statistically zero. The only structural nit: P(A>B) requires
beating ties, and per-half corner counts tie ~25–30% of the time — our NegBin comparison
already models this. Verdict: NO ACTION beyond the corners μ getting whatever small
level update Task 2's estimator grants it (corners show no strong WC26 shift).

### 1.6 The four-layer pile-up that keeps cards pinned at 50

The live CAN vs MAR run (odds_anchor_run.csv, 2026-07-04) submitted "4 or more total
cards" at **50** while the knockout-era realized rate is .20–.30 and the crowd sits ~.44.
Why can't the system move? Because every "correction" layer encodes the HISTORICAL card
level:

1. count-family Platt `[0.972, +0.181]` (fit on the 6-tournament backtest) — pushes UP;
2. Bot A's 25% base-rate blend — base computed at league-average intensities from the
   same historical fit;
3. the LIVE risk gate (s=0.5) — shrinks toward `base_rates.json["cards_ge"]=.525`,
   another historical constant;
4. the crowd-proxy prior for cards (mean ~.5) — inert here (25-pt clamp), but pointing
   the same direction.

Four independent safety layers, one shared stale assumption. No amount of per-layer
tuning fixes this; the LEVEL itself must be updated and the layers told about it (Task 2).

---

## Task 2 — Design: in-tournament count-level recalibration (shrunk, not re-fit)

### 2.1 What the evidence supports

| family | WC26 evidence | direction | strength |
|---|---|---|---|
| shots_on_target | total_sot_ge thr8: 7/7 YES (hist P=.458); total_sot_half_ge thr4 2H: 9/12; team_sot 2H realized>pred | UP ~+15–30% | overwhelming (joint P under H0 < 1%) |
| cards | pooled cards realized by date tercile .364/.300/.200 vs pred ~.52–.56; 2H cards ≥1: 1/6 realized vs .64 pred; 2H cards ≥2: .25 vs .625 | DOWN ~−10–25%, drifting further down; 2H share collapsed | strong |
| corners | team_corners_ge realized .429 vs pred .437 (n=21); totals mixed | ~none | fit s but expect ≈1 |
| offsides | already handled by the live intercept in live_calibration.json (pred .431 = realized .431 now) | — | do NOT double-treat |
| fouls | only comparison questions settle (level-invariant) | unidentifiable | skip |

### 2.2 The estimator

One multiplicative level factor `s_f` per family on the count model's mean:
`mu' = s_f * mu(family, half, lam_t, lam_o)`. Fit by penalized weighted Bernoulli MLE on
the settled WC26 binary outcomes of that family's THRESHOLD archetypes only:

```
maximize  sum_i  w_i [ y_i log p_i(s) + (1-y_i) log(1-p_i(s)) ]  -  (ln s)^2 / (2 tau^2)
  p_i(s) = the question's own handler evaluated with mu -> s*mu   (NegBin k unchanged)
  w_i    = 0.5^(days_ago_i / 14)          (time decay: captures the measured card drift)
  tau    = 0.25                            (log-normal shrinkage toward s=1: 1sd = ±28%)
```

Include: cards_ge, team_cards_ge, both_teams_cards, cards_post_hb2 (cards);
total_sot_ge, total_sot_half_ge, team_sot_ge, both_teams_sot_ge — ONLY after the
halftime-scope parser fix, and excluding known_bug rows (SoT); corners_total_ge,
team_corners_ge, corners_pre_hb1 (corners).
EXCLUDE all `*_gt` comparison archetypes — a common level factor cancels in P(A>B);
they contribute zero information about s and only add noise.

Why NOT a free re-fit: n per family is 15–40 binaries; an unshrunk fit on the SoT
family's 7/7 sample runs to s=2.0 (verified in prototype) — absurd. The prior is
load-bearing exactly where the data is degenerate.

Prototype results (this review; match-total approximation of p_i(s), NegBin k from
historical match totals, time-weighted, tau=.25):

- cards:  s_hat = 0.935 (unshrunk 0.878) → P(≥4 cards) .440 → .398
- SoT:    s_hat = 1.291 (unshrunk 2.014) → P(≥8 SoT)  .458 → .723

The production fit, using per-question anchored lambdas and per-half mus (the real
handlers), will differ somewhat; these numbers establish sign and rough magnitude. Note
the cards factor is conservative because thr-4 group-stage rows (realized .46) dominate
the unweighted sample; with the 14-day half-life the knockout-era collapse carries the
weight it deserves. If the R16 settles another 0-for on cards, s_hat falls fast — the
estimator is self-correcting matchday by matchday (refit in daily_learn).

### 2.3 Where it plugs in — and the double-correction rule

`live_calibration.json`'s count entry `[0.972, +0.1811]` was fit on the HISTORICAL
backtest (`backtest_counts.csv`, src==model). It is an UP-shift (+4 pts near .5) that is
currently anti-correcting cards live. The level factor and that Platt would double-count
the same phenomenon if stacked. Rule: **the level fix goes into the count model; the
count Platt goes to identity.**

Precise changes:
1. `src/jumpprob/models/count_model.py` — `CountModel.mu` reads an optional
   `data/processed/wc26_level.json` = `{family: s}` and multiplies. Missing file/key = 1.0
   (inert; tests unaffected).
2. New `scripts/fit_wc26_level.py` (called from daily_learn after build_settled_dataset):
   fits s per Section 2.2, writes wc26_level.json, prints per-family n / s_hat /
   unshrunk s / effective sample and refuses to write |ln s| > ln(1.5) (circuit breaker —
   a level shift beyond ±50% means a definition problem, not conditions).
3. `scripts/fit_live_calibration.py` — set `table["count"] = [1.0, 0.0]` (mirroring the
   documented player-family precedent: a non-transferring historical fit is worse than
   neutral). Keep offsides' live intercept; it absorbs a definition gap, not a level.
4. Regenerate `base_rates.json` for the gated count archetypes from the SCALED model
   (otherwise the live risk gate keeps shrinking cards back toward .525 and undoes half
   the fix — measured: s=0.5 gate would cancel ~50% of any level move).
5. Order-of-operations audit: risk gate still applies AFTER; with refreshed bases the
   gate and the level fix now agree instead of fighting.

### 2.4 Validation before going live

- Leave-one-tercile-out CV on settled WC26: fit s on two date terciles, score Brier on
  the held-out one vs the status-quo pipeline. Require: no degradation on T1/T2 held out,
  improvement on T3 (the knockout-era tercile) held out. The asymmetric requirement is
  deliberate — T3 is what R16 looks like.
- Sanity: refitted P(≥4 cards) should land between the recent crowd (.44) and the recent
  realized (.2–.3); refitted P(≥8 SoT) should approach the crowd (.62) from below. If a
  fit OVERSHOOTS the crowd on a family, cap at the crowd family mean (the crowd is the
  only other WC26-adapted estimator in the room, and it is measurably calibrated on
  cards: crowd .47 vs realized .462 tournament-wide).
- Cross-check against Idea A (Section 4.1) when it ships: a level factor estimated from
  REAL WC26 counts must agree with the binary-outcome fit within CI; disagreement means
  the problem is settlement DEFINITION (e.g. platform counts SoT like Opta, not
  StatsBomb), which is fixed in the threshold mapping, not in mu.

---

## Task 3 — R16-specific critique

### 3.1 Regulation scoping — one real leak found

The parser strips "in/of regulation (90 minutes + stoppage time)" and every goal/count
handler is 90-minute — that path is clean. The leak: **"Will X hold a lead at any point
in the match (excluding a penalty shootout)?"** carries NO regulation qualifier — the
scope INCLUDES extra time (only the shootout is excluded), and the handler is 90-minute.
MC quantification (200k draws/config): ET adds +1.5–2.5 pts for typical R16 intensities.
Combined with the undersized comeback fudge (below) the handler is 4–12 pts LOW on every
leads question. At the knockout multiplier that is the largest per-question systematic
bias in the new templates.

### 3.2 ET_TEMPO / PEN_TILT — defensible, second-order

- `advance`: q_A ≈ 0.5 ± small tilt regardless of (c, PEN_TILT); errors are second-order.
  Grounding (839 ET matches, 678 shootouts) is real. PASS.
- `pen_shootout` = P(draw90)·P(ET drawn): first-order in P(draw90) (book-anchored ✓) and
  in P(ET drawn | c). Implied P(pens | drawn 90) ≈ .78–.81, matching the empirical .81.
  CAN–MAR priced .22 — inside the typical book to-pens band (.18–.23) for an even tie.
  WC26 heat/hydration argues c if anything LOWER → pen_shootout marginally UNDER-priced;
  magnitude ±1–2 pts. No per-match to-qualify/pens market exists in The Odds API to
  anchor to. PASS, monitor after first settled shootouts.

### 3.3 The six new templates, ranked weakest-first

1. **leads_at_any_point — WRONG, fixable, validated.** Current: p_first + 0.03.
   Empirical (martj42 goal minutes, 9,559 goal-matches since 2000): P(comeback to a
   lead | opponent scored first) = .114 per match → per-team increment ≈ +.05
   unconditional — the hardcoded +.03 is too small ON AVERAGE, and the MC shows the
   error is strength-dependent: crude vs truth-incl-ET at (la,lb)=(1.3,1.3): .493 vs
   .550; (1.7,1.0): .617 vs .692; (2.5,1.2): .689 vs .807. Favorites are under-priced
   by up to 12 pts. Fix: track ever-led in the MC engine (SimResult already loops the
   90 slots; add `led_a |= score_a > score_b` per slot — 3 lines + one field + one
   answer-map entry), plus the analytic ET term
   `P(tied FT)·(lam_t/lam_tot)·(1−exp(−ET_TEMPO·lam_tot/3))·…` or simply extend the sim
   30 low-tempo slots when the question is ET-scoped. Validate against the martj42
   led-at-any-point reconstruction bucketed by Elo gap before enabling.
2. **cards_gt_goals — double exposure to the stale cards level.** P(cards > goals) with
   cards mu ~12% high and WC26 goals/SoT running high is over-priced on both margins;
   independence of cards and goals is acceptable, the LEVEL is not. Self-corrects when
   Task 2 ships (the handler reads cm.mu). Also flag: settlement definition of "total
   cards" (is a 2nd-yellow-then-red 2 cards or 3? does a straight red count 1?) is
   unverified — one settled example will pin it; check before pricing aggressively.
3. **plays_full_match — flat 0.42 ignores position.** A GK or CB named in this template
   plays 90' at ~.95/.75 vs a forward's ~.30. The WC2022 lineup+event cache
   (data/raw/apifootball/fx*.json, already used by build_event_baselines.py) contains
   everything needed for position-conditional full-90 rates; measure four buckets and
   route by the player's position from player_rates_combined / lineups. Cheap; do it
   before a defender shows up in the template. Also: "entire match in regulation" = 90'
   scope confirmed (suffix stripped), so no ET correction needed here.
4. **total_exact — sound** (direct grid readout, anchored). Only defect is shared
   plumbing (3.4).
5. **goal_each_half — sound** (half-share thinning, independence across halves is exact
   for the Poisson field; DC correction second-order here).
6. **pen_shootout — sound** (3.2).

### 3.4 Shared plumbing defects for ALL new templates (cheap, ship together)

- **Not registered in GOAL_ARCHS/COUNT_ARCHS**: `pen_shootout`, `total_exact`,
  `goal_each_half`, `leads_at_any_point` are grid-derived goal-family questions, but
  `_family()` returns None → in `predictions_for` the `ctx.anchored and fam == "goal"`
  branch never fires, so Bot A applies the 25% league-average base blend ON TOP of a
  book-anchored grid — diluting exactly the market information the architecture says to
  trust. Fix: add the four to GOAL_ARCHS (calibrate() then also correctly skips Platt
  when anchored and applies the unanchored-goal Platt otherwise). Leave `cards_gt_goals`
  and `plays_full_match` out of COUNT_ARCHS until the count Platt is neutralized
  (Task 2 step 3), else they inherit the stale +0.18 up-shift.
- **Crowd clamp blind**: `crowd_prior.json` has no entries for any of the six — the
  25-pt catastrophic clamp cannot protect them. Harvest R16 crowd rows at first
  settlement and refit fit_crowd_prior; until then these templates run without the last
  safety net, which argues for shipping the leads/full-match fixes BEFORE R16 locks
  rather than after.
- **Half-time scope bug (Finding 1.1) is knockout-live**: every R16 match carries the
  "At halftime, will both teams have at least 1 SoT?" question at ~.78–.92 when truth
  is ~.55–.65. This alone is worth ~−0.3 to −0.5 Brier per match-day at 2×.

### 3.5 Rest days / rotation

No usable rest-day covariate exists in the repo and the book's team-level lines already
impound rest and rotation; the LLM panel covers confirmed-XI news for player props. An
explicit rest-day model would be an unvalidatable prior at 2× stakes. REJECT.

### 3.6 Interaction trap: the risk gate will EAT the parser fix unless bases move too

`both_teams_sot_ge` is in the LIVE risk gate at s=0.3 with `base_rates.json` = **0.70**.
After the halftime parser fix, the handler will correctly produce ~.62–.64 for a
first-half scope — and the gate will then pull it back to
`inv_logit(logit(.70) + 0.3·(logit(.64)−logit(.70)))` ≈ **.68**, silently undoing most
of the correction. Every settled both-teams question so far is half-scoped, so the .70
base (fit on mixed/mis-parsed history) is simply wrong for the live question population.
The parser fix MUST ship together with a base update (≈.62 for the half scope, from the
628-match-half empirical .643 shaded for WC26) or a gate bypass for half-scoped rows.
The same class of bug: any future re-fit of base_rates from settled rows classified via
matchday_predictions joins inherits Finding 0's subset bias.

---

## Task 4 — New ideas (with kill switches)

### 4.1 IDEA A — Harvest ACTUAL WC26 match counts from api-football; fit levels on counts, not binaries

**Mechanism.** The binary-outcome estimator (Task 2) squeezes level information out of
threshold indicators — ~1 bit per question. The repo already has
`ingest/apifootball.fetch_fixtures(league=1, season)` and `fetch_fixture_stats(fixture_id)`
(cached WC2022 copies prove the pipeline works: fx*.json). One fixtures call + one stats
call per played WC26 match (~88 so far, ~4/day thereafter) yields per-team Shots on Goal,
Corners, Yellow/Red Cards, Fouls, Offsides for EVERY played match. Fit
`s_f = (WC26 mean) / (historical-fit prediction at those matches' lambdas)` by NegBin GLM
with an offset — hundreds of count observations instead of ~20 binaries; the shrinkage
prior becomes nearly irrelevant because the data actually identifies s. Refit nightly in
daily_learn.

**Bonus (arguably the bigger prize): settlement-definition detection.** Join provider
counts to settled platform binaries: if api-football says a match had 3 cards and the
platform settled "4 or more total cards" YES, the platform counts cards differently
(2nd-yellow rule, red=2, etc.); same for the StatsBomb-vs-Opta SoT gap the offsides
family already demonstrated. This turns an unfalsifiable worry into a table with ~88
rows. Any systematic mismatch re-routes the fix from mu to the threshold mapping.

**Data requirement:** exists (key in .env, module in repo). Quota: ~89 requests one-off —
split over two days if the free-tier 100/day cap binds; prioritize the most recent 40
matches (they carry the time-decay weight anyway).
**Expected effect:** tightens s_f CIs by ~3× vs binaries; catches definition mismatches
that no amount of mu scaling can fix. Estimated +1.5–4 RBP/match-day on count questions
in R16+.
**Risk at 2×:** low — it feeds the same shrunk estimator; the circuit breaker
(|ln s| < ln 1.5) still applies. Provider-vs-platform count mismatch is a feature
(detected), not a risk (silently absorbed).
**Validation:** predict the settled binary outcomes from provider counts directly
(should be ~100% consistent where definitions match); then LOTO-CV as in Task 2.4.
**Verdict: VALIDATE-FIRST (one afternoon), then ship.**

### 4.2 IDEA B — The crowd as a WC26-conditions sensor (drift tripwire)

**Mechanism.** The crowd is demonstrably adapted to WC26 conditions where we are not
(cards: crowd .47 ≈ realized .462 while we sat .52+; SoT: crowd .62 vs our .52 with
realized 1.0). We already harvest per-question crowd consensus post-lock (336 rows,
growing every matchday). Maintain a rolling per-family gap
`mean(crowd_p) − mean(our_submitted_p)` over the trailing 20 questions; alert (print in
daily_learn + a row in the divergence audit) when |gap| > 6 pts sustained. This is a
DETECTOR, not a corrector — it would have flagged the cards drift by June 20 and the SoT
level within the first week, without waiting for a category to hit −7.8 RBP on the
platform's own scoreboard.
**Data requirement:** exists (crowd_consensus_raw.csv + harvest pipeline).
**Expected effect:** indirect — cuts detection latency of the NEXT drift from ~3 weeks
to ~3 days. In a finite tournament, detection latency IS the loss.
**Risk:** zero (alert-only). The corrective variant (auto-nudge family means toward the
crowd, capped at 30% of the gap) is +EV by the measured deviation-risk curve (5–15-pt
deviations from crowd are the +EV band) but overlaps Task 2; only enable it for families
where Task 2/Idea A have no data (e.g. a brand-new question family in the semifinals).
**Verdict: SHIP-NOW (alert), HOLD (auto-nudge).**

### 4.3 Ideas examined and KILLED

- **Rest-day / rotation covariate** — no data in repo, book impounds it, unvalidatable
  at 2×. Dead (3.5).
- **Book anchoring for ET/shootout questions** — The Odds API carries no per-match
  to-qualify / to-pens market on this plan; outright-derived proxies are stale and
  tie-ambiguous. The internal decomposition (grounded on 839+678 matches) is better than
  a bad anchor. Dead.
- **Referee-specific card factors for R16** — the repo's own cross-tournament backtest
  showed per-ref factors don't transfer (REF_SHRINKAGE=40 note in count_model.py);
  FBref career rates are Cloudflare-walled; api-football ref names exist but per-ref n
  in WC26 is 3–5 matches — pure noise at 2×. The family-level s_cards captures the
  directive effect without pretending to know referees. Dead.
- **Dispersion (k) refit alongside the level** — with 15–40 binaries per family the
  (mu, k) pair is unidentifiable; the prototype's k is inherited from 6 tournaments of
  actual counts. Only Idea A's real-count harvest could justify touching k, and even
  then var/mean in a 60-match sample is fragile. Not now.

---

## Task 5 — Ranked actions (expected RBP × confidence it generalizes)

| # | Action | Verdict | Expected impact | Confidence | Precise change |
|---|---|---|---|---|---|
| 1 | Halftime both-teams-SoT scope fix **+ base update** | **SHIP-NOW** | ~+15–20 RBP per settled instance (1/match, every match); stops a live −.18 Brier/question bleed | Very high (bug, not model) | parser.py: add `at halftime, will both teams have at least (\d+)/(\d+) or more shots? on target` → `both_teams_sot_ge, half=1` BEFORE the no-half patterns; set base_rates.json both_teams_sot_ge ≈ .62 (or bypass risk gate when half==1); grep open questions for other `^At halftime,` count phrasings |
| 2 | WC26 count-level recalibration (Task 2, cards+SoT) + count-Platt neutralization + base_rates regen | **VALIDATE-FIRST** (LOTO-CV is one script-run; target: before first R16 lock) | +1.5–4 RBP/match-day across ~4 count questions/match; removes the four-layer stale-level pile-up | High for direction, medium for magnitude | count_model.mu × wc26_level.json; fit_wc26_level.py (estimator §2.2); live_calibration count→[1,0]; base_rates regen; circuit breaker ±ln1.5 |
| 3 | leads_at_any_point: MC ever-led + ET term (replaces p_first+0.03) | **SHIP-NOW** (validation already done here: martj42 comeback rate .114/match ⇒ +.05 avg; MC shows fav bias up to 12 pts) | +3–8 RBP per leads question (1–2/match at 2×–3×) | High | mc_engine: track `led_a/led_b` in the slot loop, add to SimResult + answer map; runner routes leads_at_any_point via ctx.sim when available, analytic ET add-on `+P(tiedFT)·share·(1−e^(−ET_TEMPO·λt/3))`; keep the crude formula as no-MC fallback with +0.05 instead of +0.03 |
| 4 | Register pen_shootout / total_exact / goal_each_half / leads_at_any_point in GOAL_ARCHS | **SHIP-NOW** | +1–3 RBP/match-day (stops 25% base-dilution of anchored grid answers on lopsided ties) | High | runner.py GOAL_ARCHS set + one regression test that an anchored ctx passes through |
| 5 | Idea A: api-football WC26 actual-count harvest + definition cross-check | **VALIDATE-FIRST** | tightens #2 by ~3×; catches definition mismatches worth multi-RBP if present | Medium-high | scripts/harvest_wc26_counts.py using existing fetch_fixtures(1, 2026)/fetch_fixture_stats; join to settled binaries |
| 6 | Idea B: crowd-gap drift tripwire (alert-only) | **SHIP-NOW** | prevents the NEXT −7.8 category; no direct RBP | High (costless) | daily_learn: trailing-20 per-family mean gap print + divergence_audit row when >6 pts |
| 7 | plays_full_match position-conditional base | **VALIDATE-FIRST** (measure WC2022 rates first) | insurance: up to ±40 pts the day a GK/DF is named; ~0 until then | Medium | build_event_baselines: full-90 rate by position bucket from fx*/lineups; runner routes via player position |
| 8 | both-teams independence correction (×0.97) | REJECT for now | <1 RBP; second-order vs the scope fix | — | revisit only if #1+#2 leave residual |
| 9 | cards_gt_goals bespoke changes | REJECT | covered by #2 automatically | — | verify card-count settlement definition on first settled instance |
| 10 | Rest-day model, ET book anchor, per-ref cards, k refit | REJECT | — | — | §4.3 |

Sequencing note: #1, #3, #4, #6 are independent, small, and testable today. #2 is the
only one with modeling risk and has a specified acceptance test (LOTO-CV, §2.4); #5
strictly improves #2 and can land after R16 starts (refits are nightly). Do NOT ship #2's
Platt neutralization without the level factor — the stale up-shift is currently the only
thing keeping total-SoT from being even more under-priced.

---

## Appendix: evidence inventory

- Fresh /results pull 2026-07-05: 927 settled, 904 usable after p=50-ambiguity drops.
- Text-based archetype classification via parse_question (Finding 0 rationale).
- Historical structure: team_half_counts.parquet, 628 match-halves, 6 tournaments
  (SoT within-half cross-team corr −0.104; both-≥k empirical/product ratios .958–.978).
- martj42 goalscorers.csv live through 2026-07-03 (47,863 goals) — comeback-to-lead
  .114/goal-match since 2000; ET-only leads .007/goal-match.
- MC validations run at 200k draws/config for leads (crude vs true, 90' and +ET).
- Level-factor prototype: penalized weighted Bernoulli MLE, tau=.25, half-life 14d —
  cards s=.935 (unshrunk .878), SoT s=1.291 (unshrunk 2.014; prior load-bearing).
- Live-run exposure check: odds_anchor_run.csv 2026-07-04 (CAN–MAR): cards_ge 4+ at 50,
  pen_shootout .22, total_exact(1) .22; crowd_prior.json missing all six new archetypes.
