# QUANT REVIEW — ENDGAME SLATES (FRA–ENG bronze, ESP–ARG final)

*Senior-quant mechanism review, 2026-07-17. Scope: all 35 open questions, the new
handlers/routes, the six planned-but-unimplemented recipes
(docs/FINAL-NEW-KINDS-PLAN.md), and the crowd machinery. Method: the full slate was
re-priced OFFLINE through the real pipeline (`Runner.predictions_for` + the
run_matchday_odds post-processing stack, incl. the P1 clamp) at the plan doc's
devigged anchors — no platform or Odds-API calls. FRA–ENG reproduced the plan λ
exactly (1.636, 1.140); ESP–ARG deep-dive numbers were recomputed at the plan λ
(1.163, 0.889). All numbers below must be re-run at lock with the live anchor.*

**Headline: the mechanisms are mostly sound. The dangers are (1) five questions that
currently parse to NOTHING and would go unsubmitted, (2) two pipeline-interaction
traps that turn a correct handler number into a wrong submission (the
`both_teams_sot_ge` risk-gate pull to 0.54, the cards Platt+risk-gate lift), and
(3) one leak-family contrarian bet (offsides 5+) the clamp only half-catches.**

---

## 1. Verdict table — all 35 questions

Verdicts: **S** = SOUND, **SC** = SOUND-WITH-CAVEAT, **F** = FLAWED (as it would
ship). "Crowd" column: T = usable clamp template (n≥5 & σ≤12, or n≥3 & σ≤6),
H = crowd hug, B = book anchor expected at lock, — = nothing protects it.
"Now" = what the pipeline shipped in the offline reprice (Bot A, %).

### FRA vs ENG (3rd place, locks 07-19 01:00Z, ~15 questions)

| # | question | handler | now | verdict | crowd | the one thing to watch |
|---|---|---|---|---|---|---|
| 1 | 9+ total corners | `corners_total_ge` | 50 | SC | T(49.7) | WC26 knockout corners realized 0.286 (n=7) vs our avg 47.7 — the pooled count-Platt up-shift is suspect for corners too; 50 at the 9-line is the high side of defensible |
| 2 | Kane scores | `player_scores` | 40* | S | T+B+lift | *unanchored until props land; bronze ROTATION is the whole risk — lineup WebPanel must run at T-60 |
| 3 | Mbappé scores | `player_scores` | 39* | S | T+B+lift | same; book anchor + P4b (+2pts) at lock |
| 4 | each team 4+ SoT | **UNIMPLEMENTED** (plan §5) | skipped | F(route) | T(64.7)! | recipe is sound (~0.20–0.23) but BOTH available routes are traps — see §3.5. Risk gate would ship **0.54**, clamp proxy 64.7 would floor a clean 23 at ~40 |
| 5 | France wins Bronze Final | `advance` (new route) | 62 | SC | — (σ=14.7) | handler 0.624 vs book outright 0.64; parse verified (whole tie incl. ET+pens). Consider pinning to the book (63) via override |
| 6 | btts | `btts` | 55 | S | T(51.4)+B | anchored-grid passthrough; nothing to do |
| 7 | H2 > H1 goals | `h2_more...h1` | 49 | S | T(50.3) | MC timing family (edge k=1.3); realized 0.444, we're calibrated |
| 8 | Dembélé G+A | `player_ga` | 29* | SC | T+B(one-sided) | unanchored player_ga is the −4.0 leak; book prop at lock is the rescue. Club rate is low (0.41 GA/90) — check it reflects his current role |
| 9 | 4+ total goals | `total_ge` | 30 | S | T(51.2)+B | anchored grid |
| 10 | tied at HT | `ht_tied` | 46 | S | T(42.7) | realized 0.50 on n=14; fine |
| 11 | any player 2+ goals | `any_player_goals_ge` | 22 | S | T(27.5) | brace-curve over the anchored grid; realized 0/4 at crowd 27.5 — low side right |
| 12 | France scores first | `first_goal_match` | 55 | S | — (σ=14.2) | competing-exponentials ratio × P(≥1); realized 0.714 family, calibrated |
| 13 | sub scores or assists | `substitute_ga` | 50 | SC | — (n=2) | ×1.64 GA-widening is the least-validated constant; crowd prices ~38 and NO clamp/hug reaches this kind — see §3.7 |
| 14 | 20+ total shots | `total_shots_ge` | 68 | S | T(59.0) | real shots family; realized 0.667 — high side has been right |
| 15 | goal before HB1 | `goal_pre_hb1` | 37 | S | T(40.8) | window handler; realized 0.40 at crowd 40.8 — do not deviate |

### ESP vs ARG (FINAL, locks 07-19 23:00Z, ~3×, 20 questions)

| # | question | handler | now | verdict | crowd | the one thing to watch |
|---|---|---|---|---|---|---|
| 1 | penalty SCORED in regulation | **UNIMPLEMENTED** (plan §1) | skipped | S(recipe) | — | thinned-Poisson form is exactly right; ship 0.16–0.18. New archetype = NO safety net at 3× — see §3.2 |
| 2 | Spain 5+ SoT | `team_sot_ge` (rerouted ✓) | 38 | S | T(48.8) | edge family (+2.1); raw P=0.349 at plan λ; semis: both team-SoT lines settled NO under crowd 50/66. Keep the low side |
| 3 | Argentina 4+ SoT | `team_sot_ge` (rerouted ✓) | 47→~43 | S | T(48.8) | raw P=0.412 at plan λ (offline run was at a hotter synthetic total — re-run at lock) |
| 4 | goal before HB1 | `goal_pre_hb1` | 34 | S | T(40.8) | fine |
| 5 | first goal has an assist (No if 0-0) | **UNIMPLEMENTED** (plan §2) | skipped | S(recipe) | — | decomposition is self-validating (see §3.3); ship ~0.54, ≤0.58. Verify the 0-0 clause's ET scope |
| 6 | either team 2+ goals in same half | `either_team_half_goals_ge` (NEW) | 41→~33 | SC | **—** | independence over-prices by ~2pts vs the coupled MC (0.331→0.312 at plan λ); NO template, NO clamp — the only fully uncontested number on the slate. §3.4 |
| 7 | btts | `btts` | 49 | S | T+B | anchored grid |
| 8 | goal after HB2 | `goal_post_hb2` | 54 | SC | T(51.2) | stoppage mass IS included (slots fold to 89) ✓; family realized 1/5 — stay ≤ template+3 |
| 9 | 5+ total cards | `cards_ge` | 41 | **F(pipe)** | T(44.9) | raw structural P=0.279 is RIGHT (WC26 realized 0.300, our shipped avg 50.6 bled all tournament); the count-Platt up-shift + risk-gate pull to a stale 0.525 base manufactures 0.41. §3.6 |
| 10 | tied after 90 → ET | `extra_time` | 29 | S | — (n=2) | = anchored draw mass 0.314; the "finals run long" history argues against shading DOWN, not up; crowd ~33 |
| 11 | sub scores or assists | `substitute_ga` | 46 | SC | — (n=2) | as bronze #13 |
| 12 | 5+ offsides total | `offsides_total_ge` | 21→26 (clamped) | **F(level)** | T(51.4) | leak family; μ=3.1 comes from pre-semi-automated-offside data + a known StatsBomb definition gap; family realized 0.500 at crowd 51.4 (crowd calibrated). 26 is still ~20 under the likely crowd — house rule says don't fight a leak family. §3.6 |
| 13 | Yamal > Messi SoT (tie→No) | **UNIMPLEMENTED** (plan §3) | skipped | S(recipe) | — | race convolution + tie asymmetry is right; independence slightly UNDER-prices Yes (helps the planned up-shade). The crowd shrink MUST be in-handler — no template means the clamp physically cannot fire. §3.4 |
| 14 | Spain more corners | `corners_gt` | 55 | S | — (σ=14.6) | comparison ~coin; Platt'd count model; fine |
| 15 | 5+ distinct Spain shooters | **UNIMPLEMENTED** (plan §4) | skipped | S(recipe) | — | occupancy curve independently re-verified (see §3.5-b); ship ~0.85; the real risk is settlement ("attempt" incl. blocked?) |
| 16 | 10+ total corners | `corners_total_ge` | 39→~35 | S | T(49.7) | at plan λ ~0.35; knockout corners run low — low side fine |
| 17 | 22+ total shots | `total_shots_ge` | 56→~52 | S | T(59.0) | fine |
| 18 | Yamal G+A | `player_ga` | 39* | SC | T+B | book anchor at lock essential; note crowd OVER-prices GA (realized 0.125 vs crowd 36.7, n=32) — do NOT chase the crowd here, chase the book |
| 19 | Messi G+A | `player_ga` | 39* | SC | T+B | model rate (0.43 g/90) is far under WC26 form (1.36) — unanchored this is badly stale; the book prop is the whole answer. Props fetch MUST succeed |
| 20 | Argentina wins World Cup | `advance` (new route) | 42 | SC | — (σ=14.7) | handler 0.420 ≈ book 0.42, but the r-based shootout tilt is wrong-signed for ARG (s_pen 0.479 vs Martínez evidence ~0.55) — worth +1.7pts. §3.1 |

*Bot B note: the pure-model column (`p_bot_b`) is already better than Bot A on
cards (0.32 vs 0.41) — the damage there is all post-processing.*

---

## 2. What the just-settled semis actually say (653-row crowd file, tail 30)

Realized rates vs crowd vs our submissions on the archetypes present in these slates:

| archetype | n | crowd | realized | ours | read |
|---|---|---|---|---|---|
| cards_ge | 30 | 44.5 | **0.300** | 50.6 | we bleed HIGH; raw model (~0.28–0.32) was right all along |
| offsides_total_ge | 8 | 51.4 | **0.500** | 47.8 | crowd perfectly calibrated; our per-question model level is too low at high thresholds |
| team_sot_ge | 35 | 48.8 | **0.343** | 41.5 | our low-side edge is real (both semi team-SoT lines: crowd 50/66, settled NO) |
| player_ga | 32 | 36.7 | **0.125** | 29.5 | crowd hugely over-prices GA; the book, not the crowd, is the target |
| corners_total_ge | 7 | 49.7 | **0.286** | 47.7 | knockout corners run low; don't sit above template |
| goal_pre_hb1 | 10 | 40.8 | 0.400 | 35.3 | crowd calibrated; stay near it |
| goal_post_hb2 | 5 | 51.2 | 0.200 | 54.2 | small n, but don't sit above template |
| substitute_scores / _ga | 5 / 2 | 31.6 / 38 | 0.600 / 0.000 | 40.6 / 47.5 | mixed; tiny n — see §3.7 |
| advance | 14 | 58.1 | 0.643 | 58.3 | calibrated; crowd hedges outrights toward 50 (Spain-advance crowd 43, settled YES) — our book-anchored outrights beat a hedging crowd on BOTH sides |
| extra_time | 2 | 33.0 | 0.000 | 27.0 | fine |
| ht_tied | 14 | 42.6 | 0.500 | 47.0 | fine |
| any_player_goals_ge | 4 | 27.5 | 0.000 | 19.5 | low side right |
| first_goal_match | 7 | 61.4 | 0.714 | 62.1 | calibrated |
| both_teams_cards | — | 66 | YES (semi) | 64 | cards exist, they just don't reach 5 |

The semis signal for the final, condensed: **counts run LOW (SoT, corners, cards),
offsides run HIGH (semi-automated era), star attacking props settle NO more than the
crowd thinks, favorites advance.** Our model agrees with all of that except
offsides — which is exactly where we should not be contrarian (measured −2.6 leak).

---

## 3. Deep dives

### 3.1 The outright route (`advance`): mechanism review

**Parse**: both outrights verified to hit the new pattern before the generic `win`
catch-all — "Will France win the third-place match (Bronze Final)?" and "Will
Argentina win the World Cup?" both parse to `advance` with the right subject
(offline confirmation + tests/test_parser.py:174). The 15-pt win-in-90 parser trap
is closed. Both questions are attached to their own match, so whole-tie semantics
are correct by construction.

**Decomposition at plan λ** (verified against the code path):

| | w90 | ET win | ET draw | s_pen (code) | advance | book |
|---|---|---|---|---|---|---|
| France | .483 | .153 | .744 | .530 | **.624** | .64 |
| Argentina | .272 | .086 | .799 | .479 | **.420** | .42 |
| Spain (mirror) | .414 | .115 | .799 | .521 | .580 | .58 |

The decomposition lands on the book outright to within 1–2pts on all three legs —
strong external validation that ET_TEMPO and the ET grid are not distorting the
post-90 mass (market-implied s_pen for ARG ≈ 0.50; code ET-drawn 0.80 vs the
market-consistent ~0.75–0.80).

**ET tempo (0.35)**: grounded on 839 ET matches; WC26 runs hot (4/8 ETs decided) but
sensitivity is negligible — raising decisiveness to the pooled 28% moves the ARG
outright by <0.2pt, because what the ET-win leg gains the shootout leg loses.
Not worth touching.

**The favorite-tilt shootout model is the one real flaw.** `s_pen = 0.5 +
0.2·(r−0.5)` with r the 90-minute win-odds ratio makes the shootout an echo of
run-of-play strength. Mechanically that is the wrong generative story: a shootout is
GK + taker + nerve, nearly decoupled from open-play dominance. The empirical
favorite tilt (0.49→0.57 over 678 shootouts) justifies the FORM of the term, but for
this final the identifiable signal points the OTHER way from r: Argentina are the
90-minute dog (r=.397 → s_pen=.479) yet hold the one shootout asset with any
persistence in the literature (the keeper: Martínez-era 4/4, ARG 5 straight) while
Spain are 2/6 since 2018. The market itself prices the ARG post-90 leg at a coin
flip. The plan's override table `{ARG: 0.55, ESP: 0.45, FRA/ENG: 0.50}` is the right
bounded fix: worth **+1.7pts on the ARG outright** (0.420→0.437), capped at +2 over
the anchor, with the ET-fatigue asymmetry (ARG two ETs played, ESP zero) as the
argument against going further. For France the r-tilt (0.53) and the evidence
(ENG 3-of-4 shootouts since 2018, Mbappé's stutter-miss) roughly cancel: 0.50 is
right and changes the number by <0.5pt — France's edge case doesn't need the
override, Argentina's does.

**3rd-place dynamics**: rotation and motivation live almost entirely in the anchored
λ (w=0.85 to the book, and books price bronze rotation); the 15% model weight uses
full-strength Elo, a small upward distortion on both teams that mostly cancels in
the win probabilities. The bronze-specific empirics in the plan (bronze games are
open, team SoT volume unchanged) support NOT applying any manual haircut. The
rotation risk that actually bites is **player props** (Kane/Mbappé/Dembélé
minutes), which is the lineup-override machinery's job, not the λ's.

### 3.2 Penalty SCORED (plan §1) — and the crowd-hug question

**Estimator structure: correct.** `1 − exp(−λ_pen · conv)` is exactly the thinned
Poisson for "≥1 of a Poisson number of pens converts", and it handles multi-pen
matches for free. The taker-mix (`P(taker on pitch) × taker conv + backup conv`)
is the right composition; taker-off-pitch risk is bounded because the Oyarzabal-off
branch already reverts to the 0.75 field rate. λ_pen = 0.206 has three independent
WC26 checks (21/102 direct, 14 scored /0.75, our settled 2/11) — that constant is
the best-sourced number in the plan.

**Should it inherit `penalty_awarded`'s CROWD_HUG? No — and the reasoning matters.**
The hug exists for kinds where our model is a stale CONSTANT and the crowd is
near-constant: there we have no information, so we submit the field level and bank
the Jensen field-variance bonus. Here we have genuine, current information the
crowd likely lacks (WC26's pen-rate collapse to ~60% of historical + the conversion
haircut + Messi's two misses). Hugging would destroy exactly the edge the question
offers. The settled evidence supports the low side: `pen_or_red` realized 0.226
while the crowd priced 0.386 — the crowd systematically OVER-prices pen-family
events this tournament.

**How contrarian to be at 3×** (crowd will likely price 0.25–0.30; `penalty_awarded`
template is 29.4 and the average forecaster will not haircut for conversion):

| true p | RBP/q vs crowd@0.28 if we submit 0.16 |
|---|---|
| 0.16 | **+1.4** (+4.3 at 3×) |
| 0.20 | +0.5 |
| 0.22 | ~breakeven |
| 0.25 | −0.7 |

Break-even truth is ~0.22; three measurements put truth at 0.15–0.17. The bet is
solidly +EV as planned. Ship **0.16–0.18** (0.17 buys robustness against the crowd
sitting at 0.24 rather than 0.30). **The naked-archetype problem**: a new archetype
has no template, no clamp, no hug — a transposed constant would ship raw at 3×.
Add a hard assert in the handler (`0.10 ≤ p ≤ 0.30`) or ship via a printed
expert-override so a fat-finger is loud.

**Settlement**: verify the rebound wording; a pen scored in ET settles No
(regulation scope) — consistent with using the regulation λ only.

### 3.3 First goal credited with an assist (plan §2)

**The exchangeability worry dissolves in the plan's own numbers** — worth making
explicit because it's the main structural assumption. Question: is
P(assisted | open-play, FIRST goal) = P(assisted | open-play, any goal)? Check:
unconditional first-goal-assisted was measured directly at 0.587 (n=288), and the
composition (1 − pen .108 − OG .080) × all-goal assisted_open .722 = **0.586**. The
direct measurement and the composition agree to 3 decimals, i.e. first open-play
goals are assisted at the same rate as all open-play goals. Early/late type-mix
differences flow entirely through the pen/OG shares — which the plan correctly
re-centers for WC26/final (pen share ↑ to 0.08 in a low-tempo final because pens
don't scale with tempo; that direction is right and subtle).

Structure: `(1−p00_anchored) × (1−pen−og) × assisted_open` — every leg is either
market-inherited or measured. **0.54 is honest and nearly crowd-neutral; ship it.**
Two checks before lock: (i) the 0-0 clause — if "ends 0-0" means after ET, the
first factor is 1 − p00·e^{−λ_ET} ≈ 0.880 not 0.8645 (+1pt, still ≤0.55);
(ii) provider assist-credit convention (deflections/dribble-fouls) is irreducible
noise — the plan's ≤0.58 cap is the right response.

### 3.4 The race and the new half-goals handler — the two "no-net" numbers

**`either_team_half_goals_ge` (new handler, runner.py ~line 942).** Structure:
1 − Π P(cell < thr) over 4 independent Poisson team-half cells at measured half
shares (.444/.556). Mechanism findings:

- *DC correction*: correctly ignorable — ρ_DC moves joint low-score mass, and a
  single team-half ≥2 tail is a marginal event. Verified immaterial.
- *Independence*: tested against the house's own state-coupled MC (200k draws,
  fitted score-state multipliers) at plan λ. **Analytic 0.331 vs MC 0.312 (final);
  0.493 vs 0.470 (bronze).** Independence OVER-prices by ~2pts. The MC's cross-half
  within-team correlations are mildly negative (−0.01/−0.02: game-state easing —
  a team 2-up stops pushing) and same-half cross-team correlation is positive
  (+0.07: open halves are open for both), both of which shave the union. The
  missing overdispersion (team-goal frailty) pushes the other way but is mostly
  absorbed by the anchored λ.
- *Direction at the final's tempo*: our shipped number should be ~0.31, not 0.33.
  Small — but this is also the ONLY question on either slate with **no template, no
  clamp, no book market and no hug**: an uncontested number at 3×. The cheapest
  robust fix is to route it through `ctx.sim` (the MC result is already computed at
  the same λ; `ht_a/ht_b/score_a/score_b` give the answer in one line), which
  bakes the −2pt coupling correction in for free and removes the hand-derivation.

**Yamal > Messi SoT (plan §3).** The convolution race with per-player minutes
mixtures is the right object, and the tie→No asymmetry (~0.19 tie mass at these
means) is correctly load-bearing. Two mechanism notes, both supporting the plan's
upward shade from the raw 0.24:

- *Game-state coupling between opposing stars is NEGATIVE* (one team chasing =
  the other defending), and negative correlation between Y and M FATTENS both tails
  of Y−M: it raises P(Y>M) as well as P(M>Y) and shrinks the tie mass. Since the
  question pays on the small tail, independence slightly UNDER-prices Yes.
- *Minutes*: Messi's late sub-off when ARG lead is a real scenario the mixture
  underweights (his exits cluster in exactly the game states where Yamal's team is
  chasing with more shot volume). Same direction.

Together with the documented SoT under-call bias (realized 40% vs predicted 29% on
player SoT), the plan's **0.28–0.31 with the crowd blend baked into the handler**
is right. Critical wiring point (mandate check c): **the P1 clamp cannot fire for
this archetype** — `crowd_clamp` requires a `crowd_prior.json` entry and a new kind
has none — so the 0.65 model / 0.35 crowd-50 logit blend MUST be inside the
handler's own output (or the override value). Nothing downstream will do it.
Also: a new archetype has family=None → no Platt, no base blend, no risk gate —
the handler's return value ships verbatim. Lineup dependency: if Messi is out, the
answer jumps to P(Yamal ≥ 1) ≈ 0.7 — wire the T-60 re-price.

### 3.5 The two joint/occupancy recipes

**(a) Each team 4+ SoT (plan §5).** The estimator is right, and the plan's
residual-correlation argument deserves an explicit yes: the raw joint/product ratio
(0.83) double-counts because the anchored marginals already encode matchup
heterogeneity (mismatches produce one-sided SoT); after conditioning on the model μ
the residual indicator corr is −0.076 → ρ_ind ≈ −0.05, i.e. **subtracting ~1.2pts
is coherent** — the remaining negative dependence is genuine within-match
possession zero-sumness net of shared-tempo effects. Marginals at plan λ verified:
P(FRA≥4)=.518, P(ENG≥4)=.389 → joint ≈ 0.19–0.20, shade up to ~0.23 for the SoT
under-call history.

**But both available implementation routes are traps (this is the biggest
found-bug of the review):**

1. The question does not parse — "each team **record** 4 or more" matches no
   pattern (verified: it lands in the template audit as unparsed). So today it
   would simply not be submitted.
2. If a parser pattern routes it to the existing `both_teams_sot_ge`, the pipeline
   MANGLES it: full trace at bronze λ shows the raw handler answer 0.248 shipping
   as **0.56**, because (i) `apply_risk` (risk_gate 0.3, RISK_GATE_ENABLE=1) pulls
   toward `base_rates["both_teams_sot_ge"] = 0.70` — a base measured at 1–2-SoT
   thresholds, threshold-blind — and (ii) the clamp proxy (template mean 64.7,
   also low-threshold rows) would floor any clean low number at ~40 anyway.
   Submitting ~0.55 on a ~0.20 event at the bronze would be a bug-class bleed.

Fix (code-level, small): give the joint-tail its own archetype (e.g.
`each_team_sot_ge`) with the ρ=−0.05 copula in-handler, NOT in `risk_gate.json`,
NOT inheriting the `both_teams_sot_ge` template; or route to `both_teams_sot_ge`
but gate `apply_risk` and `crowd_clamp` on threshold ≤2 for this archetype. The
first is cleaner. (The general lesson — `base_rates` and templates are
threshold-blind while questions are not — also produced the cards finding in §3.6.)

**(b) 5+ distinct Spain shooters (plan §4).** Independently re-verified the
occupancy curve on `player_match.parquet` (627 team-matches): population
P(≥5)=0.842 (plan 0.841 ✓); conditional P(≥5|N) matches the logistic at N=8/10/12
(.857/.953/1.0 vs curve .809/.958/.992). The sufficient-statistic claim (Spain
residual ≈ 0) held up. One honest wrinkle: at **N=6 the curve over-predicts**
(measured .375 vs curve .438) — the region that matters only if ARG parks AND
Spain manage <8 shots; the anchored shots pmf gives that tail little mass, and the
plan's shade from 0.876 to 0.85 covers it. Mechanically, the blowout-vs-tight
shooter-spread worry is second-order because the pmf mixing already moves the
volume; what it can't see is WHO shoots in a tight final (star concentration at
low N) — which is the same N≤7 tail. 0.85 is right. The dominating risk is
settlement: if the provider excludes blocked shots, the effective N drops ~25% and
the answer is ~0.80 — resolve the definition before lock, not after.

### 3.6 The two leak-family count questions (mandate check d)

**cards_ge 5+ (final): the raw model is right and the pipeline breaks it.**
Anatomy at plan λ: raw NegBin P(≥5)=0.279 (cards μ doesn't scale with λ — correct,
the backtest showed λ covariates hurt cards). Then: pooled count-Platt
(a=.972, **b=+0.18**) shifts it up — the b was earned by corners/SoT settling high,
and cards inherit it exactly backwards in a low-cards tournament (the runner's own
comment at COUNT_ARCHS acknowledges this and exempts `card_each_half` /
`card_stoppage`, but left `cards_ge` in). Then the risk gate (s=0.5) pulls toward
`base_rates["cards_ge"]=0.525` — a six-tournament base that WC26 (realized 0.300,
n=30, our shipped avg 50.6 vs crowd 44.5) has flatly contradicted. Net: 0.279 →
0.41 shipped. The irony: the risk gate was built to stop unearned deviation from
base, and here the BASE is the unearned deviation. The crowd at the 5-line will be
~38–42; truth is ~0.28. Bounded fix for this slate: pin cards_ge 5+ at **~0.33–0.36**
(override or exempt cards_ge from the count Platt like its sibling kinds and
re-center its risk base to the crowd-template level 0.449). Do not ship 0.41.

**offsides_total_ge 5+ (final): the opposite problem — the raw model is the leak.**
Model μ_offsides=3.1 → P(≥5)=0.218, clamped by P1 to 26 (template 51.4−25). But the
family realized 0.500 on WC26 (n=8) with the crowd calibrated at 51.4 — our μ comes
from six pre-semi-automated-offside tournaments PLUS a known StatsBomb→platform
definition gap. The 2026 tech flags every marginal offside; the level is simply
higher than our data. This is a measured −2.6 leak family and the house rule is
explicit: do not deviate from the crowd outside edge families. The clamp's 26 is
still a ~20pt contrarian bet. The risk gate can't help — only `offsides_ge` (team-
scoped) is in `risk_gate.json`, `offsides_total_ge` was never added (gap). Fix:
override to **~0.40–0.45** (below the crowd's likely ~48–50 respects the 5-vs-4
threshold, above the model respects the regime), and add `offsides_total_ge` to the
risk gate keyed to the template level for the future.

### 3.7 substitute_ga (both slates) — the quiet 12-point deviation

Mechanism: base P(sub scores)=0.30 (WC22, 40 matches), converted to a hazard,
λ-scaled (sound — sub goals track match tempo), then widened ×min(GA/G ratio,2)=1.64
for "or assist". The widening is the weak link: it assumes sub ASSISTS arrive at the
population GA/G ratio, but subs are disproportionately forwards brought on to
finish, not create; the ratio for subs specifically is plausibly 1.3–1.5, not 1.64.
Output: bronze 0.51, final 0.41–0.46, vs a crowd template of 38 (n=2, σ=1.4 — too
few rows for the clamp, and `substitute_ga` is deliberately not in CROWD_HUG_KINDS
because it λ-scales). Settled evidence is a coin toss (sub_scores 3/5 realized at
crowd 31.6 — crowd too LOW; sub_ga 0/2). The 5-sub era supports the high side
structurally. Verdict: keep the handler, but at the FINAL (3×, low tempo) shade the
GA widening toward 1.5 or cap the submission at ~0.45; at the bronze 0.50 is
defensible (high tempo, heavy rotation = more sub minutes).

### 3.8 Crowd-machinery coverage audit (mandate checks a–c, consolidated)

Of the 30 currently-priced questions, 24 sit under a usable clamp template; the
exceptions and what actually protects them:

| question | template? | actual protection |
|---|---|---|
| both outrights (`advance`) | entry exists but σ=14.7 > 12 → clamp OFF | book-anchored λ (the decomposition ≈ the book outright to 1–2pts) |
| `either_team_half_goals_ge` | none | **nothing** — see §3.4 |
| `extra_time` | n=2 | anchored draw mass = book X |
| `substitute_ga` | n=2 | **nothing** — see §3.7 |
| `first_goal_match`, `corners_gt` | σ>12 | anchored grid / Platt'd count model (both calibrated on settled data) |
| all six NEW kinds | none | **nothing** — every shrink/clamp must be in-handler or via loud overrides |

And one wiring fact that matters for implementation: **`expert_overrides.json`
cannot rescue an unparsed question** — overrides apply to rows the runner priced,
and unparsed questions never reach that frame. The five unimplemented kinds
need at minimum a parser pattern + a handler returning a number; the overrides can
then pin the value. (Also: the daily-learn template refit after the bronze settles
will not create templates for the final's new kinds — they've never been seen.)

---

## 4. Ranked action list

### MUST-FIX before the bronze lock (07-19 01:00Z)

1. **Implement the five unimplemented kinds** (parser pattern + handler each;
   constants from the plan doc, re-run at the live anchor). Unparsed = unsubmitted;
   forfeiting 5 of 20 final questions at ~+2.5–3 RBP/q baseline × 3 is the single
   largest expected-value item on the board. Priority order as in the plan
   (outrights are already live — then §5 each-team SoT, §1 pen scored, §3 race,
   §2 assist, §4 shooters).
2. **Defuse the `both_teams_sot_ge` routing trap before adding the "each team
   record 4+ SoT" pattern** (§3.5-a): new archetype `each_team_sot_ge` with the
   in-handler −0.05 copula, excluded from `risk_gate` and from the 64.7 template
   proxy. Verified: the naive route ships 0.56 on a ~0.20 event.
3. **cards_ge 5+ (final): do not ship 0.41.** Exempt `cards_ge` from the pooled
   count-Platt (mirror the `card_each_half` decision) or pin 0.33–0.36 by override.
   Evidence: WC26 realized 0.300 vs our shipped 50.6 (n=30).
4. **offsides_total_ge 5+ (final): do not ship 26.** Override to 0.40–0.45 (leak
   family; realized 0.500 at crowd 51.4; semi-automated-offside regime our data
   pre-dates). Add `offsides_total_ge` to the risk gate for the future.
5. **s_pen override table in `advance`** `{ARG: 0.55, ESP: 0.45, else 0.50}` —
   +1.7pts on the ARG outright (0.420→0.437); the r-based tilt is wrong-signed for
   the one team whose shootout asset (Martínez) is the persistent kind. Bounded,
   consistent with the book (+≤2 over anchor).
6. **Yamal–Messi race: bake the crowd shrink into the handler output**
   (0.65/0.35 logit toward 0.50 → ship 0.28–0.31). The clamp cannot fire for a
   template-less archetype — nothing downstream will save a raw 0.24. Wire the
   T-60 lineup re-price.
7. **Ops, both matches**: (i) player-props fetch MUST succeed at lock — 6 of the
   35 are player props and the book anchor is the entire defense of the player_ga
   leak (Messi's model rate is 3× under his WC26 form); check Odds-API credits
   now, bronze prop coverage is usually thinner; (ii) near-KO lineup WebPanel run
   for the bronze (rotation: Kane/Mbappé/Dembélé caps); (iii) re-run every plan
   worked number at the live devig (the plan's are 07-16/17 openers).

### MUST-FIX before the final lock (07-19 23:00Z)

8. **Pen-scored: ship 0.16–0.18 with a hard sanity bound in code** (0.10–0.30
   assert or loud override) — correct to NOT crowd-hug (§3.2), but a naked new
   archetype at 3× needs a fat-finger guard.
9. **Re-verify settlement definitions**: pen rebounds; "attempt a shot" incl.
   blocked (moves §4 recipe 0.85→0.80); assist-credit provider convention; the
   0-0 clause's ET scope (§3.3).

### NICE-TO-HAVE

10. Route `either_team_half_goals_ge` through `ctx.sim` (one mapping entry; the
    MC already computes it) or apply a −2pt coupling haircut: 0.33→0.31 at plan λ.
11. `substitute_ga`: cap the GA widening at 1.5 (or cap the final's submission at
    0.45); bronze can stay ~0.50.
12. Pin the France outright to the book (0.63) rather than the decomposition's
    0.62 — the book has bronze-rotation information the model lacks.
13. corners_total_ge: sit 2–3 under the model at the 9/10 lines (knockout corners
    realized 0.286, n=7) — low-conviction, template-bounded either way.
14. Endgame extremize dial: currently unset (α=1.0, verified inert). Run
    `scripts/endgame_policy.py` against the live deficit before deciding; if armed,
    it touches only `team_sot_ge` on these slates among priced questions — the two
    questions where our low-side edge is best supported. Do not arm anything wider.
15. Post-final: fold the threshold-blindness lesson (base_rates/templates keyed by
    archetype only) into the risk gate and clamp — this produced both §3.5-a and
    §3.6, and it will recur wherever thresholds drift from the historical mix.
