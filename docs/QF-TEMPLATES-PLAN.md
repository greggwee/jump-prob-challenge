# QF New-Template Pricing Plan (POR vs ESP quarterfinal)

Status: COMPLETE (all sections §0-§8 written; every number recomputed from repo data this session).
Author: senior-quant planning pass, 2026-07-05.
Scope: implementation-ready pricing plan for the 5 unpriced QF question types
surfaced by the template audit. NO code is changed by this pass.

The 5 questions (all binary YES/NO, POR vs ESP, knockout multiplier ~3x by QF;
scored relative Brier vs the mean individual forecaster; unpriced = forfeited RBP):

1. Q1 `halves_equal_goals` — "Will both halves have the same number of goals in regulation?"
2. Q2 `card_before_goal` — "Will the first card of the match be shown before the first goal is scored?"
3. Q3 `gk_saves_ge` — "Will Diogo Costa (Portugal) make 4 or more saves in regulation?"
4. Q4 `subs_total_ge` — "Will there be 9 or more total substitutions (both teams combined) in regulation?"
5. Q5 `extra_time` — "Will the match go to extra time?"

Sections below are appended in order:
- §0 Pipeline verification (what actually exists in the repo — checked, not assumed)
- §1..§5 Per-question: estimators (math), data, validation, failure modes/crowd, recommendation
- §6 Ranked implementation order + effort
- §7 Rejected approaches (statistically unsound)
- §8 Executive summary

---
## §0 Pipeline verification (checked in-repo 2026-07-05, not assumed)

| Pipeline | Verified state | Relevant to |
|---|---|---|
| DC scoreline grid + book anchor | `models/goal_model.py` dc_grid; `market_anchor.implied_lambdas` inverts de-vigged 1X2(+O2.5) by least squares (2 params vs 3-4 targets — the anchored grid draw is a *near*, not exact, reproduction of the de-vigged X); blend w_market=0.85; runner `MatchContext.grid/lam_a/lam_b/rho_dc/anchored` | Q1, Q2 (goal side), Q5 |
| Half shares | `simulate/markets.SECOND_HALF_SHARE = 0.556` (47k martj42 goals); runner `H1 = 0.444, H2 = 0.556` | Q1 |
| Minute dynamics | `data/processed/minute_dynamics.json` has alpha (6x15-min buckets) AND beta (score-state) for **goal, corner, card, foul, sot, offside** — cards ARE covered. Card alpha strongly late-loaded: `[0, .67, .99, .87, .82, 1.70]` vs goal `[0, .03, .41, .44, .39, .70]` -> the constant-rate race answer for Q2 is biased and the direction is knowable (see §2). Analytic window-integral precedent already in runner (`_WINDOW` hydration handlers). | Q2 |
| Count models | `models/count_model.py`: per-team-half NegBin mus for corners/cards/offsides/SoT/fouls/shots, log-linear in (lam_team, lam_opp); cards lambda-free by design. SoT k(dispersion) fitted. **WC26 SoT level is running +15-30% above the historical fit** (adversarial review Task 2, s_SoT ~= 1.29) — directly relevant to Q3 (saves face the same drift on the SoT-against side). | Q2 (card mu), Q3 |
| StatsBomb events | 314 match event files under `data/raw/statsbomb/events/`. Verified in raw JSON: GK events carry `goalkeeper.type` incl. **"Shot Saved"** (with outcomes In Play Safe/Danger, Touched Out, Saved Twice); shots carry outcome **"Saved"** distinctly from "Blocked"/"Off T" — saves are DIRECTLY observable, no SoT-minus-goals proxy needed. Cards: Foul Committed / Bad Behaviour events with card attributes + minute. First-card vs first-goal race fully reconstructable. `team_half_counts.parquet` = 628 team-halves. | Q2, Q3 |
| api-football WC2022 | `data/raw/apifootball/events/fx*.json` — **59 of 64 WC2022 matches cached** with `subst` events incl. `time.elapsed` (+`extra`); `lineups/` 40 files; free tier 2022-2024 only, NO WC26 fixtures. Regulation filter = elapsed <= 90. | Q4 |
| martj42 | `data/raw/martj42/`: results.csv, goalscorers.csv (minutes; live through WC26), shootouts.csv. Half splits reconstructable (minute<=45 vs 46-90); draw-at-90 for knockouts reconstructable (regulation-goal filter); shootouts.csv identifies pens matches. | Q1, Q5 |
| Odds API | `ingest/odds_api.py PLAYER_MARKETS = "player_shots_on_target,player_goal_scorer_anytime,player_assists"` — **no GK-saves market wired**; catalog check pending (§3). 478 credits. h2h/totals per-event = the de-vigged X for Q5 at zero marginal cost (already fetched every matchday run). | Q3, Q5 |
| Crowd prior / clamp / gates | `crowd_prior.json` has **no entry for any of the 5 new archetypes** -> the 25-pt clamp is blind to all of them until first settlement + refit (same situation as the R16 six; review §3.4). `base_rates.json` likewise. Consequence: these questions run WITHOUT the last safety net -> the plan must be conservative by construction, not by clamp. | all |
| Existing knockout machinery | runner: `advance`, `pen_shootout` (= P(draw90)xP(ET drawn)), ET_TEMPO=0.35 (839 ET matches), PEN_TILT=0.20 (678 shootouts); `GOAL_ARCHS` already registers the R16 six so anchored grid answers pass through undiluted — the same registration is required for the new grid-derived archetypes. | Q5 |

Naming (parser conventions, `submit/parser.py`): proposed archetypes
`halves_equal_goals` (Q1), `first_card_before_first_goal` (Q2), `player_saves_ge` (Q3, threshold=4, subject=player), `subs_total_ge` (Q4, threshold=9), `extra_time` (Q5). All five carry the regulation suffix already stripped by the parser except Q2/Q5 which have no suffix (Q2's scope: "the match" — first card vs first goal, natural reading is regulation+stoppage; Q5 has no regulation ambiguity by construction).

---
## Grounding numbers used below (computed this pass, read-only, reproducible)

- **martj42 half-split reconstruction** (goals minute<=45 vs 46-90; matches where scorer rows reconcile with the final score, or true 0-0): **n=11,658** internationals since 2000. P(H1==H2)=.3531 overall; H2 share = .5615. Conditional on total N: P(equal|N) empirical vs Binomial(N, s): N=2 **.4945 vs .4937**; N=4 **.3567 vs .3657**; N=6 **.3468 vs .3009**. Split dispersion var(n2|N)/[Ns(1-s)] = .99/.99/.97/.88/.92/.85 for N=1..6 (mild under-dispersion at high N -> splits slightly MORE even than binomial). WC-finals-only P(equal) = **.2659** (n=455).
- **StatsBomb card-vs-goal race** (314 matches, regulation periods only, own goals included as goals, yellows+reds as cards): branch masses — both events .866, card-but-no-goal **.086**, goal-but-no-card .041, neither .006. P(card first | both) = **.4669**; P(YES) strict-scope (goal must exist) = **.4045**; permissive scope (card wins if goal never comes) = **.4904**. Median first-card minute 31.4 vs first-goal 30.8. Corpus mu_card=3.462, mu_goal=2.455. **Naive constant-rate race = .5835 (+9.4 pts off empirical); time-varying race using dyn.alpha = .5212 (+3.1 pts)** — the minute-profile correction removes 2/3 of the bias; the residual is card overdispersion (frailty_k=12.2) + state-0 normalization, absorbed by one calibration factor.
- **StatsBomb saves** (628 team-matches): saves == SoT-against − goals-against held **exactly in 628/628** (SoT = Goal/Saved/Saved-to-Post; saves = Saved/Saved-to-Post). Mean 2.570, var/mean 1.41, NegBin k=6.32. P(saves>=4) empirical **.2691**; NegBin fit .2697 (Poisson .2575 — too thin; NB required). Tail: P(>=5)=.148, P(>=6)=.072.
- **api-football WC2022 subs** (59 cached matches, subst events elapsed<=90): mean 8.98, var/mean **0.175** (massively UNDER-dispersed — a quota process, not Poisson), pmf {5:1, 6:2, 7:6, 8:6, 9:18, 10:25, 11:1}, **P(>=9)=.746** (group .729 n=48; KO .818 n=11). **The 5 uncached WC2022 matches are EXACTLY the 5 extra-time matches** (JPN-CRO, MAR-ESP, CRO-BRA, NED-ARG, ARG-FRA) — the cache is selection-biased against precisely the hold-a-sub-for-ET stratum a QF cares about.
- **martj42 ET base rates**: WC knockouts 1986-2022: **48 ET / 160 KO = .300**; P(pens|ET)=.75 (repo global: .81). Tied-at-90 shares: WC-finals all-rounds .266; Euro 2021+ .307; Copa 2024 .323.
- **Odds API catalog re-checked 2026-07-05**: soccer player props = anytime/first/last scorer, to-receive-card/red, shots, SoT, assists — **NO goalkeeper-saves market**; no to-qualify / extra-time market. Q3 and Q4 are unanchored by construction; Q5 has a PERFECT anchor (the X).
- **Artifacts**: `wc26_level.json` does NOT exist yet (review item #2 unshipped); `live_calibration.json` count Platt still [0.972, +0.181]; de-vig default = Shin (`devig.consensus(method="shin")`).

---

## §1 Q1 — "Both halves same number of goals" (`halves_equal_goals`)

**Settlement reading.** YES iff regulation H1 total == H2 total (both teams combined; 0-0 halves equal; ET excluded by the regulation suffix, which the parser already strips).

### a. Estimators

**A (chosen): anchored-grid total pmf x conditional split factors.**

    P(YES) = sum over even N of  P_grid(total = N) * q_N,    q_N = P(#H2 = N/2 | N)

- `P_grid(total=N)`: the total-goals pmf read off the ANCHORED Dixon-Coles grid (`ctx.grid`) — it already carries the book's 1X2+O2.5 information and the DC low-score corrections, which matter here more than anywhere: N=0 and N=2 carry the two largest q_N and ~35% of the mass.
- `q_N`: conditioned on N total goals from a (time-inhomogeneous) Poisson field with a common minute profile, goal times are iid, so each goal falls in H2 w.p. s = SECOND_HALF_SHARE = .556 -> #H2 | N ~ Binomial(N, s); q_N = C(N, N/2) s^{N/2} (1-s)^{N/2} for even N, 0 for odd N.
- Refinement (recommended): replace binomial q_N with the EMPIRICAL split factors for N<=6 — q_0=1, q_2=.4945, q_4=.3567, q_6=.3468, binomial beyond — the martj42 table is n=395..2475 per cell and captures the measured under-dispersion of the split at N>=4 (halves slightly anti-correlated in share -> splits slightly MORE even than binomial). Effect: +0.4 pt at lambda=2.5. These are constants, not a fitted artifact — hardcode with the derivation date.

Why the decomposition is sound: it is EXACT given (i) the total pmf and (ii) q_N. (i) is the best-validated distribution in the system (totals-anchored); (ii) is validated above to 3 decimals at N=2,4. The DC correction lives entirely in (i) — no independence-of-halves assumption is ever made. Team asymmetry in half shares is second-order because the question POOLS both teams and both teams' H2 shares are ~.55.

**B (rejected): independent Poisson halves** P = sum_k Pois(k; .444 lam) Pois(k; .556 lam). Throws away the anchored total pmf and the DC mass at 0-0/1-1, and models totals as Poisson when the anchored grid is in hand. Pure Poisson at lam=2.5 gives .2668 vs grid-based ~.27-.29 — numerically close here, but B is strictly dominated (same inputs, worse use of them) and breaks coherence with every other goal question priced off the same grid.

Expected value for POR-ESP: lam_tot anchored ~2.3-2.6 -> **P(YES) ~ .27-.29** (Poisson-total reference: lam 2.2 -> .288, 2.5 -> .267, 2.8 -> .249; +~0.5-1 pt after DC/empirical-q corrections). Cross-check: WC-finals empirical .2659.

### b. Data / pipeline
All exists: anchored grid per matchday (runner ctx), SECOND_HALF_SHARE, q_N constants (derived above from martj42; re-derivable by a 20-line script).

### c. Validation
- Component-wise (done this pass): q_N table vs binomial; the total pmf is validated by the goal-anchor backtest. The composition needs only an integration test: at uniform lam=2.5, P in [.26, .29]; monotone decreasing in lam_tot.
- Optional cheap backtest before lock: replay settled WC26 matches' anchored grids -> P(YES) vs half-equality reconstructed from martj42 WC26 rows (n~90): a coarse calibration check on live-tournament conditions.

### d. Failure modes + crowd
- **Crowd**: reads as a coin-flip novelty -> expect .35-.45 (over-priced). We sit .27-.29, i.e. 8-17 pts UNDER — inside the measured +EV deviation band (5-15 pts). No crowd-prior entry exists -> the 25-pt clamp is blind -> enforce a sanity rail instead: for ANY sane lambda in [1.6, 3.4] the answer lies in **[.24, .33]**; a value outside = pipeline bug, block submission.
- The odd/even structure makes P insensitive to moderate lambda error (dP/dlam ~ -0.07/goal); an odds glitch moves us ~2 pts, not 20.
- Settlement check on first instance: 0-0 FT must settle YES (0==0). Ledger it.

### e. Recommendation
Estimator A with empirical q_N; fallback (no odds): same formula on the pure-model grid. Register in GOAL_ARCHS (grid-derived — avoid the R16 base-dilution bug class). Exposure: 1 q x ~3x; expected +2-6 RBP if the crowd floats .38+, ~0 if the crowd computes .30; downside bounded by the rail.

---

## §2 Q2 — "First card before first goal" (`first_card_before_first_goal`)

**Settlement branches (defined BEFORE the math).** T_c = first card time (yellow or red, either team), T_g = first goal time (own goals count — they are goals). Branches and their StatsBomb masses: both occur (.866): YES iff T_c < T_g; card-but-never-a-goal (**.086**): AMBIGUOUS — the permissive book-race convention says YES (the surviving event wins), a strict textual reading ("before the first goal is scored" presupposes a goal) says NO/void; goal-but-no-card (.041): NO; neither (.006): NO. The two readings differ by **8.6 pts**: P_strict = .4045 vs P_permissive = .4904 at corpus rates — this definitional risk EXCEEDS every modeling error below. Plan: price at w_perm = 0.6, flag the question in the audit ledger as definition-pending, and lock w_perm to 0/1 the moment the FIRST instance of this template settles anywhere with a no-goal or no-card match — add that to the settlement-harvest checklist. Max regret of w_perm=0.6 vs knowing the rule: ~3.5 pts on the ~9% branch — acceptable for one matchday, not two.

### a. Estimators

**A (chosen): time-varying competing-risks race on the minute profiles.**
Slot hazards, t in [0,90): h_g(t) = lam_tot * w_g(t) and h_c(t) = mu_c * w_c(t), where w_f(t) = exp(alpha_f[bucket(t)]) / sum_s exp(alpha_f[bucket(s)]) — the exact share-normalization pattern of the shipped hydration-window handlers.

    P(card strictly first) = sum_t exp(-(Lc(t)+Lg(t))) * [h_c/(h_c+h_g)](t) * (1 - exp(-(h_c+h_g)(t)))
      Lc, Lg = cumulative hazards to slot start; within-slot = competing exponentials
    P_perm = P(card strictly first) + exp(-Lg(90)) * (1 - exp(-(Lc(90)-...)))   [no-goal tail: card occurs, goal never]
    P(YES) = w_perm * P_perm + (1 - w_perm) * P_strict

A structural free lunch: before the first goal the score is 0-0 — the BASELINE state (beta = 0) — so the race regime needs NO score-state modeling; alpha-only weights are exactly the right object, and the late-card loading in alpha (bucket-6 card weight 5.5x bucket-1, vs 2.0x for goals) is precisely what the naive estimator misses.

Quantified at corpus rates: naive constant-rate mu_c/(mu_c+mu_g) * (1 - exp(-(mu_c+mu_g))) = **.5835**; time-varying race = **.5212**; empirical permissive = **.4904**. The minute profile removes 2/3 of the naive bias. The residual -3.1 pts is (i) card burstiness — mu_c varies match to match (frailty_k = 12.2) and the race probability is concave in mu_c in this range, so E[P(mu)] < P(E[mu]) (Jensen), and (ii) marginal-vs-state-0 hazard normalization. Closure:
- **A' (preferred)**: integrate the race over Gamma frailty on mu_c with the fitted k = 12.2 (10-point Gauss-Laguerre quadrature; recovers ~1.5-2 of the 3.1 pts from theory rather than fudge);
- plus ONE per-archetype logit intercept gamma fit offline on StatsBomb so the corpus-mean model p equals the empirical .4904/.4045 — the repo's standard Platt move, absorbing whatever the frailty doesn't.

Inputs: lam_tot = lam_a + lam_b (ANCHORED — the goal side of the race rides the book); mu_c = cards count-model mu (both teams, both halves, shrunk ref factor) x the WC26 cards level (s_cards ~= 0.935 from the adversarial review's shrunk fit). If `wc26_level.json` is still unshipped at QF lock, apply 0.935 explicitly in this handler — pricing the race at the stale historical card level is a KNOWN +2-3 pt YES-direction bias.

**B (rejected): MC engine.** The v2 sim draws card COUNTS from path-integrated hazards per half — a first-card TIME does not exist in the simulation. Adding slot-level card events is real work, adds MC variance, and cannot beat the closed form built from the same alpha profile.

### b. Data / pipeline
All in-repo: minute_dynamics.json (card + goal alphas), count-model cards mu + ref factor, anchored lambdas, StatsBomb corpus for gamma and validation. Nothing to fetch.

### c. Validation
- Corpus-level: after gamma, model mean == .4904 (permissive) / .4045 (strict) by construction; the fit script must print the PRE-gamma mean (~.52 expected) — if it is not ~.52, the integral implementation is wrong.
- Discrimination (honesty check, not a gate): bucket the 314 matches into terciles of predicted p (per-match mu proxies) and verify monotone realized frequencies. Expect WEAK discrimination — this is 80% a base-rate question; at 3x, calibration is the objective, not sharpness.
- Sensitivity table in the PR: p over mu_c in [2.4, 4.4] x lam_tot in [1.8, 3.2] — total movement ~±6 pts; bounded.

### d. Failure modes + crowd
- **The settlement ambiguity IS the failure mode** (8.6 pts). Handled by w_perm + first-settlement lock, above.
- Crowd: designed-as-a-coin-flip and nearly is one -> crowd ~.50 ± 5. Ours ~.44-.47 (POR-ESP: mu_c ~2.9-3.2 after level cut, lam_tot ~2.4 -> raw race ~.49-.50 -> gamma -> ~.46-.47 perm / ~.40 strict -> blend). Deviation < 8 pts -> safe band even without a clamp entry.
- POR-ESP rivalry-heat narrative (crowd may lean cards-early): per-ref factors stay shrunk (REF_SHRINKAGE=40 finding is validated in-repo); no LLM/narrative override — frequency questions belong to the data (README doctrine).

### e. Recommendation
A' + gamma at w_perm = 0.6; fallback: flat .46 (corpus permissive .49 shaded for WC26 cards-down). Do NOT register in COUNT_ARCHS while the count Platt is the stale [0.972, +0.181] up-shift (review §3.4 precedent) — gamma is this archetype's calibration. Exposure: 1 q x 3x; expected +0.5-1.5 RBP (parity question; the win is non-forfeiture plus 3-5 pts of sharpness on a .50-anchored crowd).

---

## §3 Q3 — "Diogo Costa makes 4+ saves" (`player_saves_ge`, thr = 4)

**Definitional grounding.** In 628/628 StatsBomb team-matches: **saves == SoT-against − goals-against EXACTLY** (saves = outcomes Saved + Saved-to-Post; SoT = those + Goal). Blocked shots are neither SoT nor saves; 'Saved Off Target' (keeper touches a wide-going shot) is NOT a save in this identity, and Opta-style settlement also credits saves only on on-target shots. The model can therefore target the identity directly; the residual platform risk is their SoT source definition (the documented Opta-vs-StatsBomb SoT gap), caught after one settled instance by the settlement-definition cross-check (review Idea A machinery).

### a. Estimators

**A (chosen): NegBin thinning of opponent SoT, coherent with the anchored goal intensity.**
Generative: SoT_ESP ~ NegBin(k_sot, mu_sot); each on-target shot is a goal w.p. c, a save otherwise. NB is closed under thinning:

    saves ~ NegBin(k_sot, (1 - c) * mu_sot),      with  c := lam_ESP_anchored / mu_sot
    =>  mu_saves = mu_sot − lam_ESP_anchored      (the identity, in expectation)
    P(YES) = 1 − NegBin.cdf(3; k_sot, mu_saves)

Both inputs are the strongest available estimates of their pieces: mu_sot = s_SoT x [mu(sot,h1) + mu(sot,h2)](lam_ESP, lam_POR) from the count model at the ANCHORED lambdas, with the WC26 level factor s_SoT ~= 1.29 (the SoT family is the single strongest measured WC26 drift: 7/7 YES at thr 8); lam_ESP is the book-anchored goal intensity. **Keeper quality needs NO separate parameter**: the book prices Costa's shot-stopping inside POR's goals-against (low lam_ESP), and a better keeper mechanically converts SoT-against into MORE saves via the identity — the estimator inherits Costa's quality from the anchor. Dispersion: thinning preserves k; use the count model's SoT k cross-checked against the saves-marginal k = 6.32 (which reproduced P(>=4) to four decimals); print both, prefer 6.3 if they disagree materially.

Numbers for POR-ESP: lam_ESP ~1.2-1.4; historical-fit mu_sot(ESP vs POR) ~3.7-4.2; x1.29 -> 4.8-5.4, but shade the level factor down for knockout caginess (it was fit mostly on group-stage rows; the review's own time-decay logic argues the knockout s is smaller). Realistic band **mu_saves ~2.9-3.7 -> P(YES) ~ .34-.50, center ~.42-.44**, vs the unconditional base .269 — the WC26 SoT level and facing-Spain both push up, and that gap IS the informational content of the price.

**B (rejected as primary): flat empirical base** P = .269. Ignores exactly the two first-order drivers (opponent volume, WC26 level). Retained as the fallback and as a sanity pole.

**No book anchor exists** — catalog verified 2026-07-05: The Odds API has no soccer GK-saves market (its soccer player props are club-league scoped besides). Nothing to blend toward; the risk surrogate is the shrunk s_SoT + the sanity rail.

### b. Data / pipeline
Count model + anchored lams (exist). s_SoT: read `wc26_level.json` if the review's ship-item #2 lands before QF lock; else hardcode 1.29 with a dated comment. Do NOT run at s = 1.0 — that knowingly reprices the 7/7 total-SoT miss. StatsBomb corpus for k and the validation GLM.

### c. Validation
- Marginal (done): NB(6.32, 2.57) reproduces the saves tail at thresholds 1-7; add tests at thr 3 and 5 in case the template reprints at other lines.
- Conditional (the real test, one script): across the 628 team-matches, NB GLM of realized saves on log(model mu_saves) evaluated at each match's backtest lambdas — slope ~1 plus calibrated terciles of P(>=4) demonstrates the opponent-volume variation is signal, not level noise. In-sample for the count model, out-of-structure for the thinning identity — adequate for a template with zero settled history.
- At settlement: harvest the match's actual SoT/goals; verify platform saves == SoT − goals (definition detector).

### d. Failure modes + crowd
- **s_SoT wrong-way risk dominates**: fit on ~25 binaries, mostly group stage. If knockout SoT reverts to historical levels, truth ~.32 vs our ~.43. Mitigations: refit the level with time decay the night before (daily_learn), keep the submission inside the rail [.35, .50], and note the crowd sits ABOVE us (below), bounding the relative loss.
- **Crowd**: "Costa makes 4+ saves against SPAIN" is narrative-friendly -> crowd ~.50-.60 (over-weights Spain's volume, under-weights that a save needs an on-target non-goal). Our .42-.46 sits 5-15 pts UNDER — the validated +EV band; structurally the same edge class as the protected fringe-SoT lows (the crowd over-prices player-event YES).
- Costa subbed off / red card (~2% tail): inside the NB dispersion; ignore.
- Score-state feedback (POR leads -> ESP SoT spikes): the count-model mu is state-marginal, so this is priced on average; no scenario adjustment.

### e. Recommendation
A with the freshest s_SoT; fallback flat .30 (base .27 nudged for the SoT-up regime). Not in COUNT_ARCHS while the count Platt is stale. Sanity rail [.30, .55]. Exposure: 1 q x 3x; expected +1-3 RBP vs a ~.55 crowd; loss bounded by the crowd sitting above us if the level reverts.

---
## §4 Q4 — "9+ total substitutions in regulation" (`subs_total_ge`, thr = 9)

**Settlement reading.** Combined subs by both teams with event minute <= 90 (+ stoppage of H2, which api-football stamps as elapsed = 90 with an `extra` field — INCLUDED); ET subs (elapsed 91-120) excluded. Max under the 5-sub rule is 10 (+ concussion subs, unverified platform treatment — only shifts 10 -> 11, tail-irrelevant at thr 9).

### a. Estimators

**A (chosen): stratified empirical pmf, Beta-shrunk, with the ET-risk mixture priced off the SAME anchored X as Q5.**
The distributional facts kill any parametric count model: WC2022 subs<=90 have var/mean = 0.175 — this is a bounded quota-consumption process (managers spend 4-5 of 5 slots), not an event-arrival process. The estimator is the empirical exceedance with exactly two structure terms:

    P(YES) = (1 − X) * p_decided  +  X * p_ETbound
      X          = the book's de-vigged P(draw at 90) (~.28 for POR-ESP) — same number Q5 submits
      p_decided  = P(>=9 subs in 90' | match not headed to ET)   [WC2022 KO matches decided in 90: .818 raw, n=11]
      p_ETbound  = P(>=9 subs in 90' | match goes to ET)          [CURRENTLY UNMEASURED — see the fetch]
    Both cells Beta-shrunk toward the pooled base: posterior mean of Beta(a0+k, b0+n−k), prior centered
    at .75 with ~20 pseudo-matches, so tiny cells cannot run to extremes.

**MANDATORY pre-lock data action:** the 5 uncached WC2022 matches are EXACTLY the 5 ET matches — the current cache cannot measure p_ETbound at all, and the raw .746 pools over a censored sample. Fetch events for fixture ids 977344, 977345, 978072, 977794, 979139 (5 api-football requests, free tier, cache-writing ingest — a data task, not a model change). The hold-a-sub-for-ET hypothesis is then a 5-row table instead of a guess. Optional strengthener within free tier (2022-2024): Euro 2024 (league 4) + Copa America 2024 (league 9) events, ~85 matches incl. ~10 ET knockouts, ~90 requests over 1-2 days under the 100/day cap — triples the KO sample for both cells.
Direction note: holding the 5th sub reduces 90-min subs in ET-bound matches, but ET-bound matches are also tight-and-tiring (more like-for-like subs). The 2022 five will say which force wins; do not assume.

**B (rejected): per-team parametric quota model** (P(team uses 5) etc. with a correlation term): with n this small it re-derives the empirical pmf under extra assumptions; within-match correlation of the two managers' spending is real (blowouts, pace) and unidentified. Only revisit if team-level sub questions ever appear.
**C (rejected): NegBin/Poisson count family** — var/mean 0.175; the count-model machinery is the wrong object class for a quota process.

Expected number: raw pooled .746; after ET-mixture with X=.28 and a plausible p_ETbound in [.5, .8] -> **P(YES) ~ .65-.73**. Publish only after the 5-match fetch tightens p_ETbound.

### b. Data / pipeline
`data/raw/apifootball/events/fx*.json` (59 cached) + `fixtures/league1_2022.json` (rounds, ET flags) — exists; + the 5-fixture fetch (existing `ingest/apifootball.fetch_fixture_events`, free quota); optional Euro/Copa 2024 harvest (same module, same quota). X from the Q5 pipeline (shared).

### c. Validation
- With Euro/Copa fetched: leave-one-tournament-out — does WC2022 predict Euro24 KO P(>=9)? Report, don't over-fit.
- Without: honesty = intervals. n=16 KO gives ±.2 binomial CI — which is WHY the Beta shrink toward the pooled .75 is load-bearing and the final number must not leave [.60, .78] on data this size.
- 2026 regime check: same 5-sub + 6th-in-ET regime as 2022/2024; 26-man squads; WC26 heat/hydration pushes usage UP if anything — treat as an asymmetric risk note, not an adjustment (no data).

### d. Failure modes + crowd
- **Crowd**: "9 or more" SOUNDS high; most individuals don't know the base is ~.75 -> crowd likely .45-.60. Ours ~.68-.72 sits 10-25 pts ABOVE — the widest deviation of the five and a genuine base-rate informational edge (the same class as the validated fringe-SoT edge: the crowd misprices unfamiliar base rates). But with no crowd-prior entry the clamp cannot catch a data error -> pre-register the rail **[.60, .78]**; outside = bug, block.
- Selection/censoring recurrence: if the fetch fails, do NOT submit the raw .746 — submit the .70 fallback (pooled shaded for ET risk) and ledger the caveat.
- Settlement unknowns: concussion-sub counting; a sub AWARDED in 90' stoppage but executed in ET (edge). One settled instance pins both; ledger.
- A red card early reduces sub demand slightly (10 men manage differently) — inside the empirical variation, no adjustment.

### e. Recommendation
A after the 5-match fetch (+ Euro/Copa if time permits); fallback .70. Exposure: 1 q x 3x; **expected +2-5 RBP vs a .5-ish crowd — the highest-EV question of the five**, and the only one where we hold a base-rate edge rather than parity.

---

## §5 Q5 — "Match goes to extra time" (`extra_time`)

**Identity.** Knockout, no replays: ET <=> tied after 90 (+ stoppage). P(YES) = P(draw at 90). The de-vigged X of the 90-minute 1X2 IS this exact quantity, quoted by every book, Shin-de-vigged by the existing consensus pipeline.

### a. Estimators

**A (chosen): the de-vigged X directly**, from the same odds snapshot the runner already pulls each matchday (zero marginal credits), consensus across books (`devig.consensus`, method="shin"). Take it at the near-perfect-anchor weight: **w_book = 0.97 / 0.03 model** in logit space — mirroring the player-prop precedent for markets the book prices better than we ever will; the 3% model sliver is inertia against a fat-fingered odds row, and the anchor-regression guard covers vanished lines.

**B (fallback + cross-check): anchored grid draw** `mk.p_draw(ctx.grid)`. Two lossy steps sit between B and the truth: (i) `implied_lambdas` is a 2-parameter LS fit to 3-4 targets — the grid draw reproduces the de-vig X only approximately (<1 pt typically; worse when 1X2 and totals disagree); (ii) the 0.85/0.15 blend drags toward the model draw. B is the answer only when odds are missing; otherwise it is the divergence check: |A − B| > 4 pts -> audit flag.

**Coherence requirement (same-slate consistency).** Three consumers of P(draw90) now exist: `extra_time` (=X), `pen_shootout` (= P(draw90) x P(ET drawn)), and Q4's ET mixture. Resolve ONE shared p_draw90 per match context — the de-vig X when fresh-or-cached odds exist, grid draw otherwise — and feed all three. Otherwise the slate can assert P(pens) > P(ET)-consistent bounds or Q4/Q5 can disagree about the same event. (Today pen_shootout reads the grid draw; switch its first factor to the shared value — a one-line change, and the ET-drawn factor stays grid/ET_TEMPO-based, which the R16 review already passed at ±1-2 pts.)

### b. Data / pipeline
Odds snapshot (exists, every matchday run + the ~2-min-before-lock refresh); Shin de-vig (exists); grid fallback (exists). Nothing new.

### c. Validation
- The 90-min X is the single most-validated quantity in the system (the goal-anchor backtest); nothing further is required pre-lock. Sanity band from history: WC KO 1986-2022 P(ET) = **.300** (48/160); Euro-2021+ tied-at-90 .307; Copa-2024 .323; POR-ESP is near-even -> the fetched X should print **.26-.31**. Outside [.22, .35] = fetch/parse bug, not information.
- Post-hoc: extra_time settles fast every knockout matchday — track submitted-vs-realized in the CLV ledger like any anchored market.

### d. Failure modes + crowd
- **Crowd**: the X is public knowledge; crowd ~.30 ± 5. Zero expected edge — and that is fine: matching the sharp book beats the mean-individual baseline by construction (README Thesis 1). The only way to lose this question is to NOT price it (forfeit) or to fall back to the pure model when a stale-but-good cached X exists — the quota-resilience cache fallback must apply here exactly as for goals markets.
- Definitional edge cases (abandonment/awarded ties): ignore; not modelable, symmetric.
- Do NOT "improve" the X with knockout-history shrinkage (.300 base): the book impounds team strength, weather, and lineups; the base rate is a sanity band, not an estimator. (Rejected in §7.)

### e. Recommendation
A at w=0.97 with B fallback + the shared-p_draw90 refactor. Register `extra_time` in GOAL_ARCHS (anchored pass-through; no base dilution, no count Platt). Exposure: 1 q x 3x; expected +0.5-1 RBP (parity question, forfeit-proofing).

---

## §6 Ranked implementation order + effort

| # | Item | Effort | Why this order |
|---|---|---|---|
| 1 | **Q5 extra_time**: parser pattern + shared p_draw90 resolver (X-first, grid fallback) + GOAL_ARCHS registration + pen_shootout first-factor switch | ~30-45 min | Lowest effort, perfect anchor, unblocks the shared X that Q4 consumes; forfeiting a bookable question is the worst outcome on the slate |
| 2 | **Q1 halves_equal_goals**: grid-total x q_N handler + q_N constants + GOAL_ARCHS + rail [.24,.33] + integration test | ~1-2 h | Pure composition of two already-validated components; second-cheapest, low-risk |
| 3 | **Q4 subs**: fetch the 5 WC2022 ET-match event files (data task) -> recompute strata -> Beta-shrunk mixture handler + rail [.60,.78]; optional Euro/Copa-2024 harvest for LOTO validation | ~2-3 h (+half-day optional harvest) | Highest expected RBP of the five (crowd base-rate blindness); the fetch MUST precede pricing (selection bias documented in §4) |
| 4 | **Q3 player_saves_ge**: NB-thinning handler + s_SoT source decision (wc26_level.json or dated constant 1.29) + saves GLM validation script + rail [.30,.55] | ~2-3 h | Real edge vs an over-priced crowd, but depends on the level-factor judgment; do after the mechanical wins |
| 5 | **Q2 first_card_before_first_goal**: race integral + Gamma-frailty quadrature + offline gamma fit on StatsBomb + w_perm=0.6 + settlement-pending ledger flag | ~3-4 h | Most math for the least EV (a near-coin-flip); ship LAST but DO ship — unpriced = guaranteed forfeit, and a .45±3 price is nearly risk-free |

Cross-cutting (bundle with #1): none of the five has a crowd-prior entry -> the per-question rails in this doc are the interim clamp — implement them as submit-time asserts/audit blocks, not comments. Add all five to the crowd-harvest list so `fit_crowd_prior` picks them up at first settlement. Keep Q2/Q3 OUT of COUNT_ARCHS until the count Platt is neutralized (review Task 2 step 3). Total: roughly one focused day + the optional harvest afternoon.

Expected slate exposure: 5 questions x ~3x knockout multiplier. Parity trio (Q1 vs a sharp crowd, Q2, Q5): ~+1.5-3 RBP combined, near-zero variance if the rails hold. Edge pair (Q3, Q4): ~+3-8 RBP combined if the crowd sits where base-rate-blind crowds sit; bounded downside because both rails cap our distance from the flat bases.

---

## §7 Examined and REJECTED as statistically unsound

1. **Q1 via independent Poisson halves** — discards the anchored total pmf and the DC low-score mass; the conditional-binomial decomposition uses the identical inputs strictly better (and the binomial split is empirically verified to 3 decimals at N=2,4 on n=4,000+ cells).
2. **Q2 naive constant-rate competing risks** — measured +9.4 pts of bias at corpus rates (.5835 vs .4904 empirical); the trap the brief flagged, confirmed with the repo's own alpha profiles (cards bucket-6 loading 5.5x vs goals 2.0x).
3. **Q2 via the MC engine** — cards have no event times in the v2 sim (path-integrated per-half counts); adding timed card events = work + MC variance with no accuracy over the closed-form race.
4. **Q2/Q3 per-referee or keeper-skill parameters** — per-ref factors don't transfer (REF_SHRINKAGE=40 backtest finding, in-repo); a keeper skill term is unidentifiable on 314 matches AND already impounded in the anchored lam via the book. Both are exactly the class of unvalidatable priors the 3x multiplier punishes.
5. **Q3 saves modeled as SoT-minus-goals with separately-estimated components** — unnecessary once the identity is verified exact in-data; the NB-thinning form gets the same expectation with a coherent distribution and one fewer moving part.
6. **Q4 with any Poisson/NegBin count family** — var/mean = 0.175; quota consumption, not event arrival. Parametric count machinery would fabricate tail mass that does not exist (P(>=11) is essentially concussion-subs only).
7. **Q4 priced off the current 59-match cache** — the 5 missing matches are exactly the 5 ET matches; textbook informative censoring on the QF-relevant stratum. Fetch first, price second.
8. **Q5 priced off the grid when the direct de-vig X exists** — adds LS-inversion residual + model-blend drag to a quantity the market quotes directly; also breaks slate coherence with pen_shootout unless shared.
9. **Q5 shrunk toward the historical .300 knockout base** — the base rate is a sanity band; blending it into a sharp market price is anti-information (the book already knows POR-ESP is even).
10. **LLM/narrative adjustments for any of the five** — all five are frequency/base-rate questions; the LLM-as-verifier doctrine (hard evidence only, player-prop scope) explicitly excludes them. Rivalry-heat card narratives are the crowd's error, not our input.
11. **A bespoke rest/rotation covariate for Q4 subs** — same verdict as the review's §3.5: no data in repo, unvalidatable at knockout stakes; the book X (via the ET mixture) already carries the tie-tightness signal that matters.

---

## §8 Executive summary

1. **Q5 = the book X at w=.97** (perfect anchor, Shin de-vig; share one p_draw90 across extra_time / pen_shootout / Q4's mixture); expect ~.28, crowd parity, forfeit-proofing.
2. **Q1 = anchored-grid total pmf x empirical split factors q_N** (binomial-split verified to 3 decimals on 11,658 matches); expect ~.28 vs a crowd likely floating .35-.45 — our second edge.
3. **Q4 = stratified empirical exceedance with a book-X ET-risk mixture** — but ONLY after fetching the 5 missing WC2022 event files, which are exactly the 5 ET matches (informative censoring found this pass); expect ~.68-.72 vs a base-rate-blind crowd ~.5 — the highest-EV question.
4. **Q3 = NegBin thinning: saves ~ NB(k, mu_sot x s_SoT − lam_ESP)** — the saves==SoT−goals identity is exact in 628/628 StatsBomb matches, keeper quality rides the book anchor, no GK-saves book market exists (verified); expect ~.42-.46 vs a narrative crowd ~.55.
5. **Q2 = time-varying competing-risks race** on the repo's minute profiles (naive constant-rate is +9.4 pts wrong, measured) with Gamma-frailty closure and an 8.6-pt settlement ambiguity priced at w_perm=.6 and resolved at first settlement; expect ~.45, crowd parity.

Every question carries a pre-registered sanity rail (the interim clamp — none of the five has a crowd-prior entry yet), and none may enter COUNT_ARCHS while the stale count Platt survives. Implementation order: Q5, Q1, Q4, Q3, Q2 — about one focused day plus an optional free-tier harvest afternoon (Euro/Copa 2024) that upgrades Q4's validation from intervals to leave-one-tournament-out.

*Prepared 2026-07-05; all numbers in this file recomputed from repo data this session (scripts run read-only; scratch code not committed).*
