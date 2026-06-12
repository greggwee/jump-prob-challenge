# Jump Trading Probability Cup — Modeling & Data Plan

Target: 2026 World Cup binary-question forecasting on SportsPredict, scored by **relative Brier**.
Goal: build a system that produces well-calibrated probabilities for every market archetype, beats the
field average consistently, and runs as two API bots.

---

## 0. What the scoring rule tells us to build (read this first)

Your points per question = `(field average Brier) − (your Brier)`. Three consequences drive the whole design:

1. **Honest probabilities are optimal.** Brier is a *strictly proper scoring rule*: for a true probability
   `p`, the expected Brier `E[(q − Y)²] = q² − 2pq + p` is minimized exactly at `q = p`. The relative part
   (subtracting the field average) is a constant w.r.t. your choice, so it changes nothing about what to
   submit — **report your best estimate, never shade it**.
2. **Answer everything.** Expected points are `E[field Brier] − E[your Brier]`. If your forecast is even
   slightly better-calibrated than the crowd on average, skipping a question throws away positive EV.
   The leaderboard is cumulative, not average.
3. **You only score by being different AND right.** Matching the field average earns exactly zero. The crowd
   will be roughly "betting odds + noise" on main markets (match winner, total goals) and much weaker on
   obscure markets (offsides, half-specific corners, player shots on target). **The biggest edge per question
   lives in the markets bookmakers don't price and casuals can't intuit.** That's where the count models
   (Section 4) matter most.

Practical constraints from the API docs:
- Probabilities are **integers 1–99**; the docs state "Edge values (0, 100) are rejected" (so extreme
  certainty is capped at 0.99 / 0.01 — good, it protects us).
- Markets close at match start time (`closing_time`); predictions can be **PATCH-updated** until then →
  submit early, revise after lineups drop (typically 60–90 min before kickoff; confirm exact WC 2026
  release timing on matchday 1).
- Batch endpoint (50 per call, >50 → 400), 60 req/min rate limit per IP (REST + MCP combined).
- **Stage multipliers: group 1×, knockout 2×, final 3×** (docs: "Stage weighting applies") → knockout
  questions are worth disproportionate effort; budget model-refresh time accordingly.
- Up to 2 active bots per user, and **each API key is a separate leaderboard entry** (per the API docs) —
  the two-strategy design in §6 is therefore sound. One ambiguity remains: docs say "one prediction per
  market per user," which may or may not allow both bots to answer the same market — **test empirically on
  matchday 1** (see assumptions register, §9).

---

## 1. Question taxonomy → model mapping

Every observed question archetype reduces to a functional of a simulated match. One simulation engine
answers all of them coherently.

| Archetype | Example | Statistical object |
|---|---|---|
| Match result | "Will South Korea win?" | P(home goals > away goals) from goal model |
| Goals threshold / BTTS / combos | "BTTS AND 3+ goals?" | Joint scoreline distribution |
| Team scores in a half | "Will SK score in 2nd half?" | Half-split goal intensities |
| Count comparisons | "More corners at HT than opponent?" | Bivariate count model (corners) per half |
| Count thresholds | "2+ cards in 2nd half?", "2+ offsides?" | Poisson / negative-binomial counts |
| Player to score | "Son Heung-min anytime scorer?" | Player goal model: minutes × xG share |
| Player shots on target | "Schick 1+ SoT?" | Player SoT rate model |
| Rare events | "Penalty awarded?" | Base rate + style adjustments |

**Key principle: simulate the whole match once, answer every question by counting frequencies.** If you
model each question independently you get incoherent answers (e.g. P(BTTS AND 3+ goals) > P(3+ goals)).
A Monte Carlo simulator makes all answers automatically consistent and captures correlations (a match with
many goals tends to have more shots on target; a tight match has more cards late; etc.).

---

## 2. Core goal model (the engine everything hangs off)

### 2.1 Theory

The canonical family, in increasing sophistication:

- **Maher (1982):** goals scored by team *i* vs team *j* ~ independent Poissons with
  `λ_ij = attack_i × defence_j × c`.
- **Dixon & Coles (1997):** same, plus (a) a dependence correction `τ` for low scores (0-0, 1-0, 0-1, 1-1
  are empirically more common than independent Poisson predicts), and (b) **exponential time-decay
  weighting** of past matches: a match played `t` days ago gets weight `exp(−ξ t)`. We fit `ξ` by maximizing
  out-of-sample predictive likelihood (typically half-life ≈ 1–2 years for internationals).
- **Karlis & Ntzoufras (2003):** bivariate Poisson via a shared latent component `λ₃` that induces positive
  goal correlation: `X = U + W, Y = V + W` with `U~Poi(λ₁), V~Poi(λ₂), W~Poi(λ₃)`.

References:
- [Dixon-Coles applications & betting-market study](https://www.academia.edu/118422937/Application_of_Poisson_and_Dixon_Coles_models_on_football_match_outcome_prediction_and_research_of_a_positive_return_over_investment_in_betting_market)
- [Bivariate Poisson predictive modeling](http://article.sapub.org/10.5923.j.ajms.20201003.01.html)
- [Bivariate Poisson model for Euro 2016 (LMU technical report)](https://epub.ub.uni-muenchen.de/29028/1/TR_EM2016.pdf)
- [Bayesian state-space models for football (JRSS-C 2025)](https://academic.oup.com/jrsssc/article/74/3/717/7929974)

### 2.2 Our specification

For internationals, pure attack/defence ratings are noisy (national teams play ~10 matches/year against
wildly varying opposition). So we use a **log-linear intensity model with Elo as the backbone**:

```
log λ_team = β₀ + β₁ · (Elo_team − Elo_opp)/100 + β₂ · neutral_venue + β₃ · importance + team_attack_resid − opp_defence_resid
```

- **Elo from [eloratings.net](https://www.eloratings.net)** (World Football Elo) is the best single
  predictor of international results — better than FIFA rankings (Lasek et al. 2013). The Elo difference
  maps to expected goal difference nearly linearly.
- `team_attack_resid` / `defence_resid` are shrunk (ridge / hierarchical prior) Dixon-Coles-style residual
  ratings fit on the last ~4 years of internationals with exponential decay, capturing "scores more/less
  than Elo implies" (e.g. teams that are defensive grinders).
- Fit by weighted maximum likelihood. Validate `β`s and decay `ξ` by backtesting (Section 6).

### 2.3 Half-splits

Empirically ~46% of goals come in the 1st half, ~54% in the 2nd (more across all tournaments; verify on our
data). We model halves as **thinned Poissons**: `λ_1H = ρ·λ`, `λ_2H = (1−ρ)·λ`, with `ρ` estimated globally
and allowed to shift with in-match state in the simulator (trailing teams attack more late). For pre-match
questions, the static split is sufficient and robust.

### 2.4 Market anchor — the single biggest accuracy win

Closing odds at sharp books (Pinnacle, Betfair Exchange) are the best public forecast of match outcomes;
decades of efficiency literature confirm a model that ignores them is leaving accuracy on the table.

But raw inverse-odds overstate every outcome (the overround) and carry **favourite-longshot bias**. De-vig
properly:

- **Shin's method** (models a fraction of insider traders; corrects FLB endogenously) — use the
  [`shin` Python package](https://github.com/mberk/shin).
- Štrumbelj ([2014](https://www.researchgate.net/publication/264349990_On_determining_probability_forecasts_from_betting_odds),
  [2016](https://journals.sagepub.com/doi/10.1177/1527002513519329)) shows Shin probabilities beat basic
  normalization for soccer; the advantage shrinks for high-liquidity markets, where the *power method*
  (`p_i = (1/odds_i)^k`, solve k so probabilities sum to 1) is a fine alternative. Implement both, pick per
  market via backtest.
- From de-vigged 1X2 + Over/Under 2.5 (+ Asian handicap if available), **invert to implied (λ_home, λ_away)**:
  find the intensity pair whose Dixon-Coles scoreline distribution best reproduces the de-vigged market
  probabilities. Note this is generally an *overdetermined* system — 1X2 gives two independent constraints
  (three outcomes summing to 1) and each totals line adds another, against two unknowns (holding the
  dependence parameter ρ fixed at its historical estimate) — so solve by weighted least squares over all
  available constraints rather than exact root-finding; the redundancy is a consistency check on the odds
  themselves. This gives market-implied goal expectations to blend with our model's.

**Blending** happens in log-intensity space (for λs) or logit space (for final probabilities):

```
logit(p_final) = w · logit(p_market) + (1−w) · logit(p_model)
```

with `w` fit by backtest per market type. Expect `w ≈ 0.7–0.85` for main markets, `w → 0` for markets with
no odds available (offsides, half-specific corners).

### 2.5 Second anchor class: prediction markets (Polymarket, Kalshi)

Both platforms carry [extensive WC 2026 markets](https://www.oddpool.com/explore/world-cup) — outright
winner, group qualification, and **individual match outcomes** — with
[over $2B combined volume](https://predictionnews.com/story/kalshi-and-polymarket-odds-diverge-on-2026-world-cup-favorite-d4631f3a).
They are a *different kind* of probability source than bookmaker odds and get their own ingestion module:

- **Mechanics differ:** orderbook midpoint of (bid, ask), no bookmaker overround to strip — but account for
  bid-ask spread (uncertainty band), platform fees, and crypto/US-retail crowd composition.
- **Independent information:** [Polymarket's global vs Kalshi's US user base produce 5–8¢ divergences on
  the same contracts](https://www.oddschecker.com/us/insight/soccer/20260609-kalshi-vs-polymarket-where-to-trade-on-the-2026-world-cup).
  Cross-platform divergence is itself a signal: large spread between Pinnacle, Polymarket, and Kalshi on the
  same event = genuine disagreement = widen our uncertainty and lean on the model; tight agreement = trust
  the consensus more.
- **Weighting:** each source enters the logit-space ensemble with weight increasing in its liquidity/volume
  on that specific contract. A $10M match-winner book is near-Pinnacle quality; a $2k offsides novelty
  market is noise. Default prior: sharp bookmaker closing line ≥ deep prediction market > shallow prediction
  market — but this ordering is *tested in the backtest, not assumed* (Kalshi/Polymarket historical prices
  are retrievable via their public APIs for settled markets).
- **Caveat:** prediction markets exhibit their own favourite-longshot bias at the extremes (longshots
  overpriced below ~5¢); apply the same bias-correction calibration we fit for bookmaker probabilities.

### 2.6 Not one paper: the model zoo and how we choose

Dixon-Coles is a *member* of the candidate set, not the law. The rational procedure is: enumerate the
credible model families the literature offers, fit each in the same pipeline, and let **out-of-sample
predictive performance on our backtest tournaments** — plus forecast combination — decide. The zoo:

| Family | Key papers | What it adds | Risk |
|---|---|---|---|
| Independent Poisson | Maher 1982 | baseline, nearly unbiased | misses low-score dependence |
| Dixon-Coles | Dixon & Coles 1997 | low-score correction, time decay | ad-hoc dependence form |
| Bivariate Poisson | [Karlis & Ntzoufras 2003](http://article.sapub.org/10.5923.j.ajms.20201003.01.html) | principled positive correlation | can't produce negative correlation |
| Negative binomial / COM-Poisson | various | overdispersion (Poisson assumes Var=mean; real counts, esp. corners/cards, are overdispersed) | more parameters on little data |
| Skellam (goal difference) | Karlis & Ntzoufras 2009 | models the margin directly, robust for win/lose questions | loses the joint scoreline |
| Dynamic state-space / Bayesian | Rue & Salvesen 2000; Koopman & Lit 2015; [JRSS-C 2025](https://academic.oup.com/jrsssc/article/74/3/717/7929974) | ratings evolve over time with principled uncertainty | complexity, fitting cost |
| **Hybrid ML** | [Groll, Ley, Schauberger & Van Eetvelde — hybrid random forest, WC 2018](https://arxiv.org/abs/1806.03208) ([journal version](https://www.researchgate.net/publication/334375045_A_hybrid_random_forest_to_predict_soccer_matches_in_international_tournaments)) | random forest / gradient boosting on covariates (Elo, market value, age, confederation, …) **with statistical ability ratings injected as features** — beat both pure RF and pure Poisson regression | needs careful CV; opaque |

Two findings from this literature shape the design:

1. **Hybrids win.** Groll et al.'s best WC predictor was the random forest *fed with* ranking-model ability
   parameters — neither pure ML nor pure statistics. Our equivalent: a gradient-boosted model (XGBoost/LightGBM)
   trained on the same feature rows, whose predictions enter the ensemble alongside the generative Poisson
   simulator. The simulator remains the coherence backbone (only it can answer arbitrary combos); the GBM
   acts as a check and a correction signal on marginal probabilities.
2. **Combine, don't crown.** The forecast-combination literature
   ([Winkler et al., "Averaging Probability Forecasts: Back to the Future"](https://www.hbs.edu/ris/Publication%20Files/19-039_8af5ea92-6191-4774-aeb6-5eee0a5fd4c2.pdf);
   Bates & Granger 1969 onward) consistently finds that simple averages of diverse, individually-decent
   forecasts beat the single best model chosen ex post — selection error is real, and model diversity hedges
   it. So the production forecast is an ensemble: `{simulator, GBM, bookmaker, prediction markets}` combined
   in log-odds space with backtest-fitted weights, and we only drop a component if it *hurts* out-of-sample.

---

## 3. Match simulator

Monte Carlo, ~100k iterations per match (seconds in NumPy). Each iteration draws a full match narrative:

1. **Goals:** draw scoreline from the bivariate Poisson / Dixon-Coles distribution; allocate goals to
   minutes via the empirical within-match hazard (rising toward the end of each half); split by half.
2. **Shots / shots on target:** conditional on goals and team strength. Calibrate the joint distribution
   from StatsBomb event data: shots ~ NegBin with mean linked to λ and possession-style covariates; SoT|shots
   ~ Binomial with team-specific accuracy. Enforce SoT ≥ goals scored from open play (coherence).
3. **Corners:** NegBin per team per half. Drivers (from literature and our own fits): attacking intensity,
   *game state* (trailing favourites generate corner floods late), and style (cross-heavy teams). The
   corners–goals correlation is weak — the exact magnitude is **estimated from StatsBomb event data in
   Phase 4, not assumed** — so model corners on shots/territory, not goals directly.
4. **Cards:** Poisson per team per half with **referee effect as a first-class covariate** — referee
   strictness varies substantially (the spread is quantified from FBref referee logs in Phase 4; club-level
   data shows per-game card averages differing by 2× and more across referees) and FIFA announces match
   referees ~1–2 days ahead. Other covariates:
   match importance, rivalry/derby flag, Elo gap (underdogs foul more), in-sim game state (late, close
   matches produce more cards; 2nd half has ~2× the cards of the 1st). See
   [Modelling bookings in association football](https://eprints.whiterose.ac.uk/id/eprint/229624/1/Modelling_bookings_in_association_football.pdf).
5. **Offsides:** Poisson, mean from team style (high-line defences faced, direct attacking play). Low counts
   (mean ≈ 1.5–2.5/team) so thresholds like "2+" sit near the distribution's heart — get the mean right and
   shrink hard toward the tournament base rate.
6. **Penalties:** base rate first (VAR-era World Cups: 2018 had 29 pens awarded in 64 matches ≈ 0.45/match;
   2022 fewer; estimate P(≥1 in match) ≈ 25–35%). Adjust modestly for box-touch volume of attacking teams
   and defensive style. Wide uncertainty → stay near base rate.

Every binary question = `mean(indicator over simulations)`, including arbitrary combos.

### 3.1 Player props

Two markets: anytime scorer, player 1+ shots on target. Decompose:

```
P(player scores) = Σ_minutes  P(plays minute m) × scoring rate per minute
```

- **Minutes model:** starter probability (from lineups when available — PATCH predictions after team news;
  before that, from recent starts + press reports) × expected minutes if starting (sub patterns).
- **Rate model:** player's non-penalty xG/90 and SoT/90 from club season (FBref) + internationals
  (StatsBomb), shrunk toward position means, scaled by `team λ in this match / team average λ` — the
  opponent adjustment comes free from the goal model. Penalty-taker flag adds the penalty channel:
  `P(pen to team) × P(player takes) × ~0.78 conversion`.
- Then `P(1+ goal | plays M minutes) = 1 − exp(−rate × M)` (Poisson thinning), and similarly for SoT with
  the player's SoT/90. Careful: because `1 − exp(−rate·M)` is concave in `M`, plugging in *expected*
  minutes overstates the probability (Jensen's inequality). Compute it properly by mixing over the minutes
  distribution — `P = Σ_M P(minutes = M) × (1 − exp(−rate·M))` — i.e. simulate minutes scenarios (start +
  full game / start + subbed ~60–75' / bench cameo / unused) and average. The gap is small for nailed-on
  starters but material for rotation risks, which is exactly where the field will be sloppy.

---

## 4. Data pipeline

### 4.1 Sources (all backtestable, layered by reliability)

| Layer | Source | What we take | Access |
|---|---|---|---|
| Results history | [martj42 international results (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | All international scorelines, venue, competition → fit goal model & decay | CSV, free |
| Ratings | [eloratings.net](https://www.eloratings.net) | World Football Elo per team, point-in-time | scrape/CSV, free |
| Event data (backtest gold) | [StatsBomb open data](https://github.com/statsbomb/open-data) | Verified against `competitions.json`: WC 2018 (comp 43, season 3), WC 2022 (43, 106), Euro 2020 (55, 43), Euro 2024 (55, 282), Copa América 2024 (223, 282), AFCON 2023 (1267, 107) — every shot, corner, card, offside, minute-stamped, per player | JSON via `statsbombpy`, free |
| Aggregated match stats | [FBref](https://fbref.com) | Team & player tournament/season stats, referee per match | scrape (rate-limited), free |
| Live tournament feed | [API-Football](https://www.api-football.com/news/post/fifa-world-cup-2026-guide-to-using-data-with-api-sports) or [BALLDONTLIE FIFA API](https://fifa.balldontlie.io/) | Fixtures, lineups (~1h pre-KO), referee assignments, match stats incl. corners/cards/offsides, player stats | REST, free tier / cheap |
| Odds | The Odds API / OddsPortal / Betfair | 1X2, totals, BTTS, props where offered; closing + live | REST/scrape |
| Prediction markets | Polymarket (Gamma API) / Kalshi API | match & outright contract prices, bid/ask, volume; historical prices for backtest | REST, free |
| Player club form | [Understat](https://understat.com) | club-season npxG/90, shots/90, SoT/90 per player (top-5 leagues) → player prop rate models | scrape, free |
| Legacy ratings | [FiveThirtyEight SPI archive](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) | historical SPI ratings + match forecasts (discontinued 2023, but archived) → backtest feature & a third opinion on historical strength | CSV, free |
| Opta-grade aggregates | FBref (Opta-powered) / [The Analyst](https://theanalyst.com) | Opta's raw feed is paywalled enterprise; we get Opta-derived numbers through FBref tables and The Analyst's published WC model — use the latter as an ensemble *comparison*, not an input we can't audit | scrape, free |
| Weather | open-meteo / AccuWeather / Windy | Venue forecast (rain/heat → totals, card tempo; June heat in US/Mexico venues → late-match fatigue, more 2nd-half goals) — minor covariate | REST, free |
| Team news | BBC Sport / ESPN FC / The Athletic | Injuries, suspensions (accumulated yellows!), rotation hints → minutes model priors and PATCH triggers | manual/LLM-assisted |
| Expert analysis | Tifo, tactical podcasts, beat reporters | qualitative tactical reads (high line → offsides; cross-heavy → corners) → encoded as structured style flags with explicit author + date, never free-form vibes | manual, structured |
| AI tools | Claude / Perplexity etc. | research accelerant: news triage, question parsing fallback, source summarization — **never a probability source** (LLM "probabilities" are uncalibrated; everything quantitative must trace to data) | tooling |

**Coverage vs the brief's resource list:** every category the brief names is mapped above — team news
(BBC/ESPN/The Athletic → minutes model), weather (AccuWeather/Windy/open-meteo → totals covariate), betting
markets (Pinnacle/Betfair/OddsPortal → §2.4 anchor), statistical models (FiveThirtyEight archive, Opta via
FBref/The Analyst, FBref, Understat → ratings and player rates), AI tools (research tooling only), expert
analysis (structured style flags). Plus sources the brief doesn't list: StatsBomb event data, Elo,
prediction markets (§2.5).

**Redundancy is deliberate, not waste.** Multiple sources per quantity serve three distinct functions:
1. *Cross-validation* — StatsBomb vs FBref aggregates for the same match catches parser bugs and silent
   data drift; disagreement is an alert, not a tiebreak.
2. *Independent information* — Understat (club form) and StatsBomb (international tournaments) measure a
   player in different contexts; both enter the hierarchical player model as separate likelihood terms with
   context covariates, rather than being averaged blindly.
3. *Failover* — live feeds break mid-tournament. Every quantity the matchday loop needs has a designated
   primary and fallback source, tested before kickoff day.

**Qualitative sources get quantified or they don't enter.** A podcast saying "Korea press high" becomes a
`high_line: true` flag with a source and date, feeding the offsides covariate — auditable and reversible.
What never happens: nudging a probability by hand because an expert "feels" confident. If an insight can't
be expressed as a feature or a prior, it doesn't touch the numbers.

### 4.2 Pipeline architecture

```
ingest/      one module per source, raw → parquet, append-only with fetch timestamps
             (point-in-time discipline: store WHEN we knew it, never overwrite)
normalize/   entity resolution: team & player name canonicalization (single mapping table,
             unit tests on every alias we hit), timezone-safe kickoff times
features/    per-match feature rows: Elo diff, rest days, decay-weighted rolling rates
             (corners/cards/SoT for+against per 90), referee strictness, importance flag
models/      goal_model.py, counts_model.py, player_model.py, market_devig.py — each fit()
             /predict() with versioned parameters
simulate/    match_simulator.py → 100k draws → answer_market(question) for any archetype
blend/       logit-space ensembling, calibration layer
submit/      API client: discover markets, map question text → simulator query (the parser
             is a real component — archetype regexes + LLM fallback with human review),
             batch submit, PATCH on lineup news
backtest/    the harness below
```

**Cleaning rules that actually matter here:** dedupe replayed/abandoned matches; verify half splits sum to
full-match totals; cross-check StatsBomb aggregates vs FBref for the same matches (catches parser bugs);
flag friendlies (lower intensity — separate importance covariate, never equal weight with competitive
matches); penalty shootout goals excluded from goal models.

### 4.3 Backtesting harness (this is where the rigor lives)

Replay **WC 2018, WC 2022, Euro 2020, Euro 2024, Copa América 2024, AFCON 2023** as if live — all six are
covered by StatsBomb open data (verified above), so settlement of every count and player archetype comes
from event data, not third-party aggregates:

1. Freeze all data to what was knowable pre-kickoff (point-in-time joins — the reason for append-only ingest).
2. Auto-generate the question archetypes from the brief for every match (we know the templates).
3. Produce probabilities from each model variant; settle against actual outcomes from event data.
4. Score: **mean Brier per archetype**, vs three baselines:
   - climatological base rate (e.g. "P(2+ second-half cards)" historical frequency),
   - de-vigged closing odds (where they exist) — *the bar to beat for main markets*,
   - a "naive crowd" proxy (odds rounded to nearest 5, regressed 20% toward 50 — approximates the field).
5. **Reliability diagrams + calibration error** per archetype. If "70%" events happen 63% of the time, fit a
   correction (Section 5).
6. Ablations: with/without market anchor, with/without referee covariate, decay half-life sweep, blend
   weight `w` sweep. Keep what pays in out-of-sample Brier; discard what doesn't.

Guard against overfitting: 6 tournaments ≈ 314 matches (64+64+51+51+32+52) ≈ 3,000+ question-settlements —
enough for main archetypes, thin for rare ones (penalties). For thin archetypes, stay close to base rates
by design. AFCON and Copa also stress-test the model away from European conditions — useful given a
48-team World Cup full of cross-confederation matchups.

---

## 5. Bayesian estimation, uncertainty, and calibration

### 5.1 Hierarchical priors (why Bayes here is not decoration)

National teams play ~10 matches/year — frequentist point estimates of attack/defence ratings are noise.
Everywhere we "shrink", that's a Bayesian posterior in disguise; we make it explicit:

- Team ratings get a **hierarchical prior** centered on what Elo and confederation predict:
  `attack_i ~ Normal(f(Elo_i), σ²)`. A team with 6 data points sits near its prior; a team with 60 earns its
  own estimate. This is exactly the partial pooling that wins on out-of-sample error.
- Player rates: `SoT/90_player ~ prior centered on position × league-strength mean`. A striker with 400
  career minutes gets heavy shrinkage; Son Heung-min doesn't.
- Fit via penalized MLE (MAP — fast, same math as ridge) for the production loop, with a full posterior
  (PyMC/Stan or a Laplace approximation) fit weekly to check that MAP isn't hiding multimodality.

### 5.2 Posterior predictive simulation — propagating parameter uncertainty

A subtle but important upgrade to the simulator: instead of simulating 100k matches at the point estimate
`λ̂`, **each Monte Carlo draw first samples the parameters from their posterior**, then simulates the match:

```
for each draw:  θ ~ posterior(θ | data)   →   match ~ Model(θ)
```

This is the *posterior predictive distribution*. The effect: probabilities get pulled toward the middle by
exactly the amount our parameter uncertainty warrants. A fragile estimate that says "78%" becomes "73%"
after honestly admitting we don't know λ precisely — and that 5-point difference is pure Brier savings when
the fragile estimate is wrong. Overconfidence at the extremes is the #1 killer under Brier scoring, and this
kills it at the source rather than patching it post-hoc.

### 5.3 What credible intervals are for (and what they're not for)

One thing to be clear-eyed about: under a proper scoring rule you submit the **posterior mean** —
a credible interval on the probability does not change the optimal submission. Its value is upstream:

- **Blend weights:** when the model's CI for a market is wide (thin data, weird matchup), shift ensemble
  weight toward the market anchor and base rates; when it's tight, let the model speak. Operationally:
  per-question weights `w ∝ 1/σ²` (precision weighting), the classic optimal-combination result.
- **Bot divergence sizing (§6):** Bot B's deliberate deviation from the consensus is sized within the
  model's credible region — we diverge where we have *grounds* to diverge, never beyond what the posterior
  supports.
- **Monitoring:** a settled outcome landing far outside the posterior predictive band is a red flag for a
  data bug or model misspecification — much more diagnostic than a single bad Brier score.

### 5.4 In-tournament Bayesian updating

Ratings are not frozen on June 11. After every match, posteriors update: yesterday's posterior is today's
prior, with the new match entering the likelihood (this is the state-space/Kalman view of Elo —
[Rue & Salvesen 2000; JRSS-C 2025](https://academic.oup.com/jrsssc/article/74/3/717/7929974)). Tournament
group games are highly informative (common conditions, current squads) — but small-sample: 2 matches must
shift beliefs by the *posterior-correct* amount, not by recency hype. The hierarchical machinery does this
automatically; the field, anchoring on narratives ("Brazil looked shaky"), will over- or under-update — a
recurring edge in knockout rounds, exactly where the 2×/3× multipliers live.

### 5.5 Output calibration layer

Models are systematically overconfident at the extremes even after §5.2. After backtesting:

- Fit **Platt scaling** (logistic: `p' = σ(a·logit(p) + b)`) per archetype family — only 2 parameters, won't
  overfit our sample sizes. Isotonic regression only if we have >500 settlements for that family.
- **Shrinkage to base rate** for low-data markets: `p' = (n·p + k·p₀)/(n + k)` — a Beta-prior posterior mean
  where `k` is our distrust of the model.
- Final clip to [0.01, 0.99] and round to integer percent (API requires 1–99).

---

## 6. Playing the relative game

### 6.1 The game theory, precisely

Per question, points = `B̄_field − B_you`. Two different objectives give two different optimal strategies,
and it pays to know which one you're playing:

- **Maximizing expected cumulative points:** `B̄_field` is outside your control — an additive constant per
  question. So the objective reduces to minimizing your own expected Brier, and the strictly-proper-rule
  result applies unchanged: **report your honest posterior mean.** The brief's "you can't game it" claim is
  exactly right for this objective.
- **Maximizing P(finishing #1):** a *rank* objective is not linear in points, and here the classic result of
  [Lichtendahl & Winkler (2007), "Probability Elicitation, Scoring Rules, and Competition Among Forecasters"](https://pubsonline.informs.org/doi/10.1287/mnsc.1070.0729)
  kicks in: when forecasters care about relative standing, proper scoring rules stop being proper, and the
  equilibrium strategy **exaggerates toward your private information and away from the crowd consensus** —
  more extreme probabilities than your honest belief. (See also
  [Witkowski et al., "Incentive-Compatible Forecasting Competitions"](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4410),
  which exists precisely because standard contests like this one are *not* incentive-compatible for rank-seekers.)

The resolution isn't to pick one — it's to note that **risk appetite should depend on leaderboard state**.
Ahead and protecting a lead → honest/conservative (maximize expected points, minimize variance). Behind with
the field compressing → controlled extremizing buys rank-jump probability at a small expected-points cost.
This is the same logic as trailing-team strategy in any cumulative contest.

### 6.2 Modeling the field

We can't see others' predictions before settlement, but `GET /results` returns the **field-average Brier per
settled question**. Given the known outcome `y`, the field's mean squared error per question lets us
reverse-engineer how the field behaves by archetype: e.g., persistent overconfidence on favorites, lazy
50s on offsides questions, herding to round numbers on odds-quoted markets. We maintain a running **field
model** (initialized as "de-vigged odds shrunk 20% toward 50 where odds exist; base rate ± noise elsewhere",
then updated against observed field Briers). The field model tells us *where the points are*: archetypes
where `E[B̄_field]` is large and our backtest Brier is small are the high-yield questions — worth the most
research time per question.

### 6.3 Two-bot implementation

Relative scoring means correlation with the field is wasted variance. The two API bots are two points on the
honest-to-competitive spectrum:

- **Bot A — "Anchor":** market-heavy ensemble, fully honest posterior means. Maximizes expected points;
  reliably collects on obscure markets where the crowd guesses, matches-or-beats on main markets. This bot
  should track near the top steadily.
- **Bot B — "Alpha":** model-heavy ensemble + a *bounded* Lichtendahl-Winkler tilt: where our posterior and
  the estimated field disagree, extremize away from the field — but never beyond the edge of our posterior
  credible region (§5.3), and with tilt size increasing in stage multiplier (group 1× ≈ honest; knockout
  2×/final 3× ≈ fuller tilt, since late questions dominate rank). Higher variance, decorrelated from the
  herd — the rank-seeking bot.

Bot A's deviations from honesty: zero. Bot B's deviations are deliberate, bounded, and only deployed where
the field model says the crowd is predictably wrong — that's strategy under a known objective, not
miscalibration. Track both bots' realized points by archetype; if Bot B's tilt isn't paying after ~30
settled questions, dial it back toward A.

---

## 7. Operations cadence (per matchday)

```
T−24h   referee announced → refresh card model; odds snapshot; submit initial batch
T−3h    odds refresh (sharp closing line forming); PATCH main markets
T−90m…T−60m   official lineups land (exact timing confirmed matchday 1) → minutes model goes from
        probabilistic to known; PATCH all player props
T−15m   final odds snapshot → final PATCH; verify all markets answered (answer everything, §0.2)
post    settle, log Brier vs field average per archetype, weekly recalibration review
```

---

## 8. Build phases

1. **Phase 1 — Data foundation (build first, everything depends on it):** ingest + normalize for Kaggle
   results, Elo, StatsBomb open data; point-in-time storage; entity resolution tables.
2. **Phase 2 — Goal model + simulator:** Dixon-Coles/Elo hybrid, half splits, scoreline simulator; backtest
   on match-result and goals archetypes vs odds baseline.
3. **Phase 3 — Market anchors:** odds ingestion + Shin/power de-vig; Polymarket/Kalshi ingestion with
   liquidity-weighted entry; λ inversion; logit-space ensemble; re-backtest (including the GBM hybrid, §2.6).
4. **Phase 4 — Count models:** corners, cards (referee!), offsides, SoT; integrate into simulator; backtest
   count archetypes vs base-rate baseline.
5. **Phase 5 — Player props:** minutes + rate models from FBref/StatsBomb.
6. **Phase 6 — Calibration + question parser + submission client:** Platt per archetype, API integration,
   dry-run on live markets without submitting, then go live.
7. **Phase 7 — Live ops:** matchday cadence, monitoring dashboard (cumulative RBP, Brier by archetype),
   weekly model review.

Phases 1–3 alone produce a competitive bot on main markets. Phases 4–5 are the edge. Given the tournament
starts now, ship 1–3 fast, run Bot A on it, and bring 4–6 online during the group stage (markets exist for
every match, so early simple submissions still collect points).

---

## 9. Assumptions register

Every empirical number in this plan is either (a) verified against a primary source, or (b) listed here as
an assumption with its verification step. Nothing else gets trusted.

**Verified against primary sources (2026-06-11):**
- All API mechanics in §0 — re-fetched from the [API docs](https://sportspredict.com/probabilitycup/api):
  integers 1–99 with 0/100 rejected, stage multipliers 1×/2×/3×, PATCH before close, close = match start,
  batch ≤50, 60 req/min, 2 bots with **separate leaderboard entries**.
- StatsBomb open-data coverage — checked directly against the repo's `competitions.json`: all six backtest
  tournaments present (IDs in §4.1 table).

**Assumptions pending empirical verification (owner: the pipeline, not vibes):**

> **Execution update (2026-06-12):** measured from the 6-tournament event data —
> A2 = 0.556 second-half goal share · A3 = 1.84× cards 2H/1H · A4 = 2.29× referee
> spread *in-sample, but factors do NOT transfer out-of-sample at tournament sample
> sizes — live model uses FBref career rates* · A5 = 0.11 corners–goals corr (0.60
> corners–shots) · A6 = P(≥1 pen) 0.293 · A10 = anchor weight ~0.85 optimal
> (flat 0.65–1.0), fit on WC22+Euro20 vs 538 forecasts. A1, A7–A9 remain for live.

| # | Assumption | Current basis | Verification step |
|---|---|---|---|
| A1 | ~~Both bots can predict the same market~~ **RESOLVED NEGATIVE (2026-06-12):** second key gets "Prediction for this market already exists in this lobby" — limit is per ACCOUNT. All submissions go through KEY_A as one consolidated entry; KEY_B is backup/ops. The two-strategy design (§6.3) is moot; we submit the backtested pure calibrated model. | live A1 test | done |
| A2 | Half-goal split ρ ≈ 0.46/0.54 | general empirical lore | Estimate from 6 backtest tournaments (Phase 2) |
| A3 | 2nd half ≈ 2× the cards of the 1st | club-football studies | Estimate from StatsBomb events (Phase 4) |
| A4 | Referee strictness spread ≥2× across referees | club-level data | FBref referee logs for the assigned WC referee pool (Phase 4) |
| A5 | Corners–goals correlation weak | literature | Estimate from StatsBomb (Phase 4) |
| A6 | P(≥1 penalty per match) ≈ 25–35% | WC 2018: 29 pens/64 matches (≈0.45/match → 36% under Poisson); 2022 lower | Recompute from all 6 tournaments incl. VAR-era only (Phase 4) |
| A7 | Lineups public 60–90 min pre-KO | typical FIFA practice | Observe matchday 1, lock the cadence (§7) |
| A8 | Sharp closing line ≥ deep prediction market accuracy | efficiency literature | Backtest both on settled 2022/2024 markets (Phase 3) |
| A9 | Field ≈ de-vigged odds shrunk ~20% toward 50 | prior guess | Calibrate against observed field-average Briers from `GET /results`, rolling (Phase 7) |
| A10 | Blend weight w ≈ 0.7–0.85 on main markets | literature prior | Fit by backtest (Phase 3); never hand-set |
| A11 | Game-state dynamics transfer across tournaments | **measured (v2, 2026-06-12, `scripts/estimate_dynamics.py`):** goal multipliers trail2+/trail1/lead1/lead2+ = 1.33/1.28/0.90/1.18; corners 1.40/1.37/0.61/0.58; SoT 1.67/1.43/0.55/0.41; cards late-bucket e^1.70 ≈ 5.5×; goal-family estimation must use full likelihood (totals-conditioning biased under feedback — synthetic-verified) | point-in-time per-tournament refits in `backtest_v2.py`; coupling ablation |
| A12 | Lineup XI-share exponent gamma = 1 | not estimable without lineup panels | revisit after group stage with live lineup data |
| A13 | Tournament roster proxy = players who appeared in tournament | rosters public pre-tournament; residual selection bias unmeasured | acceptable for the (rejected) squad backtest only |
| A14 | v2 simulator covers regulation 90' only | knockout "win" market settlement semantics (90' vs ET) unknown | observe first knockout settlement; extend simulator if needed |
| A15 | P(win given scored first) logit shift +1.34 | measured 0.757 over 15,381 internationals (first-goal combo markets) | recompute from live settlements (Phase 7) |

> **v2 execution update (2026-06-12):** squad-quality feature **REJECTED** (ridge-MAP
> b_sq ≈ ±0.01, OOS Brier 0.2206 vs 0.2204 plain — `scripts/backtest_squad.py`; the
> unregularized fit pegged at the bound on 48 training matches, a cautionary tale).
> GBM ensemble member **REJECTED** (standalone 0.2068 vs v1 0.2020; expanding-OOS blend
> weight chose w=0 in 4/5 tournaments — `scripts/backtest_gbm.py`). MC convergence:
> ±0.005 across seeds at 100k draws (`test_seed_convergence`); backtests run at 50k.
> v2 Platt refit: `data/processed/v2_calibration.json` (narrow-support archetypes
> intercept-only).
