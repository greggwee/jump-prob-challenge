# jump-prob-challenge

A forecasting system built to compete in the **Jump Trading Probability Cup**, a prediction contest run on SportsPredict over the 2026 World Cup. Before each match the platform posts a slate of binary questions — *Will Brazil win? Will Mbappé manage two shots on target? Will a goal be scored before the first hydration break?* — and entrants submit a probability for each one.

This repository prices those questions without a human at the desk. It ingests football data, fits models, anchors them to bookmaker odds, walks each answer through a stack of calibration and safety layers, and submits through the contest API on a schedule, including a final pass on the closing line roughly two minutes before a match locks.

Two bots ran off this code. **gigabot1** is the calibration-first entrant; **gigabot2** is an experimental rank-chaser that takes more risk where we had measured an edge.

**It finished 67th of 4,013 entrants, and 16th of 498 in the bot category** — the top 1.7% overall. Along the way the platform's own analytics put it at +2.7 relative Brier points per forecast against the crowd, better than 80% of forecasters, with a calibration rating of "elite". The contest rules are reproduced in [brief.md](brief.md).

## The insight the whole system rests on

Scoring is Brier loss measured against **the average individual forecaster**, multiplied by a stage weight that peaks at the final. That baseline is soft, and the reason is Jensen's inequality: the *average of individuals* is a good deal worse than the *consensus of individuals*. Averaging errors is not the same as averaging opinions, and the gap between the two is free money.

Which leads to an unglamorous conclusion. You do not need to beat the bookmaker. You need to beat the average punter, and a sharp de-vigged bookmaker price already does that comfortably, before anyone gets clever.

So the system is **market-first**: anchor to the book wherever the book has an opinion, and spend the modelling effort on the two places it cannot help. First, the many questions no bookmaker quotes — hydration-break windows, whether a substitute will score, how many different players will get a shot away. Second, coherence, so that the answers to forty questions about one match cannot contradict each other.

The second rule follows from the stage multiplier. Because the knockouts double and then triple every mistake, the standing instruction is **calibration over cleverness**: no deviation from a well-calibrated price without measured evidence that this particular kind of deviation has paid before.

## How a question gets priced

The pipeline is a ladder. Each rung only moves the number if it has something real to add.

**A statistical base.** Self-computed, strictly point-in-time Elo feeds a goal model that turns team strength into scoring intensities, with a Dixon–Coles correction producing a full scoreline grid. Above that sit count models for shots, corners, cards, fouls and offsides — negative binomial, with means driven by match tempo — plus a minute-level intensity profile for timing questions and per-90 player rates shrunk toward priors. The data comes from martj42's international results, StatsBomb event files, api-football league rates, and The Odds API for live prices.

**A market anchor.** Rather than anchoring each question to its own book line, the system de-vigs the 1X2 and totals markets, inverts them into implied goal intensities, and blends those with the model's at 85% market weight. Every goal-family question then derives from that same anchored grid. Player props anchor harder still, at roughly 97%, because on stars the book is simply better informed than any public model. A name-resolution layer handles the joins that break otherwise: "Erling Haaland" against "erling braut haaland", or Trezeguet against Mahmoud Hassan.

**Question parsing.** An ordered pattern list maps each question's text to an archetype and a handler. The design rule was learned the hard way, and it is the single most important lesson in the repository: every knockout round introduces new phrasings, so anything that fails to parse is **loudly ledgered** rather than quietly skipped. A misrouted question that prices confidently wrong costs far more than one left unanswered.

**A narrow LLM layer.** An offline panel of models reviews the numbers for things statistics cannot see: a confirmed lineup, a suspension, a change of penalty taker. The live pipeline never calls a model — it reads the panel's cached output, and follows it against our own numbers only on hard, verifiable evidence. Opinions do not move prices.

**Calibration and safety.** Per-family Platt calibration, a confidence floor, and a risk gate that pins the archetypes where our own match-to-match variation measured as *anti*-signal — fouls and offsides — near their flat calibrated base. Every submission runs through audits: a book-versus-model divergence flag, a name-join check, an anchor-regression guard so a run whose book line momentarily vanishes cannot revert an anchored prop to the bare model, and a crowd-proxy clamp that makes large unbacked deviations structurally impossible.

## What actually ran

Windows Task Scheduler drove five jobs. A nightly learning job refreshed results and refitted every calibration artefact, including the crowd priors harvested from settled matches. Two panel jobs prepared the LLM review and scraped team news near kickoff, writing lineup statuses that the pricing run consumes. Two submit ticks fired in the closing window before each match: refresh the odds *and* the player props — lineups drop about an hour out and the book reprices on them — then re-price the whole slate and PATCH only what changed.

Everything is dry-run by default. Going live is an explicit opt-in:

```powershell
.venv\Scripts\python scripts\run_matchday_odds.py                     # dry-run a full pricing pass
.venv\Scripts\python scripts\run_matchday_odds.py --live --matches "ESP vs ARG" --refresh-props
```

## How the system changed over the tournament

The interesting story is not the architecture, which is conventional enough. It is that almost every significant change was forced by evidence from matches already played, and that the system got steadily *less* aggressive as it learned more.

**Weeks one and two: building, and discovering the data holes.** The first version had Elo, a Dixon–Coles goal model, count models, a question parser and a submit path. What it did not have was players. A coverage census found that Understat's club rates missed 44% of the squads being asked about; adding api-football's broader league coverage cut that hole to 16%. In the same stretch two quiet parser bugs surfaced — a greedy player regex was swallowing "both teams have at least one shot on target" and pricing it around 9% against a fair 75%, and offsides carried a definition gap between StatsBomb and the platform. Both were bleeding points invisibly, which set the tone for everything that followed.

**The first settled matches changed the plan.** Once real outcomes and crowd numbers arrived, an assumption collapsed: the crowd consensus is only visible *after* a market locks. It could never be a live signal. It was demoted to what it actually is — a calibration input, harvested post-lock and refitted nightly. That distinction is enforced in the code to this day.

**The LLM layer arrived, then got smaller.** It went in over a single day in stages: shadow-only plumbing, a forward-test gate, a fitted per-archetype gate, and finally a near-kickoff web-search pass for team news. The ambition had been a tactical blender that would improve estimates broadly. The measurements did not support it. What survived is narrow: the panel is a verifier for player props and no-edge markets, it is excluded entirely from anything the book prices sharply, and archetypes where we had proven edge are protected from it outright.

**Two safety layers turned out to be fighting each other.** The extremizer amplifies deviation from the field where a family has earned leverage; the risk gate shrinks no-edge archetypes back toward their base rate. For a while both were active on the same questions, so one layer inflated exactly the noise the other then had to deflate. Excluding risk-gated archetypes from extremizing settled it. This is the kind of bug that never throws an exception and never shows up in a test.

**The knockouts forced a market-first rewrite.** Group-stage phrasings had been stable; the round of 32 was not. A drop audit counted how many live questions the parser was silently discarding, and the answer was 56. Closing that gap drove roughly a week of work: knockout phrasings recovered, hydration-break windows derived analytically from the minute profile (and then corrected — the breaks fall at the 22nd minute of each half, not the midpoint), advancement modelled properly as *winning the tie* through extra time and penalties rather than winning in 90 minutes, and player props anchored to de-vigged book lines. The audit went 56 → 12 → 0. Backtesting the recovered markets against settled outcomes showed +0.023 Brier of skill, so this was not merely tidying.

**One bad match produced most of the safety stack.** A 92-point loss traced to two causes: a guard that backfired, and a question about a team scoring that was misrouted to a *player* handler, so an obscure nation was priced like an unknown striker at 8% when the true number was nearer 40%. The response was structural rather than a patch. A crowd-proxy clamp now makes large unbacked deviations impossible to submit, since historically every one of them had been a bug. Fouls and offsides, where per-match variation measured as anti-signal, were pinned near flat. A template audit began surfacing every unpriced question loudly at submit time. Even the logging was hardened, after a Unicode error in a `print` statement killed a submission run outright.

**Each subsequent round added templates, and each addition was audited.** Six new question types for the round of 16, five for the quarter-finals, eight more as phrasings proliferated, then a further wave for the semi-finals — a first scorer's shirt number, which side substitutes first, whether VAR sends the referee to the monitor. The discipline was to build each from primitives already validated, then check the result against the crowd. That check earned its keep: a routine audit found the correlation between "more corners" and "more shots" set six times too low, which had made it the one handler positioned to lose money against the field.

**The most expensive bugs were name joins, repeatedly.** A player subject formatted as "Kylian Mbappé (France, #10)" matched nothing, so the best striker on the pitch collapsed to the population prior. De Bruyne was benched against the United States, the book had not repriced by our closing tick, and a crowd lift then pushed a stale starter's price *higher*. Nordic characters were being dropped rather than transliterated, so Ødegaard and Sørloth silently lost their book anchors. Each fix is now a test.

**The endgame was about resisting temptation.** With stage multipliers at their highest, the obvious move is to gamble. A Monte-Carlo simulation of the endgame said otherwise: blanket variance has zero mean and the multipliers amplify losses just as eagerly as gains, giving +481 ± 379 for honest reporting against +475 ± 760 for a blanket gamble. Same expectation, twice the spread. So the variance dial that exists is off by default, restricted to a short list of kinds where the crowd demonstrably uses naive heuristics, and requires someone to set an environment variable deliberately. For the final and third-place matches, judgement was staged in advance instead: conditional rules that fire only when a confirmed lineup matches, so a decision like "if Kane is benched, the substitute-scores question moves up" executes automatically at the closing tick.

**A recurring pattern, worth stating plainly.** Several constants in this repository sit deliberately below their fitted optimum. The player-prop crowd lift is the clearest case: the Brier-versus-shift curve is monotone with no interior optimum, the underlying edge is borderline at p ≈ 0.065, and its direction is close to a coin flip. A fitted value would have been a confident answer to a question the data cannot resolve. The shipped value keeps most of the modelled gain for about half the downside, and the cap above it does the real work. Where the estimate is honest, ship it; where the estimate is unidentified, ship less than you think.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python scripts\fetch_data.py       # results, Elo inputs, StatsBomb metadata
.venv\Scripts\python scripts\fetch_events.py     # StatsBomb event files
.venv\Scripts\python -m pytest -q
```

Secrets live in `.env`, which is never committed: `SPORTSPREDICT_KEY_A` and `_B`, `THE_ODDS_API_KEY`, `API_FOOTBALL_KEY`, `ANTHROPIC_API_KEY`.

## Repository layout

```
src/jumpprob/
  ingest/       results, StatsBomb, api-football, Odds API + player props
  normalize/    team and player entity resolution
  features/     Elo
  models/       goal model (Dixon-Coles), counts, de-vig, market anchor,
                minute dynamics, calibration, strategy gates
  simulate/     scoreline-grid markets + minute-level Monte-Carlo engine
  agents/       offline LLM expert panel + gated verifier
  submit/       constants.py   every tuned number, with its justification
                parser.py      question text -> archetype
                handlers/      one handler per archetype, grouped by family
                runner.py      match context + the pricing ladder
                api_client.py, audit.py, env.py
  pipeline.py   end-to-end wiring

scripts/        the scheduler-invoked live chain + model-fitting utilities
data/processed/ fitted artefacts (raw feeds and run-time ledgers are git-ignored)
docs/           METHODS.md — the mathematics and the validation of each layer
                MARKET-FIRST-ARCHITECTURE.md — why the market anchor is the spine
tests/          pytest suite, 206 tests
```

Two files repay a look before the rest. [`submit/constants.py`](src/jumpprob/submit/constants.py) is where the judgement lives: every tuned number, what it was measured on, and whether to trust it. [`submit/runner.py`](src/jumpprob/submit/runner.py) documents the eleven-step pricing ladder in its module docstring, and the order of those steps is itself a set of decisions — confirmed team news is applied last, after the book, the crowd and the models, because it beats all of them.

For the mathematics behind each question archetype and the validation of every layer against settled data, start with [docs/METHODS.md](docs/METHODS.md) and [docs/MARKET-FIRST-ARCHITECTURE.md](docs/MARKET-FIRST-ARCHITECTURE.md).
