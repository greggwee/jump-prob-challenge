# jump-prob-challenge

A live forecasting system for the **Jump Trading Probability Cup** (SportsPredict, FIFA World Cup 2026) — a contest of binary YES/NO questions on each match ("Will Brazil win?", "Will Mbappé have 2+ shots on target?", "Will a goal be scored from outside the box?").

Two automated entrants run off this codebase: **gigabot1** (Bot A, calibrated) and **gigabot2** (Bot B, a risk-seeking rank-chaser). Competition rules: [brief.md](brief.md).

---

## How you are scored (and why the design follows from it)

Each question is scored by **Brier loss relative to the average *individual* forecaster**, multiplied by a **stage multiplier** (knockout Round-of-32 = 2×). Two consequences drive every design choice:

1. **You are scored against the mean of individuals, not the crowd consensus.** By Jensen's inequality the aggregated consensus is *sharper* (lower Brier) than the average of the individuals' Briers — by ~0.026 Brier on this field. So **matching the sharp market consensus beats the scoring baseline almost by construction.** The system is therefore *market-first*: anchor to the de-vigged book, and only add structure where the book is silent.
2. **At 2×, calibration is paramount** — a single over-confident wrong call costs double. The guiding principle in knockout rounds is *no over- or under-confidence*: conservative blending, protect validated edges, and never move a well-calibrated price without evidence.

---

## Architecture

The pipeline is a stack: a **statistical base layer**, a **market-anchoring layer**, an out-of-band **LLM expert-review layer**, and **calibration/safety layers** on top. The live daemon never calls an LLM — it reads a cache the panel produced offline.

```
 ingest ─▶ Elo + goal model ─▶ Dixon-Coles scoreline grid ─┐
 (results,   (point-in-time)     (anchored to book λ)        │
  StatsBomb,                                                 ├─▶ per-question answer
  api-football,  book odds ─▶ de-vig ─▶ implied λ ──────────┘        │
  Odds API)      player props ─▶ threshold-aware anchor              │
                                                                     ▼
                          calibration · evidence-gated extremize · risk gate
                                                                     │
   LLM expert panel (offline) ─▶ ai_panel.json cache ─▶ verifier ────┤
   (reviews the model; hard-evidence only)                           ▼
                                              audits ─▶ submit (dry-run default)
```

### 1. Ingest & entity resolution — `ingest/`, `normalize/`
- International results + goal-scorers (martj42, live through WC26), StatsBomb event data, broad-league per-90 player rates (api-football), and live book odds + player props (The Odds API).
- Team and player name normalization (`normalize/`) — accent/middle-name/initial tolerant, so a subject like *"Erling Haaland"* joins the book's *"erling braut haaland"*.

### 2. Statistical base layer — `features/`, `models/`, `simulate/`
- **Elo** (`features/elo.py`) — self-computed, strictly point-in-time, with recency weighting.
- **Goal model** (`models/goal_model.py`) — Elo → team goal intensities `(λ_a, λ_b)`, a Poisson field with a **Dixon-Coles** low-score correction. Produces a full scoreline grid.
- **Count models** (`models/count_model.py`) — shots/SoT/corners/cards/offsides, which scale with the match's goal intensity.
- **Minute dynamics + Monte-Carlo** (`models/dynamics.py`, `simulate/mc_engine.py`) — for timing/window markets (hydration-break windows at 22′ of each half, stoppage-time goals) that the analytic engine only approximates.
- **Player props** — shrunk per-90 rates; multi-threshold SoT derived via the Poisson intensity (see Theory).

### 3. Market-anchoring layer — `models/devig.py`, `models/market_anchor.py`, `ingest/odds_api.py`
- De-vig book 1X2 + totals (basic / power / **Shin**) → **implied goal intensities** → blend with the model's λ at `w_market = 0.85` → **one coherent Dixon-Coles grid** from which *every* goal-family question (win, BTTS, team goals, N+ totals, halves, first goal) is priced consistently. Threshold-independent by construction.
- Player props anchor to the book's de-vigged scorer / SoT / assist lines.
- Safety audits run every submit (`submit/audit.py`): **name-join** (player has book data but no name match), **book-vs-model divergence** (a large gap flags a model data-gap or a market mismatch), and an **anchor-regression guard** (a run whose book line momentarily vanished must not revert a previously anchored prop to the bare model).

### 4. LLM expert-review layer — `agents/`
- An **out-of-band** multi-model panel (`agents/expert.py`) that **reviews** the statistical estimate rather than re-forecasting it — adjusting only for nuance the model is blind to: player availability, recent form, controversy, a coaching change. It emits a structured `evidence_strength` (hard / soft / none) plus factors, and can web-search live team news near kickoff.
- The live runner reads the panel's cache (`agents/panel.py`) and applies a **deconfliction verifier**: on a *large* model-vs-LLM disagreement it follows the LLM **only on hard, verifiable evidence** (confirmed XI, suspension) for player props, otherwise keeps the model. Gated per-archetype, shadow-by-default. See [docs/LLM-EXPERT-PLAN.md](docs/LLM-EXPERT-PLAN.md).

### 5. Calibration & risk — `models/calibration.py`, `models/strategy.py`
- Per-family Platt calibration, evidence-gated extremization, an asymmetric confidence floor, and an optional risk gate that shrinks no-edge archetypes toward the base rate.

### 6. Submission — `submit/`
- `submit/parser.py` classifies a question into an *archetype*; `submit/runner.py` prices it, applies the anchor/LLM/calibration ladder, and submits through a throttled, dry-run-by-default API client (`submit/api_client.py`).

---

## Theory

**Why market-first.** A de-vigged, liquid book is a sharp, professionally-priced estimate that our structural model cannot reliably beat on the markets it covers. Since we are scored against the (softer) mean-individual baseline, tracking that consensus already earns positive relative Brier. So we anchor heavily to the book and reserve the structural model for (a) markets the book doesn't quote and (b) coherence across related questions.

**Coherence from implied intensities.** Rather than anchoring each goal question to its own book line, we invert 1X2 + totals into implied goal intensities `(λ_a, λ_b)`, blend once, and derive *all* goal-family answers from the resulting Dixon-Coles grid. This guarantees the answers are mutually consistent (P(win), P(BTTS), P(3+ goals) all come from the same scoreline distribution) and is automatically correct for any threshold.

**Poisson thinning for goal-type markets.** Markets like "a goal from outside the box", "any player 2+ SoT", or a player's "2+ shots on target" are *thinnings* of a rate we already model. If goals are `Poisson(μ)` and a fraction `p` are of some type, the type-count is `Poisson(pμ)`, so `P(≥1) = 1 − e^{−pμ}` — an **unconditional** probability that correctly accounts for the chance no goal is scored at all. The share `p` comes from data (~13% of open-play goals are from outside the box). The same identity backs a player's shot intensity out of the book's 1+ SoT line (`μ = −ln(1−P≥1)`) to price 2+/3+ SoT with the book's player ranking instead of a flat model. **We validate the thinning before trusting it** — for own-goals it was *rejected* (the rate doesn't scale with total goals), so that market uses a flat, competition-calibrated base rate.

**LLM as reviewer, not oracle.** The model is blind to things a human reads off team news — who's actually starting, a suspension, a mid-tournament controversy. The LLM supplies exactly that nuance and nothing else: it reviews the model's number and moves it only when it can cite a *verifiable* fact. Frequencies and base rates (own-goals, offsides, card counts) stay with the data/statistical model, where the LLM has nothing to add. This keeps variance low at 2×.

**Calibration over cleverness.** The scoring rewards being right *about your uncertainty*. Every layer above the market anchor is conservative and shadow-validated on settled data before it can move a live submission, because at 2× an unbacked deviation from a calibrated price is a doubled loss.

Deeper math and validation: [docs/METHODS.md](docs/METHODS.md), [docs/MARKET-FIRST-ARCHITECTURE.md](docs/MARKET-FIRST-ARCHITECTURE.md), [docs/KNOCKOUT-AND-TIMING-ESTIMATION.md](docs/KNOCKOUT-AND-TIMING-ESTIMATION.md).

---

## The two bots
- **Bot A / gigabot1** — the calibrated entrant. Market-anchored, conservatively blended, calibration-first.
- **Bot B / gigabot2** — a risk-seeking rank-chaser: it extremizes where the model is *directionally right* vs the field (validated edge markets) and stays flat where a coin-flip, aiming for top single-match relative Brier (`scripts/submit_gigabot2.py`).

---

## Live operation

Windows Task Scheduler runs five jobs (both bots share the base layer):

| Task | Script | Cadence | Role |
|---|---|---|---|
| DailyLearn | `daily_learn.py` | daily | refresh results/Elo, re-fit calibration + gates, score shadow hypotheses |
| AIPanel | `ai_panel_live.py` | a few ×/day | generate the LLM panel for the imminent slate (gated markets only) |
| WebPanel | `ai_panel_web.py` | near kickoff | web-search team news + top up book props for the imminent match |
| BotA-Submit | `scheduler_tick.py` → `run_matchday_odds.py` | per kickoff | refresh odds, price, submit Bot A |
| BotB-Submit | `scheduler_tick_b.py` → `submit_gigabot2.py` | per kickoff | submit Bot B |

The submit path is **dry-run by default**; `--live` is required to PATCH the contest.

---

## Setup

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python scripts\fetch_data.py       # results + Elo inputs + StatsBomb metadata
.venv\Scripts\python scripts\fetch_events.py     # StatsBomb event files (for the count/dynamics models)
.venv\Scripts\python -m pytest -q                # test suite
```

Secrets live in `.env` (never committed): `SPORTSPREDICT_KEY_A` / `_B`, `THE_ODDS_API_KEY`, `API_FOOTBALL_KEY`, `ANTHROPIC_API_KEY`.

Preview a match end-to-end (no submission):

```powershell
.venv\Scripts\python scripts\run_matchday_odds.py           # dry-run: fetch odds, price all open markets, show before/after
.venv\Scripts\python scripts\run_matchday_odds.py --live    # PATCH Bot A
```

---

## Repository layout

```
src/jumpprob/
  ingest/       results, StatsBomb, api-football, Odds API + player props
  normalize/    team & player entity resolution
  features/     Elo
  models/       goal (Dixon-Coles), counts, de-vig, market anchor, dynamics,
                calibration, strategy (gates), uncertainty
  simulate/     scoreline-grid markets + minute-level Monte-Carlo engine
  agents/       LLM expert panel (out-of-band) + gated blend/verifier
  submit/       question parser, runner (pricing), API client, submit-time audits
  pipeline.py   end-to-end wiring

scripts/        live chain (scheduler-invoked) + artifact builders + utilities
docs/           methods, market-first architecture, knockout/timing, LLM plan
tests/          pytest suite
```

Model artifacts and gates live under `data/processed/` (e.g. `live_calibration.json`, `expert_gate.json`, `event_baselines.json`); raw feeds and runtime caches are git-ignored.
