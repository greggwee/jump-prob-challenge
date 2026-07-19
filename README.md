# jump-prob-challenge

This is a live forecasting system for the **Jump Trading Probability Cup** — a World Cup 2026 prediction contest run on SportsPredict. Every match, the platform posts a slate of binary questions ("Will Brazil win?", "Will Mbappé have 2+ shots on target?", "Will a goal be scored before the first hydration break?") and you submit a probability for each. This repo prices those questions automatically, end to end: it ingests data, builds models, anchors them to bookmaker odds, applies a stack of calibration and safety layers, and submits through the contest API on a schedule — including a final pass on the closing line about two minutes before each match locks.

Two bots run off this codebase: **gigabot1** (the main, calibration-first entrant) and **gigabot2** (an experimental rank-chaser that takes more risk where we have measured edges). As of the tournament's final weekend, gigabot1 sits around the top 7% of the leaderboard with the platform's own analytics rating it "elite calibration" — beating the crowd by about +2.7 points per question. The competition rules live in [brief.md](brief.md).

## The one insight everything follows from

Scoring is **Brier loss relative to the average individual forecaster**, times a stage multiplier that peaks at the final. That baseline — the *average* of individuals, not their aggregated consensus — is soft: by Jensen's inequality, the consensus is sharper than the average individual by a meaningful margin. So simply matching a sharp, de-vigged bookmaker price already beats the baseline, before any cleverness.

That makes the system **market-first**: anchor everything the book prices, and spend the modeling effort on (a) the many questions the book doesn't quote — hydration-break windows, "will a substitute score", distinct-shooter counts — and (b) keeping every answer mutually coherent. And because the multiplier doubles or triples mistakes in the knockouts, the second rule is **calibration over cleverness**: no deviation from a well-calibrated price without measured evidence that this kind of deviation has paid before.

## How a question gets priced

The pipeline reads like a ladder; each rung only moves the number if it has something real to add.

**Statistical base.** Self-computed, strictly point-in-time Elo feeds a goal model that turns team strength into goal intensities, with a Dixon–Coles correction producing a full scoreline grid. On top of that sit count models (shots, corners, cards, fouls, offsides — negative-binomial, scaled to match tempo), minute-level dynamics for timing questions, and per-90 player rates shrunk toward priors. Data comes from martj42's international results, StatsBomb event files, api-football league rates, and The Odds API for live prices.

**Market anchor.** Rather than anchoring each question to its own book line, we de-vig the 1X2 and totals markets, invert them into *implied goal intensities*, and blend those with the model's at 85% market weight. Every goal-family question — win, both-teams-score, 4+ goals, halftime states, first goal — is then derived from the *same* anchored grid, so the answers can't contradict each other. Player props anchor even harder (~97% book weight) because on stars, the book is simply better informed than any public model. A name-resolution layer handles the join ("Erling Haaland" ↔ "erling braut haaland"; Trezeguet ↔ Mahmoud Hassan).

**Question parsing.** A single ordered pattern list maps each question's text to an archetype and handler. The design rule learned over the tournament: every knockout stage introduces new phrasings, so anything that fails to parse is *loudly ledgered* by a template audit rather than silently skipped — a misrouted question that prices confidently wrong is far more expensive than a skipped one.

**LLM expert layer, used narrowly.** An offline multi-model panel reviews the model's numbers for things statistics can't see — a confirmed lineup, a suspension, a taker change. The live pipeline never calls an LLM; it reads the panel's cached output, and follows it against the model **only on hard, verifiable evidence**. Vibes don't move prices.

**Calibration and safety.** Per-family Platt calibration, a confidence floor, and a risk gate that pins archetypes where our per-match variation measured as anti-signal (fouls, offsides) to near-flat calibrated bases. Submissions run through audits every pass: a book-vs-model divergence flag, a name-join check, an anchor-regression guard (a run whose book line momentarily vanished can't revert an anchored prop to the bare model), and a crowd-proxy clamp that makes 25+-point unbacked deviations — historically all bugs — structurally impossible, while whitelisting the one extreme that reliably earns (low prices on fringe-player props the crowd overrates).

**Human-in-the-loop overrides.** Two small JSON files let judgment participate without touching code: `expert_overrides.json` pins a specific market to a value with a written justification, and `lineup_branch_overrides.json` holds *conditional* pins that fire only when a confirmed lineup status matches — so decisions like "if Kane is benched, the substitute-scores question goes up" can be staged in advance and execute automatically at the closing tick with nobody at a desk.

## What actually runs

Windows Task Scheduler drives five jobs. DailyLearn refreshes results and refits every calibration artifact each day (including the crowd priors, harvested from settled matches — crowd data is strictly post-lock calibration input, never a live signal). AIPanel and WebPanel prepare the LLM review and scrape team news near kickoff, writing lineup statuses the pricing run consumes. The two submit ticks fire in the closing window before each match: refresh the odds *and* the player props (lineups drop ~60 minutes out, and the book reprices on them — the props refresh exists precisely so we submit post-lineup prices), re-price the whole slate through the ladder above, and PATCH only what changed.

Everything is dry-run by default; `--live` is an explicit opt-in:

```powershell
.venv\Scripts\python scripts\run_matchday_odds.py                      # dry-run a full pricing pass
.venv\Scripts\python scripts\run_matchday_odds.py --live --matches "ESP vs ARG" --refresh-props
```

## Setup

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python scripts\fetch_data.py       # results, Elo inputs, StatsBomb metadata
.venv\Scripts\python scripts\fetch_events.py     # StatsBomb event files
.venv\Scripts\python -m pytest -q
```

Secrets live in `.env` (never committed): `SPORTSPREDICT_KEY_A` / `_B`, `THE_ODDS_API_KEY`, `API_FOOTBALL_KEY`, `ANTHROPIC_API_KEY`.

## Repository layout

```
src/jumpprob/
  ingest/       results, StatsBomb, api-football, Odds API + player props
  normalize/    team & player entity resolution
  features/     Elo
  models/       goal model (Dixon-Coles), counts, de-vig, market anchor,
                dynamics, calibration, strategy gates
  simulate/     scoreline-grid markets + minute-level Monte-Carlo engine
  agents/       LLM expert panel (offline) + gated verifier
  submit/       question parser, pricing runner, API client, submit audits
  pipeline.py   end-to-end wiring

scripts/        scheduler-invoked live chain + model-fitting utilities
data/processed/ fitted artifacts, overrides, ledgers (raw feeds git-ignored)
docs/           METHODS.md (the math and validation), MARKET-FIRST-ARCHITECTURE.md,
                plus the living review docs for the current slate
tests/          pytest suite
```

For the deeper material — the estimators behind each question archetype, the validation of every layer on settled data, and the review process that gates changes — start with [docs/METHODS.md](docs/METHODS.md) and [docs/MARKET-FIRST-ARCHITECTURE.md](docs/MARKET-FIRST-ARCHITECTURE.md).
