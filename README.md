# jump-prob-challenge

Forecasting system for the Jump Trading Probability Cup (SportsPredict, WC 2026).
Design rationale and math: [PLAN.md](PLAN.md). Competition rules: [brief.md](brief.md).

## Setup

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python scripts\fetch_data.py      # results + Elo inputs + StatsBomb metadata
.venv\Scripts\python scripts\fetch_events.py    # ~2 GB StatsBomb event files (Phase 4+)
.venv\Scripts\python -m pytest -q               # 34 tests
```

## Pipeline (what exists, all backtested point-in-time)

| Component | Module | Validation |
|---|---|---|
| Ingest + entity resolution | `ingest/`, `normalize/` | cross-source score reconciliation, dedupe rules |
| Elo (self-computed, point-in-time) | `features/elo.py` | top-10 matches eloratings.net elite |
| Goal model (Elo-anchored Poisson + Dixon-Coles) | `models/goal_model.py` | synthetic parameter recovery; 6-tournament backtest |
| De-vig (basic/power/Shin) + λ inversion + blending | `models/devig.py`, `models/market_anchor.py` | round-trip tests; blend weights fit on WC22+Euro20 vs 538 forecasts |
| Count models (corners/cards/offsides/SoT) | `models/count_model.py` | corners +0.045, SoT +0.036 Brier edge vs climatology |
| Player props (shrunk rates) | `models/player_model.py` | SoT≥1 +0.016 edge |
| Platt calibration + integer formatting | `models/calibration.py` | synthetic overconfidence recovery |
| Question parser (10 brief archetypes) | `submit/parser.py` | all brief questions parse; unknown → review |
| API client (dry-run default, throttled, logged) | `submit/api_client.py` | — |
| End-to-end | `pipeline.py` | demo: `Pipeline().predict_fixture("Czech Republic", "South Korea")` |

## v2 engine (shadow mode — see V2-PLAN.md / V2-FINDINGS.md)

| Component | Module | Validation |
|---|---|---|
| Minute event stream | `ingest/event_stream.py` | 16,606 events / 314 matches |
| Game-state dynamics (no hand-set constants) | `models/dynamics.py`, `scripts/estimate_dynamics.py` | synthetic recovery; feedback-safe goal estimator |
| Minute-level MC simulator (coupling, frailty, posterior predictive) | `simulate/mc_engine.py`, `models/uncertainty.py` | reduction-to-analytic, coherence, seed convergence ±0.005 @100k |
| v2 backtest + ablations | `scripts/backtest_v2.py` | six tournaments, point-in-time incl. dynamics |
| v2 calibration | `scripts/calibrate_v2.py` | Platt refit on v2 outputs, slope guard |
| Squad features | `features/squad.py`, `scripts/backtest_squad.py` | **rejected** by OOS decision rule (lineup hook retained) |
| GBM hybrid | `models/gbm_model.py`, `scripts/backtest_gbm.py` | **rejected** by OOS decision rule |
| Shadow ops | `scripts/shadow_v2.py`, `scripts/score_shadow.py` | logs v2 vs submitted v1 per market; scores on settlement |

## Scripts

- `scripts/run_backtest.py` — Phase 2 match-archetype backtest (half-life sweep)
- `scripts/fit_blend.py` — ensemble weight fitting vs FiveThirtyEight anchor
- `scripts/estimate_counts.py` — measures assumptions A2–A6 from event data
- `scripts/backtest_counts.py`, `scripts/backtest_players.py` — count/player backtests

## Going live

1. Create two API keys on SportsPredict; set `SPORTSPREDICT_KEY_A` / `SPORTSPREDICT_KEY_B`.
2. Set `THE_ODDS_API_KEY`; snapshot odds via `ingest/odds_api.py`, de-vig with Shin,
   pass as `anchors=` to `Pipeline.predict_fixture` (Bot A w=0.85, Bot B w=0.50).
3. `SportsPredictClient(..., dry_run=True)` first; verify the submission log in
   `data/submissions/`, then flip to live. Test assumption A1 (both bots, same
   market) on one cheap market on matchday 1.
4. Matchday cadence: PLAN.md §7. Refit calibration weekly from `results` endpoint.
