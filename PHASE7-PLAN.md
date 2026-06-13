# Phase 7 Plan — Odds anchor integration (Bot A accuracy)

> **EXECUTED & LIVE 2026-06-13.** Multi-book Shin consensus → λ-inversion →
> 0.85/0.15 log-λ blend, wired into `Runner.context()`. Ran on 68 fixtures /
> 677 open markets; **524 Bot A predictions PATCHed live**, verified by
> market_id on the API. Headline: USA-vs-Australia "USA win" 28 → **57**.
> 333/677 markets moved ≥3 pts, all directionally toward the market. Both
> pure-model and anchored values logged to `data/processed/odds_anchor_run.csv`
> for CLV/Brier measurement. 52 tests green. Refresh each matchday:
> `python scripts/run_matchday_odds.py [--live]`.

**Motivation (evidence).** USA vs Paraguay scored mean Brier 0.326 (worse than a
coin flip) because the pure-Elo model favoured Paraguay and USA won. "Will USA
win?" went in at 26 → Brier 0.55. Canada vs Bosnia, where Elo agreed with
reality, scored 0.094. The single biggest accuracy lever we are not yet pulling
is the closing line. Phase 3 already built and unit-tested the machinery
(`models/devig.py`, `models/market_anchor.py`); Phase 7 is **wiring a live feed
into it**, not new math.

## What the live Odds API actually gives us (verified 2026-06-13)

- Key `soccer_fifa_world_cup`, 68 games, ~496/500 monthly credits remaining.
- Markets: **h2h** (1X2: home/draw/away) and **totals** (over/under, usually 2.5).
- Books: paddypower, skybet, casumo, leovegas (UK/EU). Coverage varies per game;
  some games carry h2h only (no totals) → the λ-inversion must run on 1X2 alone
  (already supported — totals optional in `implied_lambdas`).
- Cost model: one `/odds` call = (#regions × #markets) credits. `regions=uk`,
  `markets=h2h,totals` = 2 credits/refresh for ALL 68 games at once. At ~3
  refreshes/matchday that is ~6 credits/day, ~180/month — well inside 500.

## What odds can and cannot anchor

| Market family | Anchor path | Expected benefit |
|---|---|---|
| win / draw | h2h → Shin de-vig → direct logit blend | **large** (the USA-type misses) |
| total goals, BTTS, half goals, first-goal combos, ht-tied | h2h+totals → λ-inversion → new (λ_h, λ_a) → DC grid / simulator | **large** (goal structure re-anchored) |
| corners / cards / offsides / SoT | λ feeds count-model means as a covariate | **second-order** (better λ ⇒ better counts) |
| player props | team λ scales player rates | small |

Odds do not directly price corners/cards/props at this tier, but **every
archetype depends on (λ_h, λ_a)**, so a better goal-intensity estimate
propagates everywhere. That is the elegance of the λ-inversion design.

## Implementation steps (file-level)

1. **`ingest/odds_api.py` (extend the stub).** `snapshot_odds()` already exists.
   Add: `load_latest_snapshot()`, and `match_odds(home, away)` that resolves a
   fixture to its odds via `normalize/teams.py` (book team names → canonical;
   add aliases + a test, same discipline as StatsBomb). Append-only raw with
   fetch timestamps (already done). **Risk:** book name spellings ("USA" vs
   "United States", "Korea Republic"); mitigated by the alias test gate.

2. **Consensus aggregation — `models/devig.py::consensus` (BUILT 2026-06-13).**
   Upgraded from "equal-weight average" after studying parkjpd's repo (see
   §"Wisdom-of-crowds findings"). Per fixture: de-vig EACH book's h2h with Shin,
   then average in probability space with weight
   `w = sharp_mult(book) / (overround + 0.005)`. The margin is an observable
   sharpness proxy — a 2% book outweighs an 8% book — and named sharp venues
   (Pinnacle/Betfair/Kalshi) get a modest extra nudge. Tested (5 cases incl.
   sharp-book dominance and prediction-market pseudo-books). Then
   `implied_lambdas(consensus)` → market (λ_h, λ_a) via OUR DC + least-squares
   inversion (kept over their independent-Poisson — strictly more accurate).

3. **`models/anchor_blend.py` (new, thin).** Blend model λ and market λ in log
   space: `λ_blend = exp(w·ln λ_mkt + (1−w)·ln λ_model)`. Then build the DC grid
   / run the simulator on λ_blend so ALL downstream archetypes inherit the
   anchor coherently (not just win/draw). `w` from Phase 3 backtest ≈ 0.85 for
   main markets; **re-fit on live settlements once ≥150 accumulate** (A10).

4. **`pipeline.py` (extend).** `predict_fixture(..., odds=match_odds(...))`:
   when odds present, replace (λ_h, λ_a) with the blend before answering;
   when absent (no book for that game), fall back to pure model (current
   behaviour). One code path, odds optional.

5. **`submit/runner.py` + `scripts/run_matchday.py`.** Snapshot odds at T−3h and
   T−15m; recompute; PATCH Bot A. The matchday cadence in PLAN.md §7 already
   specifies these refresh points — now they do something.

6. **Backtest gate (non-negotiable).** Before trusting it live: re-run the
   six-tournament point-in-time backtest with `football-data.co.uk` historical
   closing odds (free CSVs) joined to past tournaments; confirm the odds-blended
   model beats pure-model OOS on win/draw/totals. Only promote if it does.
   *Caveat:* historical international odds coverage is patchier than club; if a
   tournament lacks odds, that tournament stays model-only in the backtest.

## Plan verification (2026-06-13, before executing)

Walked the plan against the live code and the available data. Three corrections,
all making the plan more honest/feasible:

1. **The historical-book backtest gate is NOT runnable.** football-data.co.uk
   carries club leagues, not World Cup / international tournaments — there is no
   free historical international closing-odds source to join. *Resolution:* the
   blend **machinery and weight are already backtested** — Phase 3 fit w≈0.85 on
   84 real fixtures using FiveThirtyEight as the market-quality anchor (538 is
   dead now, so live books replace it; the blend math is identical, only the
   anchor source changes). Remaining validation is **live shadow logging** (log
   pure-model AND odds-blended every matchday; measure CLV and realised Brier as
   settlements arrive). Honest and sufficient; do not claim a backtest we cannot run.
2. **The live submission path applies no Platt calibration** (calibration lives
   in `pipeline.predict_fixture`, which the runner does not use). So the feared
   "pure-model Platt over-shrinks a market number" problem does not exist. Drop it.
3. **Integration point = `Runner.context()`.** Every `answer()` prices off
   `ctx.lam_a/lam_b/grid`, so building the context from BLENDED λ anchors win,
   totals, scores-in-half, ht-tied, counts (λ is their covariate) and player
   props (λ scales rates) coherently, with zero changes to `answer()`. Also:
   when anchored, skip Bot A's 25%-toward-base shrink (the market is already the
   conservative estimate; shrinking it toward league average would dilute it).

Everything else in the plan stands. Going **live** (not shadow-only) for the
goal-derived markets: the market is the efficient-market benchmark, the weight
is backtested, and we have live proof the pure model is systematically wrong on
some teams. Safety: de-vig sanity checks + a human-readable before/after table
before any PATCH.

## Verification & guardrails

- **De-vig sanity:** assert each de-vigged book sums to 1.0 ± 1e-6 and every
  implied p ∈ (0,1); log the overround per book (typical 1.05–1.08) — a book
  outside 1.02–1.15 is a parse error, skip it.
- **λ-inversion residual:** assert the inverted λ reproduces the input
  probabilities within tolerance; if the solver fails to converge (rare, thin
  odds), fall back to model λ and log it.
- **Calibration:** odds-blended outputs get their own Platt refit; do not reuse
  the pure-model table.
- **Staleness:** never anchor on odds older than the last lineup news; store the
  snapshot time and refuse anchors > 6h stale at submission.

## What can be cut / deferred

- Liquidity weighting across books (equal-weight is fine at this book count).
- Asian-handicap ingestion (h2h+totals already pin two λs).
- The `_winner` outright market (422s on h2h+totals; only useful for tournament
  futures, not per-match).

## Decision rule

Promote odds-blended Bot A **only if** it beats pure-model Bot A on the
historical-odds backtest AND on the first ≥50 live settlements where odds were
present. Until then, run it in shadow alongside v1 (same pattern as v2). This is
the same evidence discipline that rejected the squad feature and GBM.

---

## Wisdom-of-crowds findings (study of parkjpd/jumptrading-probability-cup, 2026-06-13)

That repo solves the same contest. Its "wisdom of crowds" is a **multi-book
consensus**, and studying it improved this plan in three concrete, well-backed
ways — plus two things I deliberately did NOT copy.

**Adopted (logical + backed):**
1. **Inverse-overround sharpness weighting** in the book consensus (above). Their
   `consensus()` weights each de-vigged book by `1/(overround+ε)`; the margin is
   a free, observable sharpness signal. This is strictly better than my original
   equal-weight plan and needs no tuning. *Implemented in `devig.consensus`.*
2. **Prediction markets as pseudo-books.** Their matchday-1 baseline folds
   **Kalshi** into the SAME de-vig+consensus machinery as a book (mid-price →
   odds `1/p`, ~0 overround → naturally high weight). This is the clean
   unification of "betting sites AND prediction markets" — one consensus, each
   source weighted by sharpness/liquidity. `consensus()` now accepts a
   prediction-market pseudo-book directly. *Live Kalshi/Polymarket fetch is a
   follow-on (their public APIs; The Odds API does not carry them); gate each on
   liquidity via a smaller `sharp_weights[name]`.*
3. **Blend weight cross-validation.** They independently settled on **0.85
   market / 0.15 model** — the exact figure our Phase 3 backtest produced. Two
   teams converging on 0.85 from different data raises confidence in it.

**Deliberately NOT adopted (ours is better or theirs is unverified):**
- Their market→scoreline step is **independent Poisson fit to 1X2 only**. Ours
  is Dixon-Coles fit to 1X2 **+ totals** by least squares — captures the
  low-score dependence and the goal-total shape they flag as "model risk." Keep
  ours.
- Their props use a flat **0.88 logit shrink**; we use **Platt calibration fit
  from settled data** (principled, data-driven). Keep ours — though their
  independent choice to shrink rough prop models validates our calibration layer.

**Live evidence this matters (consensus on tonight's odds vs our pure model,
`scripts/demo_consensus.py`):**
- **USA vs Australia: model 28% home, market 61% — a 33-pt gap.** Same USA
  underrating that lost us USA vs Paraguay (we had 26, market ~50, USA won).
- **Brazil vs Haiti: model 73%, market 91%.** Model underrates elite-vs-minnow.
- Most other matches agree within ~5 pts (Qatar–Switzerland 76 vs 80, Scotland–
  Morocco within 1) — the anchor corrects the systematic misses without
  thrashing the calls we already get right.

The pure-Elo model has a real, systematic blind spot on a handful of teams
(USA especially). The consensus anchor is the single highest-value fix and the
evidence is now concrete rather than theoretical.
