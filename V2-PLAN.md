# V2 Engine Plan — minute-level MC simulator, squad features, GBM ensemble

v1 (analytic DC grid + NegBin counts + shrunk player rates) stays the production
path. v2 is a parallel engine; promotion only on backtest + live shadow evidence.
**Account constraint reality (A1):** one prediction per market per ACCOUNT, so v2
cannot submit live in parallel — it runs in **shadow mode**: predictions logged per
market at submission time, scored offline against v1 on settled results.

## Component order (and cut line)

1. **Event stream + game-state dynamics estimation** (the calibration everything rests on)
2. **Minute-level MC simulator core** — uncoupled first, with reduction regression test
3. **Game-state coupling + frailty** (overdispersion) + coherence property tests
4. **Timing archetypes + posterior predictive sampling**
5. **Verification phase A**: six-tournament v2-vs-v1 backtest, convergence, calibration refit, ablations
6. **Squad features** (backtest decision via point-in-time StatsBomb-derived rates; FBref club ingest live-only)
7. **GBM ensemble member** (LightGBM, Groll-style)
8. **V2-FINDINGS.md** + PLAN.md §9 updates

If time-boxed, cut from the bottom: GBM first, FBref club ingest second. The simulator
with coupling + verification (items 1–5) is the non-negotiable core.

## File-level layout

| File | Contents |
|---|---|
| `src/jumpprob/ingest/event_stream.py` | one pass over cached events → per-event rows (match_id, team, period, slot 0–89, family ∈ {goal, corner, card, foul, sot, offside}); stoppage folded into slots 44/89; shootout/ET excluded |
| `src/jumpprob/models/dynamics.py` | conditional-multinomial estimator for minute-profile α (6 buckets) and game-state multipliers β (5 states: trail2+, trail1, level, lead1, lead2+) per family; per-match-team totals conditioned out (no nuisance offsets); normalization scalars + frailty k calibrated by simulation; `fit_dynamics(stream)` / `Dynamics.to_json` |
| `scripts/estimate_dynamics.py` | reproducible: stream → fitted dynamics → `data/processed/minute_dynamics.json` + printed table of every measured multiplier |
| `src/jumpprob/models/uncertainty.py` | sandwich covariance (Hʷ⁻¹ M Hʷ⁻¹) for goal-model betas under decay-weighted Poisson MLE; `sample_betas(n)` |
| `src/jumpprob/simulate/mc_engine.py` | vectorized simulator: sequential 90-minute goal loop (state feedback), counts drawn from state-integrated hazards (no per-minute count sampling needed — counts don't feed back), gamma frailty, posterior-predictive beta sampling, `answer(archetype, …)` for every parser archetype incl. timing |
| `tests/test_dynamics.py` | synthetic recovery: generate events from known α/β along simulated paths, assert estimator recovers them |
| `tests/test_mc_engine.py` | property tests (20 random fixtures): complements, joint ≤ min(marginals), half ≤ full, monotone thresholds; reduction test (coupling off, frailty off, pp off ⇒ marginals match `dc_grid(λa, λb, rho=0)` within MC error); convergence across seeds |
| `scripts/backtest_v2.py` | six-tournament point-in-time backtest: v2 vs v1 vs climatology per archetype; ablation flags (`--no-coupling --no-pp --no-frailty`); reliability table; Platt refit on v2 outputs → `data/processed/v2_calibration.json` |
| `src/jumpprob/features/squad.py` | squad attacking quality from point-in-time StatsBomb player rates (top-K aggregate); lineup-day XI-share hook; FBref current-season ingest (live-only, append-only raw) |
| `scripts/backtest_squad.py` | goal-model regression ± squad feature, six-tournament OOS Brier — keep only if it improves |
| `src/jumpprob/models/gbm_model.py` + `scripts/backtest_gbm.py` | LightGBM on (Elo diff, neutral, rest days, importance, confederation, squad, v1 prob) per team-match row, yearly point-in-time refits, logit-space ensemble weight fit |

## Key design decisions (and the math)

**Dynamics estimation without magic numbers.** For each family, events allocate across
a match's 90 (slot, state) cells. Conditioning on each (match, team) total eliminates
per-match intensity nuisances exactly (conditional Poisson ≡ multinomial):
`LL = Σ_events [α(bucket) + β(state)] − Σ_mt n_mt log Σ_slots exp(α + β(path_mt))`.
9 free parameters per family; BFGS; synthetic-recovery tested. Dynamics are refit
per backtest tournament on prior tournaments only (point-in-time), pooled over all
six for live use.

**Counts are conditionally Poisson given the goal path** — only goals feed back into
state. So the simulator runs the sequential loop for goals only (90 vectorized steps
over N draws), then draws each family count from its path-integrated hazard in one
Poisson sample per (draw, team, family, half). Frailty: Gamma(k) multiplier on the
integrated hazard reproduces NegBin marginals in the uncoupled limit (Gamma-Poisson
mixture) — k calibrated so simulated var/mean matches the measured values (1.42
corners, 1.89 shots, …) net of what coupling already generates.

**Reduction test target is the independent grid** (`rho=0`), not the τ-corrected DC
grid: an uncoupled Poisson minute process is independent by construction. The
interesting verification is the converse — with coupling ON, the simulator should
*endogenously* generate low-score dependence; we measure the implied rho and compare
to v1's fitted rho_dc in V2-FINDINGS.

**Posterior predictive.** Decay-weighted MLE ⇒ sandwich covariance
`V = H⁻¹ (Σ w² (y−λ)² xxᵀ) H⁻¹`; sample β per draw-block. rho_dc and dynamics
parameter uncertainty not sampled in v2.0 (registered as assumption).

## Riskiest assumptions (flagged up front)

1. **Dynamics transfer across tournaments** — referee factors didn't (Phase 4); state
   multipliers might not either. The per-tournament point-in-time backtest decides.
2. **Squad feature may not beat Elo** — Elo already prices squad quality. Decision
   rule is mechanical: keep only on OOS Brier improvement.
3. **FBref ingest fragility** (rate limits, name resolution at player scale) — why it
   is live-only and second on the cut list.
4. **ET/penalty handling**: simulator covers 90' only; knockout "win" market semantics
   (90' vs incl. ET) unknown until first knockout settles → assumptions register.
5. **GBM overfit risk** on ~300 tournament evaluation matches — guard with yearly
   point-in-time refits, early stopping, and ensemble weight fit on OOS predictions only.
