# V2 Findings — what improved, what didn't, promotion recommendation

Date: 2026-06-12. All numbers from six-tournament point-in-time backtests
(180 group matches evaluated, wc2018 training-only; scripts referenced inline).
v1 stayed green throughout: **44/44 tests pass.**

## Headline results

**Production v2 config: minute-level MC simulator with GOAL-ONLY state coupling,
posterior predictive on, frailty on** (`backtest_v2.py --goal-only`, overall
Brier 0.2293 vs 0.2308 full coupling vs 0.2294 uncoupled).

| Archetype family | v1 | v2 (goal-only) | Verdict |
|---|---|---|---|
| Match results (win/draw) | 0.2024/0.2109/0.1929 | 0.2019/0.2101/0.1931 | match, slight v2 edge on draw/win_a |
| Goals (totals/btts/halves) | ≈ | within ±0.001 | match (expected: same model) |
| Corners HT comparison | 0.2081 | **0.2074** | v2 ≥ v1 |
| Cards 2H ≥ 2 | 0.2354 | **0.2312** | v2 > v1 |
| Offsides ≥ 2 | 0.2506 | **0.2490** | v2 > v1 |
| SoT 2H comparison | 0.2218 | 0.2247 | v1 ≈ v2 (within MC noise) |
| **Timing archetypes** (ht_tied, h2>h1, first-goal combos) | crude analytic approximations | 0.2466 / 0.2472, coherent by construction | **v2 only** — this is the point of the simulator |

## Ablations (`scripts/compare_ablations.py`)

| Component | Marginal contribution | Decision |
|---|---|---|
| Game-state coupling, goals | +0.0013 draw, +0.0013 win_a, +0.0007 ht_tied | **ON** — comeback dynamics are real and predictive |
| Game-state coupling, counts | −0.0184 SoT, −0.0026 corners, −0.0024 offsides | **OFF** — in-sample multipliers (SoT trail ×1.67/lead ×0.41) do not transfer; same lesson as Phase 4 referee factors |
| Posterior predictive | ±0.0002 on tournament fixtures | ON (free; exists for thin-data teams — Curaçao, Haiti, Jordan) |
| Gamma frailty | 0.0000 on evaluated archetypes | ON (tail correctness; matches measured var/mean) |
| Squad-quality feature | b_sq ≈ ±0.01 after ridge; OOS 0.2206 vs 0.2204 plain | **REJECTED** (`backtest_squad.py`). Unregularized fit pegged at bound on 48 matches — the "KEEP" verdict before regularization was noise. Elo already prices squad quality. Lineup XI-share hook retained for live PATCH (γ=1, A12) |
| GBM hybrid | standalone 0.2068 vs v1 0.2020; expanding-OOS blend chose w=0 in 4/5 tournaments | **REJECTED** (`backtest_gbm.py`). With Elo + DC probability as features, boosting has nothing to add at this data scale; Groll et al.'s gains used richer covariates (market values). Module retained as scaffolding |

## Methodological findings worth keeping

1. **Totals-conditioning is biased for self-exciting processes.** The multinomial
   trick (conditioning out per-match rates) is valid for corners/cards/fouls
   (state path exogenous) but provably biased for goals (each goal creates the
   state that follows). Synthetic recovery caught it; the goal family uses the
   full point-process likelihood with intensity offsets. (`models/dynamics.py`)
2. **Coupling endogenously explains most count overdispersion** — corners'
   empirical var/mean 1.42 reduces to residual frailty k≈8.7; SoT needs none.
3. **Calibration slope is unidentifiable on narrow-support archetypes** —
   free Platt fits went negative on cards_2h/total_ge; guarded with
   intercept-only fallback (`calibrate_v2.py`, `v2_calibration.json`).

## Verification checklist (spec items)

1. ✅ Six-tournament v2-vs-v1-vs-base table above; no family materially worse
   (SoT gap closed by goal-only config; diagnosed + documented).
2. ✅ Convergence: ±0.005 across seeds at 100k draws (`test_seed_convergence`);
   production 100k, backtests 50k.
3. ✅ Coherence property tests, 20 random fixtures + reduction test to the
   independent grid (`tests/test_mc_engine.py`). Note: uncoupled limit is the
   rho=0 grid by construction; with coupling ON the simulator endogenously
   generates the low-score dependence that v1 hand-codes via tau.
4. ✅ Reliability table + fresh Platt per archetype on v2 outputs.
5. ✅ Ablation table above.
6. ✅ PLAN.md §9 updated (A11–A15 + v2 execution update).

## Promotion recommendation

**Run v2 in shadow mode now; promote to the production submission path when it
beats v1 on live data.** Mechanics (account constraint A1: one prediction per
market per account, so no parallel live A/B):

- Every matchday, `scripts/shadow_v2.py` logs v2 predictions for all open
  markets next to the submitted v1 values (531/700 markets simulated today;
  mean |p_v2 − p_v1| = 0.0164 under the production goal-only config — the
  differences concentrate in timing and goal-structure markets where v2
  genuinely differs; 0.041 under full coupling before the ablation decision).
- `scripts/score_shadow.py` scores both on settled outcomes (recovered from our
  own Brier returns; p=0.50 settlements dropped as unrecoverable).
- **Promotion threshold: v2 cumulative Brier ≤ v1 over ≥150 settled simulated
  markets, AND not worse by >0.01 on any archetype family with ≥30 settlements.**
  At ~530 simulated markets per matchday round and ~10 questions/match settling
  daily, the threshold is reachable within the group stage.
- If promoted: v2 becomes the model component in the runner's blend (anchor
  weights unchanged); timing archetypes switch to simulator answers immediately
  (they have no real v1 competitor).
- Until then: v1 remains the production path; the timing-archetype improvement
  alone justifies promoting **just the timing family** earlier if its shadow
  performance confirms over ≥50 settlements (it replaces v1's crudest
  approximations, e.g. the +1.34 logit shift for first-goal-and-win).

## Cut list status (V2-PLAN)

- FBref club-rate ingest: **cut** (live-only enhancement; squad feature rejected
  anyway, so it now only matters for player-prop priors — still worthwhile, not v2-blocking).
- GBM: built, evaluated, rejected — kept as scaffolding.
- ET/penalty simulation for knockouts: not built (A14); revisit before June 29.
