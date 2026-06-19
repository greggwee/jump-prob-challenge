# Plan: confidence calibration + a multi-agent expert layer

*Grounded in the 2026-06-19 settled-match audit (26 matches, 257 questions) with the
real crowd consensus harvested from `/api/probability/match-stats`. Two workstreams:
**(A)** stop the catastrophic tail losses on low-edge archetypes by calibrating our
confidence to the crowd; **(B)** add gated AI subject-matter agents for the
hard-to-model archetypes. Execution plan in §6.*

> **Revised 2026-06-19 after a quant review.** Changes from v1: the per-archetype
> 4-parameter fit (overfitting at n≤6) is collapsed to **≤1 regularized parameter per
> family**; the objective is fixed to **expected points** with §12 extremizing held
> dormant; **Phase 0 is now a hard blocker** (authoritative archetypes + harvested
> consensus, not the keyword classifier / proxy field); the crowd-consensus field is
> wired into the **fit itself** (A2 no longer trails A1); Phase 2 is demoted to
> *conditional*; Part B's per-archetype gate and inverse-variance averaging are
> downgraded to match what is testable. The §1 edge table below is **provisional**
> until regenerated in Phase 0.

---

## 1. What the data says (provisional — regenerate in Phase 0)

The contest scores us against the **average individual** forecaster (noisy); the crowd
**consensus** (`prediction_average`) is far sharper. Measured:

| benchmark | mean Brier | vs us |
|---|---|---|
| crowd **consensus** | 0.224 | **beats us by ~192 RBP** |
| **us (gigabot1)** | 0.232 | — |
| average **individual** (scored) | 0.252 | we beat by **+520 RBP** (rank #240/2818) |

We rank well because we beat the scatter, but we **lose to the consensus**, concentrated
in archetypes where we are **overconfident-low** vs a hedged crowd. Provisional edge vs
consensus by archetype (negative = consensus beats us): `both_sot` −95 *(≈entirely one
already-fixed parser bug — EXCLUDE from the fit)*, `offsides` −59, `player_ga` −52,
`win` −43, `fouls` −34, `total_sot` −23, `cards` −15; POSITIVE (we beat consensus):
`player_scores` +38, `pen_or_red` +20, `halftime` +16, `sot_gt` +16, `player_sot` +11,
`corners` +6.

> **Caveat (quant review):** these numbers were computed with a *keyword* archetype
> classifier (`review_settled.family_of`) and the *proxy* field, and several rest on
> 1–6 questions / 1–2 outlier matches (NED/JPN −136, USA/PAR −76). They are directional
> only. Phase 0 regenerates this table on the **authoritative `matchday_predictions`
> archetype map** and the **harvested consensus**, accounting for the `p=0.50`
> exclusions (sign-ambiguous in outcome recovery — disproportionately the no-read
> questions we care about). Do not fit anything until that table exists.

The failure signature where we lag: we say **7–13%** where the crowd says **22–50%**
(e.g. `player_ga`), winning small on the ~8/10 No's and losing ~50 RBP each on the rare
Yes upsets (Gyökeres, Kubo).

---

## 2. The principle, and the objective

**Objective: expected points** (= sum of relative Brier). Justification: METHODS §1.2 —
under the account-level prediction limit we maximise expected points, and at **rank
#240/2818** we are nowhere near the `τ−μ` regime where buying variance for rank #1 pays.
Therefore **§12 extremizing (`k>1`) is DORMANT**: `k≥1` is allowed *only* where a family
both (a) beats the consensus OOS and (b) we are in contention — neither holds now, so the
whole system runs `k≤1`. This removes the Part A (shrink) vs §12 (extremize) contradiction
by *picking one objective for the current standings*.

Why shrinking toward the crowd is correct **as calibration, not as a hedge**: our model
is *empirically miscalibrated* on the lagging archetypes — it predicts 7–13% where 22–50%
realises. The consensus is a better-calibrated estimate of the truth, so moving toward it
**reduces our own Brier** (this is correcting *our* calibration error; it is not a Jensen
effect and not "honesty"). Two mechanisms, both calibration:

1. **Remove bias + compress overconfidence** with a refit per-*family* Platt map.
2. **Shrink toward the leave-one-out crowd consensus** on families where the consensus is
   better calibrated than our raw model.

Plus **one bounded, deliberate departure** from proper scoring: a **per-family tail floor**
`p ∈ [f, 1−f]`. Our tail estimates are the least trustworthy (thin data, cross-tournament
regime shift); flooring caps the ruinous single-question loss. Break-even: flooring
`p=0.10→0.15` helps once the true rate exceeds ≈0.125 (which the lagging player families
clear). The floor — one number per family — is also the tail-protection that a CVaR term
was reaching for, without estimating a risk measure from 1–2 points.

---

## 3. Part A — asymmetric confidence calibration

Machinery already exists: `runner._extremize` → `strategy.extremize(p, field, k)` moves
`logit(p)` by factor `k` around `logit(field)`; `k<1` shrinks toward `field` (verified).
Changes:

**A1+A2 (together) — `k<1` toward the *real* leave-one-out consensus.** Lower the `k_for`
floor from 1.0 (strategy.py:131) so a fitted `k<1` is allowed. **Set `field` to the
measured crowd consensus** (harvested table in the fit/backtest; live in Phase 2),
leave-one-out: `c₋ᵢ = (N·c − p_us)/(N−1)`, falling back to base rate when no consensus.
*Do not ship `k<1` against the proxy `field_model` and call it crowd-shrinkage — against
the proxy it shrinks toward `inv(λ·logit p)`, a λ-scaled version of our own belief, not
the crowd.*

**A3 — refit per-FAMILY Platt** (pool: `fit_calibration_table` requires
`MIN_SETTLEMENTS=150` (calibration.py:18), so per-*archetype* groups all no-op to
identity at n=257/13). Add the live settled set to the backtest pool; families
{goal, count, player, offsides} as `fit_live_calibration.py` already pools.

**A4 — per-family tail floor** `clip(p, f, 1−f)` after calibration+shrinkage.

**A5 — fit ONE regularized shrink `kᶠ` and ONE floor `fᶠ` per family** (not 4 params per
archetype). Priors pinned at the no-op (`k=1, f=0`) with L2 shrinkage toward them (the
squad-quality lesson, METHODS §9: with tens of obs the prior does the inference). **No
CVaR** (un-estimable at this n). Objective: minimise **leave-one-MATCH-out** Brier across
the 26 matches (CV unit = match, so outliers can't be memorised). A single global shrink
`k_shrink` shared across all *lagging* families is the default; only split per-family if
LOO-CV shows it generalises. Gate per-family on **"no LOO regression"**; families we beat
(player_scores, sot_gt, corners, timing) stay at `k=1, f=0` — keep confident.

**A6 — `win`-market diagnostic.** Anchored winners should track the line yet lag the
consensus (−43, provisional). Check anchor orientation/application, whether post-anchor
Platt re-distorts it, or whether the SP crowd is sharper than our books — fix the root
cause, don't shrink blindly.

**A7 — gate & test.** Each family carries `{a, b, k, f}` in `confidence_gate.json`. Unit
tests for `k<1` shrinkage and the floor; the LOO-match backtest asserts total RBP improves
and **no family regresses**. Ship behind the gate (default = today's behaviour until LOO
confirms each family). Refit nightly in `daily_learn.py`.

---

## 4. Part B — multi-agent expert layer (research/shadow only for now)

The models are weak where **data is thin or contextual** (player props esp. `player_ga`,
compound `btts_and_total`, tactically-driven counts). AI subject-matter agents add a
*decorrelated* signal — but **the per-archetype OOS gate is not measurable at n≤6**, so
Part B is **shadow-only research** until *families* (not archetypes) accumulate ≥~30
settled questions. Components as requested, grounded so the LLMs don't hallucinate numbers.

**B1 — scope.** Agents run **only** on lagging/hard archetypes; never the markets we beat
or the anchored goal markets.

**B2 — expert roles, each given data not asked to guess blind:** lineup/availability
(confirmed XI from API-Football → §6.3 minutes mixture); player-prop expert (club per-90
rates incl. the new API-Football SoT, lineup, base rate, **our prior**, **consensus** →
adjustment + rationale); tactical-matchup (style/press/stakes → directional nudges);
compound-question (joint events where independence breaks); discipline/aggregator
(combines experts + model + consensus; breaks sharp disagreements with explicit tactical
reasoning; defaults to consensus when unsure).

**B3 — "poll several models, average."** Claude Opus 4.8 (discipline) + Sonnet 4.6 /
Haiku 4.5 (experts) at varied temps; optional one cross-provider for genuine decorrelation.
**Do NOT claim inverse-variance variance-reduction for same-provider personas** — they are
correlated and METHODS §10.1's IVW derivation assumes *uncorrelated* errors. Either use a
correlation-aware combiner (`w∝Σ⁻¹1` with an estimated correlation floor) or simple-average
without the variance claim.

**B4 — grounding & anti-miscalibration:** agents return an *adjustment to a supplied
prior*, never from scratch; every agent output is **Platt-calibrated, pooled by family**
(per-archetype no-ops at MIN_SETTLEMENTS=150); hard caps via the Part A floors.

**B5 — timing.** Lineups ~T−60; scheduler fires T−6…25. Add a T−60→T−30 pass (lineup →
experts → aggregate) blended at submit time. *Decision: API-Football free-tier coverage of
live WC-2026 lineups — may need a paid tier/alternate.*

**B6 — validation (the gate).** Shadow-mode: replay over settled matches (historical
lineups via API-Football, harvested consensus), compare agent vs model vs consensus vs
outcome **per family**, promote only where it beats `min(model, consensus)` OOS *and* the
family has ≥~30 settled questions. Same rule that rejected squad-quality and the GBM hybrid.

---

## 5. How the pieces fit

- **Part A** lifts the floor (stop bleeding on low-edge archetypes) — fast, certain,
  reversible, no new dependencies.
- **Part 2 (consensus anchor)** is the biggest *potential* win **conditional on 2a**
  (consensus observable *and stable* pre-kickoff). If the consensus is only populated
  post-submission or moves after we submit, the live anchor is unusable — so it is gated,
  not assumed.
- **Part B** is offence where neither model nor raw consensus is sharp — grounded in the
  club-rate data we already built + the consensus we unlocked, and **shadow-only** until
  samples support a real gate.

---

## 6. Execution plan (ordered)

**Phase 0 — data foundation (HARD BLOCKER; nothing fits before this)**
- `scripts/harvest_crowd.py`: persist crowd consensus per settled question
  (`market_id, crowd_p, outcome`) → `data/processed/crowd_consensus.csv` (from the
  `match-stats` capture; document refresh; investigate an authenticated path).
- `scripts/build_settled_dataset.py`: join our predictions (API) + recovered outcomes +
  **authoritative `matchday_predictions` archetype** + crowd consensus →
  `settled_with_crowd.parquet`. Account for `p=0.50` exclusions. **Regenerate the §1 edge
  table from this**; EXCLUDE the known-bug `both_sot` cases. Until this exists and the
  table is rebuilt, treat every per-archetype edge as provisional.

**Phase 1 — asymmetric calibration (the catastrophic-loss fix)**
- 1a. `scripts/diagnose_calibration.py`: per-family reliability curves, our confidence vs
  leave-one-out consensus, tail-loss table (authoritative archetypes).
- 1b. Refit per-*family* Platt on the larger pooled set (A3).
- 1c. `strategy.k_for` allow `k<1`; `_extremize` field → leave-one-out consensus/base
  (A1+A2); gate carries `{a,b,k,f}` per family (A7).
- 1d. Tail floor in `runner` after calibration (A4).
- 1e. `scripts/fit_confidence.py`: fit a single regularized `k_shrink` (+ per-family floor)
  by **leave-one-match-out** Brier across the 26 matches, L2-shrunk to the no-op; writes
  `confidence_gate.json` (A5).
- 1f. `win`-market diagnostic + fix (A6).
- 1g. Unit tests + LOO-match settled backtest gate (no family regresses); wire refit into
  `daily_learn.py`.

**Phase 2 — live crowd-consensus anchor (biggest *potential* win, gated)**
- 2a. **Verify `prediction_average` is observable AND stable pre-kickoff** (poll an ongoing
  match at the scheduler's T−6…25 window; check drift after our submit). Blocker for 2b.
- 2b. If yes: `ingest/crowd_consensus.py` + a no-line consensus-anchor blend (weight fit
  like the market anchor); gate on the LOO-match backtest.

**Phase 3 — multi-agent expert layer (shadow-only until families ≥~30)**
- 3a. `ingest/lineups.py` (API-Football `/fixtures/lineups`); confirm tier/timing.
- 3b. `agents/` roles + grounded prompts (club rates, base, prior, consensus); Claude SDK.
- 3c. Discipline/aggregator; correlation-aware (or plain-average) combiner.
- 3d. Agent-output Platt (pooled by family) + per-family OOS gate.
- 3e. `scripts/shadow_agents.py`: replay over settled matches; promotion report.
- 3f. Pipeline integration T−60→T−30; submit-time blend (only once a family clears the gate).

**Phase 4 — eval & rollout (continuous)**
- Per-family RBP-vs-consensus tracked each matchday; `daily_learn` refits `{a,b,k,f}`;
  agents stay behind the OOS gate; periodically revisit whether any family has earned `k≥1`
  *and* we are in contention (then, and only then, re-open §12).

---

## 7. Open decisions

- **Crowd-consensus automation:** manual browser refresh vs an authenticated harvest path.
- **Lineups source:** API-Football free tier vs paid vs alternate, pending live WC-2026
  coverage check.
- **Agent model mix:** Claude-only (cheaper, correlated) vs add a cross-provider
  (decorrelated, costlier).
