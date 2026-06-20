# LLM-Expert Plan — incorporating a multi-model ensemble into Bot A

**Status:** verified, ready to implement (shadow-first).
**Author:** quant/stat review, 2026-06-20.
**Scope:** how to wire an LLM expert layer into Bot A *per question archetype*, with the
explicit goal of taming the cases where our probabilities "vary very far from the crowd
consensus" — player props and SoT — without breaking the markets where we already have edge.

This plan supersedes the LLM portions of [CALIBRATION-AGENTS-PLAN.md](CALIBRATION-AGENTS-PLAN.md)
(Part B). The calibration phases there (Phase 1 floor, club-GA prior) are already shipped and
live; this document is the concrete build-out of the expert ensemble that was left as a plan.

---

## 0. What was verified before writing this

The user provided an `ANTHROPIC_API_KEY`. Before planning around it, I ran an end-to-end
verification — not a ping, but the *actual mechanism* this plan depends on:

| Check | Result |
|---|---|
| Key + SDK live call (`claude-opus-4-8`, adaptive thinking) | ✅ succeeded |
| Structured output via `client.messages.parse()` + Pydantic `{probability, rationale}` | ✅ clean typed return — no regex extraction (parkjpd had to regex a bare number) |
| **Grounded-expert correction on our worst error class** | ✅ Gyökeres SoT-in-2H: model **0.11** → expert **0.44** |
| Architecture sanity vs. an independent implementation | ✅ cross-checked against `parkjpd/jumptrading-probability-cup` |

The Gyökeres result is the proof of value. Our single most damaging error pattern (catastrophic
*low* bets on elite players the base-rate model under-rates — Gyökeres, Díaz, Kane, Balogun, Gakpo;
see [shadow_agents.py](../scripts/shadow_agents.py)) is exactly what the expert corrects, and it
corrects it to a *defensible* number (0.44 for a single half), not the blunt raw-history value
(~0.65 full-match). That "pull toward sane, don't overshoot" behavior is the whole point.

**Plan re-verified against the codebase (2026-06-20), before building Phase L0.** Every integration
seam the plan assumes exists and matches: the blend inserts in [runner.py](../src/jumpprob/submit/runner.py)
right after the Phase-1 confidence step (post-extremize, pre-output) with `q`, `r["id"]`, `q.archetype`,
`p_a`, `ctx.anchored`, `fam` all in scope; `ParsedQuestion` carries `archetype/subject/other/threshold/half`;
the forward-test parquet `settled_with_crowd.parquet` (257 rows) keys on `market_id` with
`our_p/crowd_p/outcome/our_brier/crowd_brier`; the SDK (0.111.0) exposes both `messages.parse` and
`messages.batches.create`. **One correction:** the §4 table uses friendly archetype names, but the
routing *config* must key on the real parser strings — `offsides_ge`, `fouls_gt`, `sot_gt`,
`corners_gt`, `player_sot_ge`, `team_sot_ge`, `btts_and_total_ge`, etc. (full list enumerated from
[parser.py](../src/jumpprob/submit/parser.py)). The `panel.py` ROUTING dict uses those.

---

## 1. The two source strategies, reconciled

**Strategy 1 (the top-predictor recipe)** has three components:
1. Poll several AI models before each match, average their predictions.
2. ~1h before kickoff, pull confirmed starting lineups.
3. A discipline / tactical-reasoning layer that breaks ties.

**Strategy 2 (the parkjpd repo)** is a working implementation of essentially component 1, plus
hard operational discipline. The load-bearing decisions I'm adopting from it:

| parkjpd decision | Why it matters | We adopt |
|---|---|---|
| **Multi-MODEL panel**, not multi-prompt | Model diversity decorrelates errors far more than re-prompting one model. Their first version was "one Claude with 3 lens prompts" — they explicitly upgraded away from it. | ✅ Opus 4.8 + Sonnet 4.6 + Haiku 4.5 as distinct panelists (cross-provider optional, see §7) |
| **Trim-median in log-odds** aggregation | A single rogue model can't drag the center. Plain mean lets the worst model set the price. | ✅ exact same aggregator |
| **Curate by leaderboard / shadow the weak** | They only let net-positive models into the live aggregate; weak ones (Gemini, Haiku) are logged but excluded until forward-test proves them. | ✅ shadow-first, promote on OOS evidence |
| **Out-of-band only** — daemon never calls an LLM | LLM calls are slow and flaky; the submit daemon must stay deterministic. Prep → panel → write JSON → daemon hot-reads. | ✅ non-negotiable (matches our [scheduler_tick.py](../scripts/scheduler_tick.py) discipline) |
| **Never touch a sharp source** | Market 1X2/totals are de-vigged consensus — already the sharpest signal. The AI must never move them. | ✅ guard in §3 |
| **Lineup pass ≠ tactical panel** | Player props are availability/lineup-driven; a *separate* pass owns them, not the tactical reasoner. | ✅ this is component 2, and it's where our biggest miss lives |
| **Single weight knob, default 0** | Shadow mode logs the AI number; flip the weight only after the forward-test confirms. | ✅ per-role `*_BLEND_WEIGHT`, default 0 |

So the merged design is: **Strategy 2's robust ensemble + ops discipline**, with **Strategy 1's
lineup pass and tactical tie-breaker** layered on as two of three distinct *roles*. The
user's refinement — experts must reason from player+team **history**, not just current-tournament
form — is satisfied by *grounding* every prompt with our historical rates
([grounding.py](../src/jumpprob/agents/grounding.py)), already scaffolded and shadow-validated.

---

## 2. The three roles (and why three, not one)

One monolithic "ask the LLM" prompt is the wrong design — different archetypes fail for different
reasons and need different evidence. We split into three roles, each with its own owner, its own
inputs, its own blend weight, and its own shadow gate:

### Role A — Tactical ensemble panel
- **Owns:** model-only questions with *no sharp market line* — half-specific scoring, compound
  questions (BTTS-and-3+, first-goal-and-opp-2h), offsides/fouls counts.
- **Inputs:** our model prior (`model_pct`), match context (team styles, motivation, weather),
  reasoned tactically. No lineup needed (team-level).
- **Models:** full panel (Opus + Sonnet + Haiku), trim-median log-odds.
- **Why a panel:** these are genuinely "reason about the match" questions where model diversity helps.

### Role B — Lineup / player-prop pass
- **Owns:** `player_sot_ge`, `player_ga`, `player_scores` — **the cases the user flagged.**
- **Inputs:** confirmed XI (from API-Football `/fixtures/lineups`, ~1h pre-kickoff), the player's
  **historical** club per-90 rates (our `grounding.py`), opponent defensive strength, expected minutes.
- **Models:** Opus 4.8 primary, Sonnet 4.6 as a second panelist (props are expensive to fan out
  3-wide over every quoted player; 2 is enough given the strong prior).
- **Why separate:** player props are availability-driven. A great tactical read is useless if you
  don't know the guy is benched. This pass *starts* from "is he playing and for how long," then
  layers the historical rate and opponent — exactly the chain that fixes Gyökeres-type misses.

### Role C — Discipline / aggregator
- **Owns:** reconciling, per question, {our model, the relevant ensemble/lineup number} and emitting
  the final blended probability + a confidence flag.
- **Inputs:** the two numbers, their disagreement magnitude, the archetype's edge profile (§4).
- **Model:** Opus 4.8 with adaptive thinking — but *only invoked* on questions where the ensemble
  and model disagree beyond a threshold (cost control). Otherwise a deterministic blend.
- **Why:** this is Strategy 1's "tie-breaker." When the model says 0.11 and the panel says 0.50,
  blindly averaging to 0.30 may be wrong; the discipline layer reasons about *which* is more
  trustworthy for *this* archetype and *this* evidence.

---

## 3. Hard guards (invariants — these never bend)

These mirror the pricing-side calibration guards and the parkjpd invariants. Any code path that
violates one is a bug.

1. **Never move a sharp source.** If a question is priced off the de-vigged betting market
   (goal 1X2, totals, BTTS where odds exist), the LLM is *excluded entirely*. The market is sharper
   than any LLM. (Anchoring is owned by [odds_api.py](../src/jumpprob/ingest/odds_api.py) at 85%
   market weight — untouched.)
2. **Never override a market we already beat.** Corners, SoT-comparisons, timing, and `player_scores`
   have *validated OOS edge* (backtest + live agree). The LLM runs in **shadow only** here — logged,
   never blended — unless and until a forward-test shows it *adds* over our model.
3. **Cache-miss → pure model.** If no fresh panel entry exists for a question id, submit the model
   number unchanged. The LLM is strictly additive; its absence is never a failure.
4. **Out-of-band only.** The submit daemon ([run_matchday_odds.py](../scripts/run_matchday_odds.py))
   hot-reads `data/processed/ai_panel.json`. It never calls an LLM. Zero new daemon failure modes.
5. **Crowd consensus is post-lock only.** The LLM may *not* see the crowd consensus as an input
   (it's hidden until bets lock — HARD CONSTRAINT). Consensus is used *offline* as the forward-test
   grader, never as a live prior. The LLM is a *standalone* signal whose job is to reduce our
   far-from-crowd variance on its own merits.
6. **Single weight knob per role, default 0 (shadow).** `AI_TACTICAL_WEIGHT`, `AI_LINEUP_WEIGHT`,
   `AI_DISCIPLINE_WEIGHT` — all default 0. Flip only after OOS no-regression, same gate as every
   prior component (Phase 1 floor, club-GA prior).
7. **Clip [0.03, 0.97].** No LLM number escapes the same clip the rest of the pipeline uses.

---

## 4. The per-archetype routing table — the core of the plan

This is the answer to "evaluate how best to incorporate the LLM for each type of archetype." Each
row says: who owns it, what the LLM is allowed to do, and why — keyed to our settled-results edge map.

| Archetype | Our edge (settled + backtest) | LLM role | Allowed action | Rationale |
|---|---|---|---|---|
| **Goal 1X2 / totals / BTTS** *(market-priced)* | Anchored to market (sharp) | — | **Excluded** | Guard 1. Market is the sharpest signal; LLM can only add noise. |
| **Corners (count)** | **Strong** (+0.045 OOS) | Tactical | **Shadow only** | Guard 2. We win here; don't risk it. Log for forward-test. |
| **SoT comparison (team)** | **Strong** (+0.036 OOS) | Tactical | **Shadow only** | Guard 2. Validated edge. |
| **Timing (first/last goal, half)** | **Strong** | Tactical | **Shadow only** → blend if forward-test wins | Edge is real; let the panel earn its way in. |
| **`player_scores`** | **Edge** (Brier 0.030, history *hurts* it) | Lineup | **Shadow only** | Guard 2 + shadow showed raw history *degrades* this. Protect it. |
| **`player_ga` (goal+assist)** | Was our **blind spot**; club-GA prior shipped (+24 RBP) | Lineup | **Blend (primary target)** | Assists are model-invisible; history *helps* here (shadow Brier 0.164 < our 0.194). LLM lineup pass adds opponent + minutes on top of the club prior. |
| **`player_sot_ge`** | Genuine +11.7 after de-bugging; but **catastrophic-low tail** | Lineup | **Blend, capped pull** | **The user's flagged case.** Floor (Phase 1) caps the tail from below; the lineup pass corrects the central estimate (Gyökeres 0.11→0.44). Blend toward the expert but cap how far it can pull, so we don't overshoot to the blunt history number. |
| **Offsides (count)** | **No edge** (~0.0) | Tactical | **Blend (low weight)** | We're at base rate; an informed tactical read (high-line vs. deep-block) can only help. Crowd is informed here, so gate carefully. |
| **Fouls / cards (count)** | **Weak / negative** (−0.002) | Tactical | **Blend (low weight)** | Referee/temperament-driven — exactly LLM-reasonable. We have nothing to protect. |
| **Compound (BTTS-and-3+, first-goal-and-opp-2h)** | Independence assumption breaks | Tactical | **Blend** | Joint reasoning is where an LLM beats a product-of-marginals model. |
| **Half-specific scoring (a/b scores 1H/2H)** | Thinning approximation | Tactical | **Blend (low weight)** | LLM can reason about game-state (chase vs. park-the-bus 2H). |

**Reading the table:** the LLM is *aggressive* exactly where we're weak or far from crowd (player
props, offsides, fouls, compounds) and *muzzled* (shadow-only) exactly where we have proven edge
(corners, SoT-compare, player_scores, anything market-priced). That asymmetry is deliberate — it's
the same "don't bet so low we take huge losses, but don't surrender our wins" objective from the
calibration plan, now expressed through the expert layer.

---

## 5. Aggregation math

### 5.1 Ensemble combine (trim-median in log-odds) — adopted from parkjpd
```
logit(p) = ln(p / (1-p))
panel = [logit(p_i) for each model i that returned a number]
  n >= 4:  drop min and max, take median of the rest
  n == 3:  median
  n <= 2:  mean
p_ensemble = sigmoid(combined)
```
Robust by construction: one model at 0.02 or 0.98 cannot set the price.

### 5.2 Grounding (every prompt) — satisfies "reason from history"
Each prompt injects:
- the question, the match, **"strictly PRE-MATCH, do not assume a result"**,
- our model's number (`model_pct`) as the stated baseline to reason *against*,
- the subject's **historical** rate from [grounding.py](../src/jumpprob/agents/grounding.py)
  (club per-90 SoT/GA, opponent strength, expected minutes),
- **"comparative questions are STRICT (ties = NO)"**,
- "return a single calibrated probability and one-sentence rationale" (Pydantic-parsed).

This is anchoring, not from-scratch — the LLM corrects the prior, it doesn't replace it. Verified
to produce the right magnitude of correction (§0).

### 5.3 Final blend (per role, gated)
```
p_final = sigmoid( (1-w)*logit(p_model) + w*logit(p_expert) )      # log-odds blend
w = role weight, default 0 (shadow)
```
For `player_sot_ge` specifically, add a **pull cap**: `|logit(p_final) - logit(p_model)| <= C`,
so the expert can rescue a catastrophic-low bet but cannot drag us all the way to a blunt extreme.
`C` is fit on settled data alongside `w`.

---

## 6. Implementation phases (shadow-first, each gated on OOS no-regression)

**Phase L0 — plumbing (no behavior change).**
- `src/jumpprob/agents/expert.py`: the LLM caller. `forecast_one(question, grounding, model)` →
  `Forecast{probability, rationale}` via `messages.parse()`. Panel fan-out via the **Batch API**
  (50% cost) for the non-time-sensitive tactical panel; **synchronous** for the lineup pass (must
  finish in the ~1h pre-kickoff window). Provider-skip if a key is missing (graceful degradation).
- `scripts/ai_panel_prep.py` → assembles the question + grounding context for a matchday.
- `scripts/ai_panel_write.py` → runs the panel, writes `data/processed/ai_panel.json`
  (per-question: each model's prob, the ensemble, a freshness timestamp).
- Daemon-side consumer in [runner.py](../src/jumpprob/submit/runner.py): hot-read the cache, blend
  per §5.3 gated by the (default-0) weights. **Cache miss → pure model.**

**Phase L1 — shadow forward-test (weights still 0).**
- Run the full panel out-of-band on every settled matchday we have. Log expert vs. model vs. outcome.
- Extend [shadow_agents.py](../scripts/shadow_agents.py) to score the *LLM* (not just raw history)
  per archetype, leave-one-match-out, against the realized crowd Brier from
  `settled_with_crowd.parquet`.
- **Gate:** promote a (role, archetype) pair to live blend **only** if the LLM-blended Brier beats
  the pure-model Brier OOS, with no regression on the protected-edge archetypes.

**Phase L2 — promote the winners, fit weights.**
- For each archetype that passed L1, fit `w` (and `C` for sot) by minimizing OOS Brier — single
  regularized parameter per archetype, same discipline as `fit_confidence.py`.
- Expected first promotions (from the §4 priors + §0 evidence): `player_ga`, `player_sot_ge`
  (capped), offsides, fouls, compounds. Expected to *stay shadow*: corners, SoT-compare,
  player_scores, anything market-priced.

**Phase L3 — lineup pass live + discipline layer.**
- Wire API-Football `/fixtures/lineups` into the lineup pass; re-run props at confirmed minutes.
- Add Role C (Opus discipline tie-breaker) on the *subset* of questions where ensemble and model
  disagree beyond threshold — keeps Opus cost bounded.

Each phase is independently shippable and independently reversible (set its weight back to 0).

---

## 7. Models, cost, ops

- **Panel:** Opus 4.8 (`claude-opus-4-8`) + Sonnet 4.6 (`claude-sonnet-4-6`) + Haiku 4.5
  (`claude-haiku-4-5`). Three genuinely different models > three temperatures of one. Cross-provider
  (OpenAI/Gemini) is *optional* and additive — the design degrades gracefully to Anthropic-only,
  which is all we have a key for today. parkjpd's leaderboard data (GPT-5-mini 627, GPT-5 583,
  Opus 484 all beat their bot ~440; Gemini/Haiku below) says Haiku may end up **shadow**; our own
  forward-test (Phase L1) decides, not their leaderboard.
- **Cost control:** Batch API (50%) for the tactical panel; prompt-cache the per-match context
  across all questions in a match; Opus discipline layer runs only on disagreement subset; props
  fan out 2-wide not 3-wide. Rough order: a full matchday panel is tens of cents to a few dollars,
  out-of-band, once per slate — negligible vs. the value of fixing the catastrophic-low tail.
- **Latency:** tactical panel runs the night before (no deadline). Lineup pass runs in the
  ~1h pre-kickoff window, synchronous, Opus+Sonnet only, scoped to that match's quoted players —
  comfortably inside the window. The submit daemon is unaffected either way (it only reads JSON).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM overconfident on a player it "knows" from training (stale transfers, retirements) | Grounding injects *our* current club rates as the anchor; pull cap on sot; STRICT pre-match framing. |
| LLM leaks a "result" (hallucinated score) into a pre-match estimate | Prompt hard-frames PRE-MATCH; Pydantic schema returns only a probability + rationale; rationale is logged and spot-checked in L1. |
| Cost blow-up fanning 3 models over ~700 questions | Batch API, prompt caching, 2-wide props, Opus-only-on-disagreement. |
| A model degrades silently and drags the panel | Trim-median makes one rogue model harmless; per-model logging in `ai_panel.json` surfaces drift. |
| We over-fit the blend weights to a short settled history | One regularized param per archetype, LOO-match CV, shadow-first — identical to the gate that validated Phase 1 and the club-GA prior. |
| Daemon picks up a stale panel | Freshness timestamp per entry; `AI_PANEL_MAX_AGE_H` guard; stale → pure model. |

---

## 9. Definition of done

- `expert.py` + prep/write scripts exist; `ai_panel.json` is produced out-of-band; daemon hot-reads
  it; all weights default 0 (shadow); full test coverage (mirror `test_grounding.py`/`test_confidence.py`).
- L1 forward-test report shows, per (role, archetype), whether the LLM blend beats pure model OOS.
- Only archetypes that *pass* the gate get a non-zero weight; protected-edge archetypes stay shadow.
- The Gyökeres-class catastrophic-low tail on `player_sot_ge`/`player_ga` is measurably reduced in
  the L1 report (the user's stated objective), with no regression on `player_scores` or any
  market-priced or proven-edge archetype.

---

## 10. Phase L1 forward-test results (2026-06-20)

Ran the full 3-model panel (Opus 4.8 + Sonnet 4.6 + Haiku 4.5) out of band over all 221 non-sharp
settled questions (`scripts/ai_panel_write.py`, 603s, 0 failed ensembles), then scored model-vs-blend
Brier by honest **leave-one-match-out** CV with a grid that includes w=0 (`scripts/forward_test_llm.py`).
215 questions joined (known-bug excluded), 26 archetypes, 26 matches. The flagged-case verdict:

| Archetype | route | n | model Brier | crowd | blend (CV w) | verdict |
|---|---|---|---|---|---|---|
| `player_ga` | blend | 10 | 0.1937 | 0.1418 | **0.1741** (w=1.0) | **PROMOTE** (+0.0196) |
| `fouls_gt` | blend | 21 | 0.2770 | 0.2609 | 0.2689 (w=1.0) | promote, small (+0.0081) |
| `offsides_ge` | blend | 22 | 0.2803 | 0.2534 | 0.2737 (w=1.0) | promote, small (+0.0067) |
| `player_sot_ge` | blend_capped | 29 | 0.2006 | 0.2047 | 0.2189 (w=0.35) | **DO NOT** (−0.0183) |
| `sot_gt`,`corners_gt`,`team_sot_ge`,`player_scores` | shadow | — | — | — | CV picks w=0 / blend hurts | **stay shadow (vindicated)** |

**Findings — they match every prior (shadow_agents.py, the settled audit):**
1. **`player_ga` is the clean win.** Assists are the model's blind spot; the full-ensemble blend cuts
   Brier 0.1937→0.1741, closing ~40% of the gap to the crowd (0.1418). Robust across all 10 LOO folds.
2. **`player_sot_ge` — the uniform blend HURTS (−0.0183).** The ensemble systematically *over-predicts*
   SoT (it lifts the correctly-low bets, not just the catastrophic ones — same over-prediction the raw
   history showed). **But the catastrophic-low tail IS rescued** (we bet <0.20 and it happened: Brier
   0.775→0.712). Verdict: not a uniform blend — an **asymmetric, tail-only rescue** (pull UP only when
   the expert sits far above us), mirroring the Phase-1 one-sided floor. That is the L2 refinement.
3. **The shadow protections are vindicated.** Blending `corners_gt` (−0.0076), `team_sot_ge` (−0.0084),
   or `player_scores` (would drag our 0.030 edge toward the 0.31 ensemble) all HURT — CV correctly
   selects w=0. Our proven edges must stay shadow.

**Catastrophic-low tail (we bet <0.20 but it happened) — the user's stated target:**
`player_ga` 0.829→0.431 (2 cases, massive); `player_sot_ge` 0.775→0.712 (3 cases, helped but
outweighed by body losses under a uniform blend). The fix the data points to is *targeted*
(`player_ga` full blend; `player_sot_ge` tail-only rescue), not a blanket pull.

**Operational note:** Haiku 4.5 rejects adaptive thinking (400) — `expert.py` omits it for that model;
the panel is genuinely 3-wide. `ai_panel.json` / `ai_panel_prep.json` are slate-specific,
non-deterministic caches → gitignored (regenerated per slate by prep→write).

**Recommended L2:** promote `player_ga` live at a *regularized* weight (start conservative, e.g. w≈0.5,
not the CV's aggressive 1.0, given n=10); build the asymmetric `player_sot_ge` tail-rescue; re-fit
`fouls_gt`/`offsides_ge` weights with regularization. No weight is flipped live without sign-off.

---

## 11. Phase L2 — fitted gate + the `player_sot_ge` verdict (2026-06-20)

`scripts/fit_expert_gate.py` chooses each blendable archetype's parameter by honest LOO-match CV with
a **parsimony rule** (smallest param within ε of the best OOS Brier, capped at w≤0.75 — never full
replacement) and an improvement gate. Output `data/processed/expert_gate.json`, **inert until
`AI_EXPERT_ENABLE=1`**:

| Archetype | mode | param | model → OOS Brier | catastrophic tail | ship? |
|---|---|---|---|---|---|
| `player_ga` | blend | w=0.75 | 0.1937 → **0.1834** (+0.0103) | 0.829 → 0.546 | ✅ |
| `fouls_gt` | blend | w=0.75 | 0.2770 → 0.2721 (+0.0049) | — | ✅ |
| `offsides_ge` | blend | w=0.60 | 0.2803 → 0.2784 (+0.0020) | 0.706 → 0.649 | ✅ (marginal) |
| `player_sot_ge` | asym_floor | **f=0** | 0.2006 → 0.2006 (0) | no help | ❌ stay shadow |

**The `player_sot_ge` verdict — the motivating case, and why the discipline mattered.** The intuition
that launched this work (use the LLM to rescue our catastrophic-low SoT bets, à la Gyökeres 0.11→0.44)
does **not** generalize:
- The ensemble systematically **over-predicts** player SoT. An ensemble floor at f=0.5 lifts 12
  low-model cases — but **9 were outcome=0 (false lifts) vs 3 outcome=1**: wrong more than twice as
  often as right.
- Every nonzero floor **monotonically worsens** mean Brier (body Brier 0.134 → 0.160 as f→0.7); the
  tail rescue (0.775 → 0.729) is marginal and bought at a real body cost.
- We are already **on par with the crowd** on `player_sot_ge` (0.2006 vs 0.2047) — it is *not* a
  lagging case. The single Gyökeres anecdote doesn't represent the population.
- Verdict: keep `player_sot_ge` **shadow**. The Phase-1 flat floor (0.15) is the right, validated
  risk-management here; the LLM does not improve it. The shadow-first gate protected us from a
  plausible-but-wrong intervention — exactly its job.

L2 therefore ships precisely where we genuinely lag the crowd (`player_ga`, by 0.052) plus the two
no-edge count types, and correctly declines the rest. This realises the user's directive — a *weighted*
move toward the better-calibrated center as risk management, on the uncertain question types — without
touching anything we're already good at.

**Arming (the only live change, fully reversible):**
1. Schedule `scripts/ai_panel_live.py` once per slate (after the model numbers exist, before lock) to
   generate the live panel out of band.
2. Set `AI_EXPERT_ENABLE=1` in the live scheduler env. Unset it to instantly revert to pure model.
   The per-archetype gate (`expert_gate.json`) and the routing guards bound *what* can move; the
   switch bounds *whether* anything moves.

---

## 12. Tier-1 rich grounding — and `player_sot_ge` reverses (2026-06-20)

L1/L2 ran on **thin** grounding (player career club rates only; team questions got nothing but the
question + our prior). We enriched the grounding from data we already have: the **matchup** (opponent
inferred from the player's side), the player's **full** club profile (position, shots/SoT/G/A/xG/xA/
key-passes per-90, minutes, league), and the archetype's **empirical base rate** as an explicit anchor
("treat as the anchor; do not over-price a favourite"). Re-ran the player-prop panel (Opus+Sonnet, ~$1.6)
and re-fit:

| Archetype | thin ens mean | rich ens mean | realized | thin blend | rich blend (gate) |
|---|---|---|---|---|---|
| `player_sot_ge` | 0.41 (over-predicts) | **0.34** (calibrated) | 0.31 | 0.2042 (declined) | **0.1995** w=0.5 → ships |
| `player_ga` | — | — | 0.20 | 0.1758 | **0.1631** w=0.75 |
| `player_scores` | 0.097 | 0.058 | 0.00 | 0.0297 (w=0) | 0.0297 (w=0, protected) |

**The `player_sot_ge` over-prediction was a thin-context artifact.** With the base-rate anchor +
opponent + full profile, the rich ensemble calibrates (0.41 → 0.34 vs realized 0.31), its Brier beats
*both* our model and the crowd (0.1922 vs 0.2006 vs 0.2047), and the capped blend now improves OOS
(0.1995 < 0.2006) **and** cuts the catastrophic-low tail (0.775 → 0.678). It flips from DECLINED to
SHIPPED. This validates the original instinct that the panel must reason from rich context, not two
career numbers — and shows the L1/L2 verdicts were *conditional on the grounding*, as flagged.

**Gate now (rich-validated):** `player_ga` blend w=0.75; `player_sot_ge` capped blend w=0.5.
`fouls_gt`/`offsides_ge` (thin-fit, marginal +0.005/+0.002) **dropped** pending a rich re-validation —
the gate ships only what's validated on the grounding the live pipeline generates. Tactical questions
now also receive the matchup + base rate (were fully blind). Still inert until `AI_EXPERT_ENABLE=1`.

**Not yet built:** Tier 2 — the lineup pass (API-Football confirmed XI / expected minutes); Tier 3 —
web search for live (post-cutoff) form + online/critic consensus. Budget-managed: the re-test was
scoped to the player-prop archetypes (the gate-relevant ones), not the full 221.
