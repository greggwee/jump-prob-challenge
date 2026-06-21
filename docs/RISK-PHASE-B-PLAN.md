# Risk Phase B — LLM corroboration ("earn back the deviation")

**Status:** planned, pending verification → build.
**Depends on:** Phase A (risk_gate, armed) — the statistical de-confidence shrink.

## The gap Phase A leaves

Phase A shrinks no-edge-archetype submissions (offsides/fouls/cards) toward the base rate, *blindly* —
because we have no **statistical** edge there, any deviation is treated as noise (+99.6 RBP backtested).
But sometimes a deviation *is* justified by factors the count model can't see:
- a **high-press** side genuinely draws more offsides,
- an **aggressive matchup / derby / must-win** genuinely means more fouls and cards,
- a heavy **home favourite** vs a parked-bus **underdog** skews the whole game state.

Phase A would shrink those *correct* deviations back toward 50/50. Phase B lets an LLM, reasoning over
exactly those qualitative factors, **re-justify** a deviation — turning a blind shrink into a *calculated*
risk (the user's framing: statistical **and** LLM-agreed factors).

## Mechanism

1. **Out-of-band tactical-risk panel** (reuse `expert.py` / `ai_panel_web.py`). For each no-edge question
   where the model deviates notably from base (`|logit(model) − logit(base)| > τ`), a panelist is asked:
   > "The base rate for this market is B%. A statistical model with no proven edge here estimates M%.
   > Weigh ONLY qualitative factors it can't see — home-ground advantage, recent form/momentum,
   > favourite vs underdog status, and playing style (high-press → more offsides; aggressive/derby →
   > more fouls/cards). Is a deviation from the base rate justified, and which way? Give your single
   > calibrated probability."
   Web-search-grounded (live form/news), strictly pre-match. Returns a probability + one-line rationale.

2. **Shrink toward the LLM-justified center, not the raw base.** Phase A is `shrink(model, base, s)`.
   Phase B replaces `base` with the LLM's qualitatively-grounded estimate `c_llm`:
   `submit = shrink(model, c_llm, s)` where `c_llm` sits between base (no justification) and the model
   (full justification). So:
   - LLM **corroborates** the model's deviation → `c_llm` near the model → small shrink (deviation kept).
   - LLM sees **no** justification → `c_llm` near base → full Phase-A shrink.
   - LLM **disagrees** (opposite side) → `c_llm` past base → we shrink *through* base toward the LLM.

3. **Routing & guards** (identical discipline to the expert layer):
   - Only the `risk_gate` no-edge archetypes (offsides/fouls/cards) — never proven-edge counts,
     player props, or market-anchored goals.
   - **Conservative default:** no panel entry / LLM neutral → Phase-A shrink to base. The LLM can only
     *earn back* deviation, never manufacture confidence beyond the model.
   - Out-of-band (the daemon hot-reads a cache; never calls an LLM), shadow-first (a master switch),
     budget-gated (only the *deviating* no-edge questions, ~5–10/slate).

## Architecture

- A `tactical_risk` role in `expert.py` with the prompt above; reuse the web-search path (`forecast_web`)
  for live form/momentum.
- A near-kickoff/daily pass (mirror `ai_panel_web.py`) writes `c_llm` per no-edge market into the panel
  cache with `role: "tactical_risk"`.
- `strategy.apply_risk` gains an optional per-question center: if a `tactical_risk` entry exists for the
  market, shrink toward `c_llm`; else toward the base rate. Gated by `RISK_GATE_ENABLE` (already armed)
  plus a `RISK_LLM_ENABLE` master switch (default off until forward-validated).
- Forward-test ledger (mirror `ai_panel_web_ledger.csv`): log `c_llm` vs outcome so
  `forward_test_web.py`-style scoring decides whether the LLM relaxation beats the pure base-shrink.

## Cost & validation

- ~$0.2–0.4 per deviating no-edge question (web search); ~$1–3/slate. Budget-gated, shadow-first.
- **Empirical basis (already in hand):** the L1 forward-test showed the LLM ensemble *did* help these
  archetypes a little — `offsides_ge` +0.0067 and `fouls_gt` +0.0081 Brier — small but positive, which
  is exactly the "some qualitative signal, not enough to bet the farm" that a *relaxation* (not a blind
  blend) is designed to harvest.
- Cannot be offline-backtested cleanly (web search leaks settled results; and the relaxation only matters
  where the qualitative case is real) → **forward-validated** via the ledger as no-edge questions settle,
  same shadow-then-promote gate as every prior layer.

## Open question for the gate

`s` (how much deviation the LLM can earn back) and `τ` (the deviation threshold that triggers an LLM
call) are fit on the forward-test ledger, regularized — not set a priori. Until then Phase B stays in
shadow and Phase A (the validated +99.6 RBP backbone) carries the risk management.
