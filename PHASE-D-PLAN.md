# Plan + verification: betting-review #4 (prop lines) and #5 (right-size weight)

## #4 — Bookmaker prop lines: VERIFIED INFEASIBLE via current source

**Intent:** anchor the no-line prop markets (player SoT, anytime scorer, cards,
corners) to bookmaker lines — to (a) sharpen our prop truth estimate, (b) give a
real field-line proxy, (c) measure prop CLV.

**Verification (`scripts/discover_props.py`, 2026-06-14):** probed every prop
market key on The Odds API for `soccer_fifa_world_cup`:

| market key | books offering |
|---|---|
| player_goal_scorer_anytime | **0** |
| player_shots_on_target | **0** |
| player_shots / player_assists | **0** |
| player_to_receive_card (cards) | **0** |
| btts / double_chance / draw_no_bet | 0 |
| alternate_totals | 2 (pinnacle, coolbet) |
| team_totals | 1 (pinnacle) |

**The prop lines #4 needs do not exist on The Odds API for the WC.** Player props
and cards return a valid key but zero bookmakers; corners aren't even a supported
soccer key. Only goal-based extras (alternate/team totals) are offered — which
touch the already-well-anchored, low-edge goal markets, not the no-line props.

**Decision: DEFER #4.** Pursuing it would require a *new* data source — a paid
prop-odds provider or scraping a book (bet365/Pinnacle) directly — with uncertain
WC prop coverage and real cost/effort. Not worth it now.

**Two useful consequences of the finding:**
1. **It reinforces the player-prop edge (#2).** If sharp books don't even post WC
   player-prop lines, the SportsPredict field certainly can't look one up — so
   the field is *genuinely naive* on player props, exactly why our model beats a
   sharp field there (t=18.7, p<1e-4). The absence of lines is evidence our edge
   is structural, not a measurement artifact.
2. **Prop CLV is unmeasurable** (no line to value against) — so for props we must
   keep relying on the *backtest* edge gate (#2), not live CLV. That is already
   what the system does.

**Re-scope (optional, low priority):** ingest the available `team_totals` /
`alternate_totals` from Pinnacle to sharpen goal-total markets. Marginal — those
are the low-edge main markets the anchor already covers. A "#4-lite" at best.

**The real path to extend prop edge** is therefore *modeling*, not data ingest:
improve the count models (corners/cards) so they beat a sharp field the way the
player model does. Currently count/timing did NOT clear the #2 backtest gate, so
that is where modelling effort would pay — a separate workstream, not #4.

## #5 — Right-size the odds-blend weight

**Intent:** the λ-blend is 85% market / 15% model. If the 15% model has negative
CLV on goal markets (doesn't beat the closing line), raise the market weight
toward ~0.95–1.0; if it adds value, keep or lower it.

**Key clarification (verified):** the blend happens at the *goal-intensity (λ)*
level per match, so it is a **single weight** affecting all goal-derived markets
uniformly — not a per-family knob. So #5 = right-sizing one number (`W_MARKET` /
`runner.w_market`), driven by goal-market CLV.

**Mechanism: READY.** `runner.w_market` is already parametrized (improvement #1),
and the CLV ledger logs pure (w=0) / blend (w=0.85) / closing-line (w=1) per
market each run. `analyze_clv.py` computes the three Briers per family; the
weight recommendation falls straight out:
- `Brier(market) < Brier(blend)` ⇒ the 15% model is **hurting** ⇒ raise w → ~0.95.
- `Brier(blend) ≤ Brier(market)` and `≤ Brier(pure)` ⇒ 0.85 is fine, hold.
- `Brier(pure) < Brier(blend)` ⇒ the market is hurting (unlikely) ⇒ lower w.

**Status: DATA-GATED.** The CLV ledger has no *settled* goal markets yet (it was
created this session; logged markets are still open). So #5 cannot produce a
verdict until ~1–2 matchdays settle. The mechanism and decision rule are in
place; `analyze_clv.py` now prints the weight recommendation and will fire it
automatically once data accrues.

**A-priori note (do not act on yet):** the 0.85 weight was fit against
FiveThirtyEight (Phase 3), a *weaker* anchor than sharp books. Sharp books are
sharper, so the CLV-optimal weight is *likely* higher (~0.90). But this is a
guess — wait for the live CLV data rather than pre-empting it. Disciplined
evidence-gating, same as everywhere else.

## Net
- **#4: blocked by data availability — defer** (the finding usefully confirms the
  field is naive on props, reinforcing #2). Optional #4-lite (team totals) is low
  value.
- **#5: mechanism ready, decision rule defined, data-gated** on the CLV ledger
  settling — `analyze_clv.py` now emits the recommendation. No premature weight
  change; let the closing-line data decide.
