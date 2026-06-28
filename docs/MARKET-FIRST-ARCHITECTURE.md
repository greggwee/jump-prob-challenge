# Market-first architecture — verified plan

Premise (from the critique + Tier-1 verification): we can't out-model the books from a ~8-match
international sample (team-SoT ICC was 0.20). The logical source of an outcome probability is the
**market price**, and — because we're scored vs the noisy *field*, not the market — **matching the
sharp line beats the field by construction.** This plan is verified against (a) what an odds feed
*actually* returns with our key, (b) the platform's *actual* questions, and (c) the parser that maps
one to the other.

## A. What the market actually provides (verified live, our The Odds API key)

| market key | status with our key | maps to platform question |
|---|---|---|
| `h2h`, `totals`, `btts` | ✅ 41 / 12 books | win, total_ge/le, btts (already anchored) |
| **`player_shots_on_target`** | ✅ **2 books**, outcomes `Over 0.5/1.0/2.0` | **`player_sot_ge`** — "at least 1 SoT" = **Over 0.5** |
| **`player_goal_scorer_anytime`** | ✅ **5 books**, `Yes` | **`player_scores`** — "score a goal" |
| `player_assists` | ✅ 1 book, `Over 0.5` | combine with anytime → **`player_ga`** ("score or assist") |
| `totals_corners` | ❌ **422 invalid market** | corners — *not carried by The Odds API* |
| `cards` | ❌ **422 invalid market** | cards — *not carried by The Odds API* |

**The headline:** the model's *weakest* area — player props (under-predicted; the v2 sheet had to fight
it) — is **directly market-anchorable today.** Corners/cards are **not** in The Odds API; they'd need a
paid feed ([OpticOdds](https://opticodds.com/sports/soccer) explicitly streams player/ team cards &
corners; [OddsJam](https://oddsjam.com/odds-api), [Goalserve](https://www.goalserve.com/en/sport-data-feeds/odds-api/description))
or stay on the model.

## B. Verified question → market map (de-vig math)

- **`player_sot_ge`** ← de-vig `player_shots_on_target` **Over 0.5** (two-way Over/Under at the 0.5 line; Shin de-vig) → P(≥1 SoT). Blend 85 % market / 15 % model as today; floor where only 1–2 soft books quote.
- **`player_scores`** ← de-vig `player_goal_scorer_anytime` `Yes` (vs the field's implied "No" across the book) → P(scores).
- **`player_ga`** ← `1 − (1−p_score)(1−p_assist)` from anytime-scorer + assists (mild positive correlation; cap the combine).
- **goal markets** ← unchanged (h2h/totals/btts).
- **everything else** (team SoT, corners, cards, offsides, fouls, exotic combos) ← **no market in our feed → model / LLM** (or upgrade the feed for corners/cards).

## C. Parser robustness — a live bug found and partly fixed

Auditing the **544 distinct** platform questions exposed **56 unparsed → silently dropped → no
prediction from either bot** (all currently open). Cause: knockout phrasing.

- **Pass 1 (56 → 37):** strip the `"... in regulation (90 minutes + stoppage time)"` qualifier
  (our models are 90-minute), and add tolerant patterns for `win` (was "win the match" only), full-match
  `more corners than`, `score N+ goals`, and `be ahead/winning at halftime`. Recovered 19 known
  archetypes; `win by N` is correctly **not** mis-read as `win`.
- **Pass 2 (37 → 22), the 2026-06-28 drop audit:** the residual was *not* mostly new physics — 15 were
  derivable from the already-calibrated, market-anchored **goal grid**, so they cost no new fitted
  parameters (no overfitting) and were pure forfeit-recovery:
  - `first_goal_match` (×7), `win_margin`, `clean_sheet`, `score_both_halves`, `draw` (was in
    `GOAL_ARCHS` but had **no handler**!), `half_total_ge`/`h2_more_goals` (existing handlers, new
    "produce" phrasing), `cards_ge`-in-half, `red_card` (constant).
  - All inherit the market view when goals are anchored; anchored goal markets skip extremize, so no
    risk-gate entanglement. Also fixed a general `canonical()` bug — a leading definite article ("the
    United States" / "the Netherlands") left the side unresolved and dropped *any* "the X" subject.
- **Pass 3 (22 → 14):** the **hydration / cooling-break** window markets ("goal before the first / after
  the second break", "≥2 corners / a card / an offside" in those windows) are answerable *analytically* —
  the MC `SimResult` discards per-minute times, but the minute-profile bucket weights give each window's
  share of the family's intensity, so `λ_window = μ_total · share` (goals Poisson, counts via the count
  model's dispersion). Breaks ~30'/75' → slots [0,30)/[75,90). No engine change. Verified live (goal-in-
  first-30' ~45 %, card-after-75' 66 % from the late-card spike). These archetypes are deliberately left
  out of the family sets → no extremize / risk-gate entanglement.
- **Still dropped (14), triaged:**
  - **Count-extendable (~5):** total shots (on+off ≈ SoT/0.37), total offside calls, both-teams-cards.
    *Cheap, but they introduce new no-edge count archetypes that need risk-gate entries to avoid the
    "layers fighting" extremize issue — do them with the risk-gate refresh, not ad hoc.*
  - **Beyond-90 / no model (~5):** first/second-half **stoppage-time** goals (the 6-bucket profile stops
    at 90' — need minute-resolved stoppage data for a base rate), **substitution before halftime** /
    **substitute scores** (no substitution data in the model at all).
  - **Market/ambiguous (~4):** **advance to Round of 16** (group-stage progression depends on *other*
    matches, not just this one → needs the outright/to-qualify line, not our match model), any-player
    SoT/multi-goal (max-over-players — derivable from the player table but low-value, thin).

**The lesson:** the regex parser is brittle and the system *silently drops* what it can't match, but the
durable fix is **route, don't regex** — most "new" archetypes are existing models in disguise, and the
genuine residual is small enough to triage by data source (grid / MC / market / base rate).

## D. The architecture

For each open market, resolve the estimate in this order:
1. **Market anchor** — if a feed prices it (now: goals + player SoT/scores/assists; +corners/cards on a
   paid feed), de-vig and use it. Sharpest, lowest-variance, beats the field by construction.
2. **LLM classify + reason** — (a) a fallback **classifier** that maps any *unmatched* question to an
   archetype (kills the silent-drop bug far more durably than chasing regexes), and (b) qualitative
   reasoning for thin/unpriced markets and live context (lineups, motivation, press-style → offsides).
3. **Model** — the current estimators as the last-resort floor (offsides, fouls, exotic combos, new
   archetypes with no market).

This inverts today's "model everything, anchor only goals."

## E. Build sequence

1. ✅ **Parser fix** (recovered 19 live markets on both bots) — done, commit `4be810f`.
2. ✅ **Anchor player props** — `fetch_player_props` de-vigs `player_shots_on_target` /
   `player_goal_scorer_anytime` / `player_assists`, blended into `player_sot_ge` / `player_scores` /
   `player_ga` (skips the base-blend/extremize/confidence steps that patched the model we anchor away
   from). Done, commit `125aeaf`. Coverage grows as matches enter the 36h prop-fetch window.
3. ◑ **Stop the silent drop** — the 2026-06-28 audit took 37 → **14** by *routing to existing
   handlers*, not an LLM classifier: 15 to the goal grid (`0ea7ef0`) + the hydration-break window
   markets recovered analytically (`0fc7b9a`). **Remaining work:** (a) count-archetype extensions
   *with* risk-gate entries (~5: total shots/offsides, both-teams-cards), (b) stoppage-time goals (need
   beyond-90 minute data) + substitution markets (no sub data), (c) advance-to-R16 via the to-qualify
   line, (d) an out-of-band LLM classifier as the final catch-all so *nothing* drops — populated like
   the existing panel (daemon reads a cache; never an LLM call in the near-kickoff submit path).
4. **(Optional, paid) corners/cards feed** — only if an OpticOdds/OddsJam trial confirms WC-2026 coverage
   at usable vig; else keep the model (offsides/fouls/corners are already well-calibrated).

## F. Caveats / uncertainty

- **Thin books on props** (SoT 2, assists 1) — soft lines, higher vig, missing for minor players →
  de-vig is noisier there; blend conservatively and floor where coverage is thin. Liquidity improves
  near kickoff.
- **De-vig method matters** (Shin vs multiplicative) on 2-way prop margins — a real error source.
- **Coverage gaps are real**: corners/cards/team-SoT/offsides/exotic/new-archetypes have **no** market in
  our feed → still modeled or LLM'd. Market-first is *most* of the board, not all of it.
- **Anchoring caps us at the field-vs-market gap** — we can't beat the books this way, but the
  verification says we have no proven edge to spend chasing more; the gap is the reliable farm.
- **The parser fix is partial** — 37 new archetypes still drop; the LLM classifier (step 3) is the
  durable fix, not more regexes.
