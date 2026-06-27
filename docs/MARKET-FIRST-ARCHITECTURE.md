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

- **Fixed this pass (56 → 37):** strip the `"... in regulation (90 minutes + stoppage time)"` qualifier
  (our models are 90-minute), and add tolerant patterns for `win` (was "win the match" only), full-match
  `more corners than`, `score N+ goals`, and `be ahead/winning at halftime`. Recovered 19 known
  archetypes (win, team_goals, corners_gt, ht_leading, cards_ge, total_sot) — all with working models;
  13 parser tests pass; `win by N` is correctly **not** mis-read as `win`.
- **Still dropped (37):** genuinely *new* archetypes — winning margin, clean sheet, total shots
  (on+off), score-in-both-halves, first-goal-of-**match**, red-card-alone, draw, **hydration-break**
  timing, stoppage-time goals, substitution timing, **advance to Round of 16**, any-player props.

**The lesson:** the regex parser is brittle — the platform keeps adding knockout-specific phrasings and
the system *silently drops* what it can't match. **Question understanding must be made robust**, or
~10 % of every knockout match is forfeit.

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

1. **Commit the parser fix** (recovers 19 live markets on both bots — do first, it's RBP bleeding now).
2. **Anchor player props** — extend `market_anchor.py` to fetch `player_shots_on_target` /
   `player_goal_scorer_anytime` / `player_assists`, de-vig, blend into `player_sot_ge` /
   `player_scores` / `player_ga`. Highest model-improvement per unit effort; verified available.
3. **LLM classifier fallback** for unmatched questions → stop the silent drop; route the 37 new types to
   model/market/LLM instead of `None`.
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
