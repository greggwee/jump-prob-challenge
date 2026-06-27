# Tier 1 plan — variance reduction via richer data

Tier 1 from the model critique: replace thin proxies with direct, lower-variance signals. The
survey of local assets reframes the work — **most of the data is already here**, so this is mostly
"use what we hold far better," with one binding constraint that shapes everything.

## The assets we already have

| table | grain | content | gap |
|---|---|---|---|
| `team_half_counts` | team × match × half | SoT, corners, fouls, cards, offsides, **opponent, referee** | only **~8 international matches/team** (76 teams, 314 matches) |
| `event_stream` | event × slot | per-team goal/sot/corner/card/foul/offside timing | same matches |
| `player_match` | player × match | shots, **sot**, goals, assists, minutes, **tournament** | **all national-team** comps (wc/euro/afcon/copa); no club |
| `understat_rates` | player (season) | **xG, npxG, xA, shots p90, position** | **club only** (top-5 leagues) |
| `raw/statsbomb` | event | full events for afcon23/euro/copa/wc | the international **xG** source |

## The binding constraint (read first)

**International samples are tiny.** A national team plays ~8 matches in our data (and only ~10–15/yr
even with qualifiers). So *per-team* count/xG ratings are estimated from n≈8 — high variance. The
current pooled λ-proxy model **borrows strength across all teams**, which at n≈8 may be *better* than
a noisy per-team rating. **Therefore: every per-team rating must be heavily shrunk, and must prove it
beats the pooled model out-of-sample before shipping.** This is the make-or-break question, verified
in the companion checks.

Player-level data is the exception — 2,287 players × multiple comps = far more observations per unit
than team-level. **That makes Tier 1.3 the most promising item, not 1.1/1.2.**

---

## 1.1 — Direct per-team count ratings (SoT / corners / fouls)

**Now.** `log μ_count = c0[half] + c1·log λ_team + c2·log λ_opp` — pooled; team identity enters *only*
through the goal-λ. A team that shoots a lot but scores little is mis-rated.

**Data needed.** Per-team SoT-for / SoT-against (have it in `team_half_counts`). To lift n≈8 toward
~30–40/team, **supplement with API-Football `/fixtures/statistics`** for qualifiers / Nations League /
friendlies (`raw/apifootball` is a partial pull already). This is the one place new ingest pays off.

**Model.** Hierarchical Poisson attack/defense on SoT: `log μ_sot(i vs j, h) = base[h] + α_i − δ_j`,
with `α, δ ~ N(0, σ²)` (shrunk to 0). Same for corners/fouls. Replaces the λ-proxy with a SoT-specific
rating, opponent-adjusted.

**Improvement mechanism.** SoT is driven by shot *generation*, not goals (corners-goals corr ≈ 0.11);
a direct SoT rating is a less-biased, lower-variance input — **iff** per-team SoT is a stable signal at
this sample size. **Verify split-half reliability first.**

**Risk.** At n≈8 the rating may add variance, not remove it. Mitigate with strong shrinkage + the
API-Football supplement; gate on an out-of-sample backtest vs the current pooled model.

---

## 1.2 — xG-based goal ratings (replace Elo)

**Now.** `log λ_s = b0 + b1·(elo_s−elo_opp)/100 + b2·home` — one skill scalar.

**Data needed.** Team xG-for / xG-against for internationals — **compute from `raw/statsbomb` events**
(we hold afcon/euro/copa/wc) via shot xG; club form from `understat_rates`.

**Model.** Dixon-Coles–Maher per-team attack/defense fit on **xG** (≈3–4× lower per-match variance than
goals) instead of Elo.

**Honest caveat — modest direct value.** Goal markets are **85 % market-anchored** (`log λ = 0.85·log
λ_market + 0.15·log λ_model`), so a better model λ moves only the 15 % blend and the **no-odds fallback**.
Its real leverage is as the **λ input to the count models** (§1.1) and when the odds API is dry. Same
n≈8 sparsity applies. **Lower priority than the value of "xG" makes it sound** — verify it actually
beats Elo on the internationals we have before building.

---

## 1.3 — Player club-vs-country rates + xG (the promising one)

**Now.** One global prior; a player's club and country history are pooled (in practice we only *have*
country, in `player_match`). Stars shrink to the population mean.

**Data needed.** Two complementary sources we already hold:
- **country rate** ← `player_match` (sot, minutes per national-team match),
- **club rate + xG + position** ← `understat_rates` (shots p90, xG p90, position),
joined **by player name** (the engineering risk — fuzzy match across "player" ↔ "player_name").

**Model.** Per player, estimate `r_country` and `r_club`, each Gamma-Poisson shrunk to a **position
prior** (from understat position), then blend:
`rate = w·r_country + (1−w)·r_club`, `w` rising with national-team minutes (relevance) but capped low
when country n is tiny (stability). `P(≥1 SoT | M min) = 1 − e^{−rate·M}`, minutes mixed as today.

**Improvement mechanism.** (a) club data (richer, xG-quality-adjusted) stabilizes the estimate the
current model can't see; (b) a **position** prior stops defenders and strikers shrinking to the same
mean; (c) xG p90 is a lower-variance shooting signal than raw SoT counts. This is the item most likely
to actually move Brier, because player-level data is plentiful.

**Risk / verify.** (a) **Join coverage** — what fraction of WC players appear in understat (top-5
leagues only)? (b) **Transfer** — does club shooting predict country shooting (else club data is
irrelevant)? Both checked in verification.

---

## Build sequence & validation

**Order by promise × data-readiness:** **1.3 first** (data in hand, player-level density, clear
mechanism) → **1.1** (only if the SoT-reliability check + API-Football supplement justify it) →
**1.2** (lowest direct value given the anchor; do as the count-λ upgrade, not for goal markets).

**Validation (non-negotiable, per the overfitting discipline):**
1. Build each estimator, **backtest out-of-sample** on the existing settled set (train on pre-date,
   score post-date) vs the current model — *before* any live wiring.
2. **Shadow mode** live (compute + log, don't submit) until aggregate Brier beats live Bot A.
3. **Heavy regularization** on every rating (international n is small); report the shrinkage that wins
   OOS, not in-sample.
4. Judge on **aggregate** Brier over many matches (power: ~1,000 settled to resolve +1 RBP/q), never
   per-archetype.

## Verification results (empirical, on the data we hold) — the plan changed

**1.1 — per-team count ratings: FAILS the bar. Shelve.** Team-SoT split-half reliability is **r=0.36**
(49 teams, median 9 matches), and **ICC ≈ 0.20** — only ~20 % of SoT variance is the stable team signal;
~80 % is match-to-match noise. A per-team rating would shrink almost all the way back to the pooled mean
and add little over the λ-proxy, at real build cost and overfitting risk. **The pooled model is
empirically mostly right.** The API-Football supplement raises n but ICC 0.20 caps the ceiling — even
~30 matches/team leaves the signal weak. Not worth building now.

**1.2 — xG goal ratings: same sparsity + the anchor caveat. Shelve for goal markets.** Team xG ratings
hit the identical n≈8 wall, and goal markets are already 85 % market-anchored — the achievable gain is
tiny twice over. xG's validated value turns out to be at the **player** level, not as team ratings.

**1.3 — player club-vs-country + xG: CONFIRMED. Build this.**
- **Transfer is real:** corr(country SoT/90, club shots/90) = **0.50**; with club **xG/90, 0.54** — club
  shooting genuinely predicts country shooting, and xG is the marginally better signal. This is exactly
  the information the current single-prior model cannot see.
- **Coverage:** 30 % of national-team players (≥180 min) match understat, and that 30 % is concentrated in
  the **top-5-league attackers the props are usually about**, so effective coverage on the questions that
  matter is higher. The remaining ~70 % fall back to today's prior — no regression.

**Revised Tier 1 = Tier 1.3 only.** Verification killed both team-level items (the international sample is
too thin — ICC 0.20 — for per-unit ratings to beat the pooled/anchored model) and confirmed the
player-level one. This is the overfitting discipline doing its job: the *obvious* upgrades (1.1, 1.2)
were chasing an ICC-0.20 signal; the real, evidenced win is the **player club/country/xG blend**.

### Build 1.3 (the surviving plan)
1. **Name-bridge** `player_match` ↔ `understat_rates` (normalized + fuzzy; hand-verify the top-200 WC attackers).
2. Per player: `r_country` (from `player_match` SoT/min), `r_club` (from understat **xG/shots p90**),
   each shrunk to a **position prior** (understat position).
3. Blend `rate = w·r_country + (1−w)·r_club`, `w` rising with national-team minutes but capped where
   country-n is tiny; fall back to today's prior when unmatched.
4. **Backtest OOS** (train pre-date, score post-date) vs current player model on `player_sot_ge`/`player_ga`;
   ship to **shadow** only if aggregate Brier improves; then forward-validate before going live.
