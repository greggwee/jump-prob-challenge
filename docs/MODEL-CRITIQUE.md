# Bot A — per-category model critique & estimation plan

A statistician's review of Bot A, **category by category**: what it does now, where the
589-settled calibration says it's wrong and *why* (the mechanism), the **specific data
source** that returns the right signal for that question type, and a **robust estimator**
to replace the current one. The thesis: Bot A is well-*engineered* but runs every category
off a **thin, generic signal** — one Elo number, the goal-λ as a proxy for all counts, a
single population prior for all players, and hard constants for penalties. Only the goal
markets get a sharp external anchor. Each category deserves its own data and estimator.

## ⚠ The evidence — and how little of it survives noise

`gap = mean(pred) − realized`. But a gap is evidence only if it beats sampling noise, so each is
tested with a **Spiegelhalter calibration z** (`z = Σ(yᵢ−pᵢ)/√Σpᵢ(1−pᵢ)`) and judged against the
fact that **23 archetypes were tested at once**.

| archetype | n | pred | real | gap | z | p | real signal? |
|---|---|---|---|---|---|---|---|
| `sot_gt` | 44 | .50 | .36 | +.14 | −2.0 | **.044** | no — 1 hit ≈ the 1.2 expected by chance |
| `pen_or_red` | 26 | .34 | .19 | +.15 | −1.6 | .11 | no |
| `team_scores_half` | 31 | .53 | .65 | −.12 | +1.4 | .16 | no |
| `team_sot_ge` | 21 | .43 | .33 | +.09 | −0.9 | .37 | no |
| `win` | 50 | .54 | .58 | −.04 | +0.6 | .54 | no |
| `player_sot_ge` | 62 | .25 | .29 | −.04 | +0.8 | .44 | no |
| `player_ga` | 18 | .20 | .22 | −.02 | +0.2 | .83 | no |
| `fouls / corners / offsides / btts…` | | | | ~0 | <0.4 | >.7 | calibrated ✓ |

**Honest verdict on the evidence: there is essentially none.** Of 23 archetypes, exactly **one**
clears p<0.05 (`sot_gt`), and pure chance predicts **~1.2** false positives across 23 tests — so
getting one is *exactly* the null expectation. **Zero** survive Bonferroni (p<.0022) or BH-FDR. The
per-category "biases" are **statistically indistinguishable from sampling noise.** This is the same
wall as edge estimation: with 20–60 settled per archetype and a Brier noise floor near 0.25, we
**cannot resolve a per-category bias smaller than ~±0.15**, and nearly every gap sits inside that band.

**So the sections below are a STRUCTURAL critique, not a list of confirmed defects.** The numbers
motivate *hypotheses*; none is a mandate to retune. The case for each change must rest on the two
things that *don't* depend on these noisy gaps — see the next section.

---

## What can justify a change, given the gaps are noise

Two bases survive the multiple-comparison reality — and *only* these two:

1. **Variance reduction from better DATA (the robust case).** Swapping a thin proxy for a direct,
   lower-variance observation reduces a rate's *estimation error* **whether or not any bias exists.**
   xG instead of goals (≈3–4× lower per-match variance), a team's *direct* SoT-for/against instead of
   the goal-λ proxy, a referee's *career* card rate instead of 3 tournament matches — each shrinks
   estimator variance on first principles. This is the safest class of change; it isn't a gap-chase.
2. **Structural correctness from THEORY (the weaker case).** Some assumptions are wrong on first
   principles regardless of the data — two teams' SoT are *mechanically* coupled by game state, so the
   conditional-independence `P(A>B)` is mis-specified. Fixing it is defensible, but **its empirical
   benefit here is unconfirmed** (the `sot_gt` gap is noise), so it must prove itself out-of-sample.

**Not a valid basis: the calibration gaps themselves.** Retuning `pen_or_red` 0.34→0.19, or adding a
position-prior layer because stars *looked* under-predicted, is **fitting n≈20–60 of noise** — the
textbook overfit. It would memorize this tournament's draws and likely *hurt* the next.

**Validation discipline for anything built from this doc:**
- Score every change on **held-out / forward** Brier (settlements *after* it ships), never on the data
  that suggested it. The edge gate already flipped `count` 1.3↔1.0 between two snapshots this week —
  small-sample fits are not stable.
- **Add parameters only with strong regularization** (hierarchical priors, heavy shrinkage). A
  5-way position prior or per-referee factor on this data overfits without it.
- Prefer benefit that is **mechanical (lower variance)** over benefit **inferred from a gap**.
- Respect the **irreducible floor**: these are near-coin-flip events (Brier ⌐ 0.25); a "perfect" model
  gains *thousandths*. Don't mistake the noise band for headroom.

---

## 1. Goal markets — `win`, `total_ge/le`, `btts`, `team_goals_ge`, `*_scores_half`

**Now.** `log λ_s = b0 + b1·(elo_s − elo_opp)/100 + b2·home` → independent Poissons + Dixon-Coles
ρ on the {0,1}² cells (`goal_model.py`). Goal markets are then anchored **85 % to the betting
line** (de-vig → consensus → λ-inversion). A sophisticated state-dependent point process
(`dynamics.py`) splits goals into minutes/halves.

**Critique.** Elo is **one scalar** — it cannot separate *attack* from *defense* or style. Two
teams with the same Elo gap get identical λ whether they're 3-2 merchants or 1-0 grinders. The
market anchor rescues `win`/`totals` (which is why those are ~calibrated), **but the derived
markets inherit the grid's per-team-per-half split, which rests on the *un-anchored* Elo λ**:
`team_scores_half` under-predicts **−.12 (n=31)** and `team_goals_ge` **−.15 (n=12)** — the model
allocates too little scoring to the specific team/half. And `win` under-favours the favourite
(**−.04, n=50**): a single-parameter supremacy is too timid versus the market's.

**Data.** Replace Elo with **xG-based** attack/defense:
- [StatsBomb open data](https://github.com/statsbomb/open-data) — free event-level shots with xG, incl. the 2018 World Cup and internationals (the cleanest public xG source for a national-team prior).
- [TheStatsAPI](https://www.thestatsapi.com/) / [Highlightly](https://highlightly.net/football-api/) — team xG per match at breadth; [Understat](https://understat.com) for club form.

**Proposed.** The standard **Dixon-Coles–Maher** parameterization: per-team `attack_i`,
`defense_i`, home advantage, `log λ_home = μ + attack_home − defense_away + γ`, fit on **xG**
(lower-variance than goals → tighter ratings in small samples). Anchor the **total and supremacy
separately** to the market (totals line + h2h), and let xG set the *within-match team split* that
the grid needs. This fixes the derived/half markets at the source and lets the favourite ride the
market's supremacy. Timing markets (`ht_tied/leading`, `first_goal_half`, `h2_more_goals`) are
**downstream of this λ** — they inherit the fix automatically.

---

## 2. Shots on target — `sot_gt`, `team_sot_ge`, `total_sot_ge`, `player_sot_ge`

**Now.** `log μ_sot = c0[half] + c1·log λ_team + c2·log λ_opp` (`count_model.py`) — SoT is driven
by the **goal-model λ** as an "attacking volume" proxy. Team-vs-team (`A > B`) assumes the two
sides' counts are **conditionally independent** (`p_a_gt_b`).

**Critique — two distinct errors, both visible:**
1. **`sot_gt` over-predicts +.14 (n=44)** — the **independence assumption**. Two teams' SoT in one
   match are *negatively* correlated (game state: the side on top suppresses the other), and **ties
   are common**. Under independence, `P(A>B)` over-states the favourite and ignores the tie mass.
2. **Goal-λ is the wrong driver.** SoT comes from *shot generation*, not goals (the code itself
   notes corners-goals corr is only 0.11). A team that shoots a lot but converts little is mis-rated.

**Data.** [API-Football](https://www.api-football.com/) `/fixtures/statistics` returns **per-team
shots on target per match** directly → build a team's **SoT-for** and opponent **SoT-against**
rate. `/fixtures/players` returns **per-player SoT per match**. StatsBomb/Understat give shot
locations → **xG-weighted** SoT (shot quality, not just volume).

**Proposed.**
- **Team SoT:** a direct NegBin on `SoT_for_i × SoT_against_j` (a ratings model on SoT itself), not via goal λ.
- **`A > B`:** a **bivariate** count model with a shared game-state latent (or a Gaussian copula with negative ρ), and **handle ties explicitly** — `P(A>B) + P(A=B) + P(A<B) = 1`. This removes the +.14 bias.
- **Player SoT:** per-player SoT rate from `/fixtures/players`, **split club vs national team** (§3).

---

## 3. Player props — `player_sot_ge`, `player_ga`, `player_scores`

**Now.** Gamma-Poisson shrinkage to a **single population prior**:
`rate_i = (events + prior·M0)/(minutes + M0)`, `P(≥1) = 1 − e^{−rate·minutes}`, minutes mixed over
a start/sub/benched distribution (`player_model.py`).

**Critique.** The **single global prior** is the core flaw. A defender and a striker shrink toward
the *same* mean, so **stars are pulled down**: `player_ga` realized 0.30 vs predicted 0.12 earlier;
`player_sot_ge` under by −.04 (n=62, well-powered). Two compounding gaps: (a) ~16 % of players have
no data and collapse entirely to the generic prior; (b) **no club-vs-country split** — a player's
national-team rate differs from his club (different role, teammates, opposition level), yet the
model pools them.

**Data.** [API-Football `/players/statistics`](https://www.api-football.com/) is queryable **per
competition** → pull a player's **club-league** stats *and* his **national-team** stats separately;
`/fixtures/lineups` gives **position**. [TheStatsAPI](https://www.thestatsapi.com/) has 84k players'
season/career SoT, shots, goals; StatsBomb/Understat add **xG/xA** per player.

**Proposed.** A **hierarchical** model with **position-specific priors** (ST / W / AM / CM / DEF):
shrink each player toward his *position* prior, not the global one. Estimate a **club rate** and a
**national-team rate** and blend by minutes-in-each (`w·r_country + (1−w)·r_club`, w from
international minutes share). Role/position from the live lineup. This directly lifts the
star-prop under-prediction that the v2 sheet currently has to paper over.

---

## 4. Cards — `cards_ge`, `cards_gt`, `team_cards_ge`

**Now.** `μ_cards = exp(c0[half]) × ref_factor`, with λ **removed** (it hurt) and the referee factor
**shrunk almost to neutral** (`REF_SHRINKAGE = 40` pseudo-matches) because tournament-only ref
samples are tiny and don't transfer (`count_model.py:33–38`). So cards ≈ a flat base rate.

**Critique.** Cards are driven by **referee tendency × match intensity**, and Bot A captures
**neither** — the ref factor is regularized into nothing and there's no stakes/rivalry term.
`cards_gt` under-predicts −.10 (n=16, the comparison-independence issue again). The code's own note
is the fix: *"factors should come from FBref career card rates (hundreds of matches per ref), not
tournament counts."*

**Data.** Real **referee career card rates** (large samples):
[FootyStats](https://footystats.org/stats/referee-stats),
[PlayerStats referees](https://playerstats.football/referees),
[OddAlerts referee API](https://www.oddalerts.com/referees) (programmatic),
[Soccermetrics](https://soccermetrics.github.io/fmrd-summary-api/examples/referee_performance.html).
Benchmark: ~4.4 yellow / 0.25 red per match globally, with **large league/ref variation**
([CIES study](https://football-observatory.com/IMG/sites/mr/mr57/en/)). API-Football supplies the
**ref assignment**; intensity proxies = knockout stage, rivalry, historical foul propensity.

**Proposed.** `μ_cards = base × ref_factor(career) × intensity(stage, rivalry, fouls)`, with the ref
factor from **hundreds** of matches (so light shrinkage is finally justified), and the fouls model
feeding intensity. Comparison via the §2 correlated bivariate.

---

## 5. Penalties / red cards — `pen_or_red`, `penalty_awarded`

**Now.** **Hard constants** — `runner.py:191` literally says *"constants (penalty/red) — not
calibrated."* No match-specific model at all.

**Critique.** **`pen_or_red` over-predicts +.15 (n=26)** — the constant is simply set too high
(global penalty ≈ 0.22–0.27/match, straight red ≈ 0.04–0.25/match depending on competition). Even
re-fitting the constant down is a free win; better, make it match-specific.

**Data.** Same referee sources as §4 (penalty & red rates per ref); CIES global card study for base
rates.

**Proposed.** `p_pen = base_pen × ref_pen_factor × intensity`; `p_red` similarly;
`P(pen ∨ red) = 1 − (1−p_pen)(1−p_red)`. Start by recalibrating the constant to the realized 0.19,
then layer the referee factor.

---

## 6. Fouls, corners, offsides — `fouls_gt`, `corners_gt`, `offsides_ge`

**Now.** Same count model from goal-λ. **Critique: these are already fine** — `fouls −.00 (n=49)`,
`corners +.03 (n=24)`, `offsides +.02 (n=53)`, all well-calibrated. *Do not over-engineer them.*
The only marginal upgrade is swapping the goal-λ proxy for **direct per-team for/against rates**
(API-Football `/fixtures/statistics` gives fouls, corners, offsides per team), and using the §2
correlated comparison for the `gt` variants. Low priority.

---

## The unifying verdict

Bot A's **scaffolding is right** (proper scoring, market anchor, hierarchical shrinkage, a genuinely
good dynamics model). The weakness is uniform: **every category is estimated from a generic proxy**
and only goal markets get a sharp anchor. The category-specific fixes:

| category | replace… | with… |
|---|---|---|
| goals | one Elo number | **xG attack/defense ratings** + market anchor on total & supremacy |
| SoT / counts | goal-λ as the driver; independent `A>B` | **direct per-team for/against rates**; **correlated** bivariate with ties |
| players | one global prior | **position priors** + **club-vs-country** split |
| cards | shrunk-to-neutral ref factor | **referee career rates** × intensity |
| pen/red | hard constants | **ref-specific** penalty/red rates |

The one assumption wrong on *principle* (not just by a noisy gap) cuts across §2/§4/§6: **`A > B`
questions use conditional independence, yet two teams' counts are game-state-coupled and ties are
common.** Worth fixing — but as a *structural* correction validated forward, not because `sot_gt`
"looked" +.14 (that gap is inside the noise band).

## Priority — re-ranked for the noise (robustness, not gap size)

Because the gaps are noise, "fix the biggest gap" is *itself* an overfit. Re-rank by **how little each
change leans on the unreliable per-archetype data:**

**Tier 1 — variance reduction via better data (do first; robust whether or not a bias exists):**
1. **Direct per-team count rates** (SoT / corner / foul for-&-against, API-Football) replacing the
   goal-λ proxy — lowers estimation variance everywhere, no gap-chasing.
2. **xG-based goal ratings** (StatsBomb / Understat) replacing the single Elo scalar — a lower-variance,
   richer λ that feeds every goal-derived and timing market.
3. **Player club-vs-country splits + xG** (API-Football `/players` per competition) — more relevant
   observations per player; shrinks the shrink-to-prior variance.

**Tier 2 — structural fixes (theory-motivated; ship ONLY if they beat live Bot A forward):**
4. **Correlated `A>B` with explicit ties** — independence is mis-specified on principle; empirical
   benefit unconfirmed. Validate out-of-sample first.
5. **Position priors / referee career rates** — sound in theory but add parameters; require heavy
   regularization + forward validation or they overfit.

**Tier 3 — do NOT do:** retune `pen_or_red` (or any constant) to the realized rate, or bolt on
per-archetype corrections, on the basis of these gaps. That is fitting n≈20–60 of noise and will
likely *hurt* out-of-sample.

Validate every Tier-1/2 item on **post-change settlements** vs live Bot A, on **aggregate** Brier over
many matches — recall the power math (detecting ~+1 RBP/q needs ~1,000 settled questions), so
per-archetype "it improved" claims are not trustworthy at this sample size.

## Data-source map

| need | source |
|---|---|
| per-team SoT / corners / fouls / cards / offsides per match | [API-Football `/fixtures/statistics`](https://www.api-football.com/) |
| per-player per-match SoT / shots / goals + position | API-Football `/fixtures/players`, `/fixtures/lineups` |
| player club-vs-country season/career rates | API-Football `/players/statistics` (per competition); [TheStatsAPI](https://www.thestatsapi.com/) |
| xG / shot quality (team & player), national-team prior | [StatsBomb open data](https://github.com/statsbomb/open-data); [Understat](https://understat.com) |
| referee career card / penalty / red rates | [FootyStats](https://footystats.org/stats/referee-stats), [PlayerStats](https://playerstats.football/referees), [OddAlerts](https://www.oddalerts.com/referees), [CIES](https://football-observatory.com/IMG/sites/mr/mr57/en/) |
| goal-market closing lines (keep — the anchor) | The Odds API (current) |
