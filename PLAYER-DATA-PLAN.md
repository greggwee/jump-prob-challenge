# Player-data & name-bias plan (evidence + approach)

## The evidence: the player model is data-starved (your suspicion, confirmed)

From the settled results + our data tables (`diagnose_player_data.py`,
`analyze_settled.py`, 2026-06-14):

- **44%** of players in our table have **< 180 min** of history.
- **Takefusa Kubo: 90 min, 0 SoT, 0 goals** in our *entire* data — so the model
  shrinks him to the population mean and predicted **7%** score-or-assist. He's a
  starting club attacker (true rate ~25–35%). He scored/assisted → **Brier 0.865**,
  our single worst call.
- **Cody Gakpo, Akram Afif: zero prior minutes** → pure population prior.
- Live `player_sot_ge` is mis-calibrated **−0.096** (we predict too LOW; players
  shoot more than the model thinks).

Root cause: the player model uses only **StatsBomb international-tournament**
history + shrinkage. New-tournament squads have almost no such history, so real
attackers collapse to a low population mean. We are systematically **underrating
attackers** — the opposite side of the trade we want to be on.

## Critical caveat: the #2 player edge was survivorship-biased

Improvement #2's backtest said player props were our strongest edge (t=18.7) and
turned on extremizing (k=1.3) there. But the backtest players *had* tournament
history (that's why they're in the data); live WC players often don't. So the
live player edge is not what the backtest promised, and **extremizing is now
amplifying a broken model** — it pushed Kubo's wrong-low 7% even lower.
**Immediate action: dial player extremizing back to k=1.0 until fair value is
fixed** (set the player gate off; one-line change).

## The fix: ingest club-season player rates

A club starter plays ~30–50 games/season — orders of magnitude more signal than
2–3 international tournament matches (Salah 2024-25: 38 games, 3,392 min vs our
Kubo: 90 min). **Data access verified 2026-06-14** (`discover_understat.py`):

| source | content | access (verified) |
|---|---|---|
| **Understat AJAX** | per-player club-season: games, time(min), goals, npg, xG, npxG, shots, assists, xA, key_passes (top-5 leagues, multi-season) | **WORKS with plain `requests`**: `POST /main/getPlayersStats/` form `{league, season}` → JSON `{players:[...]}` (195 KB, all players). The old embedded-`playersData` scrape is DEAD (site changed) — use the AJAX endpoint. |
| ~~**FBref** (Opta)~~ | SoT/90 directly, broad coverage | **DEAD (2026-06-15):** Cloudflare JS-challenge 403 to every HTTP client (plain `requests`, `curl_cffi` impersonating Chrome/Safari/Android) AND to `soccerdata`'s reader; `fbrapi.com` proxy also JS-walled. Needs a real headless browser — not worth it. **→ pivoted to API-Football.** |
| **API-Football** (api-sports.io) | **shots.on (SoT) directly** + broad leagues (Saudi/Qatar/Liga MX/MLS/Championship/Portugal/Eredivisie/Süper Lig) | **WORKS, keyed** (`API_FOOTBALL_KEY` in `.env`). Free tier 100/day + ~10/min. `/players/profiles?search=` (no league) → id, then `/players?id=&season=`. Per-hole-player fetch (~2 req each) keeps under quota. |
| **Transfermarkt** | market value = **"fame"** proxy | scrapeable (later, for the name-bias step) |

**Build (file-level):**
1. `ingest/understat.py` — `fetch_player_season(league, season)` via the AJAX
   endpoint; pull the top-5 leagues × last ~2 seasons; append-only raw with fetch
   timestamps. Aggregate to per-player **per-90 club rates**: shots/90, np_goals/90,
   xG/90, key_passes/90, minutes. **SoT/90 is derived** (Understat has no SoT
   field): SoT/90 = shots/90 × on-target-rate, where on-target-rate comes from the
   player's shot history (`/player/{id}` `shotsData`, result ∈ {Goal, SavedShot})
   or a position default. Entity resolution → `normalize/teams.py`-style canonical
   player names, with a completeness test (every WC squad name must resolve).
2. `models/player_model.py` upgrade — replace the tournament-only shrunk rate with
   a **hierarchical** estimate: `rate = club_rate` (high weight, ~30+ games)
   shrunk toward a **position × league-strength** mean, plus a small international
   adjustment from the existing StatsBomb history. League strength via a fixed
   multiplier table (EPL/LaLiga ≈ 1.0, weaker leagues < 1). Opponent scaling stays
   via the goal model's λ. Minutes projection unchanged (start/sub/cameo mix).
3. `scripts/backtest_player_clubfed.py` — re-backtest player props with the
   club-fed model, joining each *past-tournament* player to their *pre-tournament*
   club season (point-in-time; Understat covers 2014+). This avoids the
   survivorship bias that the old backtest had — and **only if the edge survives
   here do we re-enable player extremizing** (flip the guard in
   `analyze_prop_edge.py`).
4. Lineup-day: the announced-XI scaling hook already exists; club rates make it
   meaningful.

## The name-bias edge — real, but secondary to fixing fair value

The favourite-longshot / overweight-small-probabilities bias is well documented:
bettors overrate longshots and stars
([NBER w15923](https://www.nber.org/system/files/working_papers/w15923/w15923.pdf),
[Wikipedia](https://en.wikipedia.org/wiki/Favourite-longshot_bias),
[Boyd's Bets](https://www.boydsbets.com/favorite-longshot-bias/)). In our contest
the field is a naive crowd that likely overrates famous names on "will X score?"
props. **How quants exploit it:** build accurate fair value from rich club data
(multi-season per-90 rates, opponent + role adjustment, minutes projection,
shrinkage), then fade the gap between the crowd's name-inflated number and fair
value — measured via CLV against the prop line.

**Why we can't earn it yet:** the edge is the *gap between crowd perception and
fair value* — and our fair value is currently *broken low* (we're losing the
Kubo-type trade, not winning it). You cannot fade an overrated name with a model
that underrates everyone. **Order of operations: (1) fix fair value with club
data; (2) add a `fame` feature (market value); (3) test an explicit
"fade-the-name-premium" tilt, evidence-gated.** Step 1 alone flips us from the
wrong side to the right side of the trade and is the high-certainty win; step 3
is the speculative cherry on top (and unmeasurable without prop lines, which
Phase D showed don't exist — so it stays backtest-gated).

## Post-review revisions (statistician agent, docs/PLAYER-DATA-REVIEW.md)

The verification agent flagged four structural issues; the plan is revised to:

- **[HIGH] Coverage holes break the survivorship fix.** Understat covers ~top-5
  European leagues only, but WC squads are full of MLS / Saudi / Liga MX /
  Eredivisie / Asian / African players with *no* Understat row — they'd hit the
  prior, i.e. **Kubo's failure relocated**. ⇒ Primary source becomes **FBref via
  `soccerdata`** (far broader league coverage), Understat as the xG-quality
  supplement for top-5. The re-backtest must **stratify by data coverage and mask
  past players to match the live coverage-hole rate**, and player extremizing is
  gated on **per-prediction coverage**, not a pooled average. Coverage-hole
  players stay at k=1 (never extremized).
- **[HIGH] Don't assume the club→international adjustment** — fit it as a transfer
  regression (realized tournament per-90 ~ club per-90 + position/league +
  partial-pooled player residual), using the very join we're building.
- **[HIGH] Name-bias step is confounded** (market value ≈ ability, so fading it
  fades skill; the residual-fame estimand is unidentified while the field is
  unobservable). ⇒ Defer behind a real identification strategy; hard guardrail
  that raw market value never drives a production fade.
- **[MED] Estimator fixes:** SoT on-target ratio → Beta-Binomial position prior
  (or FBref's direct SoT/90); model `score_or_assist` as a **union of separate
  goal + assist survivals** using xG/90 and xA/90 (lower variance) — not one
  summed Poisson; **fit** league strength, don't hand-set it.
- **[MED] Minutes/availability uncertainty likely dominates rate uncertainty** —
  a start-probability (lineup) model may be a higher-priority next step than the
  name-bias tilt.

## Phased build plan (review-incorporated, iterative)

**Guiding rule** (from the review, METHODS §9): every adjustment is *estimated*
from the club→tournament join and kept only if it improves OOS Brier — no
hand-set transfer/league constants. Extremizing is gated on *per-prediction* data
coverage, never a pooled average. Name-bias is deferred behind identification.

The binding constraint is **data coverage**, so the plan is built to grow
coverage in rounds and to *never claim edge on a population it cannot see*.

### Phase 0 — DONE (ships independently, zero risk)
Player extremizing OFF (Priority-1, the survivorship guard in
`analyze_prop_edge.py`). Stops amplifying the broken model now.

### Phase 1 — Data + the coverage census (the gating measurement, H1/L3)
Goal: club-season per-player rates, clean entity resolution, and *the number that
governs the whole plan* — the live coverage-hole rate.
- `ingest/understat.py` — `POST /main/getPlayersStats/ {league, season}` for the
  top-5 + Eredivisie/Primeira/Championship/Russian, last ~2 seasons → per-player
  games, minutes, shots, xG, npxG, goals, npg, assists, xA, key_passes. Append-only.
- ~~`ingest/fbref.py` via `soccerdata`~~ → **`ingest/apifootball.py`** (FBref is
  Cloudflare-walled; see source table). Broad leagues + **direct SoT/90** via
  `shots.on` (M1: API-Football is now the *primary* SoT source; Understat the
  xG/volume backbone). Per-hole-player profile→id→stats; 429-backoff; cached.
- `normalize/players.py` — canonical player names; **fix the mojibake** (L3: the
  parquet shows `M�rio Figueira Fernandes`; decode encoding properly + fold
  diacritics) and a mis-resolution sanity gate (flag joins whose implied prob
  diverges from the position prior by > X for review).
- **Coverage census = the deliverable:** resolve every WC-2026 squad player against
  the club tables → report the fraction with NO join, broken out by
  confederation/league. This is H1's load-bearing number; it gates everything
  downstream and tells us which leagues to add next.
  - **RESULT (2026-06-15):** prop-population census = 43 players. Understat alone
    **56%** covered (44% hole). + API-Football broad leagues → **84%** (hole **16%**,
    6 residual). Residuals: diminutive mismatch (Ben Waine=Benjamin), accent/nickname
    search misses (Orkun Kökçü), no-data ids (Fajardo, Mohanad Ali, R. Mendes), and
    Al-Taamari (Ligue 1 → an Understat-side name-match fix, not a true gap).
    `scripts/coverage_census.py` (Understat) then `scripts/broad_rates.py`
    (+API-Football merge) → `player_rates_combined.parquet`.

### Phase 2 — Estimate the transfer + league strength (H2, M3 — no hand-set constants)
- Point-in-time join: past-tournament players (StatsBomb) ↔ their *pre-tournament*
  club season (2014+).
- Fit the hierarchical transfer regression per rate type (shots, SoT, xG, xA):
  `log(tourn_rate) = α + β·log(club_rate) + γ_position + δ·league_strength + u_player`.
  β<1 is the *estimated* shrinkage; α the level shift (likely negative for
  attackers); δ the *fitted* league strength; `u_player` partial-pooled.
- League strength enters twice (M3): the δ covariate (prior *level*) AND a modifier
  on the shrinkage pseudo-count M0 (weaker/less-covered leagues → smaller M0 → more
  shrinkage). Seed from a prior table the fit may move. Keep only if OOS Brier improves.

### Phase 3 — New player model (M1, M2, L1)
- Rate = club rate (xG/90, xA/90; SoT/90 from FBref) → fitted transfer offset
  (Phase 2) → Beta-Binomial-shrunk on-target ratio toward a position prior (M1).
- **xG/90 + xA/90 as the rate inputs** (M2: lower variance than realized goals).
- `score_or_assist` = union of *separate* goal & assist survivals,
  `1 − (1−P_score)(1−P_assist)`, each shrunk with its own prior + the right
  opponent scaling (M2) — not one summed Poisson.
- Recalibrate the opponent-λ reference so an average-opponent match leaves the club
  rate unchanged (L1: current `exp(b0)` is the international intercept, wrong
  population for a club rate).
- Coverage-hole players → position×league prior, **flagged**.

### Phase 4 — Coverage-stratified re-backtest + per-prediction gate (H1)
- Re-backtest player props **stratified by coverage** (real club-join vs
  prior-fallback) — report edge separately, never pooled.
- **Mask** backtest players to the live coverage-hole rate (Phase 1), matched on
  position; edge must survive on the *masked* set.
- Report **sensitivity to the assumed field sharpness λ** (sweep 0.5–0.85) — don't
  flip the dial on a guessed number (the field is unobservable; the comparison is a
  proxy, not a measurement).
- Re-enable extremizing **only** for real-coverage players, **only** if masked-set
  edge survives. The gate is a coverage flag on each prediction, not a population
  average.

### Phase 5 — Minutes / start-probability (M4 — measure first; may jump the queue)
- **First** decompose prop Brier error into rate-driven vs minutes-driven on the
  backtest (realized minutes available). Strong prior: minutes *dominates* → a
  start-probability model becomes the highest-EV step, re-ordered ahead of refinements.
- Player-specific minutes mixture from club start frequency (Understat games/minutes)
  + the existing lineup hook, replacing the frozen one-size mixture.

### Phase 6 — Name-bias (H3 — DEFERRED behind identification; hard guardrail)
- Build fair value *without* market value first (Phases 1–3).
- Estimand = field error attributable to **residual fame** (market value after
  partialling out p_fair) — currently **unidentified** (field unobservable; field
  proxy has no fame channel).
- Prereq: give the field proxy a fame channel, then backtest-gate. Test whether a
  fame term adds anything *beyond* the existing FLB / extremize-the-deviation
  machinery (much of "name premium" may just be favourite-longshot bias we already
  target). **Hard guardrail: raw market value must NEVER drive a production fade.**

### Iterative data acquisition (the through-line)
Grow coverage in rounds: start with the highest-coverage sources, run the census
(Phase 1), then each round add the leagues with the most uncovered WC players
(MLS, Saudi, Liga MX, J-League, African/Asian) via FBref/`soccerdata`. Re-run the
census + masked re-backtest each round; promote extremizing per-coverage only as
the hole shrinks. The model improves monotonically and never over-claims.

### Validation gates (what ships when)
1. Phase 1 census → know the real size of the problem before building on it.
2. Phases 2–3 → better fair value (the high-certainty Kubo fix), validated OOS.
3. Phase 4 → extremizing re-enabled *only* where masked-set edge survives.
4. Phase 5 → likely the real next win (minutes), per the variance-budget check.
5. Phase 6 → only behind identification; never with raw market value.
