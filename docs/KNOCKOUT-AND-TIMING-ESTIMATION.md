# Estimating knockout advancement & hydration-break timing

Two question types that had **no formal estimator** — "advance to the Round of 16" and the
hydration-break timing markets. Both are now derived from **market-available odds + a small,
data-grounded statistical layer**, rather than guessed. This is the analysis and the method.

---

## 1. Hydration-break timing windows

**Question forms.** "Will a goal be scored before the first hydration break?", "...after the second
break?", "≥2 corners before the first break?", "a card after the second break?", "an offside before
the first break?".

**Break timing (corrected).** Cooling breaks are at the **22' mark of each half** → match minute 22
(first) and 45+22 = **67** (second). So "before the first break" = minutes **[0,22)**, "after the
second break" = **[67,90)** (+ stoppage). (An earlier 30'/75' guess under-predicted the after-break
goal market — the real window is wider.)

**Method — analytic window integral (no simulation).** A family's minute-profile (`dynamics.alpha`,
6×15-min buckets) gives the *share* of that family's intensity falling in any minute window:

```
share_W = Σ_{slot∈W} exp(α_fam[bucket(slot)])  /  Σ_{all slots} (...)
λ_W     = μ_total_match(fam) · share_W
P(≥k in W) = 1 − Poisson_cdf(k−1, λ_W)         # goals: Poisson;  counts: via the count model's NegBin
```

`μ_total_match` is the match's expected family total (goals = λ_a+λ_b from the anchored grid; counts
from the count model). The window *shares* (state-neutral) are: goal 0.173 / 0.326, corner 0.285 /
0.227, card 0.187 / 0.347 (cards spike late), offside 0.328 / 0.212 — all matching football intuition.

**Validation (settled WC26, `scripts/backtest_recovered.py`).** Anchored skill vs base rate:
`goal_post_hb2` **+0.009** (base .57, mean_p .56 — calibrated), `goal_pre_hb1` **+0.005**. The wider,
correct [67,90) window resolved the under-prediction the [75,90) guess showed. *Card/corner/offside
windows aren't in that backtest — martj42 has only goals — so those remain forward-validation.*

---

## 2. Advance to the Round of 16  ( = win the Round-of-32 tie, incl. extra time + penalties )

In WC26's 48-team format the team is already in the R32 knockout, so **advance ⇔ win this single
tie**. A tie level after 90' goes to 30' extra time, then penalties.

**Decomposition (uses the market we already anchor).**

```
P(A advances) = P(A win in 90)  +  P(draw in 90) · q_A
        q_A   = P(A win in ET)  +  P(ET also drawn) · s_A      ( = P(A advances | drawn after 90) )
```

`P(win90)`, `P(draw90)` come straight from the de-vigged **90-minute 1X2 market** (the goal anchor).
The only new piece is `q_A`, the conditional-advance probability, built from two grounded sub-models:

**(a) Extra time** — a low-tempo 30-min grid off the *same* (anchored) intensities:
`λ_ET = (c/3)·λ_90`. `P(A win ET)`, `P(ET drawn)` from that Dixon-Coles grid.
- **ET tempo `c = 0.35`** — measured on martj42: 255 ET goals over **839** matches that reached ET =
  **0.30 goals/30min**, i.e. only **34 %** of pro-rated regulation scoring (ET is cagey). This also
  reproduces the empirical penalty rate (below) within model.

**(b) Penalty shootout** — `s_A = 0.5 + 0.20·(r_A − 0.5)`, `r_A = P(win90_A)/(P(win90_A)+P(win90_B))`:
- Over **678** historical shootouts the favorite (by Elo) wins **0.49** (even) → **0.57** (big favorite)
  — **near a coin flip** with a mild, strength-scaling tilt. `PEN_TILT = 0.20` reproduces that range.

**Why the favorite edge is modest:** **81 %** of matches that reach ET still go to penalties (only
~19 % are settled by an ET goal), and penalties wash out strength — so a drawn knockout is close to a
coin flip. The model encodes this: an even match returns **exactly 0.500**, a 58 %-in-90 favorite
(Spain v Germany) advances **0.73**, a heavy favorite (Argentina v Cape Verde) **0.94**.

**Coherence property (proven, asserted in tests):** `s_A + s_B = 1` and the ET grid is proper, so
`q_A + q_B = 1` and therefore **P(A advances) + P(B advances) = 1** — exactly one team advances, by
construction, for any input.

**Inputs & fallback.** Needs only the 90-min 1X2 (anchored when odds exist; pure model otherwise) plus
the two fixed, data-estimated constants. No separate "to-qualify" market is required — but if one is
ever sourced (it's an *outright*, not a per-match market in The Odds API), it would be a direct
cross-check / sharper anchor.

**Validation status.** The 90-min components are already validated (goal anchor); `q_A`'s parameters
are grounded on 678 shootouts + 839 ET matches. The *end-to-end* advance estimate needs settled R32
matches to score — the knockout has only just begun, so this is **forward-validation** (like the
player-prop anchor). Sanity, symmetry, monotonicity, and the sum-to-one property all hold.

**Uncertainty / caveats.**
- `c` and `PEN_TILT` are single point estimates; the advance prob is only second-order sensitive to
  them (q_A ≈ 0.5 ± a small favorite lean regardless), so this is low-risk.
- The shootout edge uses *current* Elo as the strength proxy (mild look-ahead in the historical fit);
  it only sets the small tilt, not the bulk.
- ET intensities reuse the 90-min attack strength; real ET is also more defensive/fatigued, partly
  absorbed into the low `c`.
