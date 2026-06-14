# Field Model Improvement Plan (rec L1)

## 0. The distinction that reframes everything: field ≠ market

Your question — "isn't it pulling from prediction markets and betting sites?" —
exposes a conflation worth nailing down, because two different objects are in
play:

- **The market** (sharp book consensus) = our estimate of the **true
  probability**. This is the ANCHOR (Phase 7). We *do* pull betting sites for it.
- **The field** = the ~thousands of human contestants whose **average Brier** we
  are scored against. Relative points = `field_Brier − our_Brier`. This is a
  model of *people*, not of truth.

They coincide **only where the field anchors to odds**. So pulling more market
data improves our estimate of the *truth*; it does not, by itself, tell us what
the *contestants* will do — except on the markets where contestants copy the
odds.

**Current state, precisely:**
- Betting sites → feed the **anchor** (Phase 7 consensus). ✓
- Prediction markets (Kalshi/Polymarket) → **not wired** (the `consensus()`
  pseudo-book path is ready, but nothing fetches them). ✗
- **Neither feeds the field model.** The field proxy is base-rate regression
  (`models/strategy.py::field_proxy`). That is the gap.

## 1. Where market data can and cannot help the field model

| Market type | Does the field anchor to a line? | Market data useful for field model? | Edge? |
|---|---|---|---|
| Main (win/draw/totals) | Yes — contestants read odds | **Yes**, market ≈ field | ~0 (field-correlation paradox) |
| Book-priced props (corners/cards/player SoT on big matches) | Partially — if a line exists | Partially — *if we ingest them* | Some |
| No-line (half-specific, comparisons, obscure props, small matches) | No line exists | **No** — nothing to pull | **Where our edge lives** |

The uncomfortable conclusion: **more market data mostly improves the main
markets — exactly the ~zero-edge markets we deliberately don't extremize.** The
high-value field-model work is on the **no-line markets**, where extremizing
fires (rec #2) and where *no betting or prediction market exists to pull from*.
That part cannot be solved by ingesting more odds; it needs a structural model of
crowd behavior plus the live RBP signal.

## 2. Prediction markets as a FIELD proxy (the genuinely useful reframe)

There *is* a sharp use for prediction markets here, different from the anchor:

- Sharp **book** line ≈ **truth** (de-vigged, professional money).
- Prediction-**market** price ≈ a **crowd's belief**, complete with crowd biases
  (favourite-longshot, round-number clustering, recency).
- The SportsPredict field **is a crowd**, and the brief lists Polymarket-style
  markets as a resource contestants use.

So on the markets they cover, **Kalshi/Polymarket prices may proxy the field
better than the sharp line does.** Our edge then = `sharp_line − crowd_price`
(the crowd's measurable bias). Concretely: anchor to the sharp book (truth),
estimate the field from the prediction-market price (crowd), deviate by the gap.
This also gives a **prior on the magnitude of crowd bias** that we can carry over
to the no-line markets (if the crowd is X% favourite-longshot-biased on winners,
expect similar on related props).

## 3. The binding constraint (be honest about the ceiling)

The SportsPredict API exposes **no per-question field data** — only the
**aggregate total RBP** on the profile page. So the field model can be fit from
live data only at the scalar level (one number: aggregate field Brier per
refresh). We cannot directly observe field Brier per archetype. This caps how
finely the no-line field model can ever be calibrated from live results — it
stays substantially **prior-driven**, refined slowly by aggregate RBP and by
the prediction-market crowd-bias prior.

**Phase A (do first, costs nothing): re-audit the API/site for ANY field signal**
— a leaderboard endpoint, per-question RBP, a field-average field in `/results`,
the distribution of ranks. If *any* per-question field data exists, it dominates
every proxy below and we pivot to using it directly. Verify before building
proxies.

> **Phase A RESULT (2026-06-13, `scripts/audit_field_api.py`): NO field data via
> the API. Confirmed — proceed with proxies.** Full schemas of every object
> (events, lobbies, matches, markets, predictions, results) contain only our own
> data + match/question metadata — no crowd consensus, no prediction counts, no
> field-average, no awarded-points/RBP, no rank. Every probed endpoint
> (`/leaderboard`, `/standings`, `/rankings`, `/lobbies/{id}/members`, `/me`,
> `/profile`, `/stats`, single-resource `/markets/{id}`, `/predictions/{id}`)
> returned 500/none while the documented list endpoints returned 200 in the same
> pass — i.e. they do not exist. The MCP tool set has no leaderboard tool either.
> **The only field signal remains the aggregate total RBP on the web profile
> page** (manual; the dashboard already accepts it). A possible Phase A.5 —
> scrape the authenticated web profile for any finer breakdown — is more involved
> and deferred. Net: the no-line field model is necessarily prior-driven (§3
> ceiling holds), so the structural crowd model (Phase C) is the path.

## 4. Proposed layered field model

```
field_prob(market):
  Tier 1  line exists (book OR prediction market):
            field = crowd_price   (prediction-market mid if liquid,
                    else sharp line) regressed toward 0.5 by a FITTED factor r,
                    where r is calibrated from aggregate RBP.
            edge  = anchor(sharp) − field(crowd)
  Tier 2  no line:
            field = w_base·base_rate + w_50·0.5 + w_lean·(Elo/λ obvious lean)
            mixture weights per archetype family; priors from forecasting-
            behaviour knowledge; overall level calibrated by aggregate RBP and
            the Tier-1 crowd-bias estimate.
  Tier 3  calibration loop:
            each matchday, compare predicted aggregate field Brier to realized
            (from total RBP); nudge r and the mixture weights.
```

Replaces the flat base-rate regression with a behaviourally-grounded, two-tier
model whose level is pinned by the one real field measurement we have (RBP).

## 5. Data to add
- **Kalshi API** + **Polymarket Gamma API** → match markets → (a) consensus
  pseudo-books enriching the anchor, (b) crowd-price field proxy (Tier 1). Gate
  each contract on liquidity.
- **Bookmaker prop markets** (corners/cards/player props) via The Odds API
  *additional-markets* endpoint or scrape — **audit availability + credit cost
  first**; anchors and field-proxies the *subset* of no-line markets books price
  on big matches (note: book lines are usually full-match; many SportsPredict
  questions are half-specific or comparisons, so mapping is partial).
- **Aggregate-RBP tracker** — the dashboard already accepts a pasted total RBP;
  make it persist per matchday so Tier 3 has a time series.

## 6. Validation
- Backtest the "prediction-market price = field proxy" hypothesis where both
  sharp lines and PM history exist (limited; Kalshi/Polymarket 2022+).
- Forward-test: predicted vs realized **aggregate** field Brier per matchday.
- State the ceiling plainly: without per-question field data, the no-line field
  model is a prior, not a fit.

## 7. Phasing & cut line
- **Phase A** — API field-data audit (free, do first; could obviate everything below).
- **Phase B** — prediction-market ingestion → anchor enrichment + Tier-1 field proxy.
- **Phase C** — structural crowd model for no-line markets + RBP calibration
  (the high-value, hard part where extremizing edge actually lives).
- **Phase D** — bookmaker prop-market ingestion for the book-priced no-line subset.

If time-boxed: **A + C** are the core. B and D enrich the low-edge main markets
and a partial subset of props; do them only after C, or in parallel if cheap.

## 7b. Phase B RESULT (2026-06-13): prediction markets are *efficient*, not naive

Ingested Polymarket per-match WC prices (`ingest/prediction_markets.py`) and
measured the crowd bias vs the sharp-book consensus on 8 overlapping matches
($0.35M–$17.6M volume each; `scripts/phase_b_crowd_bias.py`):

| bias (crowd − sharp), vol-weighted | value |
|---|---|
| favourite | **+0.001** |
| underdog | **+0.002** |
| draw | **−0.003** |
| mean \|crowd − sharp\| on favourite | **0.003** |

**Polymarket ≈ the sharp line to 0.3%.** A deep prediction market is arbitraged
against the books — it is a *second truth estimate*, not a naive crowd. So:
- It is **not** a useful field proxy (the SportsPredict field is naive; Polymarket
  is sharp — opposite ends).
- Its one real use is as a **fallback anchor where book coverage is thin** (a $17M
  market beats two soft books). Kept available in `prediction_markets.py`; not
  wired into the main path (it moves the anchor by ~0.3% where books already exist).
- **Implication for Phase C:** confirmed — *no* market (sharp or prediction)
  represents the naive contestant field. The field model must be structural, and
  the only ground-truth calibration signal is your aggregate RBP.

## 7c. Phase C RESULT (2026-06-13): structural field model, live and gated

Replaced the flat base-rate-regression proxy with a **per-family structural
field model** (`models/strategy.py::field_model`): the field is our belief shrunk
toward 50 by a family sharpness λ — `field = inv_logit(λ·logit(belief))`.

The elegant payoff: composed with the extremizer, the submission becomes
`inv_logit([λ + k(1−λ)]·logit(belief))`, so **λ self-modulates the leverage** —
a clueless field (λ→0) yields full extremizing (coefficient → k), a sharp field
(λ→1) yields none. The single λ per family *is* our edge over the field.

Priors (from reasoning + the Phase B efficiency finding): goal 0.85 (field reads
odds), player 0.45, count 0.40, timing 0.35 (most obscure). Wired into both the
extremizer (`runner._extremize`) and the gate (`update_edge_gate`).
`calibrate_field_sharpness()` tunes a global level from aggregate RBP when
available — coarse by necessity (Phase A: no per-family field data), so the
priors are what's active. 69 tests. Under this tougher (more honest) field
estimate the live gate stays k=1 — no family has confirmed edge yet, which is
correct.

**Net of B+C:** the field model is no longer a crude proxy — it is a
family-aware structural model whose single parameter cleanly drives the
extremizer, grounded by the Phase B finding that no market represents the naive
field. The hard ceiling remains (only aggregate RBP calibrates it), but the
structure is now right and the strategy self-modulates.

## 8. The counterintuitive recommendation
Your instinct (pull more market data) is right but aimed at the wrong leverage
point: market data improves the **truth estimate and the low-edge main markets**.
The **field model that matters** is for the **no-line markets**, where no market
exists and the only signals are (i) a structural model of how casual forecasters
behave and (ii) the aggregate RBP. So: do the free API audit first; wire
prediction markets as a *crowd* proxy (not just another anchor); but invest the
real effort in the structural no-line crowd model. And keep it gated — a better
field model sharpens the extremizer, but the extremizer still only fires on
*confirmed* edge.
