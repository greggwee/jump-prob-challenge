# Statistical review of PLAYER-DATA-PLAN.md

*Skeptical-statistician pressure-test of the player-data / club-rate / name-bias
plan, written before it is built. The job is to find holes, not to bless the
plan. Severity ordering: HIGH = will bias the production numbers or invalidate
the re-backtest if shipped as-is; MED = real error that materially degrades
estimates but is recoverable; LOW = refinement.*

---

## Executive summary

The plan's **core diagnosis is correct and the headline fix is right**: the
player model is data-starved because it uses only StatsBomb international history,
and a club starter's 30-50 games of per-90 data is the single highest-value
signal available. Ingesting Understat club rates will, on its own, fix the
Kubo-class blunders (predicting 7% for a real starting attacker). That part is
sound and worth building.

But the plan as written has **four structural statistical problems that will bite
in production or quietly invalidate the re-backtest**, plus several softer issues:

1. **The re-backtest does NOT remove the survivorship bias it claims to remove
   (HIGH).** It swaps StatsBomb-coverage survivorship for Understat-coverage
   survivorship. Live WC squads contain MLS / Saudi / Liga MX / Eredivisie /
   non-top-5 players who have *no* Understat row and fall back to the
   position×league prior — exactly Kubo's failure mode, just relocated. Unless
   the backtest forces the *same coverage holes* the live population has, "edge
   survives the re-backtest" is again measured on a non-representative
   subpopulation, and re-enabling extremizing off that result repeats the
   original mistake.

2. **The "club rate + small international adjustment" is an assumed constant where
   it should be an estimated, shrunk quantity (HIGH).** The club→international
   gap is real, heterogeneous, and signed (most players' output *drops* at
   international level: weaker chance-creation around them, tougher defences,
   fewer minutes of dominance). Asserting a "small" adjustment bakes in a guess.
   It must be *estimated from the join you are already building* (past club season
   vs realized tournament output) as a hierarchical offset, not hand-set.

3. **The SoT derivation `shots/90 × on-target-rate` is fine for the mean but the
   per-player on-target rate is estimated on a tiny, biased sample and will add
   noise, not signal (MED→HIGH).** On-target rate from a player's own shotsData
   is a ratio of two small counts; using it per-player re-introduces exactly the
   small-sample problem the club ingest is meant to solve. It must itself be
   shrunk hard toward a position/finishing-quality prior, or dropped in favour of
   a position-level on-target rate applied to the (well-estimated) shot volume.

4. **The name-bias / market-value step is a confounded trap as specified (HIGH —
   do not build it on the current plan).** Market value is overwhelmingly a proxy
   for *true ability*, not for *fame-beyond-ability*. "Fade the famous name" with
   raw market value will fade genuine skill and is anti-correlated with edge. The
   only defensible version regresses crowd/field error on the *residual* of market
   value after fair value is partialled out — and even then it is unidentified
   because the field is unobservable. Keep it backtest-gated and treat it as the
   speculative tail it is; do not let it touch production until identification is
   solved.

Plus: minutes/availability uncertainty almost certainly dominates rate
uncertainty for these props and the plan freezes it (MED); the `score_or_assist`
prop is modeled as a single summed Poisson rate, which is wrong on two counts
(MED); league-strength as a fixed scalar multiplier is the wrong functional form
and is the most likely source of miscalibration after ingest (MED); and the
opponent adjustment via the goal-model λ double-counts defensive quality once
club rates already embed opponent context (LOW-MED).

**Verdict (detail in final section): build Step 1 (club-rate ingest) — it is the
high-certainty win — but NOT as drawn.** Three things must change before it
ships: (a) the re-backtest must replicate live coverage holes or it proves
nothing; (b) the international adjustment and league strength must be *estimated*
from the join, not assumed; (c) the name-bias step must be deferred behind a real
identification strategy. Do not re-enable extremizing on the re-backtest until
(a) is fixed.

---

## HIGH severity

### H1. The re-backtest reproduces the survivorship bias under a new name

**The flaw.** The plan's headline justification for the re-backtest is that
joining "each past-tournament player to their pre-tournament club season
(point-in-time)… avoids the survivorship bias that the old backtest had." It does
not. The old bias was: *backtest players are in the data because they had
international history; live players don't.* The new bias is structurally
identical: **backtest players will have a club-season join because they played in
a top-5 league Understat covers; a large slice of live WC 2026 squads will not.**

WC squads are full of players from MLS, Saudi Pro League, Liga MX, Eredivisie,
Liga Portugal, J-League, MLS, domestic African/Asian leagues, and the Championship
— none in Understat's top-5-league scrape. Those players will hit the
position×league-strength prior, which is precisely the data-starved fallback that
produced the Kubo blunder. If the re-backtest is run on the subset of past
players who *have* an Understat join, it measures edge on the
rich-data subpopulation and is silent about the population the live system
actually faces. "Only if the edge survives here do we re-enable extremizing" then
re-enables a leverage dial on a model validated on the wrong sample — the exact
error the survivorship guard in `analyze_prop_edge.py` was added to stop.

**Why it matters.** This is the load-bearing claim of the whole plan. If it's
wrong, the re-backtest is decorative and re-enabling extremizing (k=1.3) is
unjustified — and extremizing amplifies error symmetrically (METHODS §12.4), so
on the prior-fallback players it will *re-amplify* the Kubo failure.

**Concrete fix.**
- **Stratify the re-backtest by data coverage** and report edge separately for
  (i) players with a real club-season join and (ii) players on the prior
  fallback. These are different estimators with different error; pooling them
  hides the failure.
- **Force the live coverage rate into the backtest.** Estimate the fraction of
  live WC-2026 squad players who will lack an Understat join (you have the squads;
  resolve them against the Understat name table now — the plan already demands a
  completeness test, so reuse it). Then *mask out* that same fraction of
  backtest players (ideally matched on position/role so the mask isn't random)
  and re-measure. Edge must survive on the *masked* set to justify the dial.
- **Gate extremizing on coverage at prediction time, not on a pooled backtest.**
  Players on the prior fallback should keep k=1 regardless; only players with N
  games of real club data earn the dial. This makes the guard a property of the
  individual prediction's data, not of a population average.
- **The field comparison is a proxy, not a measurement.** The re-backtest
  compares our Brier to `field_model(p, 0.70)` — a *structural prior* for the
  field, not an observed field (BETTING-REVIEW §4: the field is unobservable). So
  "we beat the field" is "we beat an assumed-sharp field." That is the honest
  current ceiling; state it. At minimum, report sensitivity of the gate decision
  to the assumed sharpness λ (sweep 0.5-0.85) so the dial isn't switched on by a
  number that was guessed.

### H2. "Club rate + small international adjustment" assumes the very thing it should estimate

**The flaw.** The plan sets `rate = club_rate shrunk toward position×league mean +
small international adjustment`. The word "small" is doing unsupported work. The
club→international transfer is not a small nuisance; it is a real, signed,
heterogeneous shift:

- **Teammate/chance-creation context changes.** A striker's club np-xG/90 is
  generated inside a system built to feed him. At international level the supply
  line is weaker and less rehearsed for most players (the few elite-nation
  attackers are the exception). Shot *volume* per 90 typically falls.
- **Opponent quality changes non-uniformly.** Group-stage minutes against weak
  sides can *raise* output; knockout minutes against elite defences lower it. The
  net is player- and draw-dependent, not a constant.
- **Minutes/role changes.** A club nailed-on starter may be a rotation option
  internationally (and vice versa) — but that's a minutes issue (H6/M-minutes),
  which must be separated from the *rate* shift or you double-count.
- **Selection/motivation.** Friendlies vs competitive tournament intensity differ.

A single "small additive adjustment" cannot represent a shift whose *sign and
size vary by player and context*. Worse, it is currently a guess, and METHODS §9
is emphatic that one hand-set parameter on thin data is the prior doing the
inference.

**Concrete fix — estimate the transfer, don't assert it.** You are already
building the exact dataset needed: past-tournament players joined to their
pre-tournament club season. That join lets you *regress realized tournament
per-90 on club per-90*:

```
log(tourn_rate_i + eps) = alpha + beta * log(club_rate_i + eps)
                          + gamma_position + delta * league_strength_i + u_i
```

- `beta < 1` is the regression-to-the-mean / transfer shrinkage you currently
  hand-set; `alpha` is the average level shift (likely negative for attackers).
- `gamma`, `delta` capture position and source-league effects.
- `u_i` is the residual player-specific transfer; partial-pool it
  (random effect) so players with more club history get more of their own signal.

This turns "small international adjustment" from an assumption into a *fitted,
backtested offset with a measured sign and magnitude and a confidence interval* —
and the fit itself tells you whether the adjustment really is small (then fine) or
whether attackers systematically lose, say, 25% of their shot rate (then the
naive "club rate" overrates them, the *opposite* of today's problem, and you'd
have over-corrected). Use the same six-tournament OOS discipline (METHODS §9):
keep the offset only if it improves OOS Brier.

### H3. The name-bias / market-value step is confounded by construction

**The flaw (this is the one the brief flagged, and it is the biggest trap).**
The plan proposes a `fame` feature from Transfermarkt market value and an explicit
"fade-the-name-premium" tilt. But **market value is, to first order, a measurement
of true ability** — it is set by clubs spending real money on expected production,
age curve, and resale. The correlation between market value and genuine
goal/assist rate is high and *legitimate*. "Fade the famous name" using raw market
value therefore fades real skill: you would systematically *underweight* Mbappé,
Haaland, Bellingham — the players most likely to actually score — which is
anti-edge. Naive name-fading is a near-perfect way to convert a fair-value model
back into a losing one.

The thing you actually want to fade is the **residual crowd error** — the part of
the field's number that is inflated by fame *beyond* what fair value justifies.
That is a fundamentally different and harder object than market value.

**Why it's not just hard but currently unidentified.**
1. You cannot separate "name premium" from "ability" without a fair-value model
   that is *independent of fame*. Market value enters both your fair value
   temptation and the crowd's bias — using it on both sides is circular.
2. The crowd's number is **unobservable** (BETTING-REVIEW §4 / METHODS §12.5). You
   are proxying the field with `base_rate + ρ·(belief − base_rate)`. There is no
   fame term in that proxy, so there is literally no estimand for "name premium"
   in the current field model — you'd be fading a quantity you cannot measure.

**What a defensible identification strategy looks like (if pursued at all).**
- Build fair value *without* market value (club rates + position + league +
  estimated transfer, per H2). Call it `p_fair`.
- The hypothesis is: field error `(field_p − p_fair)` is *increasing in residual
  fame*, where residual fame = market value after partialling out p_fair (i.e.,
  the part of market value NOT explained by your own ability estimate). Only the
  residual is a candidate "name premium."
- You cannot run this regression live (no field data). You *can* approximate it
  in backtest IF you have a field/crowd proxy that itself contains a fame channel
  — which today's proxy does not. So **step 1 of the name-bias work is to give the
  field proxy a fame channel** (e.g., crowd over-shrinks famous names toward
  "yes"), then test whether adding a fade improves OOS Brier against *that* proxy.
  That makes the result entirely a function of an assumed field model — so it is
  speculative by construction and must stay backtest-gated and out of production
  until a real field signal exists.
- Cross-check against the literature's actual mechanism: favourite-longshot bias
  is about *probability magnitude* (longshots overbet), not *fame per se*. Much of
  what looks like "name premium" on score props may just be FLB on low base-rate
  events — which your existing extremize-the-deviation machinery already targets
  without any market-value data. Test whether a fame feature adds anything *beyond*
  a base-rate/FLB term before believing it's a separate edge.

**Verdict on this step:** the plan's *ordering* is right (it correctly defers
this and calls it "the speculative cherry"). But the plan should state plainly
that **market value ≠ fame**, that raw name-fading is negative-EV, and that the
step is *unidentified* under the current unobservable-field setup. As written, a
future implementer could read "add a fame feature (market value)" and wire raw
market value into a fade. Add the guardrail explicitly.

---

## MED severity

### M1. The per-player on-target rate is a small-sample ratio that re-imports the scarcity problem

**The flaw.** `SoT/90 = shots/90 × on-target-rate`, with on-target-rate "from the
player's own shot history or a position default." Shot *volume* (shots/90) is
well-estimated from a club season (many shots). But the *on-target rate* is a
ratio `on_target_shots / total_shots`; for a midfielder or full-back with 15-25
shots in a season, that ratio has a standard error of ~0.10-0.12 — enormous
relative to the ~0.30-0.45 signal. Using the raw per-player ratio re-introduces
exactly the small-sample noise the club ingest exists to remove, now multiplied
into every SoT number. It will widen, not tighten, SoT calibration for low-volume
players — and SoT is already the mis-calibrated family (live −0.096).

There's also a subtler issue: Understat's `result ∈ {Goal, SavedShot}` as the
on-target definition. A `Goal` is on target by definition, so the on-target rate
is mechanically entangled with *finishing quality*, and blocked shots
(`BlockedShot`) are excluded from the denominator inconsistently with how Opta/
the contest may define "shot on target." Confirm the SoT definition matches the
contest's settlement source before trusting the derived rate at all.

**Concrete fix.**
- **Shrink the on-target rate hard toward a position-level (or position×league)
  prior** using a Beta-Binomial: `p_ot = (on_target + a0) / (shots + a0 + b0)`
  with `(a0,b0)` set from the position pool. Players with few shots get the
  position rate; high-volume shooters get their own. This is the same conjugate
  shrinkage already used for the goal/SoT rates (METHODS §6.2) — apply it here
  too rather than using a raw ratio.
- **Strongly consider skipping the per-player on-target rate entirely** and using
  a position-level on-target rate applied to the well-estimated shot volume. The
  per-player ratio's added variance very likely exceeds its added signal for all
  but the highest-volume shooters; verify with the OOS backtest (does per-player
  on-target beat position-default on held-out SoT Brier? If not, drop it).
- **Better still, prefer FBref's direct SoT/90** (the plan notes it's reachable
  via `soccerdata`). A directly measured SoT/90 avoids the derivation and its two
  error sources (volume × ratio) compounding. The plan treats FBref as a fallback;
  for SoT specifically it should be the *primary* source and Understat the
  volume/xG backbone. Cross-validate the two where they overlap.

### M2. The `score_or_assist` prop is modeled as a single summed Poisson rate — wrong on two counts

**The flaw.** Confirmed in `runner.py`: `player_ga` is built as
`ga = goals + assists` summed per player, one Poisson rate `rate_ga`, then
`P(≥1) = 1 − exp(−rate·M)`. Two problems:

1. **Goals and assists are different processes with different opponent/context
   scaling and different transfer behaviour.** Goals scale with the player's own
   λ; assists scale with *teammates'* finishing of the player's chances (key
   passes × team conversion). Lumping them forces one rate and one opponent scale
   on two mechanisms. A player can be a high-assist / low-goal creator (e.g. a
   deep playmaker) and the summed rate mis-allocates his probability.
2. **`P(≥1 of A or B)` from a summed Poisson assumes goals and assists are
   independent AND that double-counting is harmless.** They are negatively
   correlated within a match at the margin (a shot you score can't also be a pass
   you assisted on the same goal), and "scored OR assisted" is a *union*, not a
   count threshold. The correct object is
   `P(scored ≥1 OR assisted ≥1) = 1 − P(0 goals)·P(0 assists)` under
   conditional independence, i.e. `1 − exp(−r_g·M)·exp(−r_a·M) = 1 − exp(−(r_g+r_a)M)`
   — which *happens* to coincide with the summed-rate formula **only if you model
   goals and assists with separate rates and combine the survival functions.**
   The current code sums the *events* then shrinks, which is not the same as
   summing the *rates* after independent shrinkage (the priors and shrinkage
   strengths differ for the two), so the production number is subtly off. Once
   club data arrives with separate `np_goals/90`, `xA/90`, `key_passes/90`, model
   them separately and combine survivals.

**Concrete fix.** Keep separate goal and assist rates (the club data has both —
goals/npg and assists/xA/key_passes), shrink each with its own prior, apply the
goal-vs-assist-appropriate opponent scaling, then combine:
`P(score_or_assist) = 1 − (1 − P_score)(1 − P_assist)`. Use **xG/90 and xA/90 as
the rate estimators rather than realized goals/assists** — they are far less noisy
(this is the whole reason xG exists) and Understat hands them to you. Realized
goals are a high-variance estimator of scoring rate; xG is the lower-variance one.
The plan ingests xG/xA but the build step still describes deriving rates from
shots and goals — make xG/xA the primary rate inputs.

### M3. League strength as a fixed scalar multiplier is the wrong functional form

**The flaw.** "League strength via a fixed multiplier table (EPL/LaLiga ≈ 1.0,
weaker leagues < 1)." Three issues:

1. **It's hand-set**, against the project's own METHODS §9 doctrine (a hand-set
   constant on thin data is the prior doing the inference). League strength is
   estimable from the data you'll have: players who move between leagues, and the
   club→tournament transfer regression (H2) already contains a `league_strength`
   term you can *fit*.
2. **A single multiplier conflates two different adjustments**: (a) raw output is
   easier in a weak league (inflates everyone's rate — a *level* effect on the
   prior mean), and (b) a weak-league rate is *less informative* about
   international output (should change the *shrinkage weight*, not just the mean).
   A scalar on the rate only does (a). A Liga MX hat-trick hero and an EPL
   20-goal striker should not just have different means; the Liga MX player's club
   rate should be *trusted less* (shrunk harder toward the prior) because the
   league→international transfer is noisier. The plan's structure can't express
   that.
3. **Multiplicative on the rate** double-applies once the opponent λ scaling
   (runner.py line 383) is also in play — see L1.

**Concrete fix.** Fold league strength into the hierarchical model two ways: as a
covariate on the *prior mean* (level) AND as a modifier on the *pseudo-count* M0
(weaker/less-covered leagues → smaller effective M0 → more shrinkage). Fit the
strength coefficients from the transfer regression (H2) on the six tournaments;
don't hand-set them. If you must seed a table, treat it as a prior and let the fit
move it (the squad-feature episode in METHODS §9 is the cautionary tale).

### M4. Minutes/availability uncertainty likely dominates rate uncertainty — and the plan freezes it

**The flaw.** The plan keeps "minutes projection unchanged (start/sub/cameo mix)"
and treats the club-rate upgrade as the win. But for these props the binary
outcome is `P(≥1 event)`, and the single biggest driver of that for many players
is **whether they start and how long they play**, not the per-90 rate. METHODS
§6.3's own worked example shows a 5-point swing from the minutes mixture alone,
and that's with a *fixed, known* mixture. In reality the start probability is the
high-variance unknown: a rotation forward who starts is ~30-40% to get a SoT; the
same player on the bench is ~5%. That 25-30 point spread swamps the difference
between a club-fed rate and a prior-fed rate.

So the plan risks **precisely estimating the rate while leaving the dominant term
(minutes) as a fixed guess.** After the rate fix, the residual error in these
props will be minutes-driven, and a fixed start/sub/cameo mixture applied to every
player is clearly mis-specified (a nailed-on captain and a squad rotation option
get the same mixture).

**Concrete fix.**
- **Quantify the variance budget first.** Before/after the rate upgrade, decompose
  prop Brier error into rate-driven vs minutes-driven components on the backtest
  (you have realized minutes). If minutes dominates — likely — then the highest-EV
  next step after ingest is a *start-probability model*, not the name-bias step.
  This re-orders the roadmap.
- **Make the minutes mixture player-specific**, driven by club start rate
  (Understat gives games and minutes → minutes-per-appearance and start
  frequency) and the lineup hook the plan mentions. A fixed mixture is the weak
  link the plan under-rates.
- The Jensen correction (METHODS §6.3) is correctly applied in `runner.py`
  (`sum(w·(1−exp(−rate·m)))` over scenarios) — good. But it's only as good as the
  scenario weights, which are the thing being frozen. Jensen-correcting a wrong
  mixture is precise about the wrong number.

---

## LOW / LOW-MED severity

### L1. Opponent adjustment via the goal-model λ may double-count once club rates embed opponent context

`runner.py` scales the player rate by `(λ_a + λ_b) / (2·exp(b0))` — match
intensity vs league-average intensity. With the *old* tournament-only rates this
was a clean opponent adjustment. But a **club season per-90 rate already averages
over a season of opponents** of varying quality. Multiplying it by a fresh
opponent-intensity scalar is defensible *as a relative tilt* (this match vs the
club-season-average opponent), but only if the baseline `exp(b0)` is the right
reference for a *club* rate — it's the international goal model's intercept, a
different population. After ingest, recalibrate the scaling reference so that an
average-opponent match leaves the club rate unchanged; otherwise every player
gets a constant multiplicative bias. Low severity (it's a level shift the
calibration layer can partly absorb) but worth a check.

### L2. Poisson/independence assumptions for SoT are probably fine; for the union prop, less so

The thinned-Poisson machinery (METHODS §6.1) is sound for "≥1 SoT." The
independence assumption across a player's own shots is reasonable. The place it
strains is the union prop (M2) and any same-match correlation between a player's
goals and the team total (a player who scores is in a high-λ game). Minor for
binary ≥1 props; note it but don't over-engineer.

### L3. Entity resolution is a silent single point of failure

The plan has a "completeness test (every WC squad name must resolve)." Good —
but resolution *failure* and *mis-resolution* are different: a wrong join (two
players with similar names, or a transferred player matched to the wrong
club-season) silently feeds a real-looking rate that is someone else's. Add a
sanity gate: flag joins where the club-rate-implied prop probability diverges
from the position prior by more than X, for manual review. Diacritics are already
a known issue (the parquet shows `M�rio Figueira Fernandes` mojibake) — that
*will* break naive name matching for accented names, which is most of the squad.

### L4. xG-based rates need their own calibration check

Switching to xG/xA as rate inputs (recommended in M2) is lower-variance but
introduces model risk: Understat's xG model is not the contest's settlement
mechanism, and xG slightly *under*-predicts elite finishers and *over*-predicts
poor ones. After ingest, check whether realized-rate vs xG-rate differ
systematically by player tier; a small finishing-skill adjustment on top of xG
may be warranted for the genuine elite (the opposite end of the spectrum from the
name-bias fade). Verify, don't assume.

---

## Verdict: is this plan sound to build as-is?

**Build Step 1 (Understat club-rate ingest). Do not build it as drawn. Do not
build Step 3 (name-bias) yet at all.**

The diagnosis is right and the club-rate ingest is a genuine, high-certainty
improvement that will fix the documented Kubo-class failures. But three changes
are **prerequisites**, not nice-to-haves:

1. **Fix the re-backtest before trusting it (H1).** Stratify by data coverage and
   mask backtest players to match the live coverage-hole rate. As written, the
   re-backtest measures edge on the rich-data subpopulation and is blind to the
   prior-fallback players who are the live risk. **Do not re-enable player
   extremizing (k=1.3) on the unstratified re-backtest** — that would repeat the
   survivorship mistake the guard in `analyze_prop_edge.py` exists to prevent.
   Gate the dial on per-prediction data coverage, not a pooled average.

2. **Estimate, don't assume, the two key constants (H2, M3).** The
   "small international adjustment" and the league-strength multipliers must be
   *fitted* from the club→tournament join you are already building (a transfer
   regression with position and league terms and a partial-pooled player
   residual), under the project's existing OOS-Brier keep/reject rule. Hand-set
   constants on thin data violate the codebase's own stated methodology
   (METHODS §9) and there is no excuse here because the data to fit them is the
   same join the plan creates.

3. **Defer the name-bias step behind a real identification strategy (H3).**
   Market value ≠ fame; raw name-fading is negative-EV because it fades true
   ability. The estimand ("crowd error attributable to residual fame") is
   currently *unidentified* because the field is unobservable and the field proxy
   has no fame channel. Keep it strictly backtest-gated and add an explicit
   guardrail in the plan that raw market value must never drive a production fade.

**Also strongly recommended before claiming the upgrade is "done":**
- Use **xG/90 and xA/90 as the primary rate estimators** (M2), and model
  `score_or_assist` as a union of separate goal and assist survivals, not a summed
  rate (M2).
- Use **FBref's direct SoT/90** as the primary SoT source; if deriving, shrink the
  on-target ratio with a Beta-Binomial position prior rather than using the raw
  ratio (M1).
- **Quantify the minutes-vs-rate variance budget (M4)** — there is a strong prior
  that lineup uncertainty, not rate quality, is the dominant residual error for
  these props, which would make a start-probability model the correct *next*
  priority over the name-bias step, re-ordering the roadmap.

The "turn off player extremizing now" action (Priority 1) is correct and should
ship immediately and independently of everything above — it stops amplifying the
known-broken model at zero risk. That part of the plan needs no changes.
