# FINAL / 3RD-PLACE: six new question kinds — estimation recipes

*Ideation + reasoning pass, 2026-07-17. Two matches remain: FRA–ENG (3rd place, locks
07-19 01:00Z) and ESP–ARG (FINAL, locks 07-19 23:00Z, ~3× stage multiplier).
House method throughout: compose from existing primitives — the anchored goal grid
(λ_a, λ_b, `dc_grid`, `ctx.grid`), the NegBin count model (`cm.mu`, `cm.pmf`),
minute profiles (`self.dyn.alpha`), player thinning (`player_rate` × `MINUTE_SCENARIOS`),
and measured base rates. All worked numbers below use the 2026-07-16/17 opening lines
(devigged); **recompute at lock with the live `run_matchday_odds` anchor.**

**Shared anchored quantities** (computed offline via `Pipeline` — martj42 is live
through WC26 so Elo/goal model already include the tournament — then
`implied_lambdas` + `blend_lambdas` at w=0.85 on the devigged 1X2):

| match | devig 1X2 | λ blend | total | P(win a / draw / win b) | p00 | ET grid (win a / draw / win b) |
|---|---|---|---|---|---|---|
| **ESP–ARG** (DK +125/+200/+260, FD +130/+190/+270) | .421/.316/.263 | (1.163, 0.889) | 2.05 | .414/.314/.272 | .1355 | .115/.799/.086 |
| **FRA–ENG** (FD −110/+280/+280; O2.5 favoured) | .500/.250/.250 | (1.633, 1.137) | 2.77 | .483/.258/.259 | .0688 | .153/.744/.103 |

Count model at those λ: ESP SoT μ=3.88 / ARG 3.33; ESP shots μ=12.2 / ARG 10.8;
FRA SoT μ=3.94 / ENG 3.20. (Sources: [ESPN/FOX/Yahoo opening odds](https://www.foxsports.com/stories/soccer/spain-vs-argentina-odds-spain-favored-over-world-cup-final),
[SI/DK 3rd-place odds](https://www.si.com/betting/france-vs-england-opening-odds-for-world-cup-3rd-place-match-france-heavily-favored);
Kalshi: ESP 58% outright, FRA −178 to finish third.)

---

## 1. `9d4202c3` (final) — "Will a penalty kick be scored in regulation?"

### (a) Estimator

Poisson thinning of the penalty-awarded intensity by taker conversion:

```
P = 1 − exp(−λ_pen_awarded × conv_eff)
conv_eff = w_ESP·conv_ESP + w_ARG·conv_ARG ,  w_team = λ_team/(λ_a+λ_b)
```

The Poisson form handles the multiple-penalty case automatically (each awarded pen
is an independent conversion trial; "≥1 scored" is the complement of all failing —
`1 − exp(−λ·c)` = `1 − Σ_k Pois(k; λ)(1−c)^k`).

### (b) Inputs

| input | value | source | confidence |
|---|---|---|---|
| λ_pen_awarded, WC26 environment | **0.206/match** (21 pens in normal play / 102 matches) | [khelnow missed-pens list](https://khelnow.com/football/fifa-world-cup-all-missed-penalties-2026) + statbunker; cross-checks: martj42 WC26 = 14 reg pens *scored* → 14/0.75 ≈ 18.7 awarded; our settled WC26 `penalty_awarded` questions ran 2/11 = 0.18 | high — three independent local/web checks agree WC26 refs call ~60% of the historical rate |
| λ_pen historical (6 tournaments) | 0.347/match, P(≥1)=0.293 (= `P_PEN_MATCH`) | `team_half_counts.parquet`, n=314 | high, but stale for WC26 |
| final-match uplift | +0 to +10% (cagey final → more box scrutiny; finals sample 2/6 with a pen is too thin to fit) | wc2018+wc2022 finals had pens; euro2020/24, copa2024, afcon2023 finals none | low — treat as sensitivity band |
| WC26 field conversion | 15/21 = **0.714** | khelnow | high |
| historical WC conversion | 39/52 = **0.75** (pens_taken from StatsBomb `player_match` × pen goals from martj42, wc2018+wc2022 only — the euro/copa/afcon windows are polluted by concurrent friendlies) | local computation | med-high |
| **ESP taker** | Oyarzabal: career **53/59 = 0.898**, "yet to miss for Spain", took AND scored the SF pen vs FRA (22') | [beIN/Yahoo/Transfermarkt](https://sports.yahoo.com/articles/why-mikel-oyarzabal-not-lamine-194409518.html); martj42 confirms the SF pen | high |
| ESP conv_eff | **0.85** = P(Oyarzabal on pitch ~0.8)×0.898 + 0.2×0.75 backup (Olmo/Yamal untested this WC) | composition | med |
| **ARG taker** | Messi took 2 of ARG's 3 WC26 pens and **missed both** (wide vs AUT 5', saved vs EGY 21'); Lautaro took the other, scored (vs JOR 31'). Messi now holds the all-time WC missed-pens record (3) | khelnow, [beIN](https://www.beinsports.com/en-us/soccer/fifa-world-cup-2026/articles/messi-the-player-with-the-most-penalties-taken-and-missed-in-fifa-world-cup-history-2026-07-07), martj42 | high |
| ARG conv_eff | **0.74** = 0.65×conv_Messi(0.72: career ~0.78 Beta-updated with 0/2) + 0.35×conv_Lautaro(0.78) | composition | med — taker order after two Messi misses is genuinely uncertain |
| conv_eff (λ-weighted) | 0.55×0.85 + 0.45×0.74 = **0.80** | above | med |

### (c) Worked example

`P = 1 − exp(−0.206 × 0.80) = 0.152`; with the +10% final uplift `1 − exp(−0.23×0.80) = 0.168`.
**Recommend p ≈ 0.16.**

### (d) Shrink / clamp

Crowd priced *penalty awarded* at 0.25–0.42 all tournament (settled data) and will
likely put "scored" at 0.20–0.30. Our 0.16 is a moderate, well-sourced deviation —
exactly the §13.2 sweet-spot size. Do not go below 0.14 (one bad λ_pen estimate at
3× is expensive); land 0.16–0.18 after the P1 crowd clamp.

### (e) Settlement gotchas

- **Rebound**: saved pen + rebound converted is normally *not* "a penalty scored" —
  verify platform wording. If rebounds count, add ~+1pt.
- Regulation only: a pen scored in ET or the shootout settles **No** (final can go ET).
- Retaken penalties (encroachment) count on the final outcome of the retake.

---

## 2. `f2ad487e` (final) — "Will the first goal be credited with an assist? (No if 0-0)"

### (a) Estimator

```
P = P(≥1 goal in 90) × P(first goal assisted | ≥1 goal)
  = (1 − p00_anchored) × [ (1 − pen_share − og_share) × assisted_open ]
```

`p00` from the anchored DC grid (the 0-0 clause maps exactly to the [0,0] cell);
pen/OG/open composition of the *first* goal; only open-play goals can carry an assist.

### (b) Inputs

| input | value | source | confidence |
|---|---|---|---|
| 1 − p00 | **0.8645** (λ = 1.163, 0.889; τ00 = 1+λμ·0.053) | anchored grid | high (market-inherited) |
| first goal assisted, unconditional | **0.587** (n=288 first goals) | **StatsBomb 314 cached event files** (`data/raw/statsbomb/events`): pass `goal_assist` → shot id join | high — direct measurement |
| assisted share of open-play goals | **0.722** (n=668) | same | high |
| first-goal pen / OG share, 6 tournaments | 0.108 / 0.080 | same | high |
| first-goal pen / OG share, **WC26** | **0.041 / 0.041** (n=98 first goals) | martj42 WC26 | high — WC26 is a low-pen tournament |
| final-specific pen share | **0.08** = scored-pen intensity (0.206×0.80=0.165) / total-goal intensity (2.05) — pens don't scale down with a cagey final's tempo, so their share *rises* | composition | med |
| OG share (final) | 0.05 | 6-tournament 0.052, WC26 0.046 | med-high |
| assisted_open, team-specific | ESP **0.737** (n=38), ARG **0.692** (n=26); λ-weighted **0.717** | StatsBomb per-team | med (n small; note team spread is tiny — 0.69–0.74 across all four teams — identity flows mainly through the pen/OG mix). Messi's direct-FK/solo threat is already inside ARG's 0.692 |

### (c) Worked example

`P = 0.8645 × (1 − 0.08 − 0.05) × 0.717 = 0.8645 × 0.624 = 0.54`. **Recommend p ≈ 0.54.**
(Cross-check: unconditional 6-tournament first-goal-assisted = 0.587; our conditional
0.624 is higher because WC26's pen rate is half historical — coherent.)

### (d) Shrink / clamp

Crowd will sit near 0.5 (it reads like a coin flip). Our 0.54 is nearly crowd-neutral —
report honestly, no extra shrink needed. Confidence in the *composition* is high;
the number is dominated by P(≥1 goal), which is market-anchored.

### (e) Settlement gotchas

- **Own goal as first goal**: assumed it counts as "the first goal" and settles **No**
  (no assist ever credited on OGs). If the platform instead skips OGs ("first goal by
  a player"), the conditional rises ~+2pts → 0.56.
- Penalties: never credited with an assist (fouled player ≠ assist) — standard, but
  verify provider convention (some feeds credit the pass before a dribble-foul).
- "0-0" clause = regulation 0-0 (ET goals don't rescue a Yes).
- Assist credit source: platform's data provider, not FIFA official — deferred/deflected
  passes are provider-dependent judgment calls. This is irreducible settlement noise;
  it argues for staying ≤ 0.58.

---

## 3. `7def9dbe` (final) — "Yamal more SoT than Messi in regulation?" (tie → No)

### (a) Estimator

Discrete race via convolution of two independent count pmfs, each a Poisson mixture
over minutes scenarios (Jensen-correct, §6.3):

```
pmf_P(k) = Σ_s w_s · Pois(k; rate_P × scale × m_s/90)     scale = (λ_a+λ_b)/2exp(b0) = 2.05/2.25 = 0.91
P(Y > M) = Σ_k pmf_Y(k) · CDF_M(k−1)
```

### (b) Inputs

| input | value | source | confidence |
|---|---|---|---|
| Messi WC26 | **17 SoT / 531 min = 2.88/90** (most SoT in the tournament; 29 shots, 8 goals) | [squawka/threads stat cards](https://www.threads.com/@squawkafootball/post/Dah5SvIiqfe/), ESPN | med-high (fan-stat source; cross-check marj42 goals ✓ 8) |
| Yamal WC26 | **10 SoT / 495 min = 1.82/90** (7 matches, 6 starts, 1 goal, 3.6 xG — "quiet tournament") | [worldcuplocaltime](https://worldcuplocaltime.com/player/lamine-yamal/), Yahoo | med-high |
| Messi history | 1.65/90 (wc2018+wc2022+copa2024, 1,475 min, `player_match.parquet`); wc2022 alone 2.35/90; copa2024 (age 37) 0.85/90 | local | high |
| Yamal history | euro2024 1.24/90 (507 min); club 3.59 shots/90 (Understat, `player_rates_combined`) ≈ 1.4 SoT/90 at ~40% on-target | local | high |
| blended rates | Messi **2.4/90**, Yamal **1.6/90** (⅔ weight WC26 form, ⅓ history) | composition | med |
| minutes mixtures | Messi (0.80×90′, 0.15×75′, 0.05×60′) — 76 min/match WC26 avg but he plays long in big knockouts (120′ twice in the WC22 final run); Yamal (0.85×88′, 0.10×70′, 0.05×25′) | WC26 minutes + judgment | med |
| tempo scale | 0.91 (anchored final total 2.05 vs league avg 2.25) | runner's own `scale` formula | high |

### (c) Worked example

- Blend rates: **P(Yamal > Messi) = 0.245**, P(tie) = 0.199, P(Messi >) = 0.556.
- WC26-heavy rates: **0.225** / 0.175 / 0.600.

The tie→No asymmetry is doing real work: at these low means the tie mass is ~0.2,
so even "Yamal matches Messi" settles No.

### (d) Shrink / clamp

The crowd prices star-player SoT lines near 50 (settled WC26: Messi 3+ SoT crowd 0.51
— it settled YES while *we* said 0.27; Yamal 2+ SoT crowd 0.51, also YES). Two lessons:
(i) the crowd will likely put this race near 0.45–0.55, so our ~0.24 is a large
genuine deviation; (ii) our documented SoT *under-call* bias (§13.3, realized 40% vs
predicted 29% on player SoT) says don't take the raw model number at full confidence.
Recommend **p ≈ 0.28–0.31** (logit-blend ~0.65 model / 0.35 crowd-template 0.50).
This is plausibly the biggest single-question edge on the final slate — and a
player-family question, our highest-variance family. Cap the deviation at the P1
clamp; do not go below 0.25.

### (e) Settlement gotchas

- **Strictly more**: tie → No (confirmed reading; keep it).
- Regulation only — ET shots don't count.
- Penalty attempts on target count as SoT in most feeds (Messi took 2 pens this WC —
  the saved one vs EGY counts as SoT, the wide one doesn't). Verify provider.
- Woodwork ≠ SoT; blocked ≠ SoT.
- **Lineup risk**: if Messi were benched/injured (no sign of it) the race collapses to
  P(Yamal ≥ 1 SoT) ≈ 0.72 — watch the lineup at T-60′ and re-price. Yamal was subbed
  once mid-tournament; his 6-of-7 starts make 88′ the right modal scenario.

---

## 4. `4210db7c` (final) — "5 or more different Spain players attempt a shot?"

### (a) Estimator

Occupancy curve mixed over the team-shots pmf (identical structure to the house
`SOTCONC_BY_TEAM_SOT` / `BRACE_BY_TEAM_GOALS` curves — this is the same trick at a
new cell):

```
P = Σ_N pmf_shots,ESP(N) × P(≥5 distinct shooters | N shots)
P(≥5 | N) = σ(−5.333 + 0.847·N)    (logistic fit, 628 team-matches)
```

### (b) Inputs

| input | value | source | confidence |
|---|---|---|---|
| occupancy logistic | b0=−5.333, b1=0.847; P(≥5|N): N=6→0.44, N=8→0.81, N=10→0.96, N=12→0.99 | `player_match.parquet`, 628 team-matches, 6 tournaments | high — sharp, well-populated curve |
| team residual | **≈ zero**: Spain actual P(≥5)=0.952 vs curve-predicted 0.954 (n=21); Argentina 0.882 vs 0.922 | local | high — *shot volume is a sufficient statistic; "Spain has many shooters" is fully explained by "Spain takes many shots"* |
| Spain shots pmf | NegBin, μ = **12.2** at anchored final λ (Spain avg 17.1 shots/match historically, but the final anchor prices a much cagier game) | `cm.mu("shots", …)` | high |
| unconditional base | P(≥5 distinct) = 0.841 population; Spain historical 0.95 | local | context |

### (c) Worked example

Mixing the anchored pmf with the curve: **P = 0.876**. (Sensitivity: if Spain's real
shot volume runs at their tournament norm ~15, P→0.93; if ARG parks and Spain still
only manages 8 shots, P→0.81 — the pmf mixing already weights these.)

### (d) Shrink / clamp

The curve is the best-measured object in this whole slate; the residual risk is all
in the shots μ. Recommend **p ≈ 0.85** (a half-step toward the 0.84 population base).
Crowd likely 0.6–0.75 (it reads harder than it is) — this is a solid, defensible edge
on the Yes side, unusual for us (our bleeds were No-side SoT calls).

### (e) Settlement gotchas

- "Attempt a shot": house data counts **blocked shots as attempts** (StatsBomb) —
  verify the platform provider does too; if blocked shots are excluded, rebuild the
  curve on (shots − blocked) ≈ ×0.75 volume → P drops to ~0.80.
- Own-goal "attempts" don't count as shots.
- Regulation only (ET shots by a 6th shooter don't rescue).
- Distinctness is per-player across the whole 90′ — subs count, which helps Yes
  (Spain routinely brings on 3+ attackers).

---

## 5. `853c4ebf`-adjacent (3rd place) — "Each team 4+ shots on target?"

### (a) Estimator

Marginals from the count model at anchored λ; joint with a *small negative*
dependence correction (Gaussian-copula-lite on the indicators):

```
P = p_FRA·p_ENG + ρ_ind·sqrt(p_FRA(1−p_FRA)·p_ENG(1−p_ENG)),   ρ_ind ≈ −0.05
```

### (b) Inputs

| input | value | source | confidence |
|---|---|---|---|
| anchored λ | (1.633, 1.137), total 2.77 (O2.5 favoured everywhere; "England defense has yet to keep a knockout clean sheet") | devigged FD/DK 3rd-place lines | high |
| SoT μ | FRA 3.94, ENG 3.20 → **p_FRA(≥4)=0.534, p_ENG(≥4)=0.393** | `cm.mu`/`cm.pmf` | high |
| raw joint empirics | joint 0.178 vs marginal-product 0.215 (n=314); raw corr(SoT_a, SoT_b) = **−0.233** | `team_half_counts` | high |
| **residual dependence** | after conditioning on the model's μs (i.e. what our marginals already know), residual corr = **−0.076** → indicator ρ ≈ −0.05. Most of the raw negative correlation is matchup heterogeneity (mismatches produce one-sided SoT), which the anchored marginals already encode. **Do NOT apply the full 0.83 raw joint/product factor — that double-counts.** | local computation | med-high |
| 3rd-place empirics | team SoT lines: 2018 BEL–ENG 4–5 ✓joint, 2022 CRO–MAR 4–2 ✗, copa2024 5–5 ✓, afcon 3–3 ✗ → 2/4; mean team SoT 3.9 ≈ population 3.7 — *rotation does not depress SoT volume in bronze games; they're open* | `team_half_counts` stage field | med (n=4) |

### (c) Worked example

`P = 0.534×0.393 − 0.05×sqrt(0.249×0.239) = 0.210 − 0.012 = 0.198 ≈ 0.20`.

### (d) Shrink / clamp

Crowd template for `both_teams_sot_ge` ran mean 64.7 — at low thresholds; at the 4/4
line expect crowd ~0.40–0.55. Our SoT under-call bias (§13.3) hit exactly this family
(a −28 RBP both-teams-SoT bleed, though that one was a parser bug). Shade **up** to
**p ≈ 0.23** and let the P1 clamp bound the crowd gap. Note the existing
`both_teams_sot_ge` handler assumes independence — implementing the ρ_ind ≈ −0.05
term is a one-line improvement that also fixes the historical overprice.

### (e) Settlement gotchas

- "Each team" = AND (both must reach 4). Regulation only.
- Goals count as SoT; blocked and woodwork don't (standard, verify provider).
- Rotation shifts identity of shooters, not team volume (empirics above) — resist the
  temptation to haircut SoT for rotation.

---

## 6. Outrights (both matches) — "France wins the Bronze Final?" / "Argentina wins the World Cup?" (both incl. ET + pens)

### (a) Estimator

Anchor to the book outright where quotable; verify with the `advance` decomposition:

```
p_outright = p_win90 + p_draw90 × [ P(win ET | draw90) + P(ET drawn | draw90) × s_pen ]
```

ET leg from the existing ET grid (`ET_TEMPO=0.35`); s_pen = shootout win prob.

### (b) Inputs

| input | value | source | confidence |
|---|---|---|---|
| FRA–ENG anchored | win90 .483 / draw .258 / ENG .259; ET grid conditional: FRA .153 / drawn .744 / ENG .103 | devig + grid | high |
| ESP–ARG anchored | win90 ESP .414 / draw .314 / ARG .272; ET grid: ESP .115 / drawn .799 / ARG .086 | devig + grid | high |
| book outrights | ESP ≈ **0.58** / ARG ≈ **0.42** (Kalshi 58/42; DK −164, BetMGM −175 devig to the same); FRA ≈ **0.64** to finish third (Kalshi −178) | web (07-16/17) | high |
| market-implied s_pen for ARG | (0.42 − 0.263)/0.316 ≈ **0.50** — the market prices the post-90 leg as a coin flip *even with Martínez* | arithmetic on the two anchors | high |
| ARG shootout evidence | all-time 15/23 (0.652); **5 straight won** (NED'14, COL'21, NED'22, FRA'22, ECU'24); Martínez-era 4/4. ESP: 7/14 all-time, **2/6 since 2018** (lost to POR June '25) | `martj42/shootouts.csv` | high (the record), low (its predictive value — shootout-skill persistence is weak in the literature) |
| ET decisiveness | code's ET grid: ~25% of ETs decided pre-shootout; WC 2018+2022: 1/10; **WC26: 4/8** (BEL-SEN, ARG-CPV, ARG-SUI, NOR-ENG decided in ET). Pooled ≈ 28% — code value is fine, but WC26 runs hot (5-sub fitness era) | martj42 minute>90 reconstruction | med |
| finals go long | 5 of the last 10 WC finals (1990–2022) were drawn at 90′ — book draw 0.316 for this final is *below* the finals-history 0.50; do not shade the draw down | history | med (n=10, team-specific) |
| WC26 fatigue asymmetry | ARG has already played 2 ETs (CPV, SUI) + the SF; ESP zero ETs — favours ESP in an ET leg, offsets ARG's pen edge | martj42 | low-med |

### (c) Worked examples

- **France third place**: s_pen_FRA ≈ 0.50 (ENG won 3 of 4 shootouts since 2018 incl.
  two vs SUI; FRA lost 2 of 3, and Mbappé stutter-missed a pen *this tournament*):
  `p = 0.483 + 0.258×(0.153 + 0.744×0.50) = 0.62`. Book 0.64. **Submit the anchor: 0.63.**
- **Argentina World Cup**: with s_pen_ARG = 0.55 (evidence-tilted, bounded):
  `p = 0.272 + 0.314×(0.086 + 0.799×0.55) = 0.437`; with s_pen 0.50: 0.424. Book 0.42.
  **Submit 0.43** — the decomposition licenses at most +1–2pts over the anchor for
  Martínez; the ET-fatigue asymmetry argues against more. (Symmetric ESP question if
  it appears: 0.57.)

### (d) Shrink / clamp

These are anchored-family questions: the anchor IS the floor, never extremize. The
only deliberate deviation is the +1pt shootout tilt, capped at +2. If a
`pen_shootout` question appears for the final: `0.314 × [0.5–0.75 ET-drawn] ≈ 0.19–0.24`
— recommend 0.22 (WC26's hot ET decisiveness argues below the code's 0.25).

### (e) Settlement gotchas

- **VERIFIED: the WC26 third-place match HAS extra time** — FIFA's 2026 rules apply
  ET+pens to R32 through final *including the third-place match*
  ([Al Jazeera rules explainer](https://www.aljazeera.com/sports/2026/6/30/world-cup-2026-what-are-the-extra-time-and-penalty-shootout-rules),
  [Athlon](https://athlonsports.com/soccer/overtime-rules-extra-time-penalty-kicks)).
  No straight-to-pens special rule. Still re-verify the platform's own settlement note.
- "Win the World Cup" phrasing must parse to the *tie* winner (incl. ET+pens), not
  win-in-90 — new parser pattern needed (`outright` → `advance`-style handler); the
  existing `win` archetype would silently price win-in-90 (0.27 vs 0.42 — a 15pt
  parser trap, the single most dangerous mistake available on this slate).
- Shootout pens are not "goals" for any goals-based market; ET goals count for
  ET-inclusive markets only.

---

## Prioritized implementation order

1. **Q6 outrights** — biggest stage-weight × biggest parser-trap risk (win90 vs
   outright is 15pts). Add an `outright`/`wins_trophy` archetype routed through the
   `advance` handler with s_pen override table {ARG: 0.55, ESP: 0.45, FRA/ENG: 0.50}.
2. **Q5 both-teams SoT ≥4** — existing `both_teams_sot_ge` handler already fires;
   one-line copula term (ρ_ind −0.05) + verify threshold parse at 4. Cheapest fix,
   also corrects a known overprice in an existing archetype.
3. **Q1 penalty scored** — one new constant pair (λ_pen_WC26 = 0.21, conv table) +
   thinning formula next to `penalty_awarded`. Also update `P_PEN_MATCH` for the
   WC26 environment if we re-price `penalty_awarded` (0.29 → ~0.19) — flag: that's a
   separate decision, it changes an existing calibrated number.
4. **Q3 Yamal–Messi race** — new `player_sot_race` handler: two `player_rate` pulls
   (manual WC26 rate overrides via `expert_overrides.json`: Messi 2.4/90, Yamal 1.6/90),
   minutes mixtures, convolution. Needs the lineup check wired to the T-60 re-price.
5. **Q2 first-goal assisted** — new handler: `(1−grid[0,0]) × (1−pen_share−og_share) ×
   assisted_open`; constants from this doc (pen 0.08, og 0.05, assisted_open 0.717).
6. **Q4 distinct shooters** — new handler: occupancy logistic (−5.333, 0.847) mixed
   over `match_pmf("shots", side)`. Fit constants are ready; lowest urgency only
   because its number (0.85–0.88) is the most stable under anchor updates.

**At lock**: rerun all six with the live devigged anchor (these worked numbers used
07-16/17 openers); check lineups (Messi/Yamal start; Oyarzabal start for Q1's conv);
re-verify the five settlement items flagged above.

---

## Appendix: things found on the way

- **WC26 is a low-penalty, low-OG tournament**: 21 normal-play pens/102 matches
  (~60% of the 6-tournament rate), conversion 71.4%; first-goal pen share 4.1% vs
  10.8% historical. Every pen-family base rate should be re-centred before these
  two matches.
- **The both-teams-SoT independence assumption in the runner is wrong in the raw
  data (corr −0.23) but nearly right conditionally (−0.076)** — the correct fix is
  small, and applying the raw empirical joint/product ratio would over-correct.
- **Spain's "many shooters" is entirely volume**: distinct-shooter count given total
  shots shows zero Spain-specific residual (0.952 actual vs 0.954 predicted).
- **Messi is out-shooting Yamal ~2:1 on target this tournament** (17 vs 10 SoT,
  2.88 vs 1.82 per 90) — the "prodigy vs GOAT" framing will pull the crowd toward
  50/50 on Q3; the data says Yamal-strictly-more is a ~0.24 event.
- The market already prices the final's post-90 leg at 50/50; Martínez's shootout
  record (4/4 era, ARG 5 straight) is worth at most +1–2pts on the ARG outright, and
  ARG's two ETs already played this tournament partially offset it.
