# OPERATOR PUSHBACK REVIEW — bronze (FRA–ENG) & final (ESP–ARG)

*Independent quant sense-check, 2026-07-17. Ten operator pushback items on the open
endgame prices (quoted at the 07-16/17 devigged anchors: bronze λ (1.633, 1.137)
total 2.77; final λ (1.163, 0.889) total 2.05). Everything below was recomputed
from the local corpora — `settled_with_crowd.parquet` (1,134 settled questions),
`crowd_consensus_raw.csv`+`clv_ledger.csv` (653 crowd rows), `team_half_counts.parquet`
(314 matches, 6 tournaments, regulation only), `player_match.parquet`,
`martj42` (live through WC26) — plus web facts (ESPN per-match stats, Yahoo team
aggregates, FIFA/beIN officials announcements, FanDuel bronze odds). No platform,
no Odds-API. **Data correction to the task brief: `team_half_counts.parquet` has NO
WC26 rows (max date 2024-07-15). WC26 team counts below were reconstructed from
ESPN match pages + Yahoo aggregates + settled-question outcomes.** API-Football was
attempted for WC26 fixtures and is NOT usable (free plan, seasons 2022–24 only).*

Verdicts: **KEEP** / **ADJUST (to n)** / **EVIDENCE-THIN**. All lineup-dependent
items also have a branch table in [T-60 decision tables](#t-60-decision-tables).

---

## 1. BRONZE — "Each team 4+ SoT = 20 seems LOW" → **ADJUST to 24**

**Concern**: FRA and ENG are aggressive offences; 20 undersells the joint.

**WC26 actuals** (ESPN match pages; * = 120-minute stats):

| ENG SoT | CRO 9 | GHA 3 | PAN 6 | COD 7 | MEX 5 | NOR 8* | ARG 2 | ≥4: **5/7** |
|---|---|---|---|---|---|---|---|---|
| **FRA SoT** | SEN 8 | IRQ 5 | NOR 9 | SWE n/a (3 goals ⇒ ≥3, likely 6–7) | PAR 5 | MAR 8 | ESP 3 | ≥4: **5–6/7** |

Both teams cleared 4 in every match EXCEPT vs elite opposition (ENG 2 vs ARG,
3 vs Ghana's ultra-block; FRA 3 vs ESP). The bronze is elite-vs-elite — which is
exactly what the anchored marginals already encode — but it is also an OPEN
elite-vs-elite: FanDuel prices O2.5 at −250 and BTTS −290, Saliba is out of the
FRA defence, and both camps confirm heavy rotation.

**Statistical checks**:
- Model marginals reproduced at match-level NegBin (k=8.5): P(FRA≥4)=0.520,
  P(ENG≥4)=0.389 — the quoted 0.534/0.393 are mechanically sound.
- Raw joint/product in 314 matches: 0.178/0.216; residual copula −0.05 is right
  (don't double-count matchup heterogeneity) → joint ≈ 0.19–0.20 raw.
- 3rd-place empirics (n=4): team SoT 4,5 / 4,2 / 3,3 / 5,5 — mean 3.9, per-team
  ≥4 hit 5/8, joint 2/4. Bronze games do NOT depress SoT volume.
- Settled `team_sot_ge` at the 4-line: realized 2/7 (0.286) vs ours 0.389 — our
  LOW side has been right all tournament; this caps how far up we should go.
- The one genuine model bias: ENG's SoT μ=3.20 is λ-mapped, but England are a
  volume-over-conversion team this WC (98 shots in 7 = 14.0/match, 5.7 SoT/match);
  their μ likely runs 0.3–0.6 low for an open game against a rotated back line.

**Verdict: ADJUST 20 → 24.** That is the plan's own §5(d) shade (0.23) plus one
point for the bronze-openness/ENG-volume evidence. Do not exceed ~0.27: the joint
requires BOTH, our settled 4-line record ran 0.286, and both teams just posted
their tournament-low SoT against elite defences. **Mechanism**: in-handler shade in
the new `each_team_sot_ge` handler (raw 0.198 + documented SoT-under-call shade),
or expert override by market id. Never the `both_teams_sot_ge` template/risk-gate
route (threshold-blind 64.7 / 0.70 base — the review's found-bug stands).

---

## 2. BRONZE — "Kane 41 too high; his goals are penalties and pens are rare" → **KEEP 41** (pre-anchor, lineup-gated)

**The pens claim is factually wrong-ish**: martj42 pen flags show Kane's WC26 line
is 6 goals, **2 from the spot** (CRO 12', MEX 60') = 33% — identical to his career
international pen share (24 of 75 = 32%). This is his normal mix, not a WC26
distortion. He is 2/2 on pens this WC (the tournament's 6 misses belong to Messi
×2, Strand Larsen, Mbappé + 2 others). Non-pen: 4 npg in ~615′ ≈ 0.58 npg/90 —
ABOVE his tournament-history 0.61 g/90 *including* pens (player_match, 25 matches).

**Pen-haircut arithmetic** (steelman): WC26 awards 0.206 pens/match (~60% of the
0.347 historical). Kane's expected pen-goal leg ≈ ENG pen share (~0.5) × 0.206 ×
conv 0.75 × Kane-takes ≈ 0.077/match vs a historical-environment ~0.13 — the
haircut is worth **≈ −2pts** on P(scores), not the operator's implied collapse.
Share method sanity: Kane has 6/14 = 0.43 of ENG goals → P = 1−exp(−0.43×1.137) =
0.39 at the anchored λ.

**Our settled Kane record** (settled_with_crowd): three `player_scores` rows —
0.36/crowd 0.46 → YES (MEX), 0.45/0.54 → YES (COD), 0.37/0.44 → NO (ARG semi);
plus GA 0.52 → NO (GHA). We sat BELOW crowd every time; realized 2/3. There is no
"we're high on Kane" pattern to correct.

**What the book says**: FanDuel already quotes the bronze — **Kane anytime +115**
(0.465 raw, ≈ 0.41–0.43 after one-sided devig), Mbappé −135, Watkins' displacement
priced (Mateta +135, Toney +175). Our 41 IS the book number. The book is pricing
Kane despite the rotation reporting, i.e. it expects meaningful Kane minutes.

**The real risk is rotation, not penalties**: Tuchel — "none of our players …
want to play this match"; hitc/Yahoo report Kane "may drop out" for Watkins;
Sports Mole's XI still starts Kane; FanDuel's XI starts Watkins/Toney. Genuinely
unresolved → branch table below, resolved by the T-60 lineup gate + the book
anchor at lock (which will move if Kane is benched).

**Verdict: KEEP 41** as the pre-anchor number; the book prop at lock supersedes
it; `lineup_status.json` bench/out caps guard the desk-empty case.

---

## 3. BRONZE — "Improve Dembélé G+A calibration" → **ADJUST the rate constant (0.41 → 1.1 GA/90), keep the book-anchor plan; branch on the XI**

**The stale input is confirmed and quantified**:

| source | GA/90 |
|---|---|
| our rates table (flagged stale) | **0.41** |
| player_match history (pre-2024 tournaments) | 0.18 |
| PSG 2025-26 all comps (21G+12A, ~40 apps) | ≈ 0.95–1.05 |
| **WC26 itself: 5G+2A in 462′ (6 apps)** | **1.36** |

House blend (⅔ tournament form + ⅓ club, the same recipe as Messi/Yamal rate
overrides) → **≈ 1.2 GA/90**; shade for elite-final compression → recommend
**1.1 GA/90** as the `expert_overrides` rate. At the bronze tempo (scale
2.77/2.25 ≈ 1.23 on his λ share) that prices a STARTING Dembélé at ~0.45–0.50 GA
and a 30′ sub at ~0.30–0.34.

**But the role news cuts the other way**: Sports Mole, RotoWire and bulinews all
have Cherki starting AT Dembélé's expense; FanDuel prices him +170 anytime
(≈0.33 devigged) — i.e. the market already blends a bench-heavy role. Blended over
the realistic role mixture, fair G+A ≈ **0.36–0.40** — the current 31 is only
~5pts light, all of it in the "he actually starts" branch.

**Verdict**: the calibration fix is (i) replace the 0.41 club rate with 1.1
(mechanism: rate override in `expert_overrides.json`, same as Messi 2.4/Yamal 1.6
SoT overrides), (ii) keep the book G+A prop as the anchor at lock — the book, not
the crowd, is the target for player_ga (family realized 0.238 vs crowd 0.326;
crowd over-prices GA), and (iii) resolve start-vs-bench at T-60 (table below).
Unanchored default if props fail and no lineup: **0.37**.

---

## 4. BRONZE — "9+ corners at 52 is high for an even match" → **ADJUST to 45**

**Distributions** (team_half_counts, 314 matches, regulation):

| slice | mean corners | P(≥9) | P(≥10) |
|---|---|---|---|
| all 6 tournaments | 8.79 | 0.490 | 0.373 |
| knockout only (n=86) | 8.42 | 0.477 | 0.337 |
| the 6 finals | 8,6,7,9,12,8 | 2/6 | 1/6 |
| the 4 bronzes | 9,9,3,13 | 3/4 | 2/4 |

**WC26 is a low-corner tournament**: settled 9-lines realized 4/10 all-tournament
(ours avg 0.512, crowd 0.495); the recent/KO window ran **2/7** (crowd file: RSA-CAN
NO, MEX-ECU YES, CRO-GHA NO, AUS-EGY NO at 9; USA-BEL NO, **ENG-ARG NO** at 10;
POR-COD 2H-5+ YES); 10-lines 0/4, 8-lines 0/2. NegBin re-scaling (k≈28): the
all-tournament level implies match μ≈7.9 → P(≥9)=0.40; the KO window implies
μ≈7.0 → 0.29.

**Why not lower than 45**: the bronze is priced as the openest game of the
tournament (O2.5 −250; our λ-total 2.77) and bronze history (3/4 ≥9, tiny n) leans
high; corners scale weakly with λ (corners-goals corr 0.11) so the model's 52 is
mostly the high-tempo anchor talking. Splitting the difference between the
WC26-scaled 0.29–0.40 and the historical-KO-at-high-tempo ~0.48:

**Verdict: ADJUST 52 → 45** (defensible band 43–47; also satisfies the review's
"sit 2–3 under the model at 9/10 lines", and sits just under the 49.7 template —
the direction the whole tournament's corners evidence points). **Mechanism**:
expert override by market id (one-off; don't touch the corners constants for two
matches).

---

## 5. BRONZE — "20+ total shots at 68 seems high — what's team-specific?" → **KEEP 68**

The team-specific evidence says 68 is, if anything, LOW:

- **Combined-shots in ENG's WC26 matches**: 24 (CRO), 21 (GHA), 27 (PAN), ~14–21
  (COD; ESPN module ambiguous 9-vs-16 for ENG), 26 (MEX), 34* (NOR, 120′; regulation
  ≈28), **20 (ARG — hit the line exactly)** → ≥20 in 6 of 7.
- **FRA's matches**: 17 (SEN — the one clean miss), ~26 (IRQ), 28 (NOR), n/a (SWE),
  ~20 (PAR), 27 (MAR), 20 (ESP) → ≥20 in ~4–5 of 6 known.
- Aggregates cross-check (Yahoo): ENG 98 shots/7 + opponents 79/7 → 25.3 per
  ENG match; FRA 125/7 + opp ≈7.6 → ≈25.5. NegBin at μ=24–25 (k=45): P(≥20)=0.77.
- Settled `total_shots_ge`: 20-line **4/4 YES** (ours avg 0.693); 18-line 1/1;
  realized 0.70 across all 10 total-shots lines vs ours 0.63. High side has paid.
- Historical KO base P(≥20)=0.694; bronze matches 26,21,19,26 → 3/4; the market
  prices the game wide open (O2.5 −250).

**Verdict: KEEP 68** (central estimate from the evidence is 0.70–0.77; 70 is
also defensible; do not go below the mid-60s). No mechanism change.

---

## 6. BRONZE — "Sub scores/assists 51 — check the bench; anchor to crowd ~38?" → **KEEP ~0.50; do NOT crowd-anchor**

**(a) Mechanism**: the ×1.64 GA-widening critique is legitimate (subs skew
finisher, not creator; review §3.7 puts the honest ratio at 1.4–1.5). At widening
1.5 the bronze number is ~0.48 instead of ~0.51 — a 3pt trim, not a 13pt one.

**(b) The bench is the story, and it argues HIGH, not low.** Confirmed rotation
reporting (Sports Mole / RotoWire / hitc / FanDuel):

- FRA expected bench: **Dembélé (1.36 GA/90 this WC), Barcola, Coman, Thuram**
  (+ Rabiot/Tchouaméni), with Cherki/Doué/Olise starting.
- ENG expected bench: **Kane or Watkins/Toney (whichever doesn't start), Rashford,
  Palmer, Saka**, Foden-class depth.

That is the strongest plausible sub pool of the tournament, in the tournament's
openest match (λ 2.77, O2.5 −250), in the 5-sub era, in a dead rubber where both
managers will empty the bench early. The crowd's 38 is a 2-row template harvested
from ordinary matches with ordinary benches — it does not describe this fixture.

**Settled evidence is mixed, not damning**: `substitute_scores` realized 3/5 at
crowd 31.6 (crowd too LOW); `substitute_ga` 0/4 at crowd ~37.5 (tiny n, and two of
our four subs were 0.15 rows). Quantified: base P(sub scores)=0.30 λ-scaled to
2.77 ≈ 0.32–0.36 before the star-bench identity bump; ×1.5 → ~0.48–0.52.

**Verdict: KEEP at 0.48–0.51** (ship 0.50; the only change worth making is the
widening constant 1.64 → 1.5, which the review already recommends for the FINAL's
sibling question). Reject the crowd-38 anchor. Branch on confirmed bench strength
at T-60 (below): if rotation is LIGHTER than reported (stars start), shade to 0.45.

---

## 7. FINAL — "Penalty scored 17 too low? Pen rate? Who is the referee?" → **ADJUST 17 → 18**

**Referee (announced today, 07-17): Slavko Vinčić (Slovenia)** — first Slovenian
to referee a WC final; assistants Klančnik/Kovačič, 4th official Makhadmeh
(FIFA/beIN/Tribune). His profile:

| stat | value | source |
|---|---|---|
| career pens awarded | 88, ≈ **0.27/match** | whoscored/valuestats via search |
| WC26: 3 matches (BRA-MAR, JOR-ALG, MEX-ECU) | **0 pen goals** in those matches (martj42); 7Y+1R, 23.7 fouls/match | worldcuplocaltime + local |
| style | high foul count, low cards — lets contact go unpunished but no evidence of pen-shyness | derived |

**Recompute**: ref factor ≈ 0.27/0.25 ≈ 1.08 on the WC26 λ_pen 0.206, times the
final uplift 1.10 → λ ≈ 0.245; conv_eff 0.80 → **P = 1−exp(−0.196) = 0.18**.
Sensitivity: a genuinely pen-happy ref (0.35/m) gives 0.24; pen-shy (0.15/m) gives
0.13 — the referee is worth ±4–6pts, and Vinčić is worth about **+1pt** over the
plan's neutral 0.17. The 0.10–0.30 hard bound comfortably contains the whole ref
range — adequate, keep it.

**Crowd check**: the crowd priced `penalty_awarded` at 29.4 template and won't
haircut for conversion → expect 22–30 on "scored". The below-crowd stance stays
+EV: our own settled pen family shows the crowd systematically HIGH (pen_or_red
realized 0.226 vs crowd 0.386; awarded 2/11 vs priced ~0.29). Break-even truth vs
a crowd at 0.28 is ~0.22; three measurements put truth 0.15–0.18.

**Verdict: ADJUST 17 → 18** (mechanism: set the handler's ref factor / ship via
the loud expert override with the 0.10–0.30 assert per review §3.2). Oyarzabal
start = the conv 0.85 branch; if he's benched conv_eff → ~0.75 → P≈0.17 (T-60
table). Rebound/ET settlement wording still must be verified before lock.

---

## 8. FINAL — "Distinct shooters 86 very high — confidence?" → **ADJUST 86 → 85** (cosmetic); highest-confidence number on the slate

**Everything re-derived from scratch, and it holds**:

- Occupancy curve (player_match, 628 team-matches): measured P(≥5 shooters | N)
  = 0.375 (N=6), 0.711 (7), 0.857 (8), 0.892 (9), 0.953 (10) vs logistic 0.44 /
  0.64 / 0.81 / 0.89 / 0.96 — matches, with the known N=6 over-prediction.
- Spain residual ≈ 0 confirmed: 20/21 = 0.952 actual vs 0.954 curve-predicted.
  Volume is a sufficient statistic; "Spain has many shooters" = "Spain shoots a lot".
- Full mix at the anchored μ=12.2 with **match-level** NegBin dispersion (k=8.3,
  fatter than the runner's half-level fit): **P = 0.849**; at the runner's
  dispersion 0.876. μ=10 → 0.75, μ=15 → 0.92.
- **Failure case**: needs Spain at N≤6 (curve ≤0.44). Spain's WC26 shot counts:
  **27, 19, 11, 23, 15, 19, 10** — mean 17.7, minimum 10 (the SF vs France, their
  cagiest possible fixture). Historical minimum 6 (twice in 21; one produced the
  single <5-shooter match). Population KO teams shoot ≤6 at 14.5%, but that pools
  minnows; the anchored pmf already carries P(N≤6)=0.14 at μ=12.2 — conservative
  against Spain's actual floor.

**Verdict: ADJUST 86 → 85** — i.e. land on the plan's own shaded number; my
fatter-dispersion re-derivation (0.849) says the 0.876 end was the optimistic
half. Confidence in the estimator: highest on the slate. The REAL risk is
settlement: if the provider's "attempt a shot" excludes blocked shots, effective
N drops ~25% → ship **0.80**. Resolve the definition before lock (review MUST-FIX 9).

---

## 9. FINAL — "5+ cards pinned at 34 — could still be too high" → **ADJUST 34 → 31**

**The operator is right.** Clean threshold split of everything settled:

| line | n | realized | ours | crowd |
|---|---|---|---|---|
| 3+ total cards | 1 | 0.000 | 0.60 | 0.51 |
| 4+ (whole tournament) | 37 | 0.541 | 0.495 | 0.444 |
| **4+ (recent/KO window, crowd file)** | 23 | **0.348** | 0.497 | 0.440 |
| **5+ (all settled)** | 4 | **0.000** | 0.41 | 0.33 |

(The review's "realized 0.300, n=30" reconciles exactly as the pooled 3/4/5-line
recent window: 8/27.) Cards fell off a cliff mid-tournament: 4+ hit ~0.86 in the
early group games, 0.348 since. The regime that will referee the final is the
recent one.

**NegBin re-scaling** (match cards k=10.1): recent 4+ level 0.348 → μ=2.98 →
**P(≥5)=0.20**; final-elevation scenarios 4+=0.40/0.45 → 0.245/0.289. Finals
history (team_half_counts, regulation): 2, 4, 5, 5, 4, 2 → **2/6 at ≥5**, mean
3.67 ≈ the all-match 3.46 — finals are NOT reliably spicier in regulation.
Referee: Vinčić ran 2, 2, 4 cards in his three WC26 matches (2.7/match incl. the
red); his 4.13 Y/game club rate does not transfer (house backtest: tournament ref
factors are noise). Teams: ARG 9Y in 7 (1.3/match), ESP similar — neither is a
card magnet; FRA-ESP 4+ just settled NO at crowd 43.

**Verdict: ADJUST the existing override 34 → 31.** Evidence range 0.25–0.33:
0.31 keeps a point of finals-elevation respect while getting off the top of the
band. (Raw structural 0.279 and Bot B's 0.32 were both closer to truth than the
Platt+risk-gate 0.41 — the review's pipeline finding is re-confirmed.) The crowd
at the 5-line has priced 31–34 and lost every time; sitting AT 31, not above, is
also the crowd-respect play. **Mechanism**: edit the `7bb54c7a…` entry in
`expert_overrides.json` (value 34 → 31).

---

## 10. FINAL — "5+ offsides override 21→42 — justify the bump" → **KEEP 42**

**The bump is justified by direct WC26 measurement, not vibes**:

- Settled WC26 total-offsides lines: **4+ realized 4/6 (0.667)** (ours 0.375 —
  badly low), **3+ realized 3/5 (0.60)**, family-with-crowd 4/8 at crowd 51.4
  (crowd calibrated). No 5-line has settled this tournament — the pin is an
  extrapolation either way.
- Historical base (pass_offsides, 314 matches): mean 2.78/match, P(≥4)=0.31,
  **P(≥5)=0.15** — the raw model's 0.21 comes from this dead regime. The 2026
  semi-automated system flags every armpit; the regime shift is real and measured
  (WC26 4+ rate is ~2.1× the historical).
- **NegBin ML fit on all 11 WC26 settled outcomes** (k=9.4 shape held fixed):
  μ̂ = 4.17 → P(≥4)=0.56, P(≥3)=0.73 (both consistent with the observed 0.667/0.60)
  → **P(≥5) = 0.40**. Sensitivity: 4+ truth 0.55 → 0.39; 0.60 → 0.44; 0.667 → 0.51.
- Team check: high-line ESP and ARG are both on the offside-prone side of the
  settled team-level family (offsides_ge 2+ realized 0.441 tournament-wide; ESP
  and ARG rows settled YES).

The proposed 36–38 "compromise" would deliberately sit 4–6 under the maximum-
likelihood point of the only current-regime data we have, in a family measured as
a −2.6 RBP leak when we fight the level. 42 = the ML estimate, still ~6–9 under
where the crowd priced the 3+/4+ family (51.4) — appropriately humble on the
5-vs-4 threshold distinction.

**Verdict: KEEP 42** (band 38–45; nothing in the data supports going below 38).
Keep the `3c5f085c…` override as-is; add `offsides_total_ge` to the risk gate
post-final per the review.

---

## T-60 decision tables

*Official XIs may land at ~T-60 with nobody at the desk. Each branch below is
observable from the official lineup post alone and executable with zero context.
"Default" = what ships if NO confirmation arrives before lock. Mechanisms:
`expert_overrides.json` {market_id: {value, why}} (pins the shipped number);
`lineup_status.json` {player: {status: bench|out, team, ts}} (one-sided cap that
beats the book anchor); or "book anchor — no action" (props re-fetched at lock
impound the news).*

### Bronze locks 07-19 01:00Z

**#2 Kane scores (quoted 41)** — prior: start ~50/50 (Sports Mole XI starts him;
FanDuel's doesn't; book +115 implies big minutes either way)

| observed at lineup drop | action | value |
|---|---|---|
| Kane in starting XI | book anchor at lock; no override needed | anchor (≈0.40–0.44) |
| Kane on bench | `lineup_status.json`: Kane=bench (bench cap fires ≤ its player_scores cap) | ≈0.15–0.18 |
| Kane out of squad / injured | `lineup_status.json`: Kane=out | ≤0.05 |
| **Default (no XI seen)** | keep 41 + live book anchor (books track this news faster than we do) | 41 |

**#3 Dembélé G+A (quoted 31)** — prior: BENCH (3 outlets have Cherki starting)

| observed | action | value |
|---|---|---|
| Dembélé starts | expert override; rate constant 1.1 GA/90 | 0.43 (book G+A prop if quotable) |
| Dembélé on bench (expected) | no action — 31 already prices a sub role | **0.31 (keep)** |
| Dembélé out | `lineup_status.json`: out | ≤0.05 |
| **Default** | keep 31; book anchor if the prop posts | 31 |

**#6 Sub scores/assists (quoted 51)** — branch on BENCH strength, not one player

| observed | action | value |
|---|---|---|
| Heavy rotation confirmed (≥3 of {Dembélé, Barcola, Coman, Thuram} AND ≥2 of {Kane/Watkins-non-starter, Rashford, Palmer, Saka} among subs) | keep / nudge up | 0.51 |
| Reported rotation holds roughly (expected) | expert override | **0.50** |
| Rotation LIGHTER than reported (Mbappé+Dembélé+Kane+Saka all START) | expert override | 0.45 |
| **Default** | ship 0.50 | 0.50 |

**#1 Each team 4+ SoT (quoted 20 → 24)** — mostly lineup-insensitive (bronze
empirics: rotation shifts shooter identity, not team volume). Only branch: if
EITHER side starts a genuine B-team (≥8 changes) → revert to 0.21; else ship 24.
Default: **24**.

### Final locks 07-19 23:00Z

**#7 Penalty scored (quoted 17 → 18)** — taker branch

| observed | action | value |
|---|---|---|
| Oyarzabal starts (expected — he took+scored the SF pen) | expert override | **0.18** |
| Oyarzabal benched | conv_eff ESP → 0.75; λ-weighted conv ≈ 0.745 | 0.17 |
| **Default** | 0.18 | 0.18 |

**#8 context (Messi/Yamal race, separate question)** — per the plan: Messi out →
race collapses to P(Yamal≥1 SoT)≈0.7; Yamal benched → ≤0.10. Default: plan's
0.28–0.31. (Not one of the ten items; recorded here because it shares the T-60 run.)

**#9 cards 31 / #10 offsides 42 / #8 shooters 85** — lineup-insensitive; no
branches. Ship the override values regardless of XI.

---

## Summary table

| # | match | question | quoted | verdict | ship | mechanism |
|---|---|---|---|---|---|---|
| 1 | bronze | each team 4+ SoT | 20 | **ADJUST** | **24** | in-handler shade / expert override (never the both_teams template route) |
| 2 | bronze | Kane scores | 41 | **KEEP** | 41 → book anchor at lock | lineup_status bench/out caps at T-60 |
| 3 | bronze | Dembélé G+A | 31 | **ADJUST (input, not price)** | rate 0.41→1.1 GA/90; keep 31 if benched (expected), 0.43 if he starts | expert override rate + book anchor |
| 4 | bronze | 9+ corners | 52 | **ADJUST** | **45** | expert override |
| 5 | bronze | 20+ total shots | 68 | **KEEP** | 68 | none |
| 6 | bronze | sub scores/assists | 51 | **KEEP (≈)** | **0.50** | widening 1.64→1.5; T-60 bench check; do NOT crowd-anchor at 38 |
| 7 | final | penalty scored | 17 | **ADJUST** | **18** | ref factor (Vinčić 0.27 pens/m) via loud override; keep 0.10–0.30 assert |
| 8 | final | 5+ distinct ESP shooters | 86 | **ADJUST (cosmetic)** | **85** (0.80 if provider excludes blocked shots) | override; settle the definition pre-lock |
| 9 | final | 5+ total cards | 34 | **ADJUST** | **31** | edit existing override 7bb54c7a… |
| 10 | final | 5+ total offsides | 42 | **KEEP** | 42 | keep existing override 3c5f085c… |

**New facts surfaced during this review** (not in the plan/review docs):
1. **Final referee: Slavko Vinčić** (announced 07-17) — career 0.27 pens/match,
   0 pen goals + 8 cards total in his 3 WC26 matches; bronze referee: **Jesús
   Valenzuela** (Venezuela).
2. **Saliba is OUT of the bronze** (back injury, possible surgery) — rotated FRA
   back line supports the open-game/SoT/shots highs.
3. **FanDuel already posts bronze odds**: FRA −110/+280/+280, O2.5 −250, BTTS
   −290, Kane +115, Mbappé −135, Dembélé +170 — a live pre-lock anchor set.
4. **Bellingham faces possible disciplinary action** (incident during ARG
   celebrations) — a late scratch would move several ENG props; watch the T-60 run.
5. **Kane's pens are 2/6 of his WC26 goals** (career share 32%) — the operator's
   premise was wrong, and he's 2/2 this WC.
6. The cards regime split (4+ hit 0.86 early groups → 0.348 since) reconciles the
   review's pooled 0.300 with the whole-tournament 0.541 — threshold AND time
   slicing both matter for base-rate refits.
7. **team_half_counts.parquet has no WC26 rows** — the task brief believed it did;
   WC26 team counts must come from settled outcomes + web until someone ingests
   the tournament (post-final task: ingest WC26 into team_half_counts).
