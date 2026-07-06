# EXPERT SENSE-CHECK — Portugal vs Spain, WC26 Round of 16 (2026-07-06)

Reviewer: veteran football analyst + quantitative forecaster (LLM expert pass).
Scope: the 15 submitted probabilities. Verdict per question: SANE / QUESTIONABLE / LIKELY WRONG,
with an evidence class (HARD FACT / HISTORICAL BASE RATE / SOFT NARRATIVE).
Written incrementally — sections appended as evidence lands. Do not treat as final until the
"RANKED ACTION LIST" section exists at the bottom.

## Evidence log

### A. Repo extractions (martj42, live through WC26 — reproduced this session)

**Portugal WC26 path** (goals w/ minutes):
- G1: POR 1-1 DR Congo (Neves 6'; Wissa 45')
- G2: POR 5-0 Uzbekistan (Ronaldo 6', 39'; N.Mendes 17'; OG 60'; Leão 87')
- G3: COL 0-0 POR
- R32: POR 2-1 Croatia (Perišić 53'; Ronaldo 68' PEN; G.Ramos 90')
- Totals: 8 GF / 2 GA in 4; two 0-0-ish grinds (DRC, COL) vs two blowout/comeback games.

**Spain WC26 path**:
- G1: ESP 0-0 Cape Verde
- G2: ESP 4-0 Saudi Arabia (Yamal 10'; Oyarzabal 21', 24'; OG 49')
- G3: URU 0-1 ESP (Baena 42')
- R32: ESP 3-0 Austria (Oyarzabal 36'; Porro 66'; Oyarzabal 89')
- Totals: 8 GF / 0 GA in 4. Oyarzabal 4 goals, Yamal 1.

**POR-ESP head-to-head (martj42, last 6)**: 2-2 (2025 NL final, POR on pens), 0-1 (2022 NL), 1-1 (2022 NL), 0-0 (2021 F), 0-0 (2020 F), 3-3 (WC2018). Extremely draw-heavy fixture: 15 draws in ~40 meetings; 4 of last 6 level at 90.

**Ronaldo recent scoring**: 3 goals this WC (2 open play vs UZB, 1 pen vs CRO). Played all 270 group minutes + started R32, subbed 81' (first sub-off of his tournament). Scored vs Spain in the 2025 NL final (61').

### B. StatsBomb corpus computations (314 matches / 600 team-matches, run this session)

**P(any player on a team records 2+ SoT), regulation, SoT = Goal/Saved/Saved-to-Post:**
- Overall: **.485**
- Conditional on team total SoT k: k=1: .000 | k=2: .144 | k=3: .355 | k=4: .582 | k=5: .703 | k=6: .905 | k=7+: 1.000
- NB(k=6.3) mixture over mu_sot(POR): mu=3.0 -> .441, mu=3.5 -> .488, mu=4.0 -> .539, mu=4.5 -> .588, mu=5.0 -> .634

**Cards (regulation, Y+2Y+R)**: corpus mean 3.46/match, P(>=4)=.475, P(>=5)=.242.
Era split: WC2022 mean 3.41 P(4+)=.422 | Euro2024+ mean 3.86 P(4+)=.570 | Euro2020 2.84/.373 | WC2018-era 3.17/.406.

**A substitute scores (regulation, non-OG)**: corpus .271; 5-sub era only: WC2022 **.312**, Euro2024+ **.326**, Euro2020 .196 (pooled 5-sub ~.30).

### C. Web facts (searched 2026-07-05/06, tournament live)

- **Referee: Anthony Taylor (ENG)** confirmed for POR-ESP (3rd match of his tournament after UZB-COL and SEN-IRQ); AR Beswick/Nunn, 4th official Zwayer. [ESPN, Get Spanish Football News, Yahoo]
- **Diogo Costa saves**: **6 saves vs Colombia** (0-0, MOTM), **5 saves vs Croatia** (2-1). Two straight matches clearing the 4+ line. [OneFootball, FOX]
- **Spain SoT**: 7 SoT vs Cape Verde (0-0 — keeper made ~7 saves), **10 SoT (23 shots) vs Austria**; 2nd-most shots attempted in the tournament (49 through 4 matches). Austria managed ZERO SoT vs Spain. [ESPN, Sports Mole, fan stats]
- **Ronaldo**: starter, all 270 group minutes, R32 start + pen goal, subbed 81'. First player to score in 6 different World Cups. On penalties. [ESPN, FOX, beIN]
- **Team news**: Spain without Yeremy Pino (shoulder) and **Nico Williams (adductor) expected to miss R16**; predicted ESP XI: Simón; Porro, Cubarsí, Laporte, Cucurella; Rodri, Pedri, Olmo; Yamal, Oyarzabal, Baena. POR predicted XI: D.Costa; Cancelo, Dias, Veiga, N.Mendes; Neves, Vitinha; Neto, Bruno, Leão; Ronaldo. No fresh POR injuries. [Sports Mole, Yahoo]
- POR won the 2025 Nations League final vs Spain on penalties after 2-2 — rivalry heat + revenge narrative live on both sides.
- **Anthony Taylor's WC26 card record**: 5 yellows + 1 red across his two matches (~3 cards/match) — moderate, NOT extreme, despite "card-happy" narrative headlines. [FourFourTwo, Vietnam.vn]
- **Tournament aggregates (FOX team stats, 4 matches each)**: Spain 71 shots / **28 SoT (7.0 per match)** / **57 corners (14.25 per match)**; Portugal 53 shots / 17 SoT (4.25 per match) / 36 corners (9.0 per match). Spain fouls 42 (10.5/m), Portugal 35 (8.75/m). Spain concede ~1.0 corner/match; Austria had ZERO SoT vs Spain.
- **Diogo Costa aggregate**: **14 saves in 4 matches (3.5/match)**: ~2-3 combined vs DRC+UZB, then 6 vs COL, 5 vs CRO. MOTM vs DR Congo and vs Colombia. [matchbingo, OneFootball]
- **POR-CRO R32 detail**: POR 15 shots / 3 SoT; CRO 13 shots / 6 SoT (Costa saved 5 of 6). Bruno Fernandes had a first-time effort saved by Livakovic (1+ SoT hit). [Opta Analyst, FIFA]
- **2025 NL final POR-ESP (last competitive meeting)**: 7 yellows total over 120' (POR 3, ESP 4); ~4 in regulation (Inácio 19', Fabián 33', Neto 82', Le Normand 90+1'). First card (19') came BEFORE first goal (21'). [Wikipedia, UEFA]
- **Yamal minutes**: rotated in group (76', 45', 19' off bench), started R32; 1 goal; "the World Cup starts now" — Spain's creative hub with Nico Williams out. [Olympics.com, ESPN]
- Dallas venue = AT&T Stadium (roofed, air-conditioned) — WC26 heat narrative largely neutralized for THIS match; 3:00 PM ET kickoff.

### D. Quantitative back-solves (this session, .venv python)

- NB(k=6.3) saves tail: mu_saves 2.4 -> P(4+)=.238 | 2.9 -> .331 | 3.3 -> .404 | 3.6 -> .455 | 4.0 -> .519 | 4.5 -> .590. **The submitted 24 back-solves to mu_saves ~2.4, i.e. Spain generating only ~3.7-3.8 SoT** — vs their observed 7.0/match.
- NB(k=8) corners tail: mu 5.0 -> P(6+)=.381 | 6.0 -> .510 | 6.5 -> .568 | 7.0 -> .621. **The submitted 41 back-solves to Spain corners mu ~5.2** — vs 14.25/match observed (albeit vs weaker opposition).
- Subs mixture recompute with the fetched WC2022 ET quintet (6,8,8,5,6 — 0/5 reached 9): X=.26, p_decided .75-.82, p_ETbound shrunk .15-.35 -> **P(9+) = .60-.66**; the submitted 58 sits below the plan's pre-registered rail [.60,.78].

---

# Per-question verdicts

Format: **# — question — submitted | VERDICT | evidence class | suggested value (only where HARD or HISTORICAL)**

## 1. Spain advances — 65 (book-anchored)
**Verdict: SANE.** Book-anchored on the to-qualify/derived market; Spain's tournament (8 GF / 0 GA, 28 SoT vs 17, zero SoT allowed to Austria) fully supports a ~2:1 favorite; Portugal needed a 90+4' winner to beat Croatia. The draw-heavy H2H (4 of last 6 level at 90) is already impounded by the book's draw mass (Q14 = 26).
**Evidence: HISTORICAL + market anchor.** No hard news undermines it (no new injuries either side; Nico Williams out ADDS to Spain's edge being slightly smaller than full-strength, already in the price).
**Action: LEAVE 65.**

## 2. Cristiano Ronaldo scores (excl. OG) — 32 (book 33; model-only was 7)
**Verdict: SANE at the book number.** HARD FACTS from search: Ronaldo is a nailed-on starter (all 270 group minutes + R32 start, subbed 81' for the first time), has 3 goals in 4 matches, is the designated penalty taker (converted 68' pen vs CRO), and scored against Spain in the 2025 NL final. Book 33 is exactly where an elite-pen-taking starter vs a top defense prices. The model-only 7 is the same player-model failure family as Q15 — diagnostic of a broken player-rate route, harmlessly overridden by the anchor here.
**Evidence: HARD FACT (role/minutes confirmed) supporting the anchor.**
**Action: LEAVE 32.** Crowd note: CR7 narrative crowd will likely sit 45-55 — a 13-23 pt divergence in OUR favor; pre-decide to hold with the book.

## 3. Lamine Yamal scores or assists — 51 (book 52)
**Verdict: SANE.** Book-anchored. Yamal was load-managed in the group (76', 45', 19') but started and starred in the R32; 1 goal this WC; Spain's primary creator and set-piece taker, with Nico Williams (adductor) expected OUT — usage concentrates further onto Yamal. 51-52 for score-OR-assist for the focal attacker of a 65% favorite is right.
**Evidence: HARD FACT (fit, starting, Williams out) supporting the anchor.**
**Action: LEAVE 51.**

## 4. Bruno Fernandes 1+ shot on target — 49 (book 50)
**Verdict: SANE.** Book-anchored; Bruno starts as the 10 in the predicted XI, takes set pieces and long shots, and registered a saved effort vs Croatia. Portugal at 4.25 team SoT/match; Bruno's historical 1+ SoT rate in big matches ~.50-.55. Facing Spain's suffocation (opponents: 0 SoT for Austria) justifies the coin-flip rather than his usual ~.60 club-level number.
**Evidence: HISTORICAL + anchor.**
**Action: LEAVE 49.**

## 5. Both halves same number of goals — 26 (grid + binomial split)
**Verdict: SANE.** The repo derivation is sound (QF-TEMPLATES-PLAN §1: anchored total pmf x split factors, verified to 3 decimals on 11,658 matches; WC-finals empirical .266). 26 is internally coherent with 3+ goals at 51 (both imply lam_tot ~2.6-2.7) and sits inside the pre-registered rail [.24, .33].
**Evidence: HISTORICAL BASE RATE.**
**Action: LEAVE 26.** Crowd note: plan expects crowd .35-.45 on this novelty — a 9-19 pt divergence in our favor; pre-decide to hold.

## 6. First card shown before first goal — 49 (time-varying race; naive would say 58)
**Verdict: QUESTIONABLE (a few points high).** The submitted 49 equals the UNCALIBRATED time-varying race value. The repo's own StatsBomb calibration (314 matches) says: raw time-varying race .5212 vs empirical .4904 (permissive) / .4045 (strict — goal must exist); the plan's calibrated POR-ESP expectation was **.44-.47 blended at w_perm=0.6** (QF-TEMPLATES-PLAN §2). Specifics push the same direction: anchored lam_tot ~2.6-2.7 is ABOVE the corpus goal mean (2.455) while WC26 cards run ~6.5% below historical level (s_cards=.935) and Taylor's WC26 rate (~3 cards/match) is moderate — both teams also score early this tournament (POR 6' twice, ESP 10'/21'/36'). The 2025 NL final anecdote (card 19' before goal 21') is color, not data. It looks like the Gamma-frailty/gamma-intercept calibration step and/or the w_perm blend was not applied.
**Evidence: HISTORICAL BASE RATE (measured 3-pt calibration residual + level factors).**
**Suggested value: 45** (46-47 if the platform is confirmed to settle permissive; 41 if strict). A 4-pt fix, not an emergency.

## 7. Portugal scores the first goal — 36
**Verdict: SANE.** Consistent with the anchored lambdas: lam_POR share ~.42 x P(any goal) ~.93 ≈ .39 before DC corrections; 36 is inside the tolerance of that arithmetic given the model's draw-mass treatment. Portugal scored first in both its wins (6' both times) but failed to score vs COL and conceded first vs CRO — no evidence against.
**Evidence: HISTORICAL / internal coherence.**
**Action: LEAVE 36.**

## 8. Match has 3+ total goals — 51 (book-anchored)
**Verdict: SANE (by anchor).** The book's O/U anchors this; the fixture's grindy recent history (0-0, 0-0, 1-1, 0-1 among the last six) and Spain's 0 GA scream UNDER-lean, but both teams are top-2 in tournament shot volume and the last WC meeting was 3-3 — precisely the tension the market has already priced at ~51. Do not out-think a validated anchor with narrative.
**Evidence: market anchor; SOFT NARRATIVE both directions cancels.**
**Action: LEAVE 51.**

## 9. Diogo Costa makes 4+ saves — 24 (SoT-thinning model)
**Verdict: LIKELY WRONG — the flagged watch item is confirmed.**
Three independent strikes:
1. **Mechanical (HARD)**: the number violates the plan's own pre-registered sanity rail **[.30, .55]** (QF-TEMPLATES-PLAN §3e: "Sanity rail [.30,.55]... Do NOT run at s = 1.0"). A submission outside the rail was pre-defined as a pipeline bug. The .24 back-solves to mu_saves ~2.4, i.e. Spain producing only ~3.7 SoT — it is almost exactly the UNCONDITIONAL historical base (.269 at mu 2.57), which is what the estimator produces if the WC26 SoT level factor (s_SoT ~1.29) and/or the anchored opponent-volume term were dropped.
2. **This-tournament volumes (HISTORICAL)**: Spain average **7.0 SoT/match** (7 vs CPV, 10 vs AUT); Costa has **6 and 5 saves** in his last two matches against attacks far weaker than Spain's, and 3.5/match overall. Even shading Spain's SoT hard for opponent quality (POR >> their group), mu_SoT ~4.7-5.8, minus lam_ESP ~1.3 -> mu_saves ~3.4-4.5 -> **P(4+) = .42-.59** on the NB(6.3) tail.
3. **The plan's own expectation** was .42-.46.
**Evidence: HARD FACT (rail violation = mechanical error) + HISTORICAL BASE RATE.**
**Suggested value: 45** (any value in .42-.50 defensible). Crowd will ride the "Costa vs Spain" narrative to ~.50-.60; at 24 we'd be 26-36 pts wrong-side; at 45 we sit in the validated 5-15-pt-under +EV band. **Highest-priority fix on the slate together with Q15.**

## 10. A substitute scores — 34
**Verdict: SANE.** 5-sub-era base rates (recomputed this session, StatsBomb, regulation, non-OG): WC2022 **.312**, Euro2024+ **.326**; platform crowd prior for this archetype sits at 31. Both benches carry real scorers (Gonçalo Ramos scored the R32 winner AS A SUB; Spain bring Ferran/Morata/Pino-tier attackers off the bench). 34 = base plus a small nudge for bench quality; right zone.
**Evidence: HISTORICAL BASE RATE.**
**Action: LEAVE 34.**

## 11. 4+ total cards — 50
**Verdict: SANE.** Corpus P(4+)=.475 (mean 3.46); WC2022 .422; Euro2024 .570; WC26 running ~6.5% under historical card level -> base ~.44-.46; add knockout stakes + genuine rivalry heat (2025 final had 4 regulation yellows + 3 more in ET) and a ref who has shown 6 cards in 2 matches -> 50 is within noise of any defensible estimate. Spain's 10.5 fouls/match and POR's 8.75 support a ~3.2-3.6 card expectation.
**Evidence: HISTORICAL BASE RATE.**
**Action: LEAVE 50.**

## 12. 9+ total substitutions in regulation — 58 (mixture at P(ET)=.26)
**Verdict: QUESTIONABLE (3-6 points low).** The team correctly executed the plan's mandated fetch (the five WC2022 ET matches: 6,8,8,5,6 subs in 90' — 0/5 reached 9) and correctly priced the hold-subs-for-ET effect. But the mixture lands low: with X=.26, p_decided = .75-.82 (KO decided-in-90 was .818, n=11; pooled .746) and p_ETbound Beta-shrunk to .15-.35 (a raw 0/5 cannot honestly go below ~.10), the mixture spans **.60-.66**. 58 requires simultaneously the pooled (not KO) decided rate AND a near-unshrunk 0/5 — the most bearish corner of the parameter box, and it breaches the plan's pre-registered rail [.60, .78]. Venue note: Dallas is a roofed, air-conditioned stadium, so the WC26 heat-subs tailwind is muted here — that legitimately argues for the lower half of the band, not below it.
**Evidence: HISTORICAL BASE RATE (small-n, honestly uncertain).**
**Suggested value: 62.** Bonus: the crowd likely sits .45-.55 ("9+ sounds like a lot"), so moving up also widens the expected edge.

## 13. Spain 6+ corner kicks — 41
**Verdict: QUESTIONABLE (likely too low; medium confidence).** Spain lead the tournament tier with **57 corners in 4 matches (14.25/match)** on 17.75 shots/match; 41 back-solves to mu ~5.2 — pricing Spain to lose ~64% of their corner volume against Portugal. Portugal are a genuine step up from CPV/KSA/URU/AUT and will hold more of the ball than those sides, so a big haircut is right — but Spain vs elite opposition still historically lands 6-9 corners (they had ~7-8 in the 2025 NL final's 90 minutes), and Portugal's own matches feature ~9 corners/match for POR alone. A shaded mu of 6.0-7.0 gives P(6+) = .51-.62.
**Evidence: HISTORICAL BASE RATE (this-tournament volume; opponent-quality shading is judgment).**
**Suggested value: 50** (conservative midpoint; anywhere .48-.57 defensible). Flag: this is the softest of the CHANGE calls — if the count model's mu came from validated elite-vs-elite fits rather than a stale level, 45 would be arguable; 41 is not.

## 14. Match goes to extra time — 26 (= book draw mass)
**Verdict: SANE (by construction).** The de-vigged X is the exact object (plan §5: perfect anchor, w=.97), and the fetched value ~.26 sits inside the sanity band [.22, .35]. The fixture's four level-at-90 results in the last six meetings would tempt a narrative bump — resist; the book knows. Historical KO base .30 is a band, not an estimator.
**Evidence: market anchor + HISTORICAL sanity band.**
**Action: LEAVE 26.**

## 15. ANY Portugal player has 2+ shots on target — 15
**Verdict: LIKELY WRONG — misroute CONFIRMED, biggest error on the slate.**
**Mechanical (HARD FACT)**: 15 is a single-mid-tier-player 2+ SoT price, not a team-maximum price. The claim "priced as a single unknown player" is consistent with everything measured: the platform's own crowd prior for this exact archetype (`any_player_sot_ge`, n=1) sits at **70**, and no sane model puts the best of ~14 Portuguese players at 15 when Bruno ALONE prices ~20-25 for 2+ SoT off his 50 for 1+.
**Independent estimate (HISTORICAL)**: StatsBomb, 600 team-matches: P(any player 2+ SoT) = **.485 overall**; conditional on team SoT k: k=3 -> .355, k=4 -> .582, k=5 -> .703, k=6 -> .905. NB(6.3) mixture over Portugal's plausible SoT distribution vs Spain (POR run 4.25 SoT/match this WC; Spain allow almost nothing — 0 SoT for Austria — so mu ~3.0-4.5): **P = .44-.59, center ~.51**. Ronaldo (31 shots in WC knockouts, 3-4 attempts/match) and Bruno are both natural 2+ candidates, concentrating SoT and pushing the true value toward the upper half.
**Evidence: HARD FACT (mechanical misroute) + HISTORICAL BASE RATE (600 team-matches).**
**Suggested value: 52.** (Range .48-.56.) Crowd will sit ~.65-.75 per the archetype prior — even after our fix we expect to be 15-20 under, which is the correct side if the fix lands and catastrophic (-55 pt error) if 15 stands.

---

# RANKED ACTION LIST

## CHANGE (hard or historical evidence)
| Rank | Q | From | To | Grounds |
|---|---|---|---|---|
| 1 | Q15 any POR player 2+ SoT | **15** | **52** | HARD: mechanical misroute (single-player price for a team-max question); HISTORICAL: .485 base, .44-.59 conditional band, archetype crowd prior 70. A 37-pt error at 3x knockout scoring. |
| 2 | Q9 Costa 4+ saves | **24** | **45** | HARD: violates the plan's pre-registered rail [.30,.55] — the level/volume term was dropped (back-solves to mu_saves 2.4 vs Spain's observed 7.0 SoT/match); HISTORICAL: Costa 6 and 5 saves in his last two vs weaker attacks; NB tail gives .42-.59 across the honest mu band. |
| 3 | Q12 9+ subs | **58** | **62** | HISTORICAL: honest mixture over (p_decided .75-.82) x (p_ETbound .15-.35 after 0/5 shrink) spans .60-.66; 58 is below the pre-registered rail [.60,.78]. Small move, also widens the crowd edge. |
| 4 | Q6 first card before first goal | **49** | **45** | HISTORICAL: the repo's own measured calibration (raw race .521 vs empirical .490 perm / .405 strict) plus above-corpus goal intensity and below-historical WC26 card level; 49 = the uncalibrated race. Settle-rule caveat: 46-47 if permissive confirmed, 41 if strict. |
| 5 | Q13 Spain 6+ corners | **41** | **50** | HISTORICAL (medium confidence): Spain 14.25 corners/match this WC; 41 implies a 64% collapse vs Portugal; shaded elite-opponent mu 6-7 -> .51-.62. Softest change call — accept 45+ as a compromise if the count-model fit is trusted. |

## LEAVE (verified sane)
- **Q1 Spain advances 65** (anchor + form), **Q2 Ronaldo 32** (anchor; starter+pens confirmed), **Q3 Yamal 51** (anchor; Williams out), **Q4 Bruno 1+ SoT 49** (anchor), **Q5 halves equal 26** (validated derivation, coherent with Q8), **Q7 POR first goal 36** (coherent), **Q8 3+ goals 51** (anchor), **Q10 sub scores 34** (5-sub era base .31-.33), **Q11 4+ cards 50** (base .44-.57 band), **Q14 extra time 26** (perfect anchor).

## PRE-DECIDE: expected crowd divergences >20 pts
| Q | Us (post-fix) | Likely crowd | Divergence | Pre-decision |
|---|---|---|---|---|
| Q15 any POR 2+ SoT | 52 | 65-75 (prior: 70, n=1) | ~15-23 | HOLD 52 — the crowd over-prices any-player events; but if Q15 is NOT fixed, 15 vs 70 is a -55 catastrophe. |
| Q9 Costa 4+ saves | 45 | 50-60 (keeper narrative) | 5-15 (was 26-36 at 24) | HOLD 45 — inside the validated under-the-crowd +EV band. |
| Q2 Ronaldo scores | 32 | 45-55 (CR7 narrative) | 13-23 | HOLD 32 — book-anchored; the crowd's hero premium is exactly the measured player_scores overshoot (prior mean 36 but CR7 skews high). |
| Q5 halves equal | 26 | 35-45 (novelty coin-flip read) | 9-19 | HOLD 26 — pre-registered rail, validated derivation. |
| Q12 9+ subs | 62 | 45-55 ("9 sounds high") | 7-17 | HOLD 62 — the base-rate edge class the system is built to harvest. |

## Summary of the expert pass
Two numbers are broken (Q15 misroute 15 -> 52; Q9 dropped level/volume term 24 -> 45), three deserve calibration-size nudges (Q12 58 -> 62, Q6 49 -> 45, Q13 41 -> 50), and the ten remaining — including every book-anchored price — survive scrutiny against live-tournament facts (lineups, Ronaldo's role, Williams' injury, Taylor's appointment, Spain's suffocating volume) and repo base rates. No SOFT-NARRATIVE-only change is recommended anywhere.

*Sources consulted: martj42 (repo, live through WC26), StatsBomb corpus (314 matches, recomputed this session), api-football WC2022 events (via QF-TEMPLATES-PLAN), crowd_prior.json; web: ESPN, FIFA.com, Opta Analyst, FOX Sports team stats, OneFootball, Sports Mole, Yahoo/Get Spanish Football News (referee), Wikipedia/UEFA (2025 NL final), Olympics.com (Yamal), matchbingo (Costa saves), Racing Post/Squawka previews. Searched 2026-07-05/06 with the tournament live.*
