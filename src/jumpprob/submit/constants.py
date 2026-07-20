"""Tuned constants and archetype families for the live pricing runner.

Everything here is a number we chose deliberately, with a reason. Three kinds:

* **Measured** — estimated from data (StatsBomb events, martj42 results,
  api-football, or our own settled questions). The comment names the source.
* **Fitted** — optimised against settled outcomes by a script in ``scripts/``.
* **Judged** — a structural assumption where data was thin. These are the ones
  to distrust first, and they are marked as such.

Where a constant is deliberately conservative rather than optimal, the comment
says so: several were set below their fitted value because the fit was not
identified and the downside was asymmetric.
"""

from __future__ import annotations

from pathlib import Path

from jumpprob.simulate import markets as mk

PROCESSED = Path("data/processed")

# --------------------------------------------------------------------------
# Archetype families
# --------------------------------------------------------------------------
# Family decides two things: which Platt calibration a question gets, and which
# engine answers it. Goal-family questions skip calibration when anchored (the
# book is already calibrated); count and player questions are always calibrated.

# Timing questions the closed-form engine only approximates — routed to the
# Monte-Carlo simulator when one is available.
TIMING_ARCHS = {
    "ht_tied", "ht_leading", "h2_more_goals_than_h1", "first_goal_half",
    "first_goal_and_win", "first_goal_and_opp_2h", "goals_gt_half",
}

# Everything derived from the (anchored) scoreline grid or the goal intensities.
# Registering a template here matters: it lets an anchored answer pass straight
# through instead of being diluted 25% toward the league-average base rate.
GOAL_ARCHS = {
    "win", "draw", "btts", "total_ge", "total_le", "btts_and_total_ge",
    "pen_shootout", "total_exact", "goal_each_half", "leads_at_any_point",
    "extra_time", "halves_equal_goals", "first_goal_single_digit",
    "win_both_halves", "first_goal_2h", "margin_exactly", "total_odd",
    "team_scores_half", "half_total_ge", "team_goals_ge",
    "either_team_half_goals_ge", "first_goal_assisted", "first_goal_match",
    "win_margin", "clean_sheet", "score_both_halves", "ht_scoreless",
    "advance",
} | TIMING_ARCHS

COUNT_ARCHS = {
    "corners_gt", "sot_gt", "fouls_gt", "cards_gt", "team_corners_ge",
    "team_sot_ge", "team_cards_ge", "cards_ge", "corners_total_ge",
    "total_sot_half_ge", "total_sot_ge", "offsides_ge", "both_teams_sot_ge",
    "corners_and_shots",
}
# card_each_half and card_stoppage are deliberately NOT count-family. The count
# Platt shifts prices up, which fights the fact that cards ran low all through
# WC26. They ship raw-structural, protected by the crowd-proxy clamp instead.

PLAYER_ARCHS = {"player_scores", "player_sot_ge", "player_ga"}

# Archetypes the Monte-Carlo engine answers directly, when a simulation exists.
# The coupled simulator removes a ~2pt independence over-price that the
# closed-form same-half handler carries.
MC_TIMING = {
    "ht_tied": "ht_tied",
    "ht_leading": "ht_leading",
    "h2_more_goals_than_h1": "h2_more_goals_than_h1",
    "first_goal_half": "first_goal_half",
    "first_goal_and_win": "first_goal_and_win",
    "first_goal_and_opp_2h": "first_goal_and_opp_2h",
    "goals_gt_half": "goals_gt_half",
    "either_team_half_goals_ge": "either_team_half_goals_ge",
}

# Half shares of goal intensity, measured on 47k goals.
H1 = 1 - mk.SECOND_HALF_SHARE
H2 = mk.SECOND_HALF_SHARE

# --------------------------------------------------------------------------
# Hydration-break and stoppage windows
# --------------------------------------------------------------------------
# The platform puts cooling breaks at the 22' mark of each half, i.e. match
# minutes 22 and 67. Windows map to (dynamics family, first slot, last slot)
# and are integrated analytically from the fitted minute profile.
WINDOW = {
    "goal_pre_hb1": ("goal", 0, 22),        # before the first break
    "goal_after_hb1": ("goal", 22, 45),     # rest of H1 — carries MORE intensity
                                            # than [0,22): the hazard rises through a half
    "goal_between_hbs": ("goal", 22, 67),   # between the breaks — the fattest window,
                                            # ~55% of total goal intensity
    "goal_post_hb2": ("goal", 67, 90),
    "corners_pre_hb1": ("corner", 0, 22),
    "cards_post_hb2": ("card", 67, 90),
    "offsides_pre_hb1": ("offside", 0, 22),
}

# --------------------------------------------------------------------------
# Player props
# --------------------------------------------------------------------------
# Minutes scenarios before lineups drop: start / substitute / unused.
MINUTE_SCENARIOS = [(0.80, 80.0), (0.12, 25.0), (0.08, 0.0)]

# Players shoot and score less for country than for club — tougher opponents,
# less familiar team-mates. Flat optimum on settled player questions
# (scripts/fit_player_transfer.py). 1.0 would be no discount.
PLAYER_TRANSFER = 0.80

# The model's share of a book-anchored player prop, as a fraction of the share
# it gets on goal markets. Fitted on 51 settled anchored props, where Brier was
# monotone toward the pure book (model .250, blend .206, book .202): our model
# is data-gapped on exactly the stars the book prices well. At the live market
# weight of .85 this leaves the book with ~97%.
PLAYER_MODEL_FRAC = 0.20

# The crowd we are scored against prices player props ABOVE the book we anchor
# to, and calibrates better doing it (crowd .223 vs book .258 Brier, n=29). The
# cause is a book low-side miscalibration: prices under .34 realise .44. Since
# we cannot see the crowd before lock, we lift anchored props toward it.
#
# The magnitude is NOT identified. The Brier-vs-shift curve is monotone with no
# interior optimum, the raw edge is borderline (p~.065), and its direction is a
# 15-14 coin flip — the gain is magnitude on mid-priced props, not hit rate. So
# the lift is a deliberate under-shift, keeping ~70% of the modelled gain while
# roughly halving the downside. The cap does the real regularising: it binds on
# shots-on-target and bounds any refit.
PLAYER_CROWD_LIFT = 0.5
PLAYER_CROWD_LIFT_CAP = 0.06    # max +6 points on any one prop
PLAYER_CROWD_LIFT_CEIL = 0.85   # above this the book and crowd agree — no headroom

# Confirmed lineup news must beat the book anchor. We learned this the hard way:
# De Bruyne was benched against the USA, the book had not repriced by our closing
# tick, and the crowd lift then pushed a stale starter price higher still.
# Applied as a ceiling, so a mis-scraped "benched" caps a starter rather than
# zeroing them.
LINEUP_OUT_CEIL = 0.05          # ruled out of the squad
LINEUP_BENCH_CAP = {            # elite off-the-bench (~25 minutes) ceilings
    "player_scores": 0.15, "player_ga": 0.18, "player_sot_ge": 0.35,
}
LINEUP_FRESH_H = 12.0           # ignore a status older than this

SOT_PER_SHOT = 0.37             # Understat has no shots-on-target — derive from shots

# The three tables below are FIXTURE-SPECIFIC pins from the 2026 endgame, kept as
# a record of how named-player evidence entered the price. They fire only on the
# named players and are inert for everyone else.
#
# Per-minute form pins for players whose club-table rate is confirmed stale.
# Dembélé's table rate (0.41 GA/90) badly lagged his PSG and tournament form.
# Pinned deliberately BELOW raw form so the unanchored number lands near the
# book's if-starts price; this only ever fires with no book prop present.
PLAYER_FORM_RATE_OVERRIDES = {("player_ga", "ousmane dembele"): 0.85 * 0.80 / 90.0}

# Star shooters whose shrunk table rate lags reality badly (Messi ran ~3x his
# table rate at WC26). Two-thirds tournament form, one-third history.
SOT_RACE_FORM_RATES = {"lionel messi": 2.4 / 90.0, "lamine yamal": 1.6 / 90.0}
SOT_RACE_MINUTES = {
    "lionel messi": [(0.80, 90.0), (0.15, 75.0), (0.05, 60.0)],
    "lamine yamal": [(0.85, 88.0), (0.10, 70.0), (0.05, 25.0)],
}
# Our shots-on-target family under-calls (realised .40 against a predicted .29)
# and no crowd template exists for a player-vs-player race, so the shrink toward
# the field's ~50 lives here instead.
SOT_RACE_CROWD_W = 0.35
SOT_RACE_FLOOR = 0.25

# --------------------------------------------------------------------------
# Crowd anchoring
# --------------------------------------------------------------------------
# Per-archetype crowd levels harvested from settled questions back two safety
# nets. First, a player we know nothing about (no club prior, no tournament
# minutes) blends toward the archetype's crowd level rather than shipping the
# naked population floor — the class of error that priced Haaland at 8%.
UNRESOLVED_PLAYER_TEMPLATE_W = 0.5

# Second, on flat event kinds where our model is a constant and the crowd barely
# varies per question, we hug the crowd level. Submitting near the field's own
# mean banks the Jensen field-variance bonus instead of bleeding a fixed offset
# every match. Whitelist only — grid, count and player kinds keep their signal,
# and offsides/fouls stay flattened by the risk gate (the crowd OVER-prices
# those, so chasing it would be backwards).
CROWD_HUG_KINDS = {"pen_or_red", "penalty_awarded", "substitute_scores"}
CROWD_HUG_W = 0.6
CROWD_HUG_MAX_STD = 4.0         # hug only where the crowd is near-constant
CROWD_HUG_MIN_N = 3

# Bot A's shrink toward the league-average base rate, on unanchored questions.
BOT_A_BASE_WEIGHT = 0.25

# --------------------------------------------------------------------------
# Endgame variance dial
# --------------------------------------------------------------------------
# A graded rank-chase lever for the closing stages, off unless ENDGAME_EXTREMIZE
# is set. The endgame simulation said blanket variance is a trap — it has zero
# mean and the stage multipliers amplify losses too (honest +481±379 against
# blanket +475±760). So escalation is restricted to kinds where we have
# structural edge: grid templates where the crowd uses provably naive heuristics,
# and the team-count kinds where we measurably beat it.
#
# Never extremize: player props (the crowd beats us), offsides and fouls
# (flattened by design), anchored goal markets (the book is the floor), or card
# timing (miscalibrated).
ENDGAME_KINDS = {
    "win_both_halves", "first_goal_2h", "margin_exactly", "total_odd",
    "goal_after_hb1", "ht_scoreless", "corners_and_shots",
    "team_sot_ge", "team_corners_ge", "sot_gt", "cards_gt",
}

# --------------------------------------------------------------------------
# Match events: penalties, cards, red cards, extra time
# --------------------------------------------------------------------------
P_RED_MATCH = 0.065             # P(>=1 red), 0.067 reds/match over 6 tournaments
P_PEN_MATCH = 0.293             # P(>=1 penalty awarded), same 6 tournaments

# WC26 referees called penalties at ~60% of the historical rate, so the
# six-tournament number above is stale for this cycle's own questions.
PEN_LAMBDA_WC26 = 0.206         # awarded penalties per match, 21 in 102
PEN_FINAL_UPLIFT = 1.19         # final-specific: box scrutiny plus the appointed
                                # referee's own above-average record
# Taker-weighted conversion, for the finalists only — a fixture-specific pin.
PEN_CONV = {"Spain": 0.85, "Argentina": 0.74}
PEN_CONV_DEFAULT = 0.75         # historical World Cup conversion, 39 of 52
PEN_SCORED_BOUNDS = (0.10, 0.30)   # hard bounds — a brand-new archetype at 3x stakes

# Extra time and shootouts, grounded on 839 ET matches and 678 shootouts in
# martj42: ET scoring runs at ~0.34x the pro-rated regulation rate, 81% of ET
# matches still reach penalties, and a shootout is near a coin flip with a mild
# favourite tilt (.49 even, .57 big favourite).
ET_TEMPO = 0.35
PEN_TILT = 0.20                 # shootout tilt from the 90-minute win share
# Team shootout evidence beats that tilt where it exists: the 90-minute win share
# says nothing about penalty takers or goalkeepers, and it is wrong-signed for
# Argentina (5 straight, 4/4 in the Martínez era) against Spain (2 of 6 since 2018).
PEN_SHOOTOUT_SPEN = {"Argentina": 0.55, "Spain": 0.45}

# Share of first goals arriving from a penalty or an own goal. A cagey final's
# penalty intensity does not scale down with tempo, so the penalty SHARE rises.
FIRST_GOAL_PEN_SHARE = 0.08
FIRST_GOAL_OG_SHARE = 0.05
# Share of open-play goals carrying a credited assist — intensity-weighted from
# Spain .737 and Argentina .692 (StatsBomb, 668 open-play goals).
ASSISTED_OPEN_SHARE = 0.717

# Share of cards falling in stoppage time. Cards are late-loaded, and the 90'
# minute profile truncates added time, so this is an explicit extrapolation.
# JUDGED — low confidence.
CARD_STOPPAGE_SHARE = 0.15

# --------------------------------------------------------------------------
# First scorer's shirt number
# --------------------------------------------------------------------------
# P(first scorer wears 1-9 | a goal). WC26 itself measured 32/90 = 35.6%; WC2022
# measured 14/35 = 40.0%. A 2:1 recency blend gives 36.3%. The double-digit
# attacker pattern is stronger in 2026 — #10 is modal with 11 first goals, then
# #11 and #19. The naive "9 of ~23 numbers" intuition over-prices YES.
SINGLE_DIGIT_FIRST_SHARE = 0.365
# Better still, allocate by the two teams' actual single-digit goal shares:
# Poisson superposition makes the first scorer track each player's goal
# intensity. Self-consistency checks out — WC26's all-goal single-digit share
# (35.4%) matches its first-scorer share (35.6%).
OWN_GOAL_SHARE = 0.05           # ~5% of goals are own goals
OWN_GOAL_SINGLE_DIGIT = 0.40    # defenders skew to low numbers, so ~40%

# --------------------------------------------------------------------------
# VAR and substitutions
# --------------------------------------------------------------------------
# On-field monitor reviews per 90'. Reconstructed from WC2022 events counting
# only true monitor trips (penalty confirmed or cancelled, card upgrade,
# subjective goal review) and excluding semi-automated offside, which needs no
# monitor: lambda ~0.32. WC26 discipline is inflated — reds up ~9x — which argues
# higher. But our own settled data tempers that: the closest outcome class
# realised .226 against a crowd price of .386, so the crowd over-prices here.
# 0.33 sits above the WC22 base and below the crowd.
VAR_OFR_LAMBDA = 0.33

# First-substitution race: game-state dominated, since the trailing side subs
# first. Tilt the coin by the win-probability gap, but keep it small — manager
# idiosyncrasy and injuries hold it near 50/50.
FIRST_SUB_TILT = 0.10

# --------------------------------------------------------------------------
# Shot concentration and distinct shooters
# --------------------------------------------------------------------------
# P(>=5 distinct shooters | N team shots), logistic fit on 628 team-matches with
# zero team residual — shot volume is the sufficient statistic.
SHOOTER_OCC_B0, SHOOTER_OCC_B1 = -5.333, 0.847
DISTINCT_SHOOTERS_BASE = 0.841  # population rate; we ship a half-step toward it

# Residual correlation between the two teams both clearing a shots-on-target
# threshold. The raw within-match correlation (-0.23) is mostly matchup
# heterogeneity that the anchored marginals already encode; the conditional
# residual is -0.076, so the indicator correlation is ~-0.05.
EACH_TEAM_SOT_RHO = -0.05

# Correlation between "more corners" and "more shots". The count model treats
# them as independent given intensity, but the residual coupling beyond team
# quality is large — measured indicator correlation 0.41 on 314 matches, because
# a team having an attacking day gets both. 0.30 sits deliberately below that,
# conservative for the linearisation. An earlier value of 0.05 badly
# under-coupled this and was the one handler positioned to lose to the crowd.
CORNERS_SHOTS_RHO = 0.30

# P(some player takes >=2 of the team's goals | team scored N), and the same for
# shots on target. Measured over 314 matches and monotone-cleaned. These ride the
# team goal and shot distributions we already model, so they scale with intensity.
BRACE_BY_TEAM_GOALS = [0.0, 0.0, 0.156, 0.42, 0.77, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0]
SOTCONC_BY_TEAM_SOT = [0.0, 0.0, 0.13, 0.33, 0.58, 0.71, 0.88, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# --------------------------------------------------------------------------
# Event base rates
# --------------------------------------------------------------------------
# Markets our 90-minute minute profile cannot reach. Stoppage goals scale with
# the match's scoring; substitutions are near-fixed base rates. Defaults are
# football norms, overridden at load by data/processed/event_baselines.json
# (api-football WC2022, built by scripts/build_event_baselines.py).
EVENT_BASE_DEFAULTS = {
    "goal_share_1h_stoppage": 0.02,
    "goal_share_2h_stoppage": 0.08,
    "sub_before_ht": 0.12,            # roughly the injury base rate
    "substitute_scores": 0.18,
    "sub_at_ht": 0.52,                # 33 of 64 WC2022 matches saw a change at the
                                      # interval; Laplace-smoothed
    "own_goal": 0.10,                 # flat. Thinning by total goals was tested and
                                      # lost — own goals do not scale with scoring
    "outside_box_share": 0.13,        # share of open-play goals from outside the box
                                      # (StatsBomb). Validated as a Poisson thinning,
                                      # +0.021 Brier against flat
    "header_share": 0.21,             # headed share of open-play goals. Validated
                                      # thinning, +0.027 Brier against flat
    "player_full90": 0.42,            # P(a named starter plays all 90). Five subs over
                                      # ten outfield players leaves ~52% surviving;
                                      # attackers come off more often
    # Substitution exceedance, WC2022 all 64 matches, Laplace-smoothed. The
    # extra-time stratum comes from the five ET matches, whose regulation subs
    # ran 6, 8, 8, 5, 6 — going to ET makes teams HOLD subs back. Mixed at answer
    # time by P(draw after 90).
    "subs_ge8_nonET": 0.836, "subs_ge9_nonET": 0.738, "subs_ge10_nonET": 0.443,
    "subs_ge8_ET": 0.429, "subs_ge9_ET": 0.143, "subs_ge10_ET": 0.143,
}

# --------------------------------------------------------------------------
# Fitted artefact paths (all refreshed nightly by scripts/daily_learn.py)
# --------------------------------------------------------------------------
CROWD_TEMPLATE_PATH = PROCESSED / "crowd_prior.json"
PLAYER_CROWD_PREM_PATH = PROCESSED / "player_crowd_premium.json"
LINEUP_STATUS_PATH = PROCESSED / "lineup_status.json"
SINGLE_DIGIT_TEAM_SHARE_PATH = PROCESSED / "single_digit_team_share.json"
CLUB_RATES_PATH = PROCESSED / "player_rates_combined.parquet"
EVENT_BASE_PATH = PROCESSED / "event_baselines.json"
DYNAMICS_PATH = PROCESSED / "minute_dynamics.json"
LIVE_CALIBRATION_PATH = PROCESSED / "live_calibration.json"

# --------------------------------------------------------------------------
# Team code map (the platform names matches "ESP vs ARG")
# --------------------------------------------------------------------------
CODE_MAP = {
    "CAN": "Canada", "BIH": "Bosnia and Herzegovina", "USA": "United States",
    "PAR": "Paraguay", "QAT": "Qatar", "SUI": "Switzerland", "BRA": "Brazil",
    "MAR": "Morocco", "SCO": "Scotland", "AUS": "Australia", "TUR": "Turkey",
    "KOR": "South Korea", "CZE": "Czech Republic", "JPN": "Japan", "GER": "Germany",
    "FRA": "France", "ESP": "Spain", "ENG": "England", "POR": "Portugal",
    "NED": "Netherlands", "ARG": "Argentina", "URU": "Uruguay", "COL": "Colombia",
    "ECU": "Ecuador", "MEX": "Mexico", "CRO": "Croatia", "BEL": "Belgium",
    "SEN": "Senegal", "GHA": "Ghana", "CIV": "Ivory Coast", "NGA": "Nigeria",
    "EGY": "Egypt", "ALG": "Algeria", "TUN": "Tunisia", "RSA": "South Africa",
    "KSA": "Saudi Arabia", "IRN": "Iran", "IRQ": "Iraq", "UZB": "Uzbekistan",
    "JOR": "Jordan", "NZL": "New Zealand", "PAN": "Panama", "CRC": "Costa Rica",
    "HON": "Honduras", "NOR": "Norway", "AUT": "Austria", "POL": "Poland",
    "DEN": "Denmark", "SWE": "Sweden", "ITA": "Italy", "WAL": "Wales",
}
