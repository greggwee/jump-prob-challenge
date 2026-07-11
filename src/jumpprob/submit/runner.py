"""Live market runner: discover -> parse -> answer -> submit (dry or live).

Team identification: the two team names for a match are extracted from its own
question texts (full names appear there), validated against the Elo table;
match-name trigram codes are only a fallback. Matches whose teams cannot be
resolved are reported, not guessed.

Bot profiles (no odds anchor yet — see PLAN §6.3):
  Bot A (anchor-conservative): logit blend 75% model / 25% archetype base rate
      (base = same archetype answered with league-average intensities).
  Bot B (alpha): pure calibrated model.
Once de-vigged odds are supplied per market, A moves to w=0.85 market blending.

Player props: shrunk career rates from our event data (M0=270 SoT / 540 goals),
population prior for unseen players; minutes scenario mix
(start 80% x 80', sub 12% x 25', unused 8%) until lineups land.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.models.count_model import p_a_gt_b, p_sum_ge
from jumpprob.models.goal_model import dc_grid
from jumpprob.models.market_anchor import MatchAnchor, blend, blend_lambdas, implied_lambdas, inv_logit, logit
from jumpprob.models.calibration import submit_int
from jumpprob.normalize.teams import canonical
from jumpprob.normalize.players import build_index as _club_index_fn, resolve as _club_resolve
from jumpprob.simulate import markets as mk
from jumpprob.simulate import mc_engine
from jumpprob.models.dynamics import Dynamics, BUCKET
from jumpprob.ingest.odds_api import norm_player, resolve_player
from jumpprob.submit.parser import ParsedQuestion, parse_question
from jumpprob.agents import panel as panel_mod  # out-of-band LLM blend (daemon reads cache only)

# Archetype families (calibration + engine routing). Timing markets are the
# subset the analytic engine only approximates — route them through the v2 MC
# simulator (rec #3). Goal markets are anchored when odds exist (skip calib);
# count/player markets are model-only (always calibrate, rec #1).
TIMING_ARCHS = {"ht_tied", "ht_leading", "h2_more_goals_than_h1", "first_goal_half",
                "first_goal_and_win", "first_goal_and_opp_2h", "goals_gt_half"}
GOAL_ARCHS = {"win", "draw", "btts", "total_ge", "total_le", "btts_and_total_ge",
              # R16 grid/lambda-derived templates: registering them makes anchored answers
              # pass through instead of taking a 25% base-rate dilution (review §3.4)
              "pen_shootout", "total_exact", "goal_each_half", "leads_at_any_point",
              "extra_time", "halves_equal_goals",
              # new QF templates (2026-07-07): all pure goal-grid / per-half-grid derivations
              "win_both_halves", "first_goal_2h", "margin_exactly", "total_odd",
              "team_scores_half", "half_total_ge", "team_goals_ge",
              "first_goal_match", "win_margin", "clean_sheet", "score_both_halves",
              "ht_scoreless",
              "advance"} | TIMING_ARCHS
COUNT_ARCHS = {"corners_gt", "sot_gt", "fouls_gt", "cards_gt", "team_corners_ge",
               "team_sot_ge", "team_cards_ge", "cards_ge", "corners_total_ge",
               "total_sot_half_ge", "total_sot_ge", "offsides_ge", "both_teams_sot_ge",
               "corners_and_shots"}  # a corners+shots comparison (count-Platt like corners_gt)
# NB card_each_half / card_stoppage are deliberately NOT count-family: the count Platt shifts UP
# (b>0), which fights WC26 cards-running-LOW — price them raw-structural + P1-clamp-protected.
PLAYER_ARCHS = {"player_scores", "player_sot_ge", "player_ga"}

# parser archetype + side -> mc_engine.answer(archetype, side) for timing markets
_MC_TIMING = {
    "ht_tied": "ht_tied", "ht_leading": "ht_leading",
    "h2_more_goals_than_h1": "h2_more_goals_than_h1", "first_goal_half": "first_goal_half",
    "first_goal_and_win": "first_goal_and_win", "first_goal_and_opp_2h": "first_goal_and_opp_2h",
    "goals_gt_half": "goals_gt_half",
}

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

MINUTE_SCENARIOS = [(0.80, 80.0), (0.12, 25.0), (0.08, 0.0)]
# club->international rate discount: players shoot/score less for country than club
# (tougher opponents, less-familiar teammates). Fit on settled player props (see
# scripts/fit_player_transfer.py); 1.0 = no discount.
PLAYER_TRANSFER = 0.80   # validated flat optimum on settled player_ga (scripts/fit_player_transfer.py)
# Model share of the PLAYER-prop book blend, as a fraction of the goal-market model share
# (1-w_market). Fit 2026-07-01 on 51 settled anchored props: Brier is monotone toward the
# pure book (model 0.250 / w=.85 0.206 / book 0.202) — the model is data-gapped on stars, so
# it keeps only a 20% sliver of the goal-market share: live w_market=.85 -> player w=.97.
PLAYER_MODEL_FRAC = 0.20
# P4b player-prop crowd premium — the Jump crowd we are SCORED against prices these props
# systematically ABOVE the book we anchor to (crowd .223 vs book .258 Brier, n=29 settled anchored
# props). Quant audit 2026-07-06 confirmed the edge is real, NOT an artifact: the gap survives
# dropping any single match, outcome reconstruction is clean (0/29 mismatch), and it is LARGER
# without the anchored-only filter. Root cause is a book LOW-SIDE miscalibration — book prices in
# [0,.34) realize .44. We can't see the crowd pre-lock; lift book-anchored player props toward it
# by the fitted per-archetype premium, DAMPED and CAPPED. Refit in daily_learn.
# MAGNITUDE IS NOT DATA-IDENTIFIED (audited): the LOO-Brier-vs-shift curve is MONOTONE (more lift =
# lower Brier, no interior optimum; bootstrap argmin sits at the grid edge), the raw crowd-beats-
# book edge is only borderline (p~.065) and its DIRECTION is a 15-14 coin flip — the win is
# magnitude on mid-priced props, not hit-rate. So PLAYER_CROWD_LIFT is a deliberate CONSERVATIVE
# under-shift for safety, NOT a fitted optimum: 0.5 keeps ~70% of the modelled gain while ~halving
# the calm-match downside; let forward settlement earn any increase. The CAP is the real
# regularizer — it BINDS on SoT (biggest premium / most low-priced props, where a flat additive
# lift most overshoots) and bounds any daily_learn refit. A price-interacted or logit-space lift
# was tested and is worse/neutral at this n, so the flat-additive form stays.
# Phase-0 endgame (2026-07-10): archetype crowd templates (data/processed/crowd_prior.json,
# refit by daily_learn) back two safety nets. (1) UNRESOLVED player props (no club prior, no
# tournament minutes) blend toward the archetype's crowd-template LEVEL instead of shipping the
# naked population floor — the Haaland-8%/De Bruyne/DR-Congo catastrophic-tail class; adopted
# from parkjpd ("generic player fallback defaults to crowd level, not the floor"). (2) The flat
# event-base kinds where our model is a CONSTANT and the per-question crowd barely varies
# (template std<=4) hug the crowd level — submitting ~the field's own mean banks the Jensen
# field-variance bonus (mean-of-Briers > Brier-of-mean) instead of bleeding a fixed offset.
# Whitelist ONLY: grid/count/player kinds keep their engine signal (validated edges), and
# offsides/fouls stay risk-gate-flattened (the crowd OVER-prices those — do not chase it).
# Phase-2 endgame lever (2026-07-11, default OFF): a graded rank-chase dial for the SF/final,
# per the endgame MC — blanket variance is a trap (zero mean, multipliers hit losses too;
# honest +481±379 vs blanket +475±760), so escalation is (a) env-gated (ENDGAME_EXTREMIZE=<a>,
# e.g. 1.25/1.5; absent -> byte-identical submissions), (b) restricted to the STRUCTURAL-EDGE
# allowlist below (grid templates where the crowd uses provably-naive heuristics + the team-count
# kinds where we measurably beat the crowd), and (c) live-pass only (CLV passes stay pure).
# NEVER extremize: player props (crowd beats us), offsides/fouls (flattened by design),
# anchored goal markets (the book IS the floor), cards timing (mis-calibrated).
# Operate via scripts/endgame_policy.py: it sizes the needed rate vs the cutoff and prints the
# recommended tier; a human sets the env. Escalate only as far as the deficit demands.
ENDGAME_KINDS = {"win_both_halves", "first_goal_2h", "margin_exactly", "total_odd",
                 "goal_after_hb1", "ht_scoreless", "corners_and_shots",
                 "team_sot_ge", "team_corners_ge", "sot_gt", "cards_gt"}
CROWD_TEMPLATE_PATH = Path("data/processed/crowd_prior.json")
UNRESOLVED_PLAYER_TEMPLATE_W = 0.5   # unknown player: 50/50 model / crowd level (interior blend)
CROWD_HUG_KINDS = {"pen_or_red", "penalty_awarded", "substitute_scores"}
CROWD_HUG_W = 0.6
CROWD_HUG_MAX_STD = 4.0              # hug only where the crowd is near-constant per question
CROWD_HUG_MIN_N = 3
PLAYER_CROWD_PREM_PATH = Path("data/processed/player_crowd_premium.json")
PLAYER_CROWD_LIFT = 0.5       # deliberate conservative under-shift (magnitude unidentified, p~.065)
PLAYER_CROWD_LIFT_CAP = 0.06  # BINDING guardrail: max +6 pts/prop; caps SoT + any refit over-lift
PLAYER_CROWD_LIFT_CEIL = 0.85  # skip props already priced high — crowd premium ~0 there (both agree)
# Lineup-status override (2026-07-07): confirmed pre-match lineup news must beat the book anchor.
# We MISSED De Bruyne (benched vs USA, submitted 44 -> Brier .194, out=0): the book hadn't absorbed
# the benching by our closing tick, the WebPanel DID web-search it but the pm-guard discarded it
# (book-anchored player props skip the expert layer), and P4b then LIFTED the stale starter price.
# Fix: a benched/out flag (auto-detected by the WebPanel near KO -> lineup_status.json) overrides
# the book with a BOUNDED sub-minutes ceiling and cancels the crowd lift. Bounded = a mis-scraped
# "benched" caps a starter to the elite-sub base rate, not to zero. Benched/out named players only.
LINEUP_STATUS_PATH = Path("data/processed/lineup_status.json")
LINEUP_OUT_CEIL = 0.05        # ruled out / not in matchday squad -> ~won't feature
LINEUP_BENCH_CAP = {"player_scores": 0.15, "player_ga": 0.18, "player_sot_ge": 0.35}  # elite off-the-bench (~25') ceilings
LINEUP_FRESH_H = 12.0         # ignore stale status (the WebPanel refreshes per match near KO)
CLUB_RATES = Path("data/processed/player_rates_combined.parquet")
SOT_PER_SHOT = 0.37   # Understat has no SoT -> estimate from total shots
P_RED_MATCH = 0.065   # P(>=1 red): 0.067 reds/match measured, 6 tournaments
P_PEN_MATCH = 0.293   # A6
# Knockout advancement (advance to next round = win THIS tie, incl. extra time + penalties).
# Grounded on martj42: ET goals run 0.30/30min = 0.34x pro-rated regulation (unbiased over
# 839 ET matches), and 81% of ET matches STILL go to penalties; shootouts are ~a coin flip
# with a mild favorite tilt (0.49 even -> 0.57 big-favorite over 678 historical shootouts).
ET_TEMPO = 0.35       # extra-time scoring as a fraction of pro-rated regulation
PEN_TILT = 0.20       # shootout favorite tilt vs the 90-min relative win share

# Event markets our 90-min minute profile can't reach. Stoppage goals scale with the
# match's scoring (lambda_W = lambda_tot * goal-share); subs are ~fixed base rates.
# Defaults are football norms, overridden by data/processed/event_baselines.json
# (api-football WC2022, scripts/build_event_baselines.py).
EVENT_BASE_DEFAULTS = {
    "goal_share_1h_stoppage": 0.02, "goal_share_2h_stoppage": 0.08,
    "sub_before_ht": 0.12, "substitute_scores": 0.18,
    # "either team makes a substitution AT halftime": 33/64 WC2022 matches had a subst event
    # at elapsed=46 (the HT-interval change) -> Laplace (33+1)/(64+2)=0.515. Flat base.
    "sub_at_ht": 0.52,
    # own_goal: flat P(>=1 OG/match). Thinning-by-total-goals was TESTED and lost to flat
    # (OGs don't scale with goals). 0.10 = WC-finals prior 0.062 shrunk with WC26's ~0.14 (the
    # friendly-polluted all-comp pool of 0.019 badly under-prices the WC).
    "own_goal": 0.10,
    # outside_box_share: share of open-play goals from outside the box (StatsBomb). Validated
    # as a Poisson thinning of the goal intensity (+0.021 Brier vs flat, calibrated).
    "outside_box_share": 0.13,
    # header_share: share of open-play goals headed in (StatsBomb). Validated thinning
    # (+0.027 Brier vs flat on held-out matches, calibrated at 1-3 goals).
    "header_share": 0.21,
    # P(a named starter plays the full 90): ~5 subs over 10 outfield -> ~52% survive;
    # attackers are subbed more -> 0.42. R16 template (plays_full_match), no book market.
    "player_full90": 0.42,
    # QF subs exceedance (WC2022 all 64, Laplace-shrunk, ET stratum from the 5 fetched
    # ET matches — reg-subs 6,8,8,5,6): mixture weight = P(draw90) at answer time.
    "subs_ge8_nonET": 0.836, "subs_ge9_nonET": 0.738, "subs_ge10_nonET": 0.443,
    "subs_ge8_ET": 0.429, "subs_ge9_ET": 0.143, "subs_ge10_ET": 0.143,
}
_EVENT_BASE_PATH = Path("data/processed/event_baselines.json")

# any-player markets: P(some player has >=2 of the team's goals | team scored g) and the
# same for SoT — the "concentration" of a team's output onto one player (player_match,
# 314 matches, monotone-cleaned). any-player rides the team goal/SoT distribution we
# already model, so it lambda-scales (more team output -> more likely someone gets 2+).
BRACE_BY_TEAM_GOALS = [0.0, 0.0, 0.156, 0.42, 0.77, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0]
SOTCONC_BY_TEAM_SOT = [0.0, 0.0, 0.13, 0.33, 0.58, 0.71, 0.88, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
BOT_A_BASE_WEIGHT = 0.25

H1 = 1 - mk.SECOND_HALF_SHARE
H2 = mk.SECOND_HALF_SHARE

# Hydration/cooling-break windows: breaks at the 22' mark of EACH half, i.e. match
# minute 22 and 45+22=67 (platform definition). "before the first break" = [0,22);
# "after the second break" = [67,90) (+ stoppage). Match-level P(>=thr family events
# in the minute window), integrated analytically from the minute-profile (no MC
# change). archetype -> (dynamics family, slot_lo, slot_hi).
_WINDOW = {
    "goal_pre_hb1": ("goal", 0, 22), "goal_post_hb2": ("goal", 67, 90),
    "corners_pre_hb1": ("corner", 0, 22), "cards_post_hb2": ("card", 67, 90),
    "offsides_pre_hb1": ("offside", 0, 22),
    # "goal in H1 AFTER the first break" = [22,45); the post-break window carries MORE goal
    # intensity than [0,22) (the hazard rises through a half), so this is not simply ½·P(H1 goal).
    "goal_after_hb1": ("goal", 22, 45),
    # SF phrasing: between the two breaks = match minutes [22, 67) — the fattest window
    # (~55% of total goal intensity: late-H1 rise + all of early/mid H2)
    "goal_between_hbs": ("goal", 22, 67),
}
# New-template structural constants (2026-07-07 researcher):
CARD_STOPPAGE_SHARE = 0.15   # share of card mass in stoppage time (cards are late-loaded; the 90'
                             # profile truncates added time so this is a documented extrapolation) — LOW conf
CORNERS_SHOTS_RHO = 0.30     # WITHIN-match indicator corr of {more corners}&{more shots}. The count
                             # model treats corners _|_ shots given lambda (corr~0), but the RESIDUAL
                             # coupling BEYOND team quality is large: measured residual indicator corr
                             # 0.41 on 314 matches (per-team-baseline partial; raw within-team corr .60)
                             # -- a team having an attacking day gets BOTH. 0.30 sits below .41
                             # (conservative for the additive-copula linearization); lands an even
                             # match's joint ~.26. (2026-07-08 audit: the prior .05 under-coupled and
                             # was the one new handler positioned to LOSE to the crowd.)
_DYN_PATH = Path("data/processed/minute_dynamics.json")


def book_sot_ge_from_one_plus(p1: float, k: int) -> float | None:
    """Derive a player-differentiated P(>=k SoT) from the book's 1+ SoT (Over-0.5) line.

    The book only ever quotes the 0.5 line, i.e. P(>=1 SoT). Treating SoT as Poisson(mu),
    P(>=1) = 1 - e^-mu  =>  mu = -ln(1 - P(>=1)); then P(>=k) = 1 - Poisson.cdf(k-1, mu).
    This re-uses the book's player RANKING (elite shooters carry a high 1+ line, fringe
    players a low one) for the >=2/>=3 thresholds, instead of dropping the only
    player-differentiated signal we have and falling back to a flat undifferentiated model.

    Worked numbers (settled R32 props snapshot): Mbappe 1+=0.834 -> mu=1.80 -> P(>=2)=0.54
    (vs our flat 19, crowd 63, out=1); Sorloth 1+=0.658 -> mu=1.07 -> P(>=2)=0.29 (stays
    moderate, crowd 33, out=0); Isak 1+=0.478 -> mu=0.65 -> P(>=2)=0.14 (low, crowd 32).

    Returns None for k<=1 (caller uses the raw 1+ line directly) or a degenerate input."""
    if k <= 1:
        return None
    p1 = float(np.clip(p1, 1e-6, 1 - 1e-9))
    mu = -np.log(1.0 - p1)
    from scipy.stats import poisson
    return float(np.clip(1.0 - poisson.cdf(k - 1, mu), 0.0, 1.0))


def _norm_name(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _load_crowd_prem() -> dict:
    """Fitted per-archetype player-prop crowd premium (prob units); {} if unavailable."""
    try:
        return json.loads(PLAYER_CROWD_PREM_PATH.read_text()).get("premium", {})
    except (OSError, ValueError):
        return {}


def _load_lineup_status() -> dict:
    """Confirmed pre-match lineup status {norm_player: {status, team, ts, why}}; {} if unavailable."""
    try:
        return json.loads(LINEUP_STATUS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _load_crowd_template() -> dict:
    """Per-archetype settled-crowd level {arch: {n, mean, std}} (percent points); {} if unavailable."""
    try:
        return json.loads(CROWD_TEMPLATE_PATH.read_text())
    except (OSError, ValueError):
        return {}


@dataclass
class MatchContext:
    team_a: str          # listed first in match name
    team_b: str
    lam_a: float
    lam_b: float
    grid: np.ndarray
    rho_dc: float
    anchored: bool = False   # True if goal intensities were blended with book odds
    sim: object = None       # v2 MC SimResult (timing markets), at the same lambdas


class Runner:
    def __init__(self, pipeline, player_table: pd.DataFrame, mc=None, calibration: dict | None = None,
                 aggression: float = 1.0, w_market: float | None = None):
        self.pl = pipeline
        self.players = self._player_rates(player_table)
        self.player_transfer = PLAYER_TRANSFER
        self._load_club_rates()  # club-history priors for player props (data fix)
        self.mc = mc                    # optional v2 MCEngine for timing markets (rec #3)
        # minute profile for the hydration/stoppage window markets (reuse the MC's, or
        # load standalone so the analytic window handlers work even without an engine)
        self.dyn = mc.dyn if mc is not None else (Dynamics.from_json(_DYN_PATH) if _DYN_PATH.exists() else None)
        # stoppage/sub base rates (api-football WC2022, default to football norms)
        self.event_base = dict(EVENT_BASE_DEFAULTS)
        if _EVENT_BASE_PATH.exists():
            _d = json.loads(_EVENT_BASE_PATH.read_text())
            self.event_base.update({k: float(_d[k]) for k in EVENT_BASE_DEFAULTS if _d.get(k) is not None})
        self.calib = calibration if calibration is not None else self._load_calibration()
        from jumpprob.models.market_anchor import W_MARKET
        self.w_market = W_MARKET if w_market is None else w_market  # odds-blend weight (CLV passes set 0/1)
        # evidence-gated extremizing (rec #2). gate/base_rates default to inert.
        from jumpprob.models import strategy
        # effective gate = max(live edge gate, backtest prop-edge gate) per family
        live, bt = strategy.load_edge_gate(), strategy.load_backtest_gate()
        self.gate = {f: max(live.get(f, 1.0), bt.get(f, 1.0)) for f in set(live) | set(bt)}
        self.base_rates = strategy.load_base_rates()
        self.field_sharpness = strategy.load_field_sharpness()  # Phase C
        self.confidence = strategy.load_confidence_gate()       # Phase 1 asym. calibration
        self.risk = strategy.load_risk_gate()                   # Phase R de-confidence (off unless RISK_GATE_ENABLE)
        self.risk_on = strategy.risk_enabled()
        self.panel = panel_mod.load_panel()                     # L0 LLM expert cache (shadow by default)
        self.expert_gate = panel_mod.load_gate()                # L2 per-archetype blend gate (off unless AI_EXPERT_ENABLE)
        self.player_crowd_prem = _load_crowd_prem()             # P4b per-archetype crowd-premium lift
        self.lineup_status = _load_lineup_status()              # confirmed benched/out (near-KO WebPanel)
        self.crowd_template = _load_crowd_template()            # Phase-0 crowd levels (fallback + hug)
        import os as _os
        try:                                                    # Phase-2 endgame dial (default off)
            self.endgame_alpha = float(_os.environ.get("ENDGAME_EXTREMIZE", "1.0") or 1.0)
        except ValueError:
            self.endgame_alpha = 1.0
        self.aggression = aggression
        # league-average context for Bot A's base-rate blend
        lam0 = float(np.exp(self.pl.goal_model.b0))
        self._base_ctx = MatchContext("X", "Y", lam0, lam0,
                                      self.pl.goal_model.scoreline_grid(lam0, lam0),
                                      self.pl.goal_model.rho_dc)

    @staticmethod
    def _load_calibration() -> dict:
        p = Path("data/processed/live_calibration.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def _extremize(self, arch: str, fam: str | None, p: float, anchored: bool,
                   stage_mult: float = 1.0) -> float:
        """Evidence-gated extremizing (rec #2). No-op (returns p) unless the
        family's confirmed-edge gate grants k>1. Anchored direct-market goal
        questions are the floor — never extremized (we ARE the field there)."""
        from jumpprob.models import strategy
        if anchored and arch not in TIMING_ARCHS and fam == "goal":
            return p  # direct anchored market = calibrated floor
        # No-edge archetypes belong to the risk gate (stage 8), which shrinks them
        # toward base. Extremizing them here would amplify the very noise the risk
        # gate then has to undo, so the two layers fight (e.g. count earns k=1.3 but
        # offsides/fouls/both_teams have no edge). Leverage only the markets within a
        # family that actually earned it; leave the no-edge ones for the risk gate.
        if arch in self.risk:
            return p
        gate_fam = "timing" if arch in TIMING_ARCHS else fam
        if gate_fam is None:
            return p
        k = strategy.k_for(gate_fam, self.gate, stage_mult, self.aggression)
        if k == 1.0:
            return p
        # Phase C: structural field = belief shrunk toward 50 by family sharpness.
        field = strategy.field_model(p, self.field_sharpness.get(gate_fam, 0.5))
        return strategy.extremize(p, field, k)

    def _confidence(self, arch: str, p: float) -> float:
        """Phase 1 asymmetric calibration: floor confident-low bets per archetype
        (no-op where the gate has no entry, e.g. archetypes we have edge on)."""
        from jumpprob.models import strategy
        return strategy.apply_confidence(p, arch, self.confidence)

    def _risk(self, arch: str, p: float) -> float:
        """Phase R: shrink no-edge archetypes toward the base rate (no-op otherwise)."""
        from jumpprob.models import strategy
        return strategy.apply_risk(p, arch, self.risk, self.base_rates)

    def _lineup_override(self, subject: str | None, arch: str, teams) -> tuple[float, str] | None:
        """Bounded ceiling for a player confirmed benched/out THIS match, else None.

        Reads lineup_status.json (auto-written by the near-KO WebPanel). 'out' -> a hard
        low ceiling; 'bench' -> an elite-sub ceiling per archetype. Applied as a MIN by the
        caller, so it only ever lowers a starter-priced prop, never raises one — a bounded,
        one-sided safety override that beats the book anchor (the pm-guard otherwise drops it).
        Team-scoped + freshness-gated so a stale flag can't bleed into another match."""
        if not subject or not self.lineup_status:
            return None
        ent = self.lineup_status.get(norm_player(subject))
        if not ent:
            return None
        ts = ent.get("ts")
        if ts:
            try:
                if (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds() / 3600.0 > LINEUP_FRESH_H:
                    return None
            except Exception:
                pass
        team = ent.get("team")
        if team and teams and canonical(team) not in {canonical(t) for t in teams if t}:
            return None
        status = str(ent.get("status", "")).lower()
        if status == "out":
            return LINEUP_OUT_CEIL, status
        if status in ("bench", "benched") and arch in LINEUP_BENCH_CAP:
            return LINEUP_BENCH_CAP[arch], "bench"
        return None

    def _window_event(self, ctx: MatchContext, fam: str, lo: int, hi: int, thr: int) -> float | None:
        """Match-level P(>=thr `fam` events in slots [lo,hi)). The minute profile's
        bucket weights give the window's share of the family's intensity; scale the
        match's expected total by that share. Goals are Poisson; counts inherit the
        count model's dispersion via cm.pmf."""
        if self.dyn is None or fam not in self.dyn.alpha:
            return None
        w = np.exp(np.asarray(self.dyn.alpha[fam])[BUCKET])
        frac = float(w[lo:hi].sum() / w.sum())
        if fam == "goal":
            return float(1 - np.exp(-(ctx.lam_a + ctx.lam_b) * frac))   # all are "a goal" (thr=1)
        cm = self.pl.count_model
        if cm is None:
            return None
        cf = mc_engine.COUNT_FAMILY[fam]
        mu_tot = sum(cm.mu(cf, h, ctx.lam_a, ctx.lam_b) + cm.mu(cf, h, ctx.lam_b, ctx.lam_a) for h in (1, 2))
        return float(cm.pmf(cf, mu_tot * frac)[thr:].sum())

    def _family(self, arch: str) -> str | None:
        # offsides get their OWN calibration: a StatsBomb→SportsPredict definition
        # gap makes them over-priced live (the pooled count Platt shifts them the
        # WRONG way). Must precede the COUNT_ARCHS check (#2).
        if arch == "offsides_ge":
            return "offsides"
        if arch in COUNT_ARCHS:
            return "count"
        if arch in PLAYER_ARCHS:
            return "player"
        if arch in GOAL_ARCHS:
            return "goal"
        return None  # constants (penalty/red) — not calibrated

    def calibrate(self, arch: str, p: float, anchored: bool) -> float:
        """Apply per-family Platt (rec #1). Goal markets skip calibration when
        anchored (the market is already calibrated); count/player always."""
        fam = self._family(arch)
        if fam is None or fam not in self.calib:
            return p
        if fam == "goal" and anchored:
            return p
        a, b = self.calib[fam]
        return float(inv_logit(a * logit(p) + b))

    def _player_rates(self, pm: pd.DataFrame) -> pd.DataFrame:
        pm = pm.assign(ga=pm["goals"] + pm.get("assists", 0))
        tot = pm.groupby("player")[["minutes", "sot", "goals", "ga"]].sum()
        prior_sot = pm["sot"].sum() / pm["minutes"].sum()
        prior_goal = pm["goals"].sum() / pm["minutes"].sum()
        prior_ga = pm["ga"].sum() / pm["minutes"].sum()
        tot["rate_sot"] = (tot["sot"] + prior_sot * 270) / (tot["minutes"] + 270)
        tot["rate_goal"] = (tot["goals"] + prior_goal * 540) / (tot["minutes"] + 540)
        tot["rate_ga"] = (tot["ga"] + prior_ga * 540) / (tot["minutes"] + 540)
        tot["norm"] = [_norm_name(p) for p in tot.index]
        self.prior_sot, self.prior_goal, self.prior_ga = prior_sot, prior_goal, prior_ga
        return tot

    def _load_club_rates(self) -> None:
        """Per-minute club rates (sot, npg, ga) from the combined Understat+API-Football
        table, used as a player-specific PRIOR. Replaces the global population prior so
        established attackers are no longer shrunk to the average (the systematic
        under-pricing the settled-match audit found). SoT estimated from shots where
        Understat lacks it. Inert (falls back to population prior) if the table is absent."""
        self.club_rates: dict[str, tuple] = {}
        self.club_idx = None
        if not CLUB_RATES.exists():
            return
        df = pd.read_parquet(CLUB_RATES).drop_duplicates(subset="player_name", keep="first")
        for r in df.itertuples():
            sot90 = getattr(r, "sot_p90", None)
            if sot90 is None or (isinstance(sot90, float) and np.isnan(sot90)):
                sot90 = float(getattr(r, "shots_p90", 0) or 0) * SOT_PER_SHOT
            npg90 = float(getattr(r, "npg_p90", 0) or 0)
            ga90 = npg90 + float(getattr(r, "assists_p90", 0) or 0)
            self.club_rates[r.player_name] = (sot90 / 90.0, npg90 / 90.0, ga90 / 90.0)
        self.club_idx = _club_index_fn(df["player_name"])

    def _club_prior(self, name: str) -> tuple | None:
        """Resolve a market player name to their per-minute club rates (or None)."""
        if self.club_idx is None:
            return None
        match, _ = _club_resolve(name, *self.club_idx)
        return self.club_rates.get(match) if match else None

    def player_rate(self, name: str) -> tuple[float, float, float]:
        """Per-minute (sot, goal, ga) rates, Gamma-Poisson-shrunk off a PLAYER-CENTERED prior.

        Phase-0 endgame fix (2026-07-10): ALL THREE priors now use the transfer-discounted
        CLUB rate when the player resolves (was GA-only; SoT/scores sat on the POPULATION
        mean). The population prior + M0=270/540 shrinkage is 3-6 full matches of pull toward
        "average WC player" — textbook over-shrinkage of exactly the elite outliers the KO
        '(Player, #N) to score' questions are about. Measured leak: our player prices sat
        -18..-28pt under the crowd, the gap SCALING with crowd level (elite signature);
        Mbappe/Haaland-class no-book props collapsed to ~8-24. Re-centering the prior on the
        player fixes the MODEL side at the source (works with no book and no crowd present).
        The earlier "club over-predicts SoT" disable was an n-tiny call made WITHOUT the 0.80
        transfer discount; revisit with Phase-1 full-harvest data.
        Sets self._pr_resolved: False = neither club prior nor tournament minutes matched
        (we know NOTHING about this player) -> the caller blends toward the archetype crowd
        template instead of shipping the naked population floor (the De Bruyne/DR-Congo/
        Haaland-8% catastrophic-tail class)."""
        # Strip a '(Team, #N)' subject suffix before resolving. The QF/SF question format is
        # "Kylian Mbappé (France, #10)"; the parenthetical's tokens ('(france,', '#10)') match no
        # player, so BOTH the club-prior resolve and the token-match miss and the best striker on
        # the pitch collapses to the population prior (Mbappé scores 8 not ~32). The book/lineup
        # paths already strip this via norm_player — this is the same name-join bug class as
        # De Bruyne / Haaland / DR Congo, closed here for the model path.
        name = name.split("(")[0].strip() or name
        club = self._club_prior(name)
        prior_sot = self.player_transfer * club[0] if club else self.prior_sot
        prior_goal = self.player_transfer * club[1] if club else self.prior_goal
        prior_ga = self.player_transfer * club[2] if club else self.prior_ga
        tokens = _norm_name(name).split()
        hits = self.players[self.players["norm"].apply(lambda n: all(t in n for t in tokens))]
        self._pr_resolved = bool(club) or len(hits) == 1
        if len(hits) == 1:
            h = hits.iloc[0]
            m = float(h["minutes"])
            return ((float(h["sot"]) + prior_sot * 270) / (m + 270),
                    (float(h["goals"]) + prior_goal * 540) / (m + 540),
                    (float(h["ga"]) + prior_ga * 540) / (m + 540))
        return prior_sot, prior_goal, prior_ga   # no tournament appearances -> the prior

    # ------------------------------------------------------------------
    def resolve_teams(self, match_name: str, questions: list[str]) -> tuple[str, str] | None:
        known = set(self.pl.ratings.index)
        found: list[str] = []
        for q in questions:
            p = parse_question(q)
            if p is None:
                continue
            for nm in (p.subject, p.other):
                if nm:
                    c = canonical(nm)
                    if c in known and c not in found:
                        found.append(c)
        if len(found) == 2:
            # order them to match the "XXX vs YYY" listing when possible
            left = match_name.split(" vs ")[0].strip()
            first = CODE_MAP.get(left, left)
            first = canonical(first)
            if canonical(found[1]) == first:
                found = [found[1], found[0]]
            return found[0], found[1]
        parts = [p.strip() for p in match_name.split(" vs ")]
        if len(parts) == 2:
            a = canonical(CODE_MAP.get(parts[0], parts[0]))
            b = canonical(CODE_MAP.get(parts[1], parts[1]))
            if a in known and b in known:
                return a, b
        return None

    def context(self, team_a: str, team_b: str,
                anchor: MatchAnchor | None = None) -> MatchContext:
        """Build the match context. With an odds anchor, blend the model's goal
        intensities toward the book consensus (Phase 7) so every downstream
        answer — win, totals, half-goals, counts, props — inherits the market
        view coherently. Without one, pure model (unchanged behaviour)."""
        ra, rb = self.pl.ratings[team_a], self.pl.ratings[team_b]
        la, lb = self.pl.goal_model.intensities(ra, rb)
        rho = self.pl.goal_model.rho_dc
        anchored = False
        if anchor is not None:
            # orient the anchor to (team_a, team_b): the odds feed's home/away
            # need not match the SportsPredict match-name order.
            if anchor.home == team_a:
                p_a_win, p_b_win = anchor.p_home, anchor.p_away
            elif anchor.away == team_a:
                p_a_win, p_b_win = anchor.p_away, anchor.p_home
            else:
                p_a_win = p_b_win = None  # team mismatch -> no anchor
            if p_a_win is not None:
                la_mkt, lb_mkt = implied_lambdas(
                    p_a_win, anchor.p_draw, p_b_win, anchor.p_over25, rho=rho)
                # guard: ignore a pathological inversion, fall back to model lambdas
                if 0.05 <= la_mkt <= 6.0 and 0.05 <= lb_mkt <= 6.0:
                    la, lb = blend_lambdas((la, lb), (la_mkt, lb_mkt), self.w_market)
                    anchored = True
        # v2 MC simulation at the SAME (possibly anchored) lambdas, for the
        # timing markets the analytic engine only approximates (rec #3).
        sim = self.mc.simulate(0.0, 0.0, lam=(la, lb)) if self.mc is not None else None
        return MatchContext(team_a, team_b, la, lb,
                            self.pl.goal_model.scoreline_grid(la, lb), rho, anchored, sim)

    # ------------------------------------------------------------------
    def answer(self, q: ParsedQuestion, ctx: MatchContext) -> float | None:
        cm = self.pl.count_model
        side = None
        if q.subject:
            c = canonical(q.subject)
            side = "a" if c == ctx.team_a else "b" if c == ctx.team_b else None

        # rec #3: timing markets come from the v2 MC simulator (coherent, exact)
        # rather than the analytic approximations below, when a sim is available.
        if ctx.sim is not None and q.archetype in _MC_TIMING:
            return mc_engine.answer(ctx.sim, _MC_TIMING[q.archetype], side=side,
                                    threshold=q.threshold, half=q.half)

        def lams(s):  # (team, opp) intensities for side s
            return (ctx.lam_a, ctx.lam_b) if s == "a" else (ctx.lam_b, ctx.lam_a)

        def cpmf(fam, half, s):
            lt, lo = lams(s)
            return cm.pmf(fam, cm.mu(fam, half, lt, lo))

        def match_pmf(fam, s):
            return np.convolve(cpmf(fam, 1, s), cpmf(fam, 2, s))

        a = q.archetype
        if a in _WINDOW:
            return self._window_event(ctx, *_WINDOW[a], q.threshold or 1)
        if a in ("goal_1h_stoppage", "goal_2h_stoppage"):
            # stoppage goals beyond the 90-min profile: lambda_W = lambda_tot * goal-share
            share = self.event_base["goal_share_1h_stoppage" if a == "goal_1h_stoppage"
                                    else "goal_share_2h_stoppage"]
            return float(1 - np.exp(-(ctx.lam_a + ctx.lam_b) * share))
        if a == "sub_before_ht":
            return float(self.event_base["sub_before_ht"])          # ~injury base rate
        if a == "sub_at_ht":
            return float(self.event_base["sub_at_ht"])              # HT-interval change, WC2022 33/64
        if a in ("substitute_scores", "substitute_ga"):
            # subs come on ~72' and score ~17% of all goals (WC2022 events, lineup-verified),
            # so the chance a sub scores tracks the match's goal expectation -> scale the base
            # rate by this match's total goals vs the league average (blowout > tight game).
            lam0 = 2.0 * float(np.exp(self.pl.goal_model.b0))   # league-average total goals
            s = -np.log(1.0 - self.event_base["substitute_scores"]) / lam0
            if a == "substitute_ga":
                # "score OR ASSIST": widen the scoring hazard by the population GA/goal rate
                # ratio (assists arrive at a similar clip to goals; capped at 2x). Structural,
                # not fit — the score-or-assist phrasing first appeared in the QF round.
                s *= min(self.prior_ga / max(self.prior_goal, 1e-9), 2.0)
            return float(1 - np.exp(-(ctx.lam_a + ctx.lam_b) * s))
        if a == "any_player_goals_ge":   # "any player score 2+ goals" — over the goal grid
            if (q.threshold or 2) > 2:   # brace curve is P(some player >=2 | team goals); no 3+ data
                return None
            def _team_any(marg):
                return float(sum(marg[g] * (BRACE_BY_TEAM_GOALS[g] if g < len(BRACE_BY_TEAM_GOALS) else 1.0)
                                 for g in range(len(marg))))
            return float(1 - (1 - _team_any(ctx.grid.sum(axis=1))) * (1 - _team_any(ctx.grid.sum(axis=0))))
        if a == "win":
            return None if side is None else mk.p_win(ctx.grid, side)
        if a == "draw":
            return mk.p_draw(ctx.grid)
        if a == "first_goal_match":
            # P(side scores the match's first goal) = P(side scores first | any goal)
            # x P(>=1 goal). Inherits the market view via the (anchored) intensities.
            if side is None:
                return None
            lt, _ = lams(side)
            tot = ctx.lam_a + ctx.lam_b
            return float(lt / tot * (1 - np.exp(-tot)))
        if a == "win_margin":
            if side is None:
                return None
            n = q.threshold or 1
            idx = np.arange(ctx.grid.shape[0])
            margin = np.subtract.outer(idx, idx)            # rows(a) - cols(b)
            mask = (margin >= n) if side == "a" else (margin <= -n)
            return float(ctx.grid[mask].sum())
        if a == "clean_sheet":                              # opponent scores 0
            if side is None:
                return None
            return float(ctx.grid[:, 0].sum()) if side == "a" else float(ctx.grid[0, :].sum())
        if a == "score_both_halves":
            if side is None:
                return None
            lt, _ = lams(side)
            return float(mk.p_scores_in_half(lt, 1) * mk.p_scores_in_half(lt, 2))
        if a == "advance":
            # advance to the next round = win THIS knockout tie:
            #   P(adv) = P(win 90) + P(draw 90) * P(advance | drawn after 90),
            #   P(advance | draw) = P(win ET) + P(ET also drawn) * P(win shootout).
            # ET = a low-tempo 30-min grid off the same (anchored) intensities; the
            # shootout is ~a coin flip with a mild favorite tilt (see ET_TEMPO/PEN_TILT).
            if side is None:
                return None
            other = "b" if side == "a" else "a"
            p_win90, p_draw90, p_lose90 = mk.p_win(ctx.grid, side), mk.p_draw(ctx.grid), mk.p_win(ctx.grid, other)
            lt, lo = lams(side)
            et = dc_grid(ET_TEMPO / 3 * lt, ET_TEMPO / 3 * lo, ctx.rho_dc)
            p_win_et, p_et_draw = mk.p_win(et, "a"), mk.p_draw(et)
            r = p_win90 / (p_win90 + p_lose90) if (p_win90 + p_lose90) > 0 else 0.5
            s_pen = 0.5 + PEN_TILT * (r - 0.5)
            q = p_win_et + p_et_draw * s_pen
            return float(p_win90 + p_draw90 * q)
        # --- QF templates (2026-07-05; docs/QF-TEMPLATES-PLAN.md) ---
        if a == "extra_time":
            # ET <=> drawn after 90; the anchored grid's draw mass ~= the de-vigged book X
            return float(mk.p_draw(ctx.grid))
        if a == "halves_equal_goals":
            # P(G1h == G2h) = sum_N P(total=N) x P(N splits equally). Given N, each goal
            # falls in H1 iid w.p. H1 (the goal share): P(equal | N=2k) = Binom(2k, H1).pmf(k);
            # zero for odd N. Binomial split validated on 11,658+ martj42 matches (plan s1).
            from scipy.stats import binom
            g = ctx.grid; n = g.shape[0]
            tg = [float(sum(g[i, k - i] for i in range(k + 1) if k - i < n and i < n))
                  for k in range(2 * n - 1)]
            return float(sum(pg * binom.pmf(k // 2, k, H1) for k, pg in enumerate(tg) if k % 2 == 0))
        if a == "subs_total_ge":
            # Regulation-only subs are quota consumption, NOT a count process (WC2022
            # var/mean 0.175): stratified empirical exceedance mixed by P(ET) — ET-bound
            # matches HOLD subs back (reg-subs 6,8,8,5,6 in the five WC2022 ET matches).
            key_ne = "subs_ge%d_nonET" % (q.threshold or 0)
            key_et = "subs_ge%d_ET" % (q.threshold or 0)
            if key_ne not in self.event_base or key_et not in self.event_base:
                return None                      # un-tabled threshold -> template audit
            pd_ = float(mk.p_draw(ctx.grid))
            return float((1 - pd_) * self.event_base[key_ne] + pd_ * self.event_base[key_et])
        if a == "player_saves_ge":
            # saves == SoT-against - goals-against (identity exact in 628/628 StatsBomb
            # matches): P(saves >= t) = sum_s P(SoT_opp = s) x P(Bin(s, 1-conv) >= t), with
            # the count model's overdispersed SoT pmf and conv = lam_opp / mu_SoT_opp.
            # Keeper quality rides the (anchored) lambdas; no GK-saves book market exists.
            kteam = canonical(q.other) if q.other else None
            kside = "a" if kteam == ctx.team_a else "b" if kteam == ctx.team_b else None
            if kside is None:
                return None
            from scipy.stats import binom
            cm = self.pl.count_model
            if "shots_on_target" not in cm.params:
                return None
            lam_opp, lam_own = (ctx.lam_b, ctx.lam_a) if kside == "a" else (ctx.lam_a, ctx.lam_b)
            mu_opp = sum(cm.mu("shots_on_target", h, lam_opp, lam_own) for h in (1, 2))
            spmf = cm.pmf("shots_on_target", mu_opp)
            mu_sot = float(sum(k2 * w2 for k2, w2 in enumerate(spmf)))
            conv = min(0.9, lam_opp / mu_sot) if mu_sot > 0 else 0.3
            t = q.threshold or 1
            return float(sum(w2 * float(binom.sf(t - 1, k2, 1 - conv))
                             for k2, w2 in enumerate(spmf) if k2 >= t))
        if a == "first_card_before_goal":
            # Time-varying competing-risks race on the minute profiles: cards are LATE-
            # loaded vs goals, so the constant-rate answer overstates card-first by ~9 pts
            # (plan s2). Settlement ambiguity (card but no goal ever): permissive reading
            # (counts as YES) weighted 0.6 vs strict (a goal must follow) 0.4.
            if self.dyn is None or "card" not in self.dyn.alpha or "goal" not in self.dyn.alpha:
                return None
            cm = self.pl.count_model
            mu_c = sum(cm.mu("cards", h, ctx.lam_a, ctx.lam_b) + cm.mu("cards", h, ctx.lam_b, ctx.lam_a)
                       for h in (1, 2))
            lam_t = ctx.lam_a + ctx.lam_b
            wc = np.exp(np.asarray(self.dyn.alpha["card"])[BUCKET]); wc = wc / wc.sum()
            wg = np.exp(np.asarray(self.dyn.alpha["goal"])[BUCKET]); wg = wg / wg.sum()
            hc, hg = mu_c * wc, lam_t * wg
            S = np.exp(-(np.cumsum(hc + hg) - (hc + hg)))          # survival to slot start
            p_perm = float(np.sum(hc * S))                          # card first (goal may never come)
            tail_g = np.concatenate([np.cumsum(hg[::-1])[::-1][1:], [0.0]])
            p_strict = float(np.sum(hc * S * (1 - np.exp(-tail_g))))
            return float(np.clip(0.6 * p_perm + 0.4 * p_strict, 0.01, 0.97))
        # --- R16 templates (2026-07-04, caught by the P5 template audit) ---
        if a == "pen_shootout":
            # shootout <=> drawn after 90 AND ET also drawn — same ET grid as `advance`
            p_draw90 = mk.p_draw(ctx.grid)
            et = dc_grid(ET_TEMPO / 3 * ctx.lam_a, ET_TEMPO / 3 * ctx.lam_b, ctx.rho_dc)
            return float(p_draw90 * mk.p_draw(et))
        if a == "total_exact":
            k = q.threshold or 0
            g = ctx.grid
            return float(sum(g[i, k - i] for i in range(k + 1) if k - i < g.shape[1] and i < g.shape[0]))
        if a == "goal_each_half":
            # >=1 goal (either team) in EACH half: the match-level version of
            # score_both_halves, same half-share thinning of the total intensity.
            lam_t = ctx.lam_a + ctx.lam_b
            return float((1 - np.exp(-H1 * lam_t)) * (1 - np.exp(-H2 * lam_t)))
        if a == "win_both_halves":
            # EITHER team wins BOTH halves (strictly more goals each half). Build a grid per
            # half at the share-thinned intensities and combine assuming halves independent
            # (MC-confirmed: goal frailty across halves is negligible). Crowd over-prices this
            # (~.35 "if one team's better it wins both") vs a true ~.18.
            g1 = dc_grid(H1 * ctx.lam_a, H1 * ctx.lam_b, ctx.rho_dc)
            g2 = dc_grid(H2 * ctx.lam_a, H2 * ctx.lam_b, ctx.rho_dc)
            return float(mk.p_win(g1, "a") * mk.p_win(g2, "a")
                         + mk.p_win(g1, "b") * mk.p_win(g2, "b"))
        if a == "first_goal_2h":
            # first goal falls in H2 <=> NO goal in H1 AND >=1 in H2 (0-0 => NO). Per-half goal
            # independence is the engine convention; half-share measured on 47k goals.
            lam_t = ctx.lam_a + ctx.lam_b
            return float(np.exp(-H1 * lam_t) * (1 - np.exp(-H2 * lam_t)))
        if a == "margin_exactly":
            # decided by EXACTLY n goals (|a-b| == n; a draw is NO). Pure grid sum; the
            # crowd tends to conflate "one-goal game" with "not a draw" (~.73).
            n = q.threshold or 1
            idx = np.arange(ctx.grid.shape[0])
            return float(ctx.grid[np.abs(np.subtract.outer(idx, idx)) == n].sum())
        if a == "total_odd":
            # odd total goals. Certain-but-tiny EVEN tilt vs a crowd that anchors 0.50
            # (independent Poissons give P(odd)=(1-e^-2L)/2 < .5; DC rho<0 tilts further even).
            g = ctx.grid
            idx = np.arange(g.shape[0])
            return float(g[np.add.outer(idx, idx) % 2 == 1].sum())
        if a == "cards_gt_goals":
            # P(total cards > total goals): count-model cards pmf x the grid's total-goals
            # pmf (independence — cards/goals correlation is weak and second-order here).
            cm = self.pl.count_model
            if "cards" not in cm.params:
                return None
            mu_c = sum(cm.mu("cards", h, ctx.lam_a, ctx.lam_b) + cm.mu("cards", h, ctx.lam_b, ctx.lam_a)
                       for h in (1, 2))
            cpmf = cm.pmf("cards", mu_c)
            g = ctx.grid
            n = g.shape[0]
            tg = [float(sum(g[i, k - i] for i in range(k + 1) if k - i < n and i < n)) for k in range(2 * n - 1)]
            return float(sum(pg * float(cpmf[k + 1:].sum()) for k, pg in enumerate(tg) if k + 1 < len(cpmf)))
        if a == "leads_at_any_point":
            # P(team is strictly ahead at some point in 90') ~= P(team scores FIRST) plus a
            # small comeback allowance (fall behind, then go ahead — rare; MC-quality path
            # dynamics aren't warranted for one question). Crude but bounded and monotone.
            if side is None:
                return None
            lt, lo = lams(side)
            lam_t = lt + lo
            p_first = (lt / lam_t) * (1 - np.exp(-lam_t)) if lam_t > 0 else 0.0
            # comeback allowance 0.05 (martj42: .114 fall-behind-then-lead events/match ->
            # ~+.05 per side) and the question EXCLUDES only the shootout, so a first lead
            # in EXTRA TIME counts: + P(tied at FT) x share x P(>=1 ET goal). Review §3.3.
            p_et = float(mk.p_draw(ctx.grid)) * (lt / lam_t if lam_t > 0 else 0.5) *                    (1 - np.exp(-ET_TEMPO * lam_t / 3))
            return float(np.clip(p_first + 0.05 + p_et, 0.01, 0.97))
        if a == "plays_full_match":
            # P(a named starter plays all 90'): ~5 subs over 10 outfield starters -> ~52%
            # survive; attacking players are subbed MORE (tactics/fatigue) -> base 0.42.
            # No book market; LLM fitness/rotation nuance can be gated in later.
            return float(self.event_base["player_full90"])
        if a == "btts":
            return mk.p_btts(ctx.grid)
        if a == "total_ge":
            return mk.p_total_ge(ctx.grid, q.threshold)
        if a == "btts_and_total_ge":
            return mk.p_btts_and_total_ge(ctx.grid, q.threshold)
        if a == "team_scores_half":
            if side is None:
                return None
            lt, _ = lams(side)
            return mk.p_scores_in_half(lt, q.half)
        if a == "ht_tied":
            return float(np.trace(dc_grid(H1 * ctx.lam_a, H1 * ctx.lam_b, ctx.rho_dc)))
        if a == "ht_scoreless":   # 0-0 at HT = the [0,0] cell of the half grid (QF phrasing)
            return float(dc_grid(H1 * ctx.lam_a, H1 * ctx.lam_b, ctx.rho_dc)[0, 0])
        if a == "half_total_ge":
            share = H1 if q.half == 1 else H2
            lam = share * (ctx.lam_a + ctx.lam_b)
            from scipy.stats import poisson
            return float(1 - poisson.cdf(q.threshold - 1, lam))
        if a == "penalty_awarded":
            return P_PEN_MATCH
        if a == "pen_or_red":
            return 1 - (1 - P_PEN_MATCH) * (1 - P_RED_MATCH)
        if a == "red_card":
            return P_RED_MATCH
        if a == "own_goal":                          # near-random -> flat WC base rate (thinning tested worse)
            return float(self.event_base["own_goal"])
        if a == "goal_outside_box":                  # ~13% of open-play goals; Poisson thinning of the tempo
            return float(1.0 - np.exp(-self.event_base["outside_box_share"] * (ctx.lam_a + ctx.lam_b)))
        if a == "goal_header":                       # ~22% of open-play goals are headers; validated thinning
            return float(1.0 - np.exp(-self.event_base["header_share"] * (ctx.lam_a + ctx.lam_b)))
        if a == "total_le":
            return 1.0 - mk.p_total_ge(ctx.grid, q.threshold + 1)
        if a == "team_goals_ge":
            if side is None:
                return None
            marg = ctx.grid.sum(axis=1) if side == "a" else ctx.grid.sum(axis=0)
            return float(marg[q.threshold:].sum())
        if a == "ht_leading":
            if side is None:
                return None
            return mk.p_win(dc_grid(H1 * ctx.lam_a, H1 * ctx.lam_b, ctx.rho_dc), side)
        if a == "goals_gt_half":
            if side is None:
                return None
            share = H1 if q.half == 1 else H2
            return mk.p_win(dc_grid(share * ctx.lam_a, share * ctx.lam_b, ctx.rho_dc), side)
        if a == "first_goal_half":
            if side is None:
                return None
            lt, lo = lams(side)
            share = H1 if q.half == 1 else H2
            tot = ctx.lam_a + ctx.lam_b
            return float(lt / tot * (1 - np.exp(-share * tot)))
        if a == "first_goal_and_win":
            # P(first) x P(win | scored first), where the conditional is the
            # team's own win prob shifted by the measured first-goal boost:
            # logit shift calibrated so an average first-scorer wins 75.7%
            # (measured on 15,381 internationals, 2026-06-12).
            if side is None:
                return None
            lt, lo = lams(side)
            tot = ctx.lam_a + ctx.lam_b
            p_first = lt / tot * (1 - np.exp(-tot))
            p_win = mk.p_win(ctx.grid, side)
            z = np.log(p_win / (1 - p_win)) + 1.34
            return float(p_first / (1 - np.exp(-tot)) * (1 / (1 + np.exp(-z))) * (1 - np.exp(-tot)))
        if a == "first_goal_and_opp_2h":
            # P(subject scores first) x P(other scores in 2H) — treated as
            # independent given intensities (weak residual coupling).
            if side is None or q.other is None:
                return None
            other_c = canonical(q.other)
            other_side = "a" if other_c == ctx.team_a else "b" if other_c == ctx.team_b else None
            if other_side is None:
                return None
            lt, lo = lams(side)
            tot = ctx.lam_a + ctx.lam_b
            p_first = lt / tot * (1 - np.exp(-tot))
            lo_t, _ = lams(other_side)
            return float(p_first * mk.p_scores_in_half(lo_t, 2))
        if a == "h2_more_goals_than_h1":
            from scipy.stats import poisson
            tot = ctx.lam_a + ctx.lam_b
            xs = np.arange(25)
            return p_a_gt_b(poisson.pmf(xs, H2 * tot), poisson.pmf(xs, H1 * tot))
        if cm is None:
            return None
        def side_pmf(fam, s):
            """pmf for side s at q.half, or match-level when half is None."""
            return cpmf(fam, q.half, s) if q.half else match_pmf(fam, s)

        if a in ("corners_gt", "sot_gt", "fouls_gt", "cards_gt"):
            if side is None:
                return None
            fam = {"corners_gt": "corners", "sot_gt": "shots_on_target",
                   "fouls_gt": "fouls", "cards_gt": "cards"}[a]
            other = "b" if side == "a" else "a"
            return p_a_gt_b(side_pmf(fam, side), side_pmf(fam, other))
        if a == "corners_and_shots":
            # P(side has MORE corners AND MORE total shots than opp). Both indicators load on
            # attacking intensity, so they are weakly POSITIVELY correlated — a Gaussian-copula
            # bump over the independence product (measured indicator corr ~0.05 -> +~1.2pt).
            # Most count variance is idiosyncratic, so DON'T over-correct toward strong dependence.
            if side is None:
                return None
            other = "b" if side == "a" else "a"
            p_c = p_a_gt_b(side_pmf("corners", side), side_pmf("corners", other))
            p_s = p_a_gt_b(side_pmf("shots", side), side_pmf("shots", other))
            bump = CORNERS_SHOTS_RHO * float(np.sqrt(max(p_c * (1 - p_c) * p_s * (1 - p_s), 0.0)))
            return float(np.clip(p_c * p_s + bump, 0.01, 0.99))
        if a == "card_each_half":
            # >=1 card in EACH half. cm.mu already carries the fitted H1/H2 card split (cards are
            # 2nd-half-loaded, ~.36/.64), so read P(>=1) off each half's card pmf; halves ~indep
            # (a shared-frailty bump is <1pt). Low-edge (WC26 cards run low) -> raw structural,
            # NOT count-Platt (which shifts up); the P1 clamp bounds any crowd deviation.
            def _card_half_mu(h):
                return cm.mu("cards", h, ctx.lam_a, ctx.lam_b) + cm.mu("cards", h, ctx.lam_b, ctx.lam_a)
            p1 = 1.0 - float(cm.pmf("cards", _card_half_mu(1))[0])
            p2 = 1.0 - float(cm.pmf("cards", _card_half_mu(2))[0])
            return float(p1 * p2)
        if a == "card_stoppage":
            # >=1 card in either half's stoppage time. Cards are late-loaded so stoppage carries
            # more than its raw minute share; E[stoppage cards] ~= mu_card * CARD_STOPPAGE_SHARE.
            # LOW confidence (WC26 card level + stoppage-minute mass outside the 90' profile) ->
            # conservative, raw-structural, P1-clamp-protected.
            mu_c = sum(cm.mu("cards", h, ctx.lam_a, ctx.lam_b) + cm.mu("cards", h, ctx.lam_b, ctx.lam_a)
                       for h in (1, 2))
            return float(1.0 - np.exp(-mu_c * CARD_STOPPAGE_SHARE))
        if a == "team_corners_ge":
            return None if side is None else float(side_pmf("corners", side)[q.threshold:].sum())
        if a == "team_sot_ge":
            return None if side is None else float(side_pmf("shots_on_target", side)[q.threshold:].sum())
        if a == "team_cards_ge":
            return None if side is None else float(side_pmf("cards", side)[q.threshold:].sum())
        if a == "cards_ge":
            return p_sum_ge(side_pmf("cards", "a"), side_pmf("cards", "b"), q.threshold)
        if a == "corners_total_ge":
            return p_sum_ge(side_pmf("corners", "a"), side_pmf("corners", "b"), q.threshold)
        if a in ("total_sot_half_ge", "total_sot_ge"):
            return p_sum_ge(side_pmf("shots_on_target", "a"), side_pmf("shots_on_target", "b"), q.threshold)
        if a == "both_teams_sot_ge":
            # P(team A >= thr SoT AND team B >= thr SoT); independence given
            # intensities (same conditional-independence used elsewhere)
            pa = float(side_pmf("shots_on_target", "a")[q.threshold:].sum())
            pb = float(side_pmf("shots_on_target", "b")[q.threshold:].sum())
            return pa * pb
        if a == "offsides_ge":
            return None if side is None else float(side_pmf("offsides", side)[q.threshold:].sum())
        if a == "total_shots_ge":      # total shots (on+off) — modeled from real data, not SoT/0.37
            return p_sum_ge(side_pmf("shots", "a"), side_pmf("shots", "b"), q.threshold)
        if a == "offsides_total_ge":   # both teams' offside calls combined
            return p_sum_ge(side_pmf("offsides", "a"), side_pmf("offsides", "b"), q.threshold)
        if a == "both_teams_cards":    # P(A>=thr cards AND B>=thr cards), independence given intensities
            pa = float(side_pmf("cards", "a")[q.threshold:].sum())
            pb = float(side_pmf("cards", "b")[q.threshold:].sum())
            return pa * pb
        if a == "any_player_sot_ge":   # "any player 2+ SoT" — over each team's SoT distribution
            def _team_any_sot(s):
                pmf = side_pmf("shots_on_target", s)
                return float(sum(pmf[t] * (SOTCONC_BY_TEAM_SOT[t] if t < len(SOTCONC_BY_TEAM_SOT) else 1.0)
                                 for t in range(len(pmf))))
            return float(1 - (1 - _team_any_sot("a")) * (1 - _team_any_sot("b")))
        if a == "any_team_player_sot_ge":  # team-scoped: P(some <team> player gets 2+ SoT)
            if side is None or (q.threshold or 2) != 2:
                return None                # concentration curve is fit for the 2+ line only
            pmf2 = side_pmf("shots_on_target", side)
            return float(sum(pmf2[t] * (SOTCONC_BY_TEAM_SOT[t] if t < len(SOTCONC_BY_TEAM_SOT) else 1.0)
                             for t in range(len(pmf2))))
        if a in ("player_scores", "player_sot_ge", "player_ga"):
            rate_sot, rate_goal, rate_ga = self.player_rate(q.subject)
            rate = {"player_sot_ge": rate_sot, "player_scores": rate_goal, "player_ga": rate_ga}[a]
            # opponent adjustment: scale by this match's intensity vs league avg
            scale = (ctx.lam_a + ctx.lam_b) / (2 * np.exp(self.pl.goal_model.b0))
            # half-specific player questions: exposure = minutes x half share
            if q.half:
                scale *= H1 if q.half == 1 else H2
            # threshold-aware: "2 or more shots on target" -> P(>=k), not P(>=1).
            # scores/ga are >=1 questions (k defaults to 1, where 1 - Poisson.cdf(0, mu)
            # == 1 - exp(-mu), i.e. identical to the prior formula -> no behaviour change).
            from scipy.stats import poisson
            k = q.threshold or 1
            p = float(sum(w * (1.0 - poisson.cdf(k - 1, rate * scale * m)) for w, m in MINUTE_SCENARIOS))
            # Phase-0: an UNRESOLVED player (no club prior, no tournament minutes — we know
            # NOTHING about them) is priced off the population floor; blend toward the
            # archetype's crowd-template level instead (the Haaland-8% tail class). k>=2
            # lines excluded: the template mixes thresholds and reads as a 1+-line level.
            if not getattr(self, "_pr_resolved", True) and k == 1:
                t = (self.crowd_template or {}).get(a) or {}
                if (t.get("n") or 0) >= CROWD_HUG_MIN_N and t.get("mean") is not None:
                    p = (1 - UNRESOLVED_PLAYER_TEMPLATE_W) * p + UNRESOLVED_PLAYER_TEMPLATE_W * t["mean"] / 100.0
            return p
        return None

    # ------------------------------------------------------------------
    def predictions_for(self, market_rows: list[dict],
                        anchors: dict[tuple[str, str], MatchAnchor] | None = None,
                        player_props: dict | None = None,
                        ) -> tuple[list[dict], list[dict]]:
        """Returns (payload rows with p for both bots, unanswered rows).

        anchors: optional {(team_a, team_b): MatchAnchor} (canonical names, both
        orientations accepted). When a match is anchored, Bot A uses the blended
        answer directly (no shrink toward base — the market is already the
        conservative estimate)."""
        anchors = anchors or {}
        player_props = player_props or {}
        by_match: dict[str, list[dict]] = {}
        for m in market_rows:
            by_match.setdefault(m["match"]["name"], []).append(m)

        out, skipped = [], []
        for mname, rows in by_match.items():
            teams = self.resolve_teams(mname, [r["question"] for r in rows])
            if teams is None:
                skipped.extend({"match": mname, "question": r["question"], "reason": "teams unresolved"} for r in rows)
                continue
            anchor = anchors.get(teams) or anchors.get((teams[1], teams[0]))
            pp = player_props.get(teams) or player_props.get((teams[1], teams[0])) or {}
            team_set = {canonical(t) for t in teams}   # team-aware parse (player vs team SoT)
            ctx = self.context(*teams, anchor=anchor)
            for r in rows:
                q = parse_question(r["question"], teams=team_set)
                if q is None:
                    skipped.append({"match": mname, "question": r["question"], "reason": "unparsed"})
                    continue
                p = self.answer(q, ctx)
                if p is None:
                    skipped.append({"match": mname, "question": r["question"], "reason": "unanswerable"})
                    continue
                # rec #1: calibrate model-only markets (count/player always; goal
                # only when unanchored). Anchored goal markets pass through.
                p = self.calibrate(q.archetype, p, ctx.anchored)
                p = float(np.clip(p, 0.01, 0.99))
                fam = self._family(q.archetype)
                # Market anchor for player props (verified live: player_shots_on_target /
                # _goal_scorer_anytime / _assists). Blend the de-vigged book like a direct
                # market and SKIP the model's base-blend, extremize and confidence floor —
                # those patch the model we're now anchoring away from. See
                # docs/MARKET-FIRST-ARCHITECTURE.md.
                # resolve_player is robust to the book carrying middle names / initials
                # ('erling braut haaland' vs 'Erling Haaland') — an exact-key lookup silently
                # dropped the anchor and left the player on the shrinkage model.
                pm = resolve_player(q.subject, pp).get(q.archetype) if (fam == "player" and q.subject) else None
                # The book player_sot_ge value is the Over-0.5 (>=1 SoT) price. It must NOT be
                # stamped directly onto a ">=2 shots on target" question (that submits ~P(>=1)
                # for a >=2 market -> Brobbey 2+ SoT at 60% off a 66% 1+ line). But DROPPING it
                # was also costly: the >=2 line then falls to a flat, undifferentiated model
                # (Mbappe priced 19 like everyone, crowd 63, out=1 -> the single biggest R32 loss).
                # Instead, convert the book's player-differentiated 1+ line into P(>=k) via the
                # Poisson intensity (mu = -ln(1-P(>=1))). This keeps the book's player ranking for
                # the higher thresholds: an elite 1+ line yields a materially higher P(>=2) than a
                # fringe one, tracking the crowd, while still sitting strictly below P(>=1).
                if pm is not None and q.archetype == "player_sot_ge" and (q.threshold or 1) >= 2:
                    pm = book_sot_ge_from_one_plus(pm, q.threshold)
                if pm is not None:
                    # Player props anchor at a HIGHER book weight than goal markets. On 51
                    # settled anchored props the Brier is MONOTONE in the book weight (raw model
                    # 0.250, w=0.85 blend 0.206, pure book 0.202): the model side is data-gapped
                    # for exactly the star players the book prices well (Jimenez model 7 vs book
                    # 27, out=1; Kane raw ~9 dragging 54->45, out=1), so its logit is a large
                    # negative drag, a systematic low bias vs book AND crowd — doubled at 2x.
                    # Interpolate off w_market so the CLV passes keep their meaning
                    # (w_market=1.0 closing-line pass -> pure book; live 0.85 -> 0.97).
                    w_p = 1.0 - PLAYER_MODEL_FRAC * (1.0 - self.w_market)
                    p_a = float(np.clip(blend([p, pm], [1 - w_p, w_p]), 0.01, 0.99))
                    # P4b: lift toward the crowd we're scored against (it prices these props
                    # above the book AND calibrates better). Damped half-shift, capped, one-sided
                    # (up only), and only on the live 0.85 pass — the w=0/1 CLV passes stay pure so
                    # the divergence audit still measures the raw book anchor. Skip already-high
                    # props (>=CEIL): there the book and crowd agree and there's no headroom.
                    prem = self.player_crowd_prem.get(q.archetype, 0.0)
                    if prem and 0.0 < self.w_market < 1.0 and p_a < PLAYER_CROWD_LIFT_CEIL:
                        p_a = float(np.clip(p_a + min(PLAYER_CROWD_LIFT * prem, PLAYER_CROWD_LIFT_CAP), 0.01, 0.99))
                elif ctx.anchored and fam == "goal":
                    p_a = p  # market is already calibrated/conservative — don't shrink
                else:
                    p_base = self.answer(q, self._base_ctx)
                    p_a = blend([p_base, p], [BOT_A_BASE_WEIGHT, 1 - BOT_A_BASE_WEIGHT]) if p_base else p
                if pm is None:
                    # rec #2: evidence-gated extremizing (no-op until a family earns k>1)
                    p_a = float(np.clip(self._extremize(q.archetype, fam, p_a, ctx.anchored), 0.01, 0.99))
                    # Phase 1: asymmetric confidence calibration — floor confident-low bets on
                    # the archetypes where betting below the crowd loses catastrophically.
                    p_a = float(np.clip(self._confidence(q.archetype, p_a), 0.01, 0.99))
                # L0 LLM expert layer (out-of-band panel; shadow by default). Only when
                # there is NO sharp book anchor (pm is None): a matched player prop is the
                # sharpest signal and the LLM must not override it (else a stale panel opinion
                # fights a fresh book line). For book-anchored props the book stands.
                einfo: dict = {}
                if pm is None:
                    p_blend, einfo = panel_mod.apply_expert(r["id"], q.archetype, p_a, self.panel, self.expert_gate)
                    p_a = float(np.clip(p_blend, 0.01, 0.99))
                # Phase R risk management: shrink no-edge archetypes toward the base
                # rate (don't bet over-confidently where we have no validated edge).
                # Off unless RISK_GATE_ENABLE; no-op for archetypes not in the gate.
                # HALF-SCOPED questions bypass: the gate bases are FULL-MATCH quantities
                # (both_teams_sot_ge .70, cards .525) — shrinking a first-half price toward
                # a full-match base silently undoes the scope (review §3.6 interaction trap).
                if self.risk_on and q.half is None:
                    p_a = float(np.clip(self._risk(q.archetype, p_a), 0.01, 0.99))
                # Phase-0 crowd hug: flat event-base kinds where our model is a CONSTANT and
                # the settled crowd barely varies per question (std<=4) — submit near the
                # field's level and bank the field-variance bonus instead of bleeding a fixed
                # offset every match. Whitelist only; never grid/count/player kinds.
                if q.archetype in CROWD_HUG_KINDS and q.half is None:
                    t = (self.crowd_template or {}).get(q.archetype) or {}
                    if ((t.get("n") or 0) >= CROWD_HUG_MIN_N and t.get("mean") is not None
                            and (t.get("std") if t.get("std") is not None else 99.0) <= CROWD_HUG_MAX_STD):
                        p_a = float(np.clip((1 - CROWD_HUG_W) * p_a + CROWD_HUG_W * t["mean"] / 100.0,
                                            0.01, 0.99))
                # Phase-2 endgame dial: graded logit amplification, STRUCTURAL-EDGE kinds only,
                # live pass only, inert unless ENDGAME_EXTREMIZE is set (see ENDGAME_KINDS note).
                if (self.endgame_alpha > 1.0 and q.archetype in ENDGAME_KINDS
                        and 0.0 < self.w_market < 1.0):
                    p_a = float(np.clip(inv_logit(self.endgame_alpha * logit(
                        float(np.clip(p_a, 0.03, 0.97)))), 0.03, 0.97))
                # Lineup-status override — the LAST word for player props (beats book, lift,
                # expert, risk). A player confirmed benched/out this match is capped at a bounded
                # sub-minutes ceiling; applied as a MIN so it only lowers a starter-priced prop.
                # This is the De Bruyne fix: the pm-guard drops the WebPanel's lineup read for
                # book-anchored props, so the override has to land here, after everything else.
                lineup_cap = None
                if fam == "player":
                    _ovr = self._lineup_override(q.subject, q.archetype, (ctx.team_a, ctx.team_b))
                    if _ovr is not None:
                        cap, _st = _ovr
                        if p_a > cap:
                            p_a = float(np.clip(cap, 0.01, 0.99))
                        lineup_cap = f"{_st}<={int(round(cap * 100))}"
                out.append({
                    "market_id": r["id"], "lobby_id": r["lobby_id"], "match": mname,
                    "question": r["question"], "archetype": q.archetype,
                    "p_bot_a": round(p_a, 4), "sub_a": submit_int(p_a),
                    "p_bot_b": round(p, 4), "sub_b": submit_int(p),
                    "p_expert": einfo.get("expert"), "expert_action": einfo.get("action"),
                    "lineup": lineup_cap,
                })
        return out, skipped
