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
from jumpprob.ingest.odds_api import norm_player
from jumpprob.submit.parser import ParsedQuestion, parse_question
from jumpprob.agents import panel as panel_mod  # out-of-band LLM blend (daemon reads cache only)

# Archetype families (calibration + engine routing). Timing markets are the
# subset the analytic engine only approximates — route them through the v2 MC
# simulator (rec #3). Goal markets are anchored when odds exist (skip calib);
# count/player markets are model-only (always calibrate, rec #1).
TIMING_ARCHS = {"ht_tied", "ht_leading", "h2_more_goals_than_h1", "first_goal_half",
                "first_goal_and_win", "first_goal_and_opp_2h", "goals_gt_half"}
GOAL_ARCHS = {"win", "draw", "btts", "total_ge", "total_le", "btts_and_total_ge",
              "team_scores_half", "half_total_ge", "team_goals_ge",
              "first_goal_match", "win_margin", "clean_sheet", "score_both_halves"} | TIMING_ARCHS
COUNT_ARCHS = {"corners_gt", "sot_gt", "fouls_gt", "cards_gt", "team_corners_ge",
               "team_sot_ge", "team_cards_ge", "cards_ge", "corners_total_ge",
               "total_sot_half_ge", "total_sot_ge", "offsides_ge", "both_teams_sot_ge"}
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
CLUB_RATES = Path("data/processed/player_rates_combined.parquet")
SOT_PER_SHOT = 0.37   # Understat has no SoT -> estimate from total shots
P_RED_MATCH = 0.065   # P(>=1 red): 0.067 reds/match measured, 6 tournaments
P_PEN_MATCH = 0.293   # A6
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
}
_DYN_PATH = Path("data/processed/minute_dynamics.json")


def _norm_name(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


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
        """Per-minute (sot, goal, ga) rates. The GA prior uses the transfer-discounted
        CLUB rate (club assists are our biggest blind spot — validated on settled props:
        it cuts player_ga Brier ~0.19->0.16). SoT and scores keep the population prior:
        on settled data the club rate over-predicts SoT (the tail is better handled by
        the Phase-1 floor) and erodes our genuine player_scores edge. The tournament
        appearances then shrink off the prior (Gamma-Poisson, M0=270/540)."""
        club = self._club_prior(name)
        prior_sot, prior_goal = self.prior_sot, self.prior_goal
        prior_ga = self.player_transfer * club[2] if club else self.prior_ga
        tokens = _norm_name(name).split()
        hits = self.players[self.players["norm"].apply(lambda n: all(t in n for t in tokens))]
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
        if a in ("player_scores", "player_sot_ge", "player_ga"):
            rate_sot, rate_goal, rate_ga = self.player_rate(q.subject)
            rate = {"player_sot_ge": rate_sot, "player_scores": rate_goal, "player_ga": rate_ga}[a]
            # opponent adjustment: scale by this match's intensity vs league avg
            scale = (ctx.lam_a + ctx.lam_b) / (2 * np.exp(self.pl.goal_model.b0))
            # half-specific player questions: exposure = minutes x half share
            if q.half:
                scale *= H1 if q.half == 1 else H2
            return float(sum(w * (1 - np.exp(-rate * scale * m)) for w, m in MINUTE_SCENARIOS))
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
                pm = pp.get(norm_player(q.subject), {}).get(q.archetype) if (fam == "player" and q.subject) else None
                if pm is not None:
                    p_a = float(np.clip(blend([p, pm], [1 - self.w_market, self.w_market]), 0.01, 0.99))
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
                # L0 LLM expert layer (out-of-band panel; shadow by default). At the
                # default env (all role weights 0) apply_expert returns p_a unchanged,
                # so sub_a is byte-for-byte identical; a cache miss -> pure model. The
                # expert number is still logged for the forward-test. See LLM-EXPERT-PLAN.
                p_blend, einfo = panel_mod.apply_expert(r["id"], q.archetype, p_a, self.panel, self.expert_gate)
                p_a = float(np.clip(p_blend, 0.01, 0.99))
                # Phase R risk management: shrink no-edge archetypes toward the base
                # rate (don't bet over-confidently where we have no validated edge).
                # Off unless RISK_GATE_ENABLE; no-op for archetypes not in the gate.
                if self.risk_on:
                    p_a = float(np.clip(self._risk(q.archetype, p_a), 0.01, 0.99))
                out.append({
                    "market_id": r["id"], "lobby_id": r["lobby_id"], "match": mname,
                    "question": r["question"], "archetype": q.archetype,
                    "p_bot_a": round(p_a, 4), "sub_a": submit_int(p_a),
                    "p_bot_b": round(p, 4), "sub_b": submit_int(p),
                    "p_expert": einfo.get("expert"), "expert_action": einfo.get("action"),
                })
        return out, skipped
