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

import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jumpprob.models.count_model import p_a_gt_b, p_sum_ge
from jumpprob.models.goal_model import dc_grid
from jumpprob.models.market_anchor import blend
from jumpprob.models.calibration import submit_int
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets as mk
from jumpprob.submit.parser import ParsedQuestion, parse_question

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
P_RED_MATCH = 0.065   # P(>=1 red): 0.067 reds/match measured, 6 tournaments
P_PEN_MATCH = 0.293   # A6
BOT_A_BASE_WEIGHT = 0.25

H1 = 1 - mk.SECOND_HALF_SHARE
H2 = mk.SECOND_HALF_SHARE


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


class Runner:
    def __init__(self, pipeline, player_table: pd.DataFrame):
        self.pl = pipeline
        self.players = self._player_rates(player_table)
        # league-average context for Bot A's base-rate blend
        lam0 = float(np.exp(self.pl.goal_model.b0))
        self._base_ctx = MatchContext("X", "Y", lam0, lam0,
                                      self.pl.goal_model.scoreline_grid(lam0, lam0),
                                      self.pl.goal_model.rho_dc)

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

    def player_rate(self, name: str) -> tuple[float, float, float]:
        tokens = _norm_name(name).split()
        hits = self.players[self.players["norm"].apply(lambda n: all(t in n for t in tokens))]
        if len(hits) == 1:
            return (float(hits["rate_sot"].iloc[0]), float(hits["rate_goal"].iloc[0]),
                    float(hits["rate_ga"].iloc[0]))
        return self.prior_sot, self.prior_goal, self.prior_ga

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

    def context(self, team_a: str, team_b: str) -> MatchContext:
        ra, rb = self.pl.ratings[team_a], self.pl.ratings[team_b]
        la, lb = self.pl.goal_model.intensities(ra, rb)
        return MatchContext(team_a, team_b, la, lb,
                            self.pl.goal_model.scoreline_grid(la, lb),
                            self.pl.goal_model.rho_dc)

    # ------------------------------------------------------------------
    def answer(self, q: ParsedQuestion, ctx: MatchContext) -> float | None:
        cm = self.pl.count_model
        side = None
        if q.subject:
            c = canonical(q.subject)
            side = "a" if c == ctx.team_a else "b" if c == ctx.team_b else None

        def lams(s):  # (team, opp) intensities for side s
            return (ctx.lam_a, ctx.lam_b) if s == "a" else (ctx.lam_b, ctx.lam_a)

        def cpmf(fam, half, s):
            lt, lo = lams(s)
            return cm.pmf(fam, cm.mu(fam, half, lt, lo))

        def match_pmf(fam, s):
            return np.convolve(cpmf(fam, 1, s), cpmf(fam, 2, s))

        a = q.archetype
        if a == "win":
            return None if side is None else mk.p_win(ctx.grid, side)
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
    def predictions_for(self, market_rows: list[dict]) -> tuple[list[dict], list[dict]]:
        """Returns (payload rows with p for both bots, unanswered rows)."""
        by_match: dict[str, list[dict]] = {}
        for m in market_rows:
            by_match.setdefault(m["match"]["name"], []).append(m)

        out, skipped = [], []
        for mname, rows in by_match.items():
            teams = self.resolve_teams(mname, [r["question"] for r in rows])
            if teams is None:
                skipped.extend({"match": mname, "question": r["question"], "reason": "teams unresolved"} for r in rows)
                continue
            ctx = self.context(*teams)
            for r in rows:
                q = parse_question(r["question"])
                if q is None:
                    skipped.append({"match": mname, "question": r["question"], "reason": "unparsed"})
                    continue
                p = self.answer(q, ctx)
                if p is None:
                    skipped.append({"match": mname, "question": r["question"], "reason": "unanswerable"})
                    continue
                p = float(np.clip(p, 0.01, 0.99))
                p_base = self.answer(q, self._base_ctx)
                p_a = blend([p_base, p], [BOT_A_BASE_WEIGHT, 1 - BOT_A_BASE_WEIGHT]) if p_base else p
                out.append({
                    "market_id": r["id"], "lobby_id": r["lobby_id"], "match": mname,
                    "question": r["question"], "archetype": q.archetype,
                    "p_bot_a": round(p_a, 4), "sub_a": submit_int(p_a),
                    "p_bot_b": round(p, 4), "sub_b": submit_int(p),
                })
        return out, skipped
