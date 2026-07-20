"""The live pricing runner: match questions in, submitted probabilities out.

Given the slate of questions the platform has posted, the runner resolves which
fixture each belongs to, builds one match context per fixture, prices every
question against it, and then walks each price down a ladder of calibration and
safety layers before it is submitted.

**Team identification.** The two teams are extracted from a match's own question
texts, where full names appear, and validated against the Elo table. The
three-letter codes in the match name are only a fallback. A match whose teams
cannot be resolved is reported, never guessed.

**Two bots.** Bot A is the calibration-first entrant: it anchors hard to the
book and treats the market as a floor. Bot B is model-leaning, deliberately
decorrelated from the field. Both price the same slate from the same context.

**The pricing ladder**, in order, in :meth:`Runner.predictions_for`:

1. Answer the question from the model, via the archetype handlers.
2. Platt-calibrate by family — skipped for anchored goal markets, which the
   book has already calibrated.
3. Anchor: blend the de-vigged book line. Player props anchor hardest.
4. Shrink toward the base rate, for unanchored questions only.
5. Extremize, but only where a family has a measured edge.
6. Floor confident-low bets where betting under the crowd has proved costly.
7. Consult the LLM panel, but only where no book line exists.
8. Shrink no-edge archetypes toward their base rate.
9. Hug the crowd on flat event kinds where our model is a constant.
10. Apply the endgame variance dial, if armed.
11. Cap player props for anyone confirmed benched or out. This is last on
    purpose: confirmed team news beats every estimate above it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd

from jumpprob.agents import panel as panel_mod
from jumpprob.ingest.odds_api import norm_player, resolve_player
from jumpprob.models import strategy
from jumpprob.models.calibration import submit_int
from jumpprob.models.dynamics import Dynamics
from jumpprob.models.market_anchor import (
    MatchAnchor, blend, blend_lambdas, implied_lambdas, inv_logit, logit,
)
from jumpprob.normalize.players import ascii_lower, build_index, resolve
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import mc_engine
from jumpprob.submit import handlers
from jumpprob.submit.constants import (
    BOT_A_BASE_WEIGHT, CLUB_RATES_PATH, CODE_MAP, COUNT_ARCHS, CROWD_HUG_KINDS,
    CROWD_HUG_MAX_STD, CROWD_HUG_MIN_N, CROWD_HUG_W, CROWD_TEMPLATE_PATH,
    DYNAMICS_PATH, ENDGAME_KINDS, EVENT_BASE_DEFAULTS, EVENT_BASE_PATH,
    GOAL_ARCHS, LINEUP_BENCH_CAP, LINEUP_FRESH_H, LINEUP_OUT_CEIL,
    LINEUP_STATUS_PATH, LIVE_CALIBRATION_PATH, MC_TIMING, PLAYER_ARCHS,
    PLAYER_CROWD_LIFT, PLAYER_CROWD_LIFT_CAP, PLAYER_CROWD_LIFT_CEIL,
    PLAYER_CROWD_PREM_PATH, PLAYER_MODEL_FRAC, PLAYER_TRANSFER,
    SINGLE_DIGIT_TEAM_SHARE_PATH, SOT_PER_SHOT, TIMING_ARCHS,
)
from jumpprob.submit.parser import ParsedQuestion, parse_question


def book_sot_ge_from_one_plus(p1: float, k: int) -> float | None:
    """Turn the book's 1+ shots-on-target line into a price for the k+ line.

    Books only ever quote the 0.5 line, i.e. P(>=1). Treating shots on target as
    Poisson(mu) gives mu = -ln(1 - P(>=1)), from which P(>=k) follows. That
    re-uses the book's player *ranking* — elite shooters carry a high 1+ line,
    fringe players a low one — instead of discarding the only
    player-differentiated signal we have and falling back to a flat model.

    The alternative failure was worse in both directions. Stamping the 1+ line
    onto a 2+ question over-prices badly; dropping it entirely priced Mbappé's
    2+ line at 19 against a crowd's 63, and it landed. Worked from a settled
    snapshot: Mbappé 1+ = .834 gives P(>=2) = .54, Isak 1+ = .478 gives .14.

    Returns None for k <= 1, where the caller should use the raw line.
    """
    if k <= 1:
        return None
    from scipy.stats import poisson
    p1 = float(np.clip(p1, 1e-6, 1 - 1e-9))
    mu = -np.log(1.0 - p1)
    return float(np.clip(1.0 - poisson.cdf(k - 1, mu), 0.0, 1.0))


def _load_json(path: Path, default):
    """Read a fitted artefact, falling back to ``default`` if it is missing."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


class PlayerRates(NamedTuple):
    """Per-minute rates for one player, and whether we actually found them.

    ``resolved`` is False when neither the club prior nor tournament minutes
    matched — we know nothing about this player and the rates are the population
    average. Callers use it to fall back to the crowd level rather than shipping
    that floor as if it were an estimate.
    """

    sot: float
    goal: float
    ga: float
    resolved: bool


@dataclass
class MatchContext:
    """One fixture's pricing state, shared by every question about it."""

    team_a: str              # listed first in the match name
    team_b: str
    lam_a: float             # goal intensities, market-blended when anchored
    lam_b: float
    grid: np.ndarray         # Dixon-Coles scoreline grid at those intensities
    rho_dc: float
    anchored: bool = False   # True if intensities were blended with book odds
    sim: object = None       # Monte-Carlo result at the same intensities, if any


class Runner:
    def __init__(self, pipeline, player_table: pd.DataFrame, mc=None,
                 calibration: dict | None = None, aggression: float = 1.0,
                 w_market: float | None = None):
        self.pipeline = pipeline
        self.players = self._player_rates(player_table)
        self.player_transfer = PLAYER_TRANSFER
        self._load_club_rates()
        self.mc = mc

        # Minute profile for the window handlers. Reuse the simulator's if we
        # have one, else load it standalone so windows work without an engine.
        self.dyn = (mc.dyn if mc is not None
                    else Dynamics.from_json(DYNAMICS_PATH) if DYNAMICS_PATH.exists()
                    else None)

        self.event_base = dict(EVENT_BASE_DEFAULTS)
        if EVENT_BASE_PATH.exists():
            measured = _load_json(EVENT_BASE_PATH, {})
            self.event_base.update({k: float(measured[k]) for k in EVENT_BASE_DEFAULTS
                                    if measured.get(k) is not None})

        self.calib = calibration if calibration is not None else self._load_calibration()

        from jumpprob.models.market_anchor import W_MARKET
        # Odds-blend weight. The closing-line passes set 0 or 1 to measure the
        # raw model and the raw book against each other.
        self.w_market = W_MARKET if w_market is None else w_market

        # Per-family leverage: the better of the live edge gate and the
        # backtest prop-edge gate.
        live, backtest = strategy.load_edge_gate(), strategy.load_backtest_gate()
        self.gate = {f: max(live.get(f, 1.0), backtest.get(f, 1.0))
                     for f in set(live) | set(backtest)}
        self.base_rates = strategy.load_base_rates()
        self.field_sharpness = strategy.load_field_sharpness()
        self.confidence = strategy.load_confidence_gate()
        self.risk = strategy.load_risk_gate()
        self.risk_on = strategy.risk_enabled()

        self.panel = panel_mod.load_panel()          # cached LLM review, shadow by default
        self.expert_gate = panel_mod.load_gate()     # per-archetype blend weights
        self.player_crowd_prem = _load_json(PLAYER_CROWD_PREM_PATH, {}).get("premium", {})
        self.lineup_status = _load_json(LINEUP_STATUS_PATH, {})
        self.crowd_template = _load_json(CROWD_TEMPLATE_PATH, {})
        self.single_digit_share = _load_json(SINGLE_DIGIT_TEAM_SHARE_PATH, {})

        try:
            self.endgame_alpha = float(os.environ.get("ENDGAME_EXTREMIZE", "1.0") or 1.0)
        except ValueError:
            self.endgame_alpha = 1.0
        self.aggression = aggression

        # League-average context, used for Bot A's shrink toward the base rate.
        lam0 = float(np.exp(self.pipeline.goal_model.b0))
        self._base_ctx = MatchContext("X", "Y", lam0, lam0,
                                      self.pipeline.goal_model.scoreline_grid(lam0, lam0),
                                      self.pipeline.goal_model.rho_dc)

    @staticmethod
    def _load_calibration() -> dict:
        return _load_json(LIVE_CALIBRATION_PATH, {})

    # ------------------------------------------------------------------
    # Player rates
    # ------------------------------------------------------------------
    def _player_rates(self, pm: pd.DataFrame) -> pd.DataFrame:
        """Per-minute tournament rates, shrunk toward the population mean."""
        pm = pm.assign(ga=pm["goals"] + pm.get("assists", 0))
        tot = pm.groupby("player")[["minutes", "sot", "goals", "ga"]].sum()
        prior_sot = pm["sot"].sum() / pm["minutes"].sum()
        prior_goal = pm["goals"].sum() / pm["minutes"].sum()
        prior_ga = pm["ga"].sum() / pm["minutes"].sum()
        tot["rate_sot"] = (tot["sot"] + prior_sot * 270) / (tot["minutes"] + 270)
        tot["rate_goal"] = (tot["goals"] + prior_goal * 540) / (tot["minutes"] + 540)
        tot["rate_ga"] = (tot["ga"] + prior_ga * 540) / (tot["minutes"] + 540)
        tot["norm"] = [ascii_lower(p) for p in tot.index]
        self.prior_sot, self.prior_goal, self.prior_ga = prior_sot, prior_goal, prior_ga
        return tot

    def _load_club_rates(self) -> None:
        """Per-minute club rates, used as a player-specific prior.

        This replaced the global population prior, which shrank established
        attackers toward the average — the systematic under-pricing the settled
        match audit found. Inert if the combined table is absent.
        """
        self.club_rates: dict[str, tuple] = {}
        self.club_idx = None
        if not CLUB_RATES_PATH.exists():
            return
        df = pd.read_parquet(CLUB_RATES_PATH).drop_duplicates(subset="player_name",
                                                              keep="first")
        for r in df.itertuples():
            sot90 = getattr(r, "sot_p90", None)
            if sot90 is None or (isinstance(sot90, float) and np.isnan(sot90)):
                sot90 = float(getattr(r, "shots_p90", 0) or 0) * SOT_PER_SHOT
            npg90 = float(getattr(r, "npg_p90", 0) or 0)
            ga90 = npg90 + float(getattr(r, "assists_p90", 0) or 0)
            self.club_rates[r.player_name] = (sot90 / 90.0, npg90 / 90.0, ga90 / 90.0)
        self.club_idx = build_index(df["player_name"])

    def _club_prior(self, name: str) -> tuple | None:
        if self.club_idx is None:
            return None
        match, _ = resolve(name, *self.club_idx)
        return self.club_rates.get(match) if match else None

    def player_rate(self, name: str) -> PlayerRates:
        """Per-minute shot, goal and goal-or-assist rates for one player.

        Gamma-Poisson shrinkage toward a *player-centred* prior: the
        transfer-discounted club rate wherever the player resolves, and the
        population mean only when they do not. Centring the prior on the
        population instead was a real leak — 270 to 540 minutes of shrinkage is
        three to six full matches of pull toward "average World Cup player",
        applied to exactly the elite outliers these questions ask about. Our
        player prices sat 18 to 28 points under the crowd, and the gap grew with
        the crowd's level, which is the signature of over-shrunk stars.
        """
        name = name.split("(")[0].strip() or name
        club = self._club_prior(name)
        prior_sot = self.player_transfer * club[0] if club else self.prior_sot
        prior_goal = self.player_transfer * club[1] if club else self.prior_goal
        prior_ga = self.player_transfer * club[2] if club else self.prior_ga

        tokens = ascii_lower(name).split()
        hits = self.players[self.players["norm"].apply(
            lambda n: all(t in n for t in tokens))]
        resolved = bool(club) or len(hits) == 1
        if len(hits) == 1:
            h = hits.iloc[0]
            m = float(h["minutes"])
            return PlayerRates((float(h["sot"]) + prior_sot * 270) / (m + 270),
                               (float(h["goals"]) + prior_goal * 540) / (m + 540),
                               (float(h["ga"]) + prior_ga * 540) / (m + 540),
                               resolved)
        return PlayerRates(prior_sot, prior_goal, prior_ga, resolved)

    # ------------------------------------------------------------------
    # Lineup news
    # ------------------------------------------------------------------
    def lineup_flag(self, subject: str | None, teams) -> str | None:
        """Confirmed status for a player: 'out', 'bench', or None.

        Freshness-gated and team-scoped, so a stale flag cannot bleed into
        another match.
        """
        if not subject or not self.lineup_status:
            return None
        ent = self.lineup_status.get(norm_player(subject))
        if not ent:
            return None
        ts = ent.get("ts")
        if ts:
            try:
                age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds() / 3600.0
                if age_h > LINEUP_FRESH_H:
                    return None
            except Exception:
                pass
        team = ent.get("team")
        if team and teams and canonical(team) not in {canonical(t) for t in teams if t}:
            return None
        status = str(ent.get("status", "")).lower()
        if status == "out":
            return "out"
        return "bench" if status in ("bench", "benched") else None

    def _lineup_override(self, subject: str | None, arch: str, teams) -> tuple[float, str] | None:
        """A bounded ceiling for a player confirmed benched or out.

        Applied as a maximum by the caller, so it can only ever lower a
        starter-priced prop. That bound is the point: a mis-scraped "benched"
        caps an elite starter at the elite-substitute rate rather than zeroing
        them.
        """
        flag = self.lineup_flag(subject, teams)
        if flag == "out":
            return LINEUP_OUT_CEIL, "out"
        if flag == "bench" and arch in LINEUP_BENCH_CAP:
            return LINEUP_BENCH_CAP[arch], "bench"
        return None

    # ------------------------------------------------------------------
    # Calibration and the safety gates
    # ------------------------------------------------------------------
    def _family(self, arch: str) -> str | None:
        """Which Platt calibration a question gets, or None for constants."""
        # Offsides get their own: a definition gap between StatsBomb and the
        # platform makes them over-priced live, and the pooled count Platt moves
        # them the wrong way. This must precede the count check.
        if arch == "offsides_ge":
            return "offsides"
        if arch in COUNT_ARCHS:
            return "count"
        if arch in PLAYER_ARCHS:
            return "player"
        if arch in GOAL_ARCHS:
            return "goal"
        return None

    def calibrate(self, arch: str, p: float, anchored: bool) -> float:
        """Per-family Platt. Anchored goal markets skip it — the book is already
        calibrated — while count and player markets always take it."""
        fam = self._family(arch)
        if fam is None or fam not in self.calib:
            return p
        if fam == "goal" and anchored:
            return p
        a, b = self.calib[fam]
        return float(inv_logit(a * logit(p) + b))

    def _extremize(self, arch: str, fam: str | None, p: float, anchored: bool,
                   stage_mult: float = 1.0) -> float:
        """Amplify our deviation from the field, where a family has earned it.

        A no-op unless the family's edge gate grants leverage above 1. Anchored
        goal markets are never extremized: there we *are* the field, so the
        anchored price is a floor, not a position.

        No-edge archetypes are skipped too. They belong to the risk gate, which
        shrinks them toward base — extremizing them here would amplify the very
        noise that gate then has to undo, and the two layers would fight.
        """
        if anchored and arch not in TIMING_ARCHS and fam == "goal":
            return p
        if arch in self.risk:
            return p
        gate_fam = "timing" if arch in TIMING_ARCHS else fam
        if gate_fam is None:
            return p
        k = strategy.k_for(gate_fam, self.gate, stage_mult, self.aggression)
        if k == 1.0:
            return p
        field = strategy.field_model(p, self.field_sharpness.get(gate_fam, 0.5))
        return strategy.extremize(p, field, k)

    def _confidence(self, arch: str, p: float) -> float:
        """Floor confident-low bets on archetypes where betting below the crowd
        has proved catastrophic. No-op where we have measured edge."""
        return strategy.apply_confidence(p, arch, self.confidence)

    def _risk(self, arch: str, p: float) -> float:
        """Shrink no-edge archetypes toward their base rate."""
        return strategy.apply_risk(p, arch, self.risk, self.base_rates)

    def _crowd_hug(self, arch: str, p: float) -> float:
        """Pull flat event kinds toward the crowd's own level.

        Only where our model is a constant and the settled crowd barely varies
        per question. Submitting near the field's mean banks the Jensen
        field-variance bonus instead of bleeding a fixed offset every match.
        """
        t = (self.crowd_template or {}).get(arch) or {}
        if (t.get("n") or 0) < CROWD_HUG_MIN_N or t.get("mean") is None:
            return p
        if (t.get("std") if t.get("std") is not None else 99.0) > CROWD_HUG_MAX_STD:
            return p
        return float(np.clip((1 - CROWD_HUG_W) * p + CROWD_HUG_W * t["mean"] / 100.0,
                             0.01, 0.99))

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def resolve_teams(self, match_name: str, questions: list[str]) -> tuple[str, str] | None:
        """Identify a match's two teams from its own question texts.

        Full team names appear in the questions; the match name carries only
        three-letter codes, which are the fallback. Returns None rather than
        guessing.
        """
        known = set(self.pipeline.ratings.index)
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
            # Order them to match the "XXX vs YYY" listing where we can.
            left = match_name.split(" vs ")[0].strip()
            first = canonical(CODE_MAP.get(left, left))
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
        """Build one fixture's pricing state.

        With an odds anchor, the model's goal intensities are blended toward the
        book consensus, so every downstream answer — result, totals, half goals,
        counts, props — inherits the market view coherently. Without one, this is
        the pure model.
        """
        ra, rb = self.pipeline.ratings[team_a], self.pipeline.ratings[team_b]
        la, lb = self.pipeline.goal_model.intensities(ra, rb)
        rho = self.pipeline.goal_model.rho_dc
        anchored = False

        if anchor is not None:
            # The odds feed's home/away need not match the platform's ordering.
            if anchor.home == team_a:
                p_a_win, p_b_win = anchor.p_home, anchor.p_away
            elif anchor.away == team_a:
                p_a_win, p_b_win = anchor.p_away, anchor.p_home
            else:
                p_a_win = p_b_win = None            # team mismatch, no anchor
            if p_a_win is not None:
                la_mkt, lb_mkt = implied_lambdas(p_a_win, anchor.p_draw, p_b_win,
                                                 anchor.p_over25, rho=rho)
                # Guard against a pathological inversion from a bad quote.
                if 0.05 <= la_mkt <= 6.0 and 0.05 <= lb_mkt <= 6.0:
                    la, lb = blend_lambdas((la, lb), (la_mkt, lb_mkt), self.w_market)
                    anchored = True

        sim = self.mc.simulate(0.0, 0.0, lam=(la, lb)) if self.mc is not None else None
        return MatchContext(team_a, team_b, la, lb,
                            self.pipeline.goal_model.scoreline_grid(la, lb), rho, anchored, sim)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def answer(self, q: ParsedQuestion, ctx: MatchContext) -> float | None:
        """Price one parsed question, or None if we cannot answer it honestly."""
        side = None
        if q.subject:
            c = canonical(q.subject)
            side = "a" if c == ctx.team_a else "b" if c == ctx.team_b else None

        # Timing questions go to the Monte-Carlo simulator when one exists: it
        # is coherent and exact where the closed-form handlers approximate.
        if ctx.sim is not None and q.archetype in MC_TIMING:
            return mc_engine.answer(ctx.sim, MC_TIMING[q.archetype], side=side,
                                    threshold=q.threshold, half=q.half)

        return handlers.answer(handlers.Answering(self, q, ctx, side))

    def _anchor_player_prop(self, q: ParsedQuestion, p: float, book: float) -> float:
        """Blend a model player prop with its book line, then lift toward the crowd.

        Player props anchor at a higher book weight than goal markets. On 51
        settled anchored props the Brier was monotone in the book weight — raw
        model .250, blended .206, pure book .202 — because the model side is
        data-gapped for exactly the stars the book prices well. Its logit acts as
        a large negative drag, a systematic low bias against both book and crowd,
        and the stage multipliers double it.

        The weight interpolates off ``w_market`` so the closing-line passes keep
        their meaning: at w_market = 1 this is the pure book, at the live .85 it
        is about .97.

        The crowd lift then nudges toward the field we are actually scored
        against, which prices these props above the book and calibrates better
        doing it. One-sided, damped, capped, and only on the live pass — the
        w = 0 and w = 1 passes stay pure so the divergence audit still measures
        the raw book anchor.
        """
        w = 1.0 - PLAYER_MODEL_FRAC * (1.0 - self.w_market)
        p_a = float(np.clip(blend([p, book], [1 - w, w]), 0.01, 0.99))
        prem = self.player_crowd_prem.get(q.archetype, 0.0)
        if prem and 0.0 < self.w_market < 1.0 and p_a < PLAYER_CROWD_LIFT_CEIL:
            lift = min(PLAYER_CROWD_LIFT * prem, PLAYER_CROWD_LIFT_CAP)
            p_a = float(np.clip(p_a + lift, 0.01, 0.99))
        return p_a

    def predictions_for(self, market_rows: list[dict],
                        anchors: dict[tuple[str, str], MatchAnchor] | None = None,
                        player_props: dict | None = None,
                        ) -> tuple[list[dict], list[dict]]:
        """Price a whole slate. Returns (rows to submit, rows we could not answer).

        ``anchors`` maps a canonical team pair to its de-vigged book consensus;
        either orientation is accepted. Anything skipped is returned with a
        reason rather than dropped, so the template audit can surface it.
        """
        anchors = anchors or {}
        player_props = player_props or {}

        by_match: dict[str, list[dict]] = {}
        for m in market_rows:
            by_match.setdefault(m["match"]["name"], []).append(m)

        out, skipped = [], []
        for mname, rows in by_match.items():
            teams = self.resolve_teams(mname, [r["question"] for r in rows])
            if teams is None:
                skipped.extend({"match": mname, "question": r["question"],
                                "reason": "teams unresolved"} for r in rows)
                continue

            anchor = anchors.get(teams) or anchors.get((teams[1], teams[0]))
            props = player_props.get(teams) or player_props.get((teams[1], teams[0])) or {}
            team_set = {canonical(t) for t in teams}
            ctx = self.context(*teams, anchor=anchor)

            for r in rows:
                q = parse_question(r["question"], teams=team_set)
                if q is None:
                    skipped.append({"match": mname, "question": r["question"],
                                    "reason": "unparsed"})
                    continue
                p = self.answer(q, ctx)
                if p is None:
                    skipped.append({"match": mname, "question": r["question"],
                                    "reason": "unanswerable"})
                    continue

                p = float(np.clip(self.calibrate(q.archetype, p, ctx.anchored), 0.01, 0.99))
                fam = self._family(q.archetype)
                p_a, expert, lineup_cap = self._price_bot_a(q, ctx, p, fam, props, r["id"])
                out.append({
                    "market_id": r["id"], "lobby_id": r["lobby_id"], "match": mname,
                    "question": r["question"], "archetype": q.archetype,
                    # Bot A takes the full ladder; Bot B ships the calibrated
                    # model price, deliberately decorrelated from the field.
                    "p_bot_a": round(p_a, 4), "sub_a": submit_int(p_a),
                    "p_bot_b": round(p, 4), "sub_b": submit_int(p),
                    "p_expert": expert.get("expert"),
                    "expert_action": expert.get("action"),
                    "lineup": lineup_cap,
                })
        return out, skipped

    def _price_bot_a(self, q, ctx, p, fam, props, market_id) -> tuple[float, dict, str | None]:
        """Walk the calibrated model price down Bot A's ladder (steps 3-11).

        Returns the price, whatever the LLM panel contributed, and a note of any
        lineup cap that fired — the latter two purely so the submit run can
        print them for review.
        """
        # 3. Book anchor. Player props resolve through a fuzzy name join, since
        #    the book may carry middle names or initials.
        book = resolve_player(q.subject, props).get(q.archetype) \
            if (fam == "player" and q.subject) else None
        # A book player line is always the 1+ price; convert it for higher lines.
        if book is not None and q.archetype == "player_sot_ge" and (q.threshold or 1) >= 2:
            book = book_sot_ge_from_one_plus(book, q.threshold)

        if book is not None:
            p_a = self._anchor_player_prop(q, p, book)
        elif ctx.anchored and fam == "goal":
            p_a = p                      # the market is already the conservative estimate
        else:
            # 4. Shrink toward the league-average answer for this archetype.
            p_base = self.answer(q, self._base_ctx)
            p_a = blend([p_base, p], [BOT_A_BASE_WEIGHT, 1 - BOT_A_BASE_WEIGHT]) if p_base else p

        if book is None:
            # 5-6. Leverage where earned, then floor confident-low bets. Both
            #      patch the model, so they are skipped when we anchored away
            #      from it.
            p_a = float(np.clip(self._extremize(q.archetype, fam, p_a, ctx.anchored), 0.01, 0.99))
            p_a = float(np.clip(self._confidence(q.archetype, p_a), 0.01, 0.99))
            # 7. The LLM panel speaks only where no sharp book line exists — a
            #    matched prop is the better signal and a cached opinion must not
            #    fight a fresh line.
            p_blend, expert = panel_mod.apply_expert(market_id, q.archetype, p_a,
                                                     self.panel, self.expert_gate)
            p_a = float(np.clip(p_blend, 0.01, 0.99))
        else:
            expert = {}

        # 8. Shrink no-edge archetypes. Half-scoped questions bypass this: the
        #    gate's base rates are full-match quantities, so shrinking a
        #    first-half price toward one would silently undo the scope.
        if self.risk_on and q.half is None:
            p_a = float(np.clip(self._risk(q.archetype, p_a), 0.01, 0.99))

        # 9. Hug the crowd on the whitelisted flat event kinds.
        if q.archetype in CROWD_HUG_KINDS and q.half is None:
            p_a = self._crowd_hug(q.archetype, p_a)

        # 10. Endgame variance dial — structural-edge kinds, live pass only.
        if (self.endgame_alpha > 1.0 and q.archetype in ENDGAME_KINDS
                and 0.0 < self.w_market < 1.0):
            p_a = float(np.clip(inv_logit(self.endgame_alpha * logit(
                float(np.clip(p_a, 0.03, 0.97)))), 0.03, 0.97))

        # 11. Confirmed team news has the last word on player props, beating the
        #     book, the crowd lift, the panel and the gates alike.
        lineup_cap = None
        if fam == "player":
            override = self._lineup_override(q.subject, q.archetype, (ctx.team_a, ctx.team_b))
            if override is not None:
                cap, status = override
                if p_a > cap:
                    p_a = float(np.clip(cap, 0.01, 0.99))
                lineup_cap = f"{status}<={int(round(cap * 100))}"
        return p_a, expert, lineup_cap
