"""Questions answered from the scoreline grid or the goal intensities.

These are the questions the bookmaker effectively prices for us. When a match is
anchored, the grid has already been blended toward the de-vigged book, so every
answer here inherits the market's view — and, just as importantly, they inherit
it *coherently*. Win, both-teams-score, over 2.5, halftime states and first-goal
questions all read off one object, so they cannot contradict each other.
"""

from __future__ import annotations

import numpy as np

from jumpprob.models.goal_model import dc_grid
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets as mk
from jumpprob.submit.constants import (
    ASSISTED_OPEN_SHARE, BRACE_BY_TEAM_GOALS, ET_TEMPO, FIRST_GOAL_OG_SHARE,
    FIRST_GOAL_PEN_SHARE, H1, H2, OWN_GOAL_SHARE, OWN_GOAL_SINGLE_DIGIT,
    PEN_SHOOTOUT_SPEN, PEN_TILT, SINGLE_DIGIT_FIRST_SHARE,
)
from jumpprob.submit.handlers import Answering, register

# --------------------------------------------------------------------------
# Match result
# --------------------------------------------------------------------------


@register("win")
def win(a: Answering) -> float | None:
    return None if a.side is None else mk.p_win(a.grid, a.side)


@register("draw")
def draw(a: Answering) -> float:
    return mk.p_draw(a.grid)


@register("clean_sheet")
def clean_sheet(a: Answering) -> float | None:
    """The opponent fails to score — a marginal of the grid."""
    if a.side is None:
        return None
    return float(a.grid[:, 0].sum()) if a.side == "a" else float(a.grid[0, :].sum())


@register("win_margin")
def win_margin(a: Answering) -> float | None:
    """Win by at least N."""
    if a.side is None:
        return None
    n = a.threshold or 1
    idx = np.arange(a.grid.shape[0])
    margin = np.subtract.outer(idx, idx)          # rows (a) minus columns (b)
    mask = (margin >= n) if a.side == "a" else (margin <= -n)
    return float(a.grid[mask].sum())


@register("margin_exactly")
def margin_exactly(a: Answering) -> float:
    """Decided by exactly N goals; a draw counts as No.

    The crowd tends to conflate "a one-goal game" with "not a draw" and prices
    this near .73.
    """
    n = a.threshold or 1
    idx = np.arange(a.grid.shape[0])
    return float(a.grid[np.abs(np.subtract.outer(idx, idx)) == n].sum())


# --------------------------------------------------------------------------
# Goal totals
# --------------------------------------------------------------------------


@register("btts")
def btts(a: Answering) -> float:
    return mk.p_btts(a.grid)


@register("total_ge")
def total_ge(a: Answering) -> float:
    return mk.p_total_ge(a.grid, a.threshold)


@register("total_le")
def total_le(a: Answering) -> float:
    return 1.0 - mk.p_total_ge(a.grid, a.threshold + 1)


@register("btts_and_total_ge")
def btts_and_total_ge(a: Answering) -> float:
    return mk.p_btts_and_total_ge(a.grid, a.threshold)


@register("total_exact")
def total_exact(a: Answering) -> float:
    k = a.threshold or 0
    g = a.grid
    return float(sum(g[i, k - i] for i in range(k + 1)
                     if k - i < g.shape[1] and i < g.shape[0]))


@register("total_odd")
def total_odd(a: Answering) -> float:
    """An odd number of goals.

    There is a certain-but-small tilt toward even against a crowd that anchors
    at 0.50: independent Poissons give P(odd) = (1 - e^-2L)/2 < .5, and the
    Dixon-Coles correction (rho < 0) tilts it further.
    """
    g = a.grid
    idx = np.arange(g.shape[0])
    return float(g[np.add.outer(idx, idx) % 2 == 1].sum())


@register("team_goals_ge")
def team_goals_ge(a: Answering) -> float | None:
    if a.side is None:
        return None
    marg = a.grid.sum(axis=1) if a.side == "a" else a.grid.sum(axis=0)
    return float(marg[a.threshold:].sum())


# --------------------------------------------------------------------------
# Halves
# --------------------------------------------------------------------------


@register("ht_tied")
def ht_tied(a: Answering) -> float:
    return float(np.trace(dc_grid(H1 * a.lam_a, H1 * a.lam_b, a.rho)))


@register("ht_scoreless")
def ht_scoreless(a: Answering) -> float:
    """Goalless at halftime — the [0,0] cell of the first-half grid."""
    return float(dc_grid(H1 * a.lam_a, H1 * a.lam_b, a.rho)[0, 0])


@register("ht_leading")
def ht_leading(a: Answering) -> float | None:
    if a.side is None:
        return None
    return mk.p_win(dc_grid(H1 * a.lam_a, H1 * a.lam_b, a.rho), a.side)


@register("goals_gt_half")
def goals_gt_half(a: Answering) -> float | None:
    if a.side is None:
        return None
    share = H1 if a.half == 1 else H2
    return mk.p_win(dc_grid(share * a.lam_a, share * a.lam_b, a.rho), a.side)


@register("team_scores_half")
def team_scores_half(a: Answering) -> float | None:
    if a.side is None:
        return None
    lt, _ = a.lams(a.side)
    return mk.p_scores_in_half(lt, a.half)


@register("score_both_halves")
def score_both_halves(a: Answering) -> float | None:
    if a.side is None:
        return None
    lt, _ = a.lams(a.side)
    return float(mk.p_scores_in_half(lt, 1) * mk.p_scores_in_half(lt, 2))


@register("goal_each_half")
def goal_each_half(a: Answering) -> float:
    """At least one goal by either team in each half."""
    return float((1 - np.exp(-H1 * a.lam_tot)) * (1 - np.exp(-H2 * a.lam_tot)))


@register("half_total_ge")
def half_total_ge(a: Answering) -> float:
    from scipy.stats import poisson
    share = H1 if a.half == 1 else H2
    return float(1 - poisson.cdf(a.threshold - 1, share * a.lam_tot))


@register("either_team_half_goals_ge")
def either_team_half_goals_ge(a: Answering) -> float:
    """Some (team, half) cell reaches the threshold.

    The four cells are near-independent Poissons: the Dixon-Coles correction
    only moves the low-score joint cells, not a single team-half tail.
    """
    from scipy.stats import poisson
    n = (a.threshold or 2) - 1
    p_none = 1.0
    for lam in (a.lam_a, a.lam_b):
        for share in (H1, H2):
            p_none *= float(poisson.cdf(n, share * lam))
    return float(1.0 - p_none)


@register("win_both_halves")
def win_both_halves(a: Answering) -> float:
    """Either team wins both halves outright.

    Halves are combined as independent — the Monte-Carlo engine confirms goal
    frailty across halves is negligible. The crowd over-prices this at ~.35 on
    the reasoning "if one team is better it wins both"; the truth is nearer .18.
    """
    g1 = dc_grid(H1 * a.lam_a, H1 * a.lam_b, a.rho)
    g2 = dc_grid(H2 * a.lam_a, H2 * a.lam_b, a.rho)
    return float(mk.p_win(g1, "a") * mk.p_win(g2, "a")
                 + mk.p_win(g1, "b") * mk.p_win(g2, "b"))


@register("h2_more_goals_than_h1")
def h2_more_goals_than_h1(a: Answering) -> float:
    from scipy.stats import poisson
    from jumpprob.models.count_model import p_a_gt_b
    xs = np.arange(25)
    return p_a_gt_b(poisson.pmf(xs, H2 * a.lam_tot), poisson.pmf(xs, H1 * a.lam_tot))


@register("halves_equal_goals")
def halves_equal_goals(a: Answering) -> float:
    """The two halves produce the same number of goals.

    Sum over totals: given N goals, each falls in the first half independently
    with probability H1, so P(equal | N = 2k) is Binomial(2k, H1).pmf(k) and odd
    totals contribute nothing. The binomial split was validated on 11,658
    martj42 matches.
    """
    from scipy.stats import binom
    return float(sum(pg * binom.pmf(k // 2, k, H1)
                     for k, pg in enumerate(a.total_goals_pmf()) if k % 2 == 0))


# --------------------------------------------------------------------------
# First goal
# --------------------------------------------------------------------------


@register("first_goal_match")
def first_goal_match(a: Answering) -> float | None:
    """The side scores the match's first goal.

    P(scores first | any goal) x P(at least one goal), which inherits the market
    view through the anchored intensities.
    """
    if a.side is None:
        return None
    lt, _ = a.lams(a.side)
    return float(lt / a.lam_tot * (1 - np.exp(-a.lam_tot)))


@register("first_goal_half")
def first_goal_half(a: Answering) -> float | None:
    if a.side is None:
        return None
    lt, _ = a.lams(a.side)
    share = H1 if a.half == 1 else H2
    return float(lt / a.lam_tot * (1 - np.exp(-share * a.lam_tot)))


@register("first_goal_2h")
def first_goal_2h(a: Answering) -> float:
    """The match's first goal falls in the second half.

    Equivalently: no goal in the first half and at least one in the second. A
    goalless match counts as No.
    """
    return float(np.exp(-H1 * a.lam_tot) * (1 - np.exp(-H2 * a.lam_tot)))


@register("first_goal_and_win")
def first_goal_and_win(a: Answering) -> float | None:
    """Scores first and goes on to win.

    P(first) x P(win | scored first), where the conditional is the side's own
    win probability shifted on the logit scale so that an average first scorer
    wins 75.7% — measured on 15,381 internationals.
    """
    if a.side is None:
        return None
    lt, _ = a.lams(a.side)
    tot = a.lam_tot
    p_first = lt / tot * (1 - np.exp(-tot))
    p_win = mk.p_win(a.grid, a.side)
    z = np.log(p_win / (1 - p_win)) + 1.34
    return float(p_first / (1 - np.exp(-tot)) * (1 / (1 + np.exp(-z))) * (1 - np.exp(-tot)))


@register("first_goal_and_opp_2h")
def first_goal_and_opp_2h(a: Answering) -> float | None:
    """The side scores first AND the named opponent scores in the second half.

    Treated as independent given the intensities; the residual coupling is weak.
    """
    if a.side is None or a.q.other is None:
        return None
    other_c = canonical(a.q.other)
    other_side = ("a" if other_c == a.ctx.team_a
                  else "b" if other_c == a.ctx.team_b else None)
    if other_side is None:
        return None
    lt, _ = a.lams(a.side)
    tot = a.lam_tot
    p_first = lt / tot * (1 - np.exp(-tot))
    lo_t, _ = a.lams(other_side)
    return float(p_first * mk.p_scores_in_half(lo_t, 2))


@register("first_goal_assisted")
def first_goal_assisted(a: Answering) -> float:
    """The first goal is credited with an assist.

    Only open-play goals carry one, so strip the penalty and own-goal shares of
    first goals before applying the assisted share. The goalless case is exactly
    the grid's [0,0] cell, which is market-set when anchored.
    """
    p_goal = 1.0 - float(a.grid[0, 0])
    return float(p_goal * (1.0 - FIRST_GOAL_PEN_SHARE - FIRST_GOAL_OG_SHARE)
                 * ASSISTED_OPEN_SHARE)


@register("first_goal_single_digit")
def first_goal_single_digit(a: Answering) -> float:
    """The first scorer wears a single-digit shirt.

    P(a goal exists in regulation or extra time) x P(single digit | goal). The
    conditional is allocated bottom-up from the two teams' own single-digit goal
    shares, intensity-weighted, since Poisson superposition makes the first
    scorer track each player's goal intensity. Falls back to the tournament base
    if either team is unknown. Own goals are folded in separately — that scorer's
    shirt is effectively a random defender's.
    """
    p00 = float(a.grid[0, 0])
    lam_et = ET_TEMPO * a.lam_tot / 3.0
    p_goal_ever = 1.0 - p00 * float(np.exp(-lam_et))
    sh_a = a.runner.single_digit_share.get(a.ctx.team_a)
    sh_b = a.runner.single_digit_share.get(a.ctx.team_b)
    if sh_a is not None and sh_b is not None:
        share = (a.lam_a * sh_a + a.lam_b * sh_b) / a.lam_tot
    else:
        share = SINGLE_DIGIT_FIRST_SHARE
    share = (1 - OWN_GOAL_SHARE) * share + OWN_GOAL_SHARE * OWN_GOAL_SINGLE_DIGIT
    return p_goal_ever * share


@register("leads_at_any_point")
def leads_at_any_point(a: Answering) -> float | None:
    """The side is ahead at some point in the match.

    Approximated as P(scores first) plus a small comeback allowance — martj42
    records .114 fall-behind-then-lead events per match, roughly +.05 per side.
    The question excludes only the shootout, so a first lead taken in extra time
    still counts. Crude, but bounded and monotone; full path dynamics are not
    worth simulating for one question.
    """
    if a.side is None:
        return None
    lt, lo = a.lams(a.side)
    lam_t = lt + lo
    p_first = (lt / lam_t) * (1 - np.exp(-lam_t)) if lam_t > 0 else 0.0
    p_et = (float(mk.p_draw(a.grid)) * (lt / lam_t if lam_t > 0 else 0.5)
            * (1 - np.exp(-ET_TEMPO * lam_t / 3)))
    return float(np.clip(p_first + 0.05 + p_et, 0.01, 0.97))


# --------------------------------------------------------------------------
# Knockout: extra time, shootouts, advancement
# --------------------------------------------------------------------------


def _et_grid(a: Answering, lam_a: float, lam_b: float) -> np.ndarray:
    """A 30-minute extra-time grid at the same (anchored) intensities."""
    return dc_grid(ET_TEMPO / 3 * lam_a, ET_TEMPO / 3 * lam_b, a.rho)


@register("extra_time")
def extra_time(a: Answering) -> float:
    """Extra time happens exactly when the tie is level after 90; the anchored
    grid's draw mass is the de-vigged book's X price."""
    return float(mk.p_draw(a.grid))


@register("pen_shootout")
def pen_shootout(a: Answering) -> float:
    """A shootout needs a draw after 90 AND a draw after extra time."""
    et = _et_grid(a, a.lam_a, a.lam_b)
    return float(mk.p_draw(a.grid) * mk.p_draw(et))


@register("advance")
def advance(a: Answering) -> float | None:
    """The side advances, i.e. wins this knockout tie by any route.

        P(advance) = P(win in 90) + P(draw in 90) x P(advance | level)
        P(advance | level) = P(win ET) + P(ET also drawn) x P(win shootout)

    Team shootout evidence overrides the win-share tilt where we have it: the
    90-minute win share says nothing about penalty takers or goalkeepers, and it
    is wrong-signed for Argentina against Spain.
    """
    if a.side is None:
        return None
    other = a.other_side
    p_win90 = mk.p_win(a.grid, a.side)
    p_draw90 = mk.p_draw(a.grid)
    p_lose90 = mk.p_win(a.grid, other)
    lt, lo = a.lams(a.side)
    et = _et_grid(a, lt, lo)
    p_win_et, p_et_draw = mk.p_win(et, "a"), mk.p_draw(et)

    r = p_win90 / (p_win90 + p_lose90) if (p_win90 + p_lose90) > 0 else 0.5
    s_pen = 0.5 + PEN_TILT * (r - 0.5)
    team = a.ctx.team_a if a.side == "a" else a.ctx.team_b
    opp = a.ctx.team_b if a.side == "a" else a.ctx.team_a
    if team in PEN_SHOOTOUT_SPEN:
        s_pen = PEN_SHOOTOUT_SPEN[team]
    elif opp in PEN_SHOOTOUT_SPEN:
        s_pen = 1.0 - PEN_SHOOTOUT_SPEN[opp]

    q = p_win_et + p_et_draw * s_pen
    return float(p_win90 + p_draw90 * q)


# --------------------------------------------------------------------------
# Cross-family: goals against a count distribution
# --------------------------------------------------------------------------


@register("any_player_goals_ge")
def any_player_goals_ge(a: Answering) -> float | None:
    """Some player scores a brace.

    Rides the team goal distribution through the measured concentration curve —
    the more a team scores, the likelier one player takes two of them. The curve
    is only fit for the 2+ line.
    """
    if (a.threshold or 2) > 2:
        return None

    def team_any(marg: np.ndarray) -> float:
        return float(sum(marg[g] * (BRACE_BY_TEAM_GOALS[g] if g < len(BRACE_BY_TEAM_GOALS)
                                    else 1.0)
                         for g in range(len(marg))))

    return float(1 - (1 - team_any(a.grid.sum(axis=1))) * (1 - team_any(a.grid.sum(axis=0))))


@register("cards_gt_goals", needs_counts=True)
def cards_gt_goals(a: Answering) -> float | None:
    """More cards than goals. Independence between the two is second-order here."""
    cm = a.cm
    if "cards" not in cm.params:
        return None
    mu_c = sum(cm.mu("cards", h, a.lam_a, a.lam_b) + cm.mu("cards", h, a.lam_b, a.lam_a)
               for h in (1, 2))
    cpmf = cm.pmf("cards", mu_c)
    return float(sum(pg * float(cpmf[k + 1:].sum())
                     for k, pg in enumerate(a.total_goals_pmf()) if k + 1 < len(cpmf)))
