"""Discrete match events: timing windows, substitutions, penalties, VAR, own goals.

These are the questions no bookmaker quotes, which makes them the most valuable
ones on the slate — beating the field here needs no edge over a sharp price,
only an honest estimate where the crowd is guessing.

Two devices do most of the work. Timing windows integrate the fitted
minute-by-minute intensity profile over a slot range, so "a goal before the
first hydration break" is the profile's share of intensity in minutes 0-22
rather than a flat 22/90. And Poisson thinning turns a share into a probability:
if headers are 21% of open-play goals, P(a headed goal) is 1 - exp(-0.21 x total
intensity), which correctly makes it scale with how open the match is expected
to be.
"""

from __future__ import annotations

import numpy as np

from jumpprob.models.dynamics import BUCKET
from jumpprob.simulate import markets as mk
from jumpprob.simulate import mc_engine
from jumpprob.submit.constants import (
    ET_TEMPO, FIRST_SUB_TILT, P_PEN_MATCH, P_RED_MATCH, PEN_CONV,
    PEN_CONV_DEFAULT, PEN_FINAL_UPLIFT, PEN_LAMBDA_WC26, PEN_SCORED_BOUNDS,
    VAR_OFR_LAMBDA, WINDOW,
)
from jumpprob.submit.handlers import Answering, register

# --------------------------------------------------------------------------
# Hydration-break and stoppage windows
# --------------------------------------------------------------------------


def window_event(a: Answering, fam: str, lo: int, hi: int, thr: int) -> float | None:
    """P(at least ``thr`` events of ``fam`` in match minutes [lo, hi)).

    The minute profile's bucket weights give the window's share of the family's
    total intensity; scale the match's expected total by that share. Goals are
    Poisson; counts inherit the count model's overdispersion.
    """
    dyn = a.runner.dyn
    if dyn is None or fam not in dyn.alpha:
        return None
    w = np.exp(np.asarray(dyn.alpha[fam])[BUCKET])
    frac = float(w[lo:hi].sum() / w.sum())
    if fam == "goal":
        return float(1 - np.exp(-a.lam_tot * frac))
    cm = a.cm
    if cm is None:
        return None
    cf = mc_engine.COUNT_FAMILY[fam]
    mu_tot = sum(cm.mu(cf, h, a.lam_a, a.lam_b) + cm.mu(cf, h, a.lam_b, a.lam_a)
                 for h in (1, 2))
    return float(cm.pmf(cf, mu_tot * frac)[thr:].sum())


@register(*WINDOW)
def timing_window(a: Answering) -> float | None:
    return window_event(a, *WINDOW[a.q.archetype], a.threshold or 1)


@register("goal_1h_stoppage", "goal_2h_stoppage")
def goal_in_stoppage(a: Answering) -> float:
    """A goal in one half's added time, beyond the 90-minute profile's reach.

    Thinned from the match's total intensity by that period's measured share of
    goals — second-half stoppage carries four times the first's.
    """
    key = ("goal_share_1h_stoppage" if a.q.archetype == "goal_1h_stoppage"
           else "goal_share_2h_stoppage")
    return float(1 - np.exp(-a.lam_tot * a.runner.event_base[key]))


@register("goal_stoppage")
def goal_in_either_stoppage(a: Answering) -> float:
    """A goal in either half's added time — both windows' shares combined."""
    share = (a.runner.event_base["goal_share_1h_stoppage"]
             + a.runner.event_base["goal_share_2h_stoppage"])
    return float(1 - np.exp(-a.lam_tot * share))


# --------------------------------------------------------------------------
# Substitutions
# --------------------------------------------------------------------------


@register("sub_before_ht")
def sub_before_ht(a: Answering) -> float:
    """A substitution before halftime — essentially the injury base rate."""
    return float(a.runner.event_base["sub_before_ht"])


@register("sub_at_ht")
def sub_at_ht(a: Answering) -> float:
    """A change made during the interval itself."""
    return float(a.runner.event_base["sub_at_ht"])


@register("substitute_scores", "substitute_ga")
def substitute_scores(a: Answering) -> float:
    """A substitute scores (or, for the second archetype, scores or assists).

    Substitutes arrive around the 72nd minute and account for ~17% of all goals,
    so the chance one scores tracks the match's goal expectation: scale the base
    rate by this match's total intensity against the league average, and a
    blowout prices above a tight game.

    The "or assists" variant widens the hazard by the population assists-to-goals
    ratio, capped: the measured 1.64 is the least-validated constant in this file
    and the cap is worth a few points of protection.
    """
    s = -np.log(1.0 - a.runner.event_base["substitute_scores"]) / a.league_lam_total
    if a.q.archetype == "substitute_ga":
        s *= min(a.runner.prior_ga / max(a.runner.prior_goal, 1e-9), 1.5)
    return float(1 - np.exp(-a.lam_tot * s))


@register("subs_total_ge")
def subs_total_ge(a: Answering) -> float | None:
    """At least N substitutions in the match.

    Regulation substitutions are quota consumption, not a count process — the
    WC2022 variance-to-mean ratio is 0.175. So this uses stratified empirical
    exceedance rather than a fitted distribution, mixed by P(draw after 90),
    because matches heading for extra time visibly hold substitutions back.
    Thresholds outside the table fall through to the template audit.
    """
    key_ne = "subs_ge%d_nonET" % (a.threshold or 0)
    key_et = "subs_ge%d_ET" % (a.threshold or 0)
    if key_ne not in a.runner.event_base or key_et not in a.runner.event_base:
        return None
    p_draw = float(mk.p_draw(a.grid))
    return float((1 - p_draw) * a.runner.event_base[key_ne] + p_draw * a.runner.event_base[key_et])


@register("first_sub_team")
def first_sub_team(a: Answering) -> float | None:
    """Which side makes the first substitution.

    Game-state dominated — the trailing side changes things first — so tilt the
    coin by the win-probability gap. Kept bounded near 50/50: manager
    idiosyncrasy and injuries add a lot of noise.
    """
    if a.side is None:
        return None
    gap = float(mk.p_win(a.grid, a.other_side)) - float(mk.p_win(a.grid, a.side))
    return float(np.clip(0.5 + FIRST_SUB_TILT * gap, 0.40, 0.60))


@register("plays_full_match")
def plays_full_match(a: Answering) -> float:
    """A named starter is still on at the whistle. No book market for this."""
    return float(a.runner.event_base["player_full90"])


# --------------------------------------------------------------------------
# Penalties, cards, VAR
# --------------------------------------------------------------------------


@register("penalty_awarded")
def penalty_awarded(a: Answering) -> float:
    return P_PEN_MATCH


@register("red_card")
def red_card(a: Answering) -> float:
    return P_RED_MATCH


@register("pen_or_red")
def pen_or_red(a: Answering) -> float:
    return 1 - (1 - P_PEN_MATCH) * (1 - P_RED_MATCH)


@register("pen_scored")
def pen_scored(a: Answering) -> float:
    """A penalty is awarded AND converted.

    Poisson-thinned so multiple penalties are handled correctly. Uses WC26's own
    (low) award rate rather than the stale six-tournament average, with
    conversion intensity-weighted across the two teams' named takers.

    Deliberately not crowd-hugged: the crowd anchors this on the awarded-penalty
    family around 30, which is the wrong reference. Hard bounds are the only
    safety net a brand-new archetype gets at triple stakes.
    """
    conv = (a.lam_a * PEN_CONV.get(a.ctx.team_a, PEN_CONV_DEFAULT)
            + a.lam_b * PEN_CONV.get(a.ctx.team_b, PEN_CONV_DEFAULT)) / a.lam_tot
    p = 1.0 - float(np.exp(-PEN_LAMBDA_WC26 * PEN_FINAL_UPLIFT * conv))
    return float(np.clip(p, *PEN_SCORED_BOUNDS))


@register("var_ofr")
def var_ofr(a: Answering) -> float:
    """At least one on-field VAR monitor review, shootouts excluded.

    Poisson at the measured per-90 rate, with exposure extended by P(extra time)
    times a third of a match.
    """
    lam = VAR_OFR_LAMBDA * (1.0 + float(mk.p_draw(a.grid)) / 3.0)
    return float(1.0 - np.exp(-lam))


@register("first_card_before_goal", needs_counts=True)
def first_card_before_goal(a: Answering) -> float | None:
    """A card is shown before the first goal.

    A competing-risks race between two time-varying hazards. Cards are
    late-loaded relative to goals, so assuming constant rates overstates
    card-first by around nine points.

    Settlement is ambiguous when a card is shown but no goal ever arrives. We
    weight the permissive reading (that counts as Yes) at 0.6 against the strict
    one (a goal must follow) at 0.4.
    """
    dyn = a.runner.dyn
    if dyn is None or "card" not in dyn.alpha or "goal" not in dyn.alpha:
        return None
    cm = a.cm
    mu_c = sum(cm.mu("cards", h, a.lam_a, a.lam_b) + cm.mu("cards", h, a.lam_b, a.lam_a)
               for h in (1, 2))
    wc = np.exp(np.asarray(dyn.alpha["card"])[BUCKET]); wc = wc / wc.sum()
    wg = np.exp(np.asarray(dyn.alpha["goal"])[BUCKET]); wg = wg / wg.sum()
    hc, hg = mu_c * wc, a.lam_tot * wg
    surv = np.exp(-(np.cumsum(hc + hg) - (hc + hg)))        # survival to slot start
    p_permissive = float(np.sum(hc * surv))
    tail_goal = np.concatenate([np.cumsum(hg[::-1])[::-1][1:], [0.0]])
    p_strict = float(np.sum(hc * surv * (1 - np.exp(-tail_goal))))
    return float(np.clip(0.6 * p_permissive + 0.4 * p_strict, 0.01, 0.97))


# --------------------------------------------------------------------------
# Goal flavours
# --------------------------------------------------------------------------


@register("own_goal")
def own_goal(a: Answering) -> float:
    """Flat base rate. Thinning by total goals was tested and lost — own goals
    genuinely do not scale with how much a match is expected to produce."""
    return float(a.runner.event_base["own_goal"])


@register("goal_outside_box")
def goal_outside_box(a: Answering) -> float:
    return float(1.0 - np.exp(-a.runner.event_base["outside_box_share"] * a.lam_tot))


@register("goal_header")
def goal_header(a: Answering) -> float:
    return float(1.0 - np.exp(-a.runner.event_base["header_share"] * a.lam_tot))
