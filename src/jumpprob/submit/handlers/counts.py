"""Corners, cards, shots, fouls, offsides — the negative-binomial count family.

Every handler here reads the fitted count model, whose mean is a function of the
match's goal intensities, so counts inherit the market anchor too: a book that
makes a match one-sided implies more corners for the favourite without anyone
quoting a corners line.

The count model is required, so these handlers are all registered with
``needs_counts=True`` and return None when it is unavailable.
"""

from __future__ import annotations

import numpy as np

from jumpprob.models.count_model import p_a_gt_b, p_sum_ge
from jumpprob.submit.constants import (
    CARD_STOPPAGE_SHARE, CORNERS_SHOTS_RHO, DISTINCT_SHOOTERS_BASE,
    EACH_TEAM_SOT_RHO, SHOOTER_OCC_B0, SHOOTER_OCC_B1, SOTCONC_BY_TEAM_SOT,
)
from jumpprob.submit.handlers import Answering, register

# Question wording -> the count family it is really about.
_COMPARISON_FAMILY = {"corners_gt": "corners", "sot_gt": "shots_on_target",
                      "fouls_gt": "fouls", "cards_gt": "cards"}


def _copula_bump(p1: float, p2: float, rho: float) -> float:
    """Gaussian-copula-lite correction to an independence product.

    Two indicators that both load on attacking intensity are not independent
    even after conditioning on it. This adds rho x the geometric mean of their
    variances, which is the linearised joint correction.
    """
    return rho * float(np.sqrt(max(p1 * (1 - p1) * p2 * (1 - p2), 0.0)))


# --------------------------------------------------------------------------
# Head-to-head comparisons
# --------------------------------------------------------------------------


@register("corners_gt", "sot_gt", "fouls_gt", "cards_gt", needs_counts=True)
def comparison(a: Answering) -> float | None:
    """One side records strictly more of something than the other."""
    if a.side is None:
        return None
    fam = _COMPARISON_FAMILY[a.q.archetype]
    return p_a_gt_b(a.side_pmf(fam, a.side), a.side_pmf(fam, a.other_side))


@register("corners_and_shots", needs_counts=True)
def corners_and_shots(a: Answering) -> float | None:
    """More corners AND more total shots than the opponent.

    Both indicators load on attacking intensity, so they are positively coupled
    well beyond what conditioning on goal intensity removes — the measured
    residual correlation is 0.41 over 314 matches, because a team having an
    attacking day gets both. We use a deliberately lower 0.30.
    """
    if a.side is None:
        return None
    other = a.other_side
    p_c = p_a_gt_b(a.side_pmf("corners", a.side), a.side_pmf("corners", other))
    p_s = p_a_gt_b(a.side_pmf("shots", a.side), a.side_pmf("shots", other))
    return float(np.clip(p_c * p_s + _copula_bump(p_c, p_s, CORNERS_SHOTS_RHO), 0.01, 0.99))


# --------------------------------------------------------------------------
# Team thresholds
# --------------------------------------------------------------------------


@register("team_corners_ge", needs_counts=True)
def team_corners_ge(a: Answering) -> float | None:
    if a.side is None:
        return None
    return float(a.side_pmf("corners", a.side)[a.threshold:].sum())


@register("team_sot_ge", needs_counts=True)
def team_sot_ge(a: Answering) -> float | None:
    if a.side is None:
        return None
    return float(a.side_pmf("shots_on_target", a.side)[a.threshold:].sum())


@register("team_cards_ge", needs_counts=True)
def team_cards_ge(a: Answering) -> float | None:
    if a.side is None:
        return None
    return float(a.side_pmf("cards", a.side)[a.threshold:].sum())


@register("offsides_ge", needs_counts=True)
def offsides_ge(a: Answering) -> float | None:
    """Offsides get their own calibration, not the shared count one: a
    definition gap between StatsBomb and the platform makes them over-priced
    live, and the pooled count Platt shifts them the wrong way."""
    if a.side is None:
        return None
    return float(a.side_pmf("offsides", a.side)[a.threshold:].sum())


# --------------------------------------------------------------------------
# Match totals
# --------------------------------------------------------------------------


@register("cards_ge", needs_counts=True)
def cards_ge(a: Answering) -> float:
    return p_sum_ge(a.side_pmf("cards", "a"), a.side_pmf("cards", "b"), a.threshold)


@register("corners_total_ge", needs_counts=True)
def corners_total_ge(a: Answering) -> float:
    return p_sum_ge(a.side_pmf("corners", "a"), a.side_pmf("corners", "b"), a.threshold)


@register("total_sot_ge", "total_sot_half_ge", needs_counts=True)
def total_sot_ge(a: Answering) -> float:
    return p_sum_ge(a.side_pmf("shots_on_target", "a"),
                    a.side_pmf("shots_on_target", "b"), a.threshold)


@register("total_shots_ge", needs_counts=True)
def total_shots_ge(a: Answering) -> float:
    """Total shots on and off target — modelled from real shot data rather than
    inferred from shots on target."""
    return p_sum_ge(a.side_pmf("shots", "a"), a.side_pmf("shots", "b"), a.threshold)


@register("offsides_total_ge", needs_counts=True)
def offsides_total_ge(a: Answering) -> float:
    return p_sum_ge(a.side_pmf("offsides", "a"), a.side_pmf("offsides", "b"), a.threshold)


# --------------------------------------------------------------------------
# Both teams clear a threshold
# --------------------------------------------------------------------------


@register("both_teams_sot_ge", needs_counts=True)
def both_teams_sot_ge(a: Answering) -> float:
    """Both teams reach the threshold, treated as independent given intensity."""
    pa = float(a.side_pmf("shots_on_target", "a")[a.threshold:].sum())
    pb = float(a.side_pmf("shots_on_target", "b")[a.threshold:].sum())
    return pa * pb


@register("each_team_sot_ge", needs_counts=True)
def each_team_sot_ge(a: Answering) -> float:
    """Deliberately a separate archetype from ``both_teams_sot_ge``.

    That one sits in the risk gate behind a threshold-blind crowd template fit
    at the 1+ line, a combination that shipped ~0.56 on a ~0.20 event at the 4+
    line. This handler prices the joint honestly, with only the small negative
    residual dependence — the raw correlation is matchup mix that the anchored
    marginals already encode.
    """
    pa = float(a.side_pmf("shots_on_target", "a")[a.threshold:].sum())
    pb = float(a.side_pmf("shots_on_target", "b")[a.threshold:].sum())
    return float(np.clip(pa * pb + _copula_bump(pa, pb, EACH_TEAM_SOT_RHO), 0.01, 0.99))


@register("both_teams_cards", needs_counts=True)
def both_teams_cards(a: Answering) -> float:
    pa = float(a.side_pmf("cards", "a")[a.threshold:].sum())
    pb = float(a.side_pmf("cards", "b")[a.threshold:].sum())
    return pa * pb


# --------------------------------------------------------------------------
# Card timing
# --------------------------------------------------------------------------


@register("card_each_half", needs_counts=True)
def card_each_half(a: Answering) -> float:
    """At least one card in each half.

    The count model's mean already carries the fitted half split (cards run
    roughly .36/.64), so we read P(>=1) off each half directly. Halves are
    treated as independent; a shared-frailty correction is worth under a point.
    """
    def half_mu(h: int) -> float:
        return (a.cm.mu("cards", h, a.lam_a, a.lam_b)
                + a.cm.mu("cards", h, a.lam_b, a.lam_a))

    p1 = 1.0 - float(a.cm.pmf("cards", half_mu(1))[0])
    p2 = 1.0 - float(a.cm.pmf("cards", half_mu(2))[0])
    return float(p1 * p2)


@register("card_stoppage", needs_counts=True)
def card_stoppage(a: Answering) -> float:
    """At least one card in either half's stoppage time.

    Cards are late-loaded, so stoppage carries more than its raw share of
    minutes. Low confidence: the 90-minute profile truncates added time, and
    WC26 card levels ran below history. Priced conservatively and left for the
    crowd-proxy clamp to bound.
    """
    mu_c = sum(a.cm.mu("cards", h, a.lam_a, a.lam_b) + a.cm.mu("cards", h, a.lam_b, a.lam_a)
               for h in (1, 2))
    return float(1.0 - np.exp(-mu_c * CARD_STOPPAGE_SHARE))


# --------------------------------------------------------------------------
# Shot concentration across a squad
# --------------------------------------------------------------------------


@register("distinct_shooters_ge", needs_counts=True)
def distinct_shooters_ge(a: Answering) -> float | None:
    """At least five different players from one team attempt a shot.

    An occupancy logistic over the team's shot distribution. Shot volume is the
    sufficient statistic — the team residual is zero. Fit for the 5+ line only;
    anything else falls through to the template audit. We ship a half-step
    toward the population base because the residual risk is all in the mean.
    """
    if a.side is None or a.threshold != 5:
        return None
    pmf = a.side_pmf("shots", a.side)
    occ = 1.0 / (1.0 + np.exp(-(SHOOTER_OCC_B0 + SHOOTER_OCC_B1 * np.arange(len(pmf)))))
    p = float(np.dot(pmf, occ))
    return float(0.5 * p + 0.5 * DISTINCT_SHOOTERS_BASE)


def _team_sot_concentration(a: Answering, s: str) -> float:
    """P(some player on side ``s`` records 2+ shots on target)."""
    pmf = a.side_pmf("shots_on_target", s)
    return float(sum(pmf[t] * (SOTCONC_BY_TEAM_SOT[t] if t < len(SOTCONC_BY_TEAM_SOT) else 1.0)
                     for t in range(len(pmf))))


@register("any_player_sot_ge", needs_counts=True)
def any_player_sot_ge(a: Answering) -> float:
    """Any player in the match records 2+ shots on target."""
    return float(1 - (1 - _team_sot_concentration(a, "a")) * (1 - _team_sot_concentration(a, "b")))


@register("any_team_player_sot_ge", needs_counts=True)
def any_team_player_sot_ge(a: Answering) -> float | None:
    """The same, scoped to one named team. Curve is fit for the 2+ line only."""
    if a.side is None or (a.threshold or 2) != 2:
        return None
    return _team_sot_concentration(a, a.side)
