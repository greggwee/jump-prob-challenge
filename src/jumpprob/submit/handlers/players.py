"""Individual player questions: goals, assists, shots on target, races, saves.

This is where the model is weakest and the bookmaker strongest. Books price
stars off far better information than any public dataset carries, so anything
with a matched book line is anchored hard downstream in the pricing ladder. The
handlers here supply the model side of that blend, and the whole answer for the
many players no book quotes.

Two failure modes shaped this file, both discovered by losing points to them:

* Over-shrinkage. A population prior plus heavy shrinkage pulls elite outliers
  toward "average World Cup player" — exactly the players these questions are
  about. The prior is now centred on the individual wherever we can resolve them.
* Silent name-join misses. A subject like "Kylian Mbappé (France, #10)" matched
  no player, so the best striker on the pitch collapsed to the population floor.
  The parenthetical is stripped before resolving.
"""

from __future__ import annotations

import numpy as np

from jumpprob.models.count_model import p_a_gt_b
from jumpprob.models.market_anchor import inv_logit, logit
from jumpprob.normalize.players import ascii_lower
from jumpprob.normalize.teams import canonical
from jumpprob.submit.constants import (
    CROWD_HUG_MIN_N, H1, H2, MINUTE_SCENARIOS, PLAYER_FORM_RATE_OVERRIDES,
    SOT_RACE_CROWD_W, SOT_RACE_FLOOR, SOT_RACE_FORM_RATES, SOT_RACE_MINUTES,
    UNRESOLVED_PLAYER_TEMPLATE_W,
)
from jumpprob.submit.handlers import Answering, register


def strip_subject_suffix(name: str) -> str:
    """Drop a trailing "(Team, #N)" from a question's subject."""
    return name.split("(")[0].strip() or name


@register("player_scores", "player_sot_ge", "player_ga", needs_counts=True)
def player_prop(a: Answering) -> float:
    """P(the player reaches the threshold), over a mixture of minutes scenarios.

    The per-minute rate is scaled by this match's intensity against the league
    average, so a player in a high-tempo tie is priced higher than the same
    player in a cagey one. Half-scoped questions scale exposure by the half's
    share of the match.
    """
    from scipy.stats import poisson

    arch = a.q.archetype
    rates = a.runner.player_rate(a.q.subject)
    rate = {"player_sot_ge": rates.sot, "player_scores": rates.goal,
            "player_ga": rates.ga}[arch]
    rate = PLAYER_FORM_RATE_OVERRIDES.get(
        (arch, ascii_lower(strip_subject_suffix(a.q.subject or ""))), rate)

    scale = a.tempo_scale
    if a.half:
        scale *= H1 if a.half == 1 else H2

    k = a.threshold or 1
    p = float(sum(w * (1.0 - poisson.cdf(k - 1, rate * scale * m))
                  for w, m in MINUTE_SCENARIOS))

    # A player we could not resolve at all — no club prior, no tournament
    # minutes — would otherwise ship the naked population floor. That is the
    # error class that priced Haaland at 8%. Blend toward the archetype's crowd
    # level instead. Only for 1+ lines: the template mixes thresholds and reads
    # as a 1+ level.
    if not rates.resolved and k == 1:
        t = (a.runner.crowd_template or {}).get(arch) or {}
        if (t.get("n") or 0) >= CROWD_HUG_MIN_N and t.get("mean") is not None:
            p = ((1 - UNRESOLVED_PLAYER_TEMPLATE_W) * p
                 + UNRESOLVED_PLAYER_TEMPLATE_W * t["mean"] / 100.0)
    return p


@register("player_sot_race", needs_counts=True)
def player_sot_race(a: Answering) -> float | None:
    """One named player records strictly more shots on target than another.

    A tie settles as No, so this convolves two minutes-mixed Poissons and asks
    for a strict win. Star rates are pinned from tournament form because the
    shrunk population table lags them badly.

    No crowd template exists for this archetype, so the shrink toward the field
    lives here: a logit blend toward the ~50 the crowd puts on star races, plus
    a floor. Both are guards against pre-lineup uncertainty — once real team news
    lands, the raw number is the honest one and the guards come off.
    """
    if not a.q.subject or not a.q.other:
        return None
    from scipy.stats import poisson

    scale = a.tempo_scale
    ks = np.arange(13)
    teams = (a.ctx.team_a, a.ctx.team_b)
    flag_s = a.runner.lineup_flag(a.q.subject, teams)
    flag_o = a.runner.lineup_flag(a.q.other, teams)

    def race_pmf(name: str, flag: str | None) -> np.ndarray:
        key = ascii_lower(strip_subject_suffix(name))
        rate = SOT_RACE_FORM_RATES.get(key)
        if rate is None:
            # player_rate already folds the club-to-country discount into its
            # prior — applying it again here would double-count.
            rate = a.runner.player_rate(name).sot
        mix = ([(1.0, 0.0)] if flag == "out"
               else [(1.0, 25.0)] if flag == "bench"
               else SOT_RACE_MINUTES.get(key, MINUTE_SCENARIOS))
        return sum(w * poisson.pmf(ks, rate * scale * m) for w, m in mix)

    p = p_a_gt_b(race_pmf(a.q.subject, flag_s), race_pmf(a.q.other, flag_o))
    if flag_s or flag_o:
        return float(np.clip(p, 0.02, 0.98))
    p = float(inv_logit((1 - SOT_RACE_CROWD_W) * logit(p)))   # logit(0.5) == 0
    return max(p, SOT_RACE_FLOOR)


@register("player_saves_ge", needs_counts=True)
def player_saves_ge(a: Answering) -> float | None:
    """A goalkeeper makes at least N saves.

    Saves equal shots on target faced minus goals conceded — an identity that
    held in all 628 StatsBomb matches. So sum over the opponent's shots-on-target
    distribution, each shot converting with probability lam_opp / mu_sot. There
    is no book market for saves, but keeper quality rides the anchored
    intensities: a side the market expects to be pinned back faces more shots.
    """
    keeper_team = canonical(a.q.other) if a.q.other else None
    kside = ("a" if keeper_team == a.ctx.team_a
             else "b" if keeper_team == a.ctx.team_b else None)
    if kside is None:
        return None
    from scipy.stats import binom

    cm = a.cm
    if "shots_on_target" not in cm.params:
        return None
    lam_opp, lam_own = (a.lam_b, a.lam_a) if kside == "a" else (a.lam_a, a.lam_b)
    mu_opp = sum(cm.mu("shots_on_target", h, lam_opp, lam_own) for h in (1, 2))
    spmf = cm.pmf("shots_on_target", mu_opp)
    mu_sot = float(sum(k * w for k, w in enumerate(spmf)))
    conv = min(0.9, lam_opp / mu_sot) if mu_sot > 0 else 0.3
    t = a.threshold or 1
    return float(sum(w * float(binom.sf(t - 1, k, 1 - conv))
                     for k, w in enumerate(spmf) if k >= t))
