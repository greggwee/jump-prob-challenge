"""v2 minute-level Monte Carlo simulator with game-state coupling.

Architecture (see V2-PLAN.md):
- Goals run a sequential 90-slot loop (state feedback), vectorized across draws.
- Count families don't feed back into state, so given each draw's state path
  they are conditionally Poisson: one draw from the path-integrated hazard per
  (draw, team, family, half). Gamma frailty on the integrated hazard reproduces
  NegBin marginals in the uncoupled limit (Gamma-Poisson mixture).
- Posterior predictive: goal-model betas sampled per draw-block from the
  sandwich covariance (models/uncertainty.py).
- Hazard scale: goal hazards are lambda x m(slot, state) / Z with
  Z = E_paths[sum_slots m] estimated by pilot simulation (dynamics.scale),
  so that simulated mean totals match lambda (the unconditional expectation).

Player-prop and penalty archetypes stay on v1 (not simulated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jumpprob.models.count_model import CountModel
from jumpprob.models.dynamics import BUCKET, Dynamics, N_STATES
from jumpprob.models.goal_model import GoalModel

COUNT_FAMILY = {"corner": "corners", "card": "cards", "foul": "fouls",
                "sot": "shots_on_target", "offside": "offsides"}
N_BLOCKS = 200  # posterior-predictive beta blocks


@dataclass
class SimResult:
    score_a: np.ndarray
    score_b: np.ndarray
    ht_a: np.ndarray
    ht_b: np.ndarray
    first_team: np.ndarray        # 0 none, 1 team A, 2 team B (match)
    first_team_h2: np.ndarray     # first goal of the second half
    counts: dict                  # family -> (N, 2 teams, 2 halves)


class MCEngine:
    def __init__(self, goal_model: GoalModel, count_model: CountModel | None,
                 dynamics: Dynamics, n_draws: int = 100_000,
                 coupled: bool | set[str] = True,
                 frailty: bool = True, beta_cov: np.ndarray | None = None, seed: int = 0):
        """coupled: True = all families state-coupled; False = none; or an
        explicit set of family names. Production config is {"goal"} — the
        six-tournament ablation showed count-family state multipliers do not
        transfer out-of-sample (SoT comparisons -0.018 Brier) while goal
        coupling pays (+0.001 on draw/win, +0.0007 ht_tied). See V2-FINDINGS."""
        self.gm = goal_model
        self.cm = count_model
        self.dyn = dynamics
        self.n = n_draws
        if coupled is True:
            self.couple_set = set(dynamics.alpha)
        elif coupled is False:
            self.couple_set = set()
        else:
            self.couple_set = set(coupled)
        self.coupled = bool(self.couple_set)
        self.frailty = frailty
        self.beta_cov = beta_cov
        self.rng = np.random.default_rng(seed)
        self._m = {}  # family -> (90, N_STATES) multiplier table
        for fam in dynamics.alpha:
            a = np.asarray(dynamics.alpha[fam])
            b = np.asarray(dynamics.beta[fam])
            if fam not in self.couple_set:
                # uncoupled family: alpha profile only, exact analytic
                # normalization (calibrated scale is a coupled-family quantity)
                m = np.repeat(np.exp(a[BUCKET][:, None]), N_STATES, axis=1)
                self._m[fam] = m / m[:, 2].sum()
            else:
                m = np.exp(a[BUCKET][:, None] + b[None, :])
                self._m[fam] = m / dynamics.scale.get(fam, m[:, 2].sum())

    def _lambdas(self, elo_a, elo_b, a_home_ground):
        """(N,) intensity vectors, posterior-predictive if beta_cov given."""
        if self.beta_cov is None:
            la, lb = self.gm.intensities(elo_a, elo_b, a_home_ground)
            return np.full(self.n, la), np.full(self.n, lb)
        from jumpprob.models.uncertainty import sample_betas
        betas = sample_betas(self.gm, self.beta_cov, N_BLOCKS, self.rng)
        d = (elo_a - elo_b) / 100.0
        la_blk = np.exp(betas[:, 0] + betas[:, 1] * d + betas[:, 2] * float(a_home_ground))
        lb_blk = np.exp(betas[:, 0] - betas[:, 1] * d)
        reps = int(np.ceil(self.n / N_BLOCKS))
        return (np.repeat(la_blk, reps)[: self.n], np.repeat(lb_blk, reps)[: self.n])

    def simulate(self, elo_a: float, elo_b: float, a_home_ground: bool = False,
                 referee: str | None = None, lam: tuple[float, float] | None = None) -> SimResult:
        n = self.n
        if lam is not None:
            # fixed intensities (e.g. odds-anchored blended lambdas) — bypass the
            # Elo->lambda map and posterior predictive; the anchor has already
            # resolved the intensity, so we don't reintroduce model uncertainty.
            lam_a = np.full(n, lam[0])
            lam_b = np.full(n, lam[1])
        else:
            lam_a, lam_b = self._lambdas(elo_a, elo_b, a_home_ground)
        mg = self._m["goal"]

        score_a = np.zeros(n, dtype=np.int32)
        score_b = np.zeros(n, dtype=np.int32)
        ht_a = np.zeros(n, dtype=np.int32)
        ht_b = np.zeros(n, dtype=np.int32)
        first = np.zeros(n, dtype=np.int8)
        first_h2 = np.zeros(n, dtype=np.int8)
        fams = [f for f in self.dyn.alpha if f != "goal"]
        integ = {f: np.zeros((n, 2, 2)) for f in fams}

        for slot in range(90):
            half = 0 if slot < 45 else 1
            margin = score_a - score_b
            st_a = np.clip(margin, -2, 2) + 2
            st_b = np.clip(-margin, -2, 2) + 2
            ka = self.rng.poisson(lam_a * mg[slot, st_a])
            kb = self.rng.poisson(lam_b * mg[slot, st_b])
            for f in fams:
                mf = self._m[f]
                integ[f][:, 0, half] += mf[slot, st_a]
                integ[f][:, 1, half] += mf[slot, st_b]
            scored_a, scored_b = ka > 0, kb > 0
            new = (first == 0) & (scored_a | scored_b)
            both = scored_a & scored_b
            coin = self.rng.random(n) < 0.5
            first[new & scored_a & ~both] = 1
            first[new & scored_b & ~both] = 2
            first[new & both] = np.where(coin[new & both], 1, 2)
            if half == 1:
                new2 = (first_h2 == 0) & (scored_a | scored_b)
                first_h2[new2 & scored_a & ~both] = 1
                first_h2[new2 & scored_b & ~both] = 2
                first_h2[new2 & both] = np.where(coin[new2 & both], 1, 2)
            score_a += ka
            score_b += kb
            if slot == 44:
                ht_a, ht_b = score_a.copy(), score_b.copy()

        counts = {}
        if self.cm is not None:
            for f in fams:
                cf = COUNT_FAMILY[f]
                mu_a = self.cm.mu(cf, 1, float(np.mean(lam_a)), float(np.mean(lam_b)), referee if cf == "cards" else None) \
                     + self.cm.mu(cf, 2, float(np.mean(lam_a)), float(np.mean(lam_b)), referee if cf == "cards" else None)
                mu_b = self.cm.mu(cf, 1, float(np.mean(lam_b)), float(np.mean(lam_a)), referee if cf == "cards" else None) \
                     + self.cm.mu(cf, 2, float(np.mean(lam_b)), float(np.mean(lam_a)), referee if cf == "cards" else None)
                k = self.dyn.frailty_k.get(f, float("inf"))
                g_a = self.rng.gamma(k, 1.0 / k, n) if (self.frailty and np.isfinite(k)) else 1.0
                g_b = self.rng.gamma(k, 1.0 / k, n) if (self.frailty and np.isfinite(k)) else 1.0
                arr = np.zeros((n, 2, 2), dtype=np.int32)
                for h in (0, 1):
                    arr[:, 0, h] = self.rng.poisson(mu_a * integ[f][:, 0, h] * g_a)
                    arr[:, 1, h] = self.rng.poisson(mu_b * integ[f][:, 1, h] * g_b)
                counts[f] = arr
        return SimResult(score_a, score_b, ht_a, ht_b, first, first_h2, counts)


# ---------------------------------------------------------------------------

def answer(res: SimResult, archetype: str, side: str | None = None,
           threshold: int | None = None, half: int | None = None,
           other_side: str | None = None) -> float | None:
    """Frequency-count any team-level archetype from a SimResult."""
    s = res.score_a if side != "b" else res.score_b
    o = res.score_b if side != "b" else res.score_a
    total = res.score_a + res.score_b
    h2_a, h2_b = res.score_a - res.ht_a, res.score_b - res.ht_b
    sh = {1: (res.ht_a, res.ht_b), 2: (h2_a, h2_b)}
    side_i = 0 if side != "b" else 1
    first_val = 1 if side != "b" else 2

    def cnt(fam, si, h):
        c = res.counts[fam]
        return c[:, si, h - 1] if h else c[:, si, 0] + c[:, si, 1]

    m = {
        "win": lambda: (s > o),
        "draw": lambda: (res.score_a == res.score_b),
        "btts": lambda: (res.score_a > 0) & (res.score_b > 0),
        "total_ge": lambda: total >= threshold,
        "total_le": lambda: total <= threshold,
        "btts_and_total_ge": lambda: (res.score_a > 0) & (res.score_b > 0) & (total >= threshold),
        "team_goals_ge": lambda: s >= threshold,
        "team_scores_half": lambda: sh[half][side_i] > 0,
        "ht_tied": lambda: res.ht_a == res.ht_b,
        "ht_leading": lambda: (res.ht_a > res.ht_b) if side != "b" else (res.ht_b > res.ht_a),
        "half_total_ge": lambda: (sh[half][0] + sh[half][1]) >= threshold,
        # some (team, half) cell reaches the threshold — the sim carries the full
        # state-coupled joint, unlike the runner's independent-Poisson fallback
        "either_team_half_goals_ge": lambda: (sh[1][0] >= threshold) | (sh[1][1] >= threshold)
                                             | (sh[2][0] >= threshold) | (sh[2][1] >= threshold),
        "h2_more_goals_than_h1": lambda: (h2_a + h2_b) > (res.ht_a + res.ht_b),
        "first_goal_half": lambda: res.first_team_h2 == first_val if half == 2 else _first_h1(res) == first_val,
        "first_goal_and_win": lambda: (res.first_team == first_val) & (s > o),
        "first_goal_and_opp_2h": lambda: (res.first_team == first_val) & (sh[2][1 - side_i] > 0),
        "goals_gt_half": lambda: sh[half][side_i] > sh[half][1 - side_i],
        "corners_gt": lambda: cnt("corner", side_i, half) > cnt("corner", 1 - side_i, half),
        "team_corners_ge": lambda: cnt("corner", side_i, half) >= threshold,
        "corners_total_ge": lambda: cnt("corner", 0, half) + cnt("corner", 1, half) >= threshold,
        "fouls_gt": lambda: cnt("foul", side_i, half) > cnt("foul", 1 - side_i, half),
        "cards_gt": lambda: cnt("card", side_i, half) > cnt("card", 1 - side_i, half),
        "cards_ge": lambda: cnt("card", 0, half) + cnt("card", 1, half) >= threshold,
        "team_cards_ge": lambda: cnt("card", side_i, half) >= threshold,
        "offsides_ge": lambda: cnt("offside", side_i, half) >= threshold,
        "sot_gt": lambda: cnt("sot", side_i, half) > cnt("sot", 1 - side_i, half),
        "team_sot_ge": lambda: cnt("sot", side_i, half) >= threshold,
        "total_sot_ge": lambda: cnt("sot", 0, half) + cnt("sot", 1, half) >= threshold,
        "total_sot_half_ge": lambda: cnt("sot", 0, half) + cnt("sot", 1, half) >= threshold,
    }
    fn = m.get(archetype)
    if fn is None:
        return None
    return float(np.mean(fn()))


def _first_h1(res: SimResult) -> np.ndarray:
    """First scorer of the FIRST half: first_team where HT goals exist."""
    any_h1 = (res.ht_a + res.ht_b) > 0
    return np.where(any_h1, res.first_team, 0)
