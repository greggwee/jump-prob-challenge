"""Game-state dynamics: minute-profile and score-state multipliers per family.

Model: an event of family f for a team in slot m (0..89) arrives with hazard
    h(m) = base_team_match_rate x exp(alpha_f[bucket(m)] + beta_f[state(m)])
bucket = 15-minute bins (6), state = team's score margin in {<=-2, -1, 0, +1, >=+2}
(indices 0..4, level=2 is the baseline, beta[2] := 0; alpha[0] := 0).

Estimation — two regimes, and the distinction matters:

* Non-goal families (corner, card, foul, sot, offside): the state path is
  EXOGENOUS to the family's own events (it is driven by goals), so conditioning
  on each (match, team) total is valid and eliminates base rates exactly
  (conditional Poisson = multinomial over the match's 90 cells):
    LL = sum_events [alpha(b_e) + beta(s_e)]
       - sum_(match,team) n_mt * log( sum_slots exp(alpha(b) + beta(state_path)) )

* The GOAL family has feedback: each goal creates the state that follows it, so
  totals are not ancillary and the multinomial conditioning is BIASED (verified
  by synthetic test — recovered alphas inflate monotonically). The valid
  estimator is the full point-process likelihood on the realized filtration
  (Aalen/Cox): Poisson regression over all 90 cells with a per-(match, team)
  intensity offset (log lambda from the goal model; any common constant is
  absorbed by a global intercept mu):
    LL = sum_cells [ k * (o + mu + alpha + beta) - exp(o + mu + alpha + beta) ]

State paths are built from the goal stream (own goals included — they change
the score).

No hand-set constants: everything here is estimated; scripts/estimate_dynamics.py
reproduces data/processed/minute_dynamics.json from the raw event cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FAMILIES = ["goal", "corner", "card", "foul", "sot", "offside"]
N_BUCKETS = 6
N_STATES = 5  # margin <=-2, -1, 0, +1, >=+2
BASE_STATE = 2

BUCKET = np.minimum(np.arange(90) // 15, N_BUCKETS - 1)


def margin_state(margin: int) -> int:
    return int(np.clip(margin, -2, 2)) + 2


@dataclass
class Dynamics:
    alpha: dict[str, list[float]] = field(default_factory=dict)   # per family, len 6, [0]=0
    beta: dict[str, list[float]] = field(default_factory=dict)    # per family, len 5, [2]=0
    scale: dict[str, float] = field(default_factory=dict)         # simulation normalizers
    frailty_k: dict[str, float] = field(default_factory=dict)     # gamma frailty shape

    def weights(self, family: str, state: int) -> np.ndarray:
        """Unnormalized slot weights for a fixed state (len 90)."""
        a = np.asarray(self.alpha[family])
        return np.exp(a[BUCKET] + self.beta[family][state])

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"alpha": self.alpha, "beta": self.beta, "scale": self.scale,
             "frailty_k": self.frailty_k}, indent=1))

    @classmethod
    def from_json(cls, path: Path) -> "Dynamics":
        d = json.loads(path.read_text())
        return cls(d["alpha"], d["beta"], d.get("scale", {}), d.get("frailty_k", {}))


def state_paths(stream: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """Per match: per team, the margin state index at the START of each slot."""
    out: dict[int, dict[str, np.ndarray]] = {}
    goals = stream[stream["family"] == "goal"]
    for mid, g in stream.groupby("match_id"):
        home = g["home_team"].iloc[0]
        away = g["away_team"].iloc[0]
        diff = np.zeros(91)  # home margin entering each slot
        for r in goals[goals["match_id"] == mid].itertuples():
            sgn = 1 if r.team == home else -1
            diff[r.slot + 1:] += sgn
        sh = np.array([margin_state(int(d)) for d in diff[:90]])
        sa = np.array([margin_state(int(-d)) for d in diff[:90]])
        out[mid] = {home: sh, away: sa}
    return out


def fit_family(stream: pd.DataFrame, paths: dict, family: str) -> tuple[np.ndarray, np.ndarray]:
    ev = stream[stream["family"] == family]
    # observed cell assignments
    ev_buckets, ev_states, groups = [], [], []
    for (mid, team), g in ev.groupby(["match_id", "team"]):
        path = paths.get(mid, {}).get(team)
        if path is None:
            continue
        slots = g["slot"].to_numpy()
        ev_buckets.append(BUCKET[slots])
        ev_states.append(path[slots])
        groups.append((path, len(slots)))
    if not groups:
        raise ValueError(f"no events for family {family}")
    ev_b = np.concatenate(ev_buckets)
    ev_s = np.concatenate(ev_states)
    # stack paths for vectorized denominator: (n_groups, 90)
    P = np.stack([p for p, _ in groups])
    n_mt = np.array([n for _, n in groups], dtype=float)

    def nll(theta):
        alpha = np.concatenate([[0.0], theta[: N_BUCKETS - 1]])
        beta = np.insert(theta[N_BUCKETS - 1:], BASE_STATE, 0.0)
        num = alpha[ev_b].sum() + beta[ev_s].sum()
        cell = alpha[BUCKET][None, :] + beta[P]          # (G, 90)
        denom = np.log(np.exp(cell).sum(axis=1))
        return -(num - float(n_mt @ denom))

    theta0 = np.zeros(N_BUCKETS - 1 + N_STATES - 1)
    res = minimize(nll, theta0, method="BFGS")
    alpha = np.concatenate([[0.0], res.x[: N_BUCKETS - 1]])
    beta = np.insert(res.x[N_BUCKETS - 1:], BASE_STATE, 0.0)
    return alpha, beta


def fit_goal_family(stream: pd.DataFrame, paths: dict,
                    offsets: dict[tuple[int, str], float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Full-likelihood estimator for the goal family (feedback-safe).

    offsets: log relative intensity per (match_id, team) — e.g. log lambda from
    the goal model. None = all equal (fine when match strengths are homogeneous;
    production always passes model lambdas)."""
    ev = stream[stream["family"] == "goal"]
    groups, counts, offs = [], [], []
    for (mid, team), path in ((k, p) for mid_, d in paths.items() for k, p in
                              (((mid_, t), pp) for t, pp in d.items())):
        slots = ev[(ev["match_id"] == mid) & (ev["team"] == team)]["slot"].to_numpy()
        k = np.zeros(90)
        np.add.at(k, slots, 1)
        groups.append(path)
        counts.append(k)
        offs.append(0.0 if offsets is None else offsets.get((mid, team), 0.0))
    P = np.stack(groups)            # (G, 90) state indices
    K = np.stack(counts)            # (G, 90) event counts
    O = np.asarray(offs)[:, None]   # (G, 1)

    def nll(theta):
        mu = theta[0]
        alpha = np.concatenate([[0.0], theta[1: N_BUCKETS]])
        beta = np.insert(theta[N_BUCKETS:], BASE_STATE, 0.0)
        log_h = O + mu + alpha[BUCKET][None, :] + beta[P]
        return float(np.sum(np.exp(log_h)) - np.sum(K * log_h))

    theta0 = np.concatenate([[np.log(max(K.sum() / K.size, 1e-6))],
                             np.zeros(N_BUCKETS - 1 + N_STATES - 1)])
    res = minimize(nll, theta0, method="BFGS")
    alpha = np.concatenate([[0.0], res.x[1: N_BUCKETS]])
    beta = np.insert(res.x[N_BUCKETS:], BASE_STATE, 0.0)
    return alpha, beta


def fit_dynamics(stream: pd.DataFrame,
                 goal_offsets: dict[tuple[int, str], float] | None = None) -> Dynamics:
    paths = state_paths(stream)
    dyn = Dynamics()
    present = [f for f in FAMILIES if (stream["family"] == f).any()]
    for fam in present:
        if fam == "goal":
            a, b = fit_goal_family(stream, paths, goal_offsets)
        else:
            a, b = fit_family(stream, paths, fam)
        dyn.alpha[fam] = [round(float(x), 4) for x in a]
        dyn.beta[fam] = [round(float(x), 4) for x in b]
    return dyn
