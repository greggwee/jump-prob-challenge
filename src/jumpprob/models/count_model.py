"""Count models for corners, cards, offsides, shots on target (per team-half).

Mean model per family f and half h:
    log mu = c0[f,h] + c1[f] * log(lam_team) + c2[f] * log(lam_opp)
where lam_team/lam_opp are the goal model's match intensities — the "how much
attacking will this team do against this opponent" signal (corners-shots corr
0.60 measured; corners-goals only 0.11, so lambda enters as a proxy for
attacking volume, not via goals).

Cards additionally get a shrunk multiplicative referee factor:
    mu_cards *= ref_factor,  ref_factor = (cards_ref + s*g) / (g*(n_ref + s))
with g = global cards per team-half, s = shrinkage strength (pseudo-matches).

Distribution: negative binomial with method-of-moments dispersion per family
(measured var/mean 2026-06-12: corners 1.42, shots 1.89, yellows 1.15,
offsides ~1.0). NegBin(k, p) with p = k/(k+mu); k -> inf recovers Poisson.

Team-vs-team comparison questions assume conditional independence of the two
sides' counts given the intensities (v1 simplification, noted in PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

FAMILIES = ["corners", "cards", "offsides", "shots_on_target", "fouls", "shots"]
MAX_COUNT = 30
REF_SHRINKAGE = 40.0  # pseudo team-halves of global rate mixed into each referee
# NOTE: cross-tournament backtest showed referee factors with light shrinkage
# (8) HURT out-of-sample cards Brier — small per-ref samples don't transfer
# across tournaments. Heavy shrinkage keeps the covariate nearly neutral in
# backtest; for live use, factors should come from FBref career card rates
# (hundreds of matches per ref), not tournament-only counts. See A4 register.


@dataclass
class CountModel:
    params: dict[str, dict] = field(default_factory=dict)  # family -> {c0_1, c0_2, c1, c2, k}
    ref_factors: dict[str, float] = field(default_factory=dict)
    global_card_rate: float = 1.0

    def mu(self, family: str, half: int, lam_team: float, lam_opp: float, referee: str | None = None) -> float:
        p = self.params[family]
        c0 = p["c0_1"] if half == 1 else p["c0_2"]
        mu = float(np.exp(c0 + p["c1"] * np.log(lam_team) + p["c2"] * np.log(lam_opp)))
        if family == "cards" and referee is not None:
            mu *= self.ref_factors.get(referee, 1.0)
        return mu

    def pmf(self, family: str, mu: float) -> np.ndarray:
        k = self.params[family]["k"]
        xs = np.arange(MAX_COUNT + 1)
        if np.isinf(k):
            pm = poisson.pmf(xs, mu)
        else:
            pm = nbinom.pmf(xs, k, k / (k + mu))
        return pm / pm.sum()


def _dispersion_k(counts: np.ndarray) -> float:
    """MoM: var = mu + mu^2/k  ->  k = mu^2 / (var - mu). Poisson if var <= mu."""
    mu, var = counts.mean(), counts.var()
    if var <= mu * 1.02:
        return float("inf")
    return float(mu**2 / (var - mu))


def fit_count_model(train: pd.DataFrame) -> CountModel:
    """train: one row per (match, team, period) with columns
    [family columns..., lam_team, lam_opp, referee]."""
    model = CountModel()
    for family in FAMILIES:
        if family not in train.columns:
            continue  # caller's training frame doesn't carry this family — skip it
        y = train[family].to_numpy(dtype=float)
        h2 = (train["period"] == 2).to_numpy(dtype=float)
        lt = np.log(train["lam_team"].to_numpy(dtype=float))
        lo = np.log(train["lam_opp"].to_numpy(dtype=float))
        # Cards don't scale with attacking intensity (backtest: lambda
        # covariates *hurt* the cards family) — intercept + half + referee only.
        use_lams = family != "cards"

        def nll(beta):
            log_mu = beta[0] + beta[1] * h2
            if use_lams:
                log_mu = log_mu + beta[2] * lt + beta[3] * lo
            mu = np.exp(np.clip(log_mu, -8, 5))
            return float(np.sum(mu - y * log_mu))  # Poisson deviance core (consistent for mean)

        x0 = np.array([np.log(max(y.mean(), 0.05)), 0.0, 0.0, 0.0] if use_lams else [np.log(max(y.mean(), 0.05)), 0.0])
        res = minimize(nll, x0=x0, method="BFGS")
        b = res.x
        model.params[family] = {
            "c0_1": float(b[0]),
            "c0_2": float(b[0] + b[1]),
            "c1": float(b[2]) if use_lams else 0.0,
            "c2": float(b[3]) if use_lams else 0.0,
            "k": _dispersion_k(y),
        }

    # referee factors for cards (per team-half rate, shrunk)
    g = train["cards"].mean()
    model.global_card_rate = float(g)
    by_ref = train.groupby("referee")["cards"].agg(["sum", "count"])
    for ref, row in by_ref.iterrows():
        model.ref_factors[ref] = float((row["sum"] + REF_SHRINKAGE * g) / (g * (row["count"] + REF_SHRINKAGE)))
    return model


# ---- question probabilities (independence across sides given intensities) ----

def p_ge(pmf: np.ndarray, k: int) -> float:
    return float(pmf[k:].sum())


def p_a_gt_b(pmf_a: np.ndarray, pmf_b: np.ndarray) -> float:
    """P(A > B) for independent counts."""
    cdf_b = np.cumsum(pmf_b)
    # P(A > B) = sum_a pmf_a[a] * P(B <= a-1)
    return float(sum(pmf_a[a] * cdf_b[a - 1] for a in range(1, len(pmf_a))))


def p_sum_ge(pmf_a: np.ndarray, pmf_b: np.ndarray, k: int) -> float:
    conv = np.convolve(pmf_a, pmf_b)
    return float(conv[k:].sum())
