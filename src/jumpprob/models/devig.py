"""Convert bookmaker odds to probabilities: normalization, power, Shin.

Inputs are decimal odds o_i for a mutually exclusive, exhaustive outcome set.
Raw inverse odds pi_i = 1/o_i sum to 1 + vig (the overround).

- basic: p_i = pi_i / sum(pi)               (proportional; keeps FLB bias)
- power: p_i = pi_i^k with k solved so sum(p) = 1   (k < 1 shrinks longshots
  harder, partially correcting favourite-longshot bias)
- shin:  Shin (1993) insider-trading model. With z = insider fraction,
      p_i = ( sqrt(z^2 + 4 (1-z) pi_i^2 / B) - z ) / (2 (1-z)),  B = sum(pi)
  and z solved so sum(p_i) = 1. As z -> 0 this reduces to basic normalization.
  Strumbelj (2014): Shin probabilities beat basic normalization for soccer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied_raw(odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if (odds <= 1.0).any():
        raise ValueError("decimal odds must exceed 1.0")
    return 1.0 / odds


def devig_basic(odds) -> np.ndarray:
    pi = implied_raw(np.asarray(odds))
    return pi / pi.sum()


def devig_power(odds) -> np.ndarray:
    pi = implied_raw(np.asarray(odds))

    def excess(k: float) -> float:
        return np.sum(pi**k) - 1.0

    k = brentq(excess, 0.5, 3.0, xtol=1e-12)
    return pi**k


def devig_shin(odds) -> np.ndarray:
    pi = implied_raw(np.asarray(odds))
    booksum = pi.sum()
    if booksum <= 1.0:  # no overround: already fair
        return pi / booksum

    def shin_probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / booksum) - z) / (2.0 * (1.0 - z))

    def excess(z: float) -> float:
        return shin_probs(z).sum() - 1.0

    # z in (0, 1); excess(0) = booksum normalized... bracket carefully
    z = brentq(excess, 1e-12, 0.5, xtol=1e-12)
    return shin_probs(z)


def overround(odds) -> float:
    """Book margin = sum(1/o) - 1. ~0.02 for a sharp book, ~0.08 for a soft one,
    ~0 for an exchange / prediction-market mid-price."""
    return float(implied_raw(odds).sum() - 1.0)


# Named-book sharpness priors. The inverse-overround term below already
# up-weights tight-margin books, so this is a small nudge for books known to
# be sharp beyond what their margin shows (limits, who-moves-first). Kept
# modest deliberately. A prediction-market mid-price is passed as a pseudo-book.
SHARP_BOOKS = {
    "pinnacle": 2.0, "betfair_ex_eu": 2.0, "betfair": 2.0,
    "kalshi": 2.0, "polymarket": 2.0, "bet365": 1.3,
}


def consensus(books, method: str = "shin", sharp_weights: dict | None = None,
              eps: float = 0.005) -> np.ndarray:
    """Wisdom-of-crowds consensus across books for ONE market.

    books: [(name, [decimal odds in a fixed outcome order]), ...]. Each book is
    de-vigged individually (Shin), then averaged in PROBABILITY space with
    weight  w = sharp_mult(name) / (overround + eps).

    Why probability-space (not logit)? Books are not independent forecasters —
    they are correlated copies of the same sharp line plus noise. The right
    pool for many noisy estimates of ONE quantity is a (precision-weighted)
    arithmetic mean. Logit/geometric pooling is for COMBINING DIVERSE sources
    (our model vs the market), which happens later in market_anchor.blend.

    Why 1/(overround+eps)? The margin is an observable sharpness proxy: a 2%
    book is sharper than an 8% book, so it earns more weight. eps keeps a
    zero-margin exchange/prediction-market from getting infinite weight.

    A prediction market (Kalshi/Polymarket) enters as a pseudo-book: pass its
    per-outcome odds as 1/mid_price; its ~0 overround naturally earns it high
    weight, and SHARP_BOOKS gives real-money venues a further nudge. Cap a thin
    market's influence by passing a smaller sharp_weights[name] (liquidity).
    """
    sharp = {**SHARP_BOOKS, **(sharp_weights or {})}
    n = len(books[0][1])
    acc = np.zeros(n)
    wsum = 0.0
    for name, odds in books:
        if len(odds) != n:
            raise ValueError(f"book {name}: {len(odds)} outcomes, expected {n}")
        p = devig(odds, method)
        w = sharp.get(name, 1.0) / (max(overround(odds), 0.0) + eps)
        acc += w * p
        wsum += w
    out = acc / wsum
    return out / out.sum()


def devig(odds, method: str = "shin") -> np.ndarray:
    fn = {"basic": devig_basic, "power": devig_power, "shin": devig_shin}.get(method)
    if fn is None:
        raise ValueError(f"unknown devig method: {method}")
    return fn(odds)
