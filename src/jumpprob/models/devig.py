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
