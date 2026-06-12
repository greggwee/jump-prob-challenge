"""Unit tests for count model math — synthetic, no data files."""

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from jumpprob.models.count_model import _dispersion_k, p_a_gt_b, p_ge, p_sum_ge


def test_dispersion_recovery():
    rng = np.random.default_rng(3)
    k_true, mu = 6.0, 2.2
    draws = nbinom.rvs(k_true, k_true / (k_true + mu), size=200_000, random_state=rng)
    k_hat = _dispersion_k(draws.astype(float))
    assert k_hat == pytest.approx(k_true, rel=0.15)


def test_dispersion_poisson_fallback():
    rng = np.random.default_rng(4)
    draws = poisson.rvs(1.5, size=100_000, random_state=rng).astype(float)
    assert _dispersion_k(draws) == float("inf")


def test_p_a_gt_b_brute_force():
    pmf_a = poisson.pmf(np.arange(31), 2.3)
    pmf_b = poisson.pmf(np.arange(31), 1.7)
    brute = sum(
        pmf_a[i] * pmf_b[j] for i in range(31) for j in range(31) if i > j
    )
    assert p_a_gt_b(pmf_a, pmf_b) == pytest.approx(brute, abs=1e-9)


def test_p_sum_and_threshold_coherence():
    pmf_a = poisson.pmf(np.arange(31), 1.0)
    pmf_b = poisson.pmf(np.arange(31), 1.0)
    # sum of two Poisson(1) = Poisson(2)
    expected = 1 - poisson.cdf(1, 2.0)
    assert p_sum_ge(pmf_a, pmf_b, 2) == pytest.approx(expected, abs=1e-6)
    assert p_ge(pmf_a, 0) == pytest.approx(1.0)
