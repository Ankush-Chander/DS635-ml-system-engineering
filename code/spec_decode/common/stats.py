"""Chi-square goodness of fit, without a SciPy dependency.

Speculative decoding's guarantee is distributional, so the only honest check is
a statistical one: draw from the sampler often enough that a biased residual
would show up, and test the counts against the target distribution.
"""
from __future__ import annotations

import math

import numpy as np


def chi2_sf(stat: float, dof: int) -> float:
    """Upper tail of the chi-square distribution: Q(dof/2, stat/2)."""
    if dof <= 0:
        return 1.0
    a, x = dof / 2.0, stat / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1.0:  # series for the lower tail, then complement
        term = total = 1.0 / a
        for n in range(1, 1000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1.0 - a, 1e300, 1.0 / (x + 1.0 - a)  # Lentz continued fraction
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi_square_test(counts: np.ndarray, probs: np.ndarray, min_expected: float = 5.0):
    """Test `counts` against `probs`, pooling low-expectation categories.

    A 150k-symbol vocabulary has a long tail of cells expecting far less than one
    observation, where the chi-square approximation does not hold; those cells are
    pooled into a single bucket. Returns (stat, dof, pvalue, kept) with `kept` the
    indices tested individually.
    """
    counts = np.asarray(counts, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n = counts.sum()
    expected = probs * n

    kept = np.flatnonzero(expected >= min_expected)
    obs = list(counts[kept])
    exp = list(expected[kept])

    tail = np.setdiff1d(np.arange(len(counts)), kept)
    if tail.size and expected[tail].sum() > 0:
        obs.append(counts[tail].sum())
        exp.append(expected[tail].sum())

    obs_a, exp_a = np.array(obs), np.array(exp)
    stat = float(((obs_a - exp_a) ** 2 / exp_a).sum())
    dof = len(obs_a) - 1
    return stat, dof, chi2_sf(stat, dof), kept
