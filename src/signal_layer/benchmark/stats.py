"""Statistical machinery for the benchmark.

Exchange-rate outcomes are strongly autocorrelated: two signals three days apart
observe almost the same market. Every routine here is chosen to survive that.

* :func:`newey_west_tstat` — mean significance with a HAC (Bartlett) variance,
  so overlapping horizons do not manufacture t-statistics.
* :func:`moving_block_bootstrap_ci` — confidence interval for a mean by
  resampling contiguous blocks rather than individual observations.
* :func:`permutation_pvalue` — the benchmark's headline test. The null is not
  "zero"; it is *a client who transfers on random days at the same cadence*.
  Sampling that null directly is exact and needs no distributional assumption.
* :func:`benjamini_hochberg` — the run tests many (strategy, corridor) cells, so
  raw p-values would produce a false positive by construction.

There is no SciPy in this project's dependency set, so everything is NumPy.
"""

from __future__ import annotations

import numpy as np


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def newey_west_tstat(values: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    """``(mean, hac_stderr, tstat)`` for the mean of an autocorrelated series.

    ``lags`` defaults to the usual ``floor(4 (n/100)^(2/9))`` rule. With a single
    observation the standard error is undefined and the t-statistic is NaN.
    """
    clean = _finite(values)
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if n < 3:
        return mean, float("nan"), float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))

    deviations = clean - mean
    variance = float(deviations @ deviations) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(deviations[lag:] @ deviations[:-lag]) / n
        variance += 2.0 * weight * covariance
    if variance <= 0:
        return mean, float("nan"), float("nan")
    stderr = float(np.sqrt(variance / n))
    return mean, stderr, mean / stderr if stderr > 0 else float("nan")


def moving_block_bootstrap_ci(
    values: np.ndarray,
    *,
    block_length: int,
    trials: int = 1_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling contiguous blocks of ``block_length``."""
    clean = _finite(values)
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    length = int(min(max(1, block_length), len(clean)))
    starts = np.arange(len(clean) - length + 1)
    blocks_needed = int(np.ceil(len(clean) / length))
    rng = np.random.default_rng(seed)
    means = np.empty(trials, dtype=float)
    for trial in range(trials):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([clean[s : s + length] for s in chosen])[: len(clean)]
        means[trial] = sample.mean()
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def permutation_pvalue(
    observed: float,
    null_samples: np.ndarray,
    *,
    alternative: str = "greater",
) -> float:
    """Monte-Carlo p-value of ``observed`` against an empirical null.

    Uses the ``(hits + 1) / (trials + 1)`` correction so a p-value is never
    reported as exactly zero from a finite number of draws.
    """
    null = _finite(null_samples)
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        centre = float(np.mean(null))
        hits = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    else:
        raise ValueError(f"Unknown alternative {alternative!r}")
    return (hits + 1) / (len(null) + 1)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """``(rejected, qvalues)`` under Benjamini-Hochberg FDR control.

    NaN p-values are carried through as not-rejected with a NaN q-value so the
    caller can keep a row per test.
    """
    raw = np.asarray(pvalues, dtype=float)
    rejected = np.zeros(len(raw), dtype=bool)
    qvalues = np.full(len(raw), np.nan)
    valid = np.flatnonzero(np.isfinite(raw))
    if len(valid) == 0:
        return rejected, qvalues

    order = valid[np.argsort(raw[valid], kind="stable")]
    m = len(order)
    ranks = np.arange(1, m + 1)
    adjusted = raw[order] * m / ranks
    # Enforce monotonicity from the largest p-value downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    qvalues[order] = np.minimum(adjusted, 1.0)
    rejected[order] = qvalues[order] <= alpha
    return rejected, qvalues
