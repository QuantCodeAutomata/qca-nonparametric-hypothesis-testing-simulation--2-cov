"""
Simulation runner utilities for ARCH-m(X) hypothesis testing experiments.

Provides parallelized simulation infrastructure for running Monte Carlo
experiments across multiple parameter configurations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Any
from joblib import Parallel, delayed
import warnings


def run_single_replication(
    rep_seed: int,
    sim_func: Callable,
    sim_kwargs: dict[str, Any],
) -> Any:
    """
    Run a single simulation replication with a given seed.

    Parameters
    ----------
    rep_seed : int
        Random seed for this replication.
    sim_func : callable
        Simulation function to call.
    sim_kwargs : dict
        Keyword arguments for sim_func.

    Returns
    -------
    Any
        Result of sim_func(**sim_kwargs, rng=rng).
    """
    rng = np.random.default_rng(rep_seed)
    try:
        return sim_func(**sim_kwargs, rng=rng)
    except Exception as e:
        warnings.warn(f"Replication {rep_seed} failed: {e}")
        return None


def run_simulation(
    sim_func: Callable,
    sim_kwargs: dict[str, Any],
    n_reps: int,
    base_seed: int = 42,
    n_jobs: int = -1,
) -> list[Any]:
    """
    Run n_reps replications of a simulation function in parallel.

    Parameters
    ----------
    sim_func : callable
        Simulation function with signature func(**kwargs, rng=rng) -> result.
    sim_kwargs : dict
        Keyword arguments passed to sim_func (excluding rng).
    n_reps : int
        Number of replications.
    base_seed : int
        Base random seed; replication i uses seed base_seed + i.
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs).

    Returns
    -------
    list
        List of results from each replication.
    """
    seeds = [base_seed + i for i in range(n_reps)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_single_replication)(seed, sim_func, sim_kwargs)
        for seed in seeds
    )

    return results


def compute_rejection_rate(
    results: list[bool | None],
) -> float:
    """
    Compute empirical rejection rate from simulation results.

    Parameters
    ----------
    results : list of bool or None
        List of rejection indicators (True/False) or None for failed reps.

    Returns
    -------
    float
        Empirical rejection rate (proportion of True values).
    """
    valid = [r for r in results if r is not None]
    if len(valid) == 0:
        return np.nan
    return np.mean(valid)


def binomial_ci(
    rate: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute Wilson confidence interval for a proportion.

    Parameters
    ----------
    rate : float
        Observed proportion.
    n : int
        Number of trials.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple of (lower, upper)
        95% confidence interval bounds.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    center = (rate + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = z * np.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return max(0.0, center - margin), min(1.0, center + margin)
