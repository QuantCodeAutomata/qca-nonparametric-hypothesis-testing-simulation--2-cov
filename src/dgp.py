"""
Data Generating Processes (DGP) for semiparametric ARCH-m(X) models.

Implements 8 DGP models for both 2-covariate (exp_1) and 5/10-covariate
(exp_2, exp_3, exp_4) settings as described in the paper.

Custom — Context7 found no library equivalent for these specific ARCH-m(X) DGPs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Literal


# ---------------------------------------------------------------------------
# Innovation samplers
# ---------------------------------------------------------------------------

def sample_innovations(
    n: int,
    distribution: Literal["normal", "laplace", "t7"],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Draw n standardized innovations from the specified distribution.

    Parameters
    ----------
    n : int
        Number of innovations to draw.
    distribution : str
        One of 'normal', 'laplace', 't7'.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (n,) with unit-variance innovations.
    """
    if distribution == "normal":
        return rng.standard_normal(n)
    elif distribution == "laplace":
        # Laplace(0, 0.5) has variance 2*(0.5)^2 = 0.5; divide by sqrt(0.5) = 0.5*sqrt(2)
        raw = rng.laplace(loc=0.0, scale=0.5, size=n)
        return raw / (0.5 * np.sqrt(2.0))
    elif distribution == "t7":
        # t(7) has variance 7/(7-2) = 7/5; divide by sqrt(7/5)
        raw = rng.standard_t(df=7, size=n)
        return raw / np.sqrt(7.0 / 5.0)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Choose 'normal', 'laplace', or 't7'.")


# ---------------------------------------------------------------------------
# Covariate generation
# ---------------------------------------------------------------------------

def generate_covariates(
    n_total: int,
    d: int,
    rho: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Generate covariates X_t ~ MVN(mu, Sigma) where mu = (2.5,...,2.5)^T
    and Sigma_{ij} = rho^|i-j|.

    Parameters
    ----------
    n_total : int
        Total number of observations including burn-in.
    d : int
        Number of covariates.
    rho : float
        Correlation parameter (0 or 0.5).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (n_total, d).
    """
    mu = np.full(d, 2.5)
    # Sigma_{ij} = rho^|i-j|
    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n_total)


# ---------------------------------------------------------------------------
# Conditional variance functions for 2-covariate DGPs (exp_1)
# ---------------------------------------------------------------------------

def sigma2_2cov(
    model: int,
    eps_lag: NDArray[np.float64],
    x: NDArray[np.float64],
    c: float,
) -> float:
    """
    Compute sigma_t^2 for 2-covariate DGP models (exp_1).

    Parameters
    ----------
    model : int
        Model index (1-8).
    eps_lag : NDArray[np.float64]
        Lagged epsilon values; eps_lag[0] = eps_{t-1}, eps_lag[1] = eps_{t-2}.
    x : NDArray[np.float64]
        Covariate vector at time t-1, shape (2,). x[0]=x_1, x[1]=x_2.
    c : float
        Signal strength parameter.

    Returns
    -------
    float
        Conditional variance sigma_t^2 (guaranteed positive).
    """
    x1, x2 = x[0], x[1]
    e1 = eps_lag[0] ** 2  # eps_{t-1}^2

    if model == 1:
        # p=1: sigma_t^2 = 0.2 + 0.3*eps_{t-1}^2 + 0.5*(x1-2.5)^2 + 2c*(x2-2.5)^2
        val = 0.2 + 0.3 * e1 + 0.5 * (x1 - 2.5) ** 2 + 2.0 * c * (x2 - 2.5) ** 2
    elif model == 2:
        # p=1: sigma_t^2 = 0.2 + 0.3*eps_{t-1}^2 + 0.5*(x1 + 2c*x2 - 10)^2
        val = 0.2 + 0.3 * e1 + 0.5 * (x1 + 2.0 * c * x2 - 10.0) ** 2
    elif model == 3:
        # p=2: sigma_t^2 = 0.2 + 0.3*eps_{t-1}^2 + 0.2*eps_{t-2}^2 + 0.5*(x1-2.5)^2 + 2c*(x2-2.5)^2
        e2 = eps_lag[1] ** 2
        val = 0.2 + 0.3 * e1 + 0.2 * e2 + 0.5 * (x1 - 2.5) ** 2 + 2.0 * c * (x2 - 2.5) ** 2
    elif model == 4:
        # p=2: sigma_t^2 = 0.2 + 0.3*eps_{t-1}^2 + 0.2*eps_{t-2}^2 + 0.5*(x1 + 2c*x2 - 10)^2
        e2 = eps_lag[1] ** 2
        val = 0.2 + 0.3 * e1 + 0.2 * e2 + 0.5 * (x1 + 2.0 * c * x2 - 10.0) ** 2
    elif model == 5:
        # p=1: sigma_t^2 = 0.2 + 0.4*eps_{t-1}^2 + sin^2(pi*x1/2) + 10c*sin^2(pi*x2/2)
        val = 0.2 + 0.4 * e1 + np.sin(np.pi * x1 / 2.0) ** 2 + 10.0 * c * np.sin(np.pi * x2 / 2.0) ** 2
    elif model == 6:
        # p=1: sigma_t^2 = 0.2 + 0.4*eps_{t-1}^2 + 10c*sin^2(x1/2 + pi*c*x2/2)
        val = 0.2 + 0.4 * e1 + 10.0 * c * np.sin(x1 / 2.0 + np.pi * c * x2 / 2.0) ** 2
    elif model == 7:
        # p=1: sigma_t^2 = 0.2 + 0.4*eps_{t-1}^2 + 0.25*x1 + c*x2
        val = 0.2 + 0.4 * e1 + 0.25 * x1 + c * x2
    elif model == 8:
        # p=1: sigma_t^2 = 0.2 + 0.4*eps_{t-1}^2 + c*x1*x2
        val = 0.2 + 0.4 * e1 + c * x1 * x2
    else:
        raise ValueError(f"Unknown model: {model}. Choose 1-8.")

    return max(val, 1e-8)


def get_p_2cov(model: int) -> int:
    """Return ARCH order p for 2-covariate model."""
    return 2 if model in (3, 4) else 1


# ---------------------------------------------------------------------------
# Conditional variance functions for 5-covariate DGPs (exp_2, exp_3)
# ---------------------------------------------------------------------------

def sigma2_5cov(
    model: int,
    eps_lag: NDArray[np.float64],
    x: NDArray[np.float64],
    c: float,
) -> float:
    """
    Compute sigma_t^2 for 5-covariate DGP models (exp_2).
    Testing X_3 significance (null: m(x) = m_3(x^{-3})).

    Parameters
    ----------
    model : int
        Model index (1-8).
    eps_lag : NDArray[np.float64]
        Lagged epsilon values.
    x : NDArray[np.float64]
        Covariate vector at time t-1, shape (5,).
    c : float
        Signal strength parameter.

    Returns
    -------
    float
        Conditional variance sigma_t^2.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    e1 = eps_lag[0] ** 2

    if model == 1:
        # p=1: sigma_t^2 = 0.2 + 0.3*e1 + 0.5*sum_{i!=3}(xi-2.5)^2 + 5c*(x3-2.5)^2
        sum_others = (x1 - 2.5) ** 2 + (x2 - 2.5) ** 2 + (x4 - 2.5) ** 2 + (x5 - 2.5) ** 2
        val = 0.2 + 0.3 * e1 + 0.5 * sum_others + 5.0 * c * (x3 - 2.5) ** 2
    elif model == 2:
        # p=1: sigma_t^2 = 0.2 + 0.3*e1 + 0.5*(x1+x2+5c*x3+x4+x5-25)^2
        val = 0.2 + 0.3 * e1 + 0.5 * (x1 + x2 + 5.0 * c * x3 + x4 + x5 - 25.0) ** 2
    elif model == 3:
        # p=2
        e2 = eps_lag[1] ** 2
        sum_others = (x1 - 2.5) ** 2 + (x2 - 2.5) ** 2 + (x4 - 2.5) ** 2 + (x5 - 2.5) ** 2
        val = 0.2 + 0.3 * e1 + 0.2 * e2 + 0.5 * sum_others + 5.0 * c * (x3 - 2.5) ** 2
    elif model == 4:
        # p=2
        e2 = eps_lag[1] ** 2
        val = 0.2 + 0.3 * e1 + 0.2 * e2 + 0.5 * (x1 + x2 + 5.0 * c * x3 + x4 + x5 - 25.0) ** 2
    elif model == 5:
        # p=1: sigma_t^2 = 0.2 + 0.4*e1 + sum_{i!=3}sin^2(pi*xi/2) + 10c*sin^2(pi*x3/2)
        sum_others = (np.sin(np.pi * x1 / 2.0) ** 2 + np.sin(np.pi * x2 / 2.0) ** 2
                      + np.sin(np.pi * x4 / 2.0) ** 2 + np.sin(np.pi * x5 / 2.0) ** 2)
        val = 0.2 + 0.4 * e1 + sum_others + 10.0 * c * np.sin(np.pi * x3 / 2.0) ** 2
    elif model == 6:
        # p=1: sigma_t^2 = 0.2 + 0.4*e1 + sin^2(pi*x1/2)*sin^2(pi*x2/2)
        #                  + 10c*sin^2(pi*x3/2)*sin^2(pi*x4/2) + sin^2(pi*x5/2)
        val = (0.2 + 0.4 * e1
               + np.sin(np.pi * x1 / 2.0) ** 2 * np.sin(np.pi * x2 / 2.0) ** 2
               + 10.0 * c * np.sin(np.pi * x3 / 2.0) ** 2 * np.sin(np.pi * x4 / 2.0) ** 2
               + np.sin(np.pi * x5 / 2.0) ** 2)
    elif model == 7:
        # p=1: sigma_t^2 = 0.2 + 0.4*e1 + 0.25*(x1+x2+x4+x5) + 8c*x3
        val = 0.2 + 0.4 * e1 + 0.25 * (x1 + x2 + x4 + x5) + 8.0 * c * x3
    elif model == 8:
        # p=1: sigma_t^2 = 0.2 + 0.4*e1 + c*x3*(x1+x2+x4+x5)
        val = 0.2 + 0.4 * e1 + c * x3 * (x1 + x2 + x4 + x5)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 1-8.")

    return max(val, 1e-8)


def get_p_5cov(model: int) -> int:
    """Return ARCH order p for 5-covariate model."""
    return 2 if model in (3, 4) else 1


# ---------------------------------------------------------------------------
# Conditional variance functions for 5-covariate variable selection DGPs (exp_3)
# Active covariates: X_1, X_3, X_4 (1-based)
# ---------------------------------------------------------------------------

def sigma2_5cov_varsel(
    model: int,
    eps_lag: NDArray[np.float64],
    x: NDArray[np.float64],
) -> float:
    """
    Compute sigma_t^2 for 5-covariate variable selection DGPs (exp_3).
    Active covariates: X_1, X_3, X_4 (1-based indexing).
    No signal strength parameter c (fixed at full signal).

    Parameters
    ----------
    model : int
        Model index (1-8).
    eps_lag : NDArray[np.float64]
        Lagged epsilon values.
    x : NDArray[np.float64]
        Covariate vector at time t-1, shape (5,).

    Returns
    -------
    float
        Conditional variance sigma_t^2.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    e1 = eps_lag[0] ** 2

    if model == 1:
        # alpha=0.3, p=1: m(x) = 2*(x1-2.5)^2 + 2*(x3-2.5)^2 + 2*(x4-2.5)^2
        val = 0.2 + 0.3 * e1 + 2.0 * (x1 - 2.5) ** 2 + 2.0 * (x3 - 2.5) ** 2 + 2.0 * (x4 - 2.5) ** 2
    elif model == 2:
        # alpha=0.3, p=1: m(x) = 0.5*(2*x1+2*x3+2*x4-15)^2 + 0.001
        val = 0.2 + 0.3 * e1 + 0.5 * (2.0 * x1 + 2.0 * x3 + 2.0 * x4 - 15.0) ** 2 + 0.001
    elif model == 3:
        # alpha=(0.3,0.2), p=2: same m(x) as Model 1
        e2 = eps_lag[1] ** 2
        val = (0.2 + 0.3 * e1 + 0.2 * e2
               + 2.0 * (x1 - 2.5) ** 2 + 2.0 * (x3 - 2.5) ** 2 + 2.0 * (x4 - 2.5) ** 2)
    elif model == 4:
        # alpha=(0.3,0.2), p=2: m(x) = 0.5*(2*x1+2*x3+2*x4-15)^2 + 0.001
        e2 = eps_lag[1] ** 2
        val = 0.2 + 0.3 * e1 + 0.2 * e2 + 0.5 * (2.0 * x1 + 2.0 * x3 + 2.0 * x4 - 15.0) ** 2 + 0.001
    elif model == 5:
        # alpha=0.4, p=1: m(x) = 8*sin^2(pi*x1/2) + 8*sin^2(pi*x3/2) + 8*sin^2(pi*x4/2)
        val = (0.2 + 0.4 * e1
               + 8.0 * np.sin(np.pi * x1 / 2.0) ** 2
               + 8.0 * np.sin(np.pi * x3 / 2.0) ** 2
               + 8.0 * np.sin(np.pi * x4 / 2.0) ** 2)
    elif model == 6:
        # alpha=0.4, p=1: m(x) = 10*sin^2(pi*x1/2)*sin^2(pi*x3/2) + 10*sin^2(pi*x4/2)
        val = (0.2 + 0.4 * e1
               + 10.0 * np.sin(np.pi * x1 / 2.0) ** 2 * np.sin(np.pi * x3 / 2.0) ** 2
               + 10.0 * np.sin(np.pi * x4 / 2.0) ** 2)
    elif model == 7:
        # alpha=0.4, p=1: m(x) = 8*x1 + 8*x3 + 8*x4
        val = 0.2 + 0.4 * e1 + 8.0 * x1 + 8.0 * x3 + 8.0 * x4
    elif model == 8:
        # alpha=0.4, p=1: m(x) = x3*(x1+x4)
        val = 0.2 + 0.4 * e1 + x3 * (x1 + x4)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 1-8.")

    return max(val, 1e-8)


def get_p_5cov_varsel(model: int) -> int:
    """Return ARCH order p for 5-covariate variable selection model."""
    return 2 if model in (3, 4) else 1


# ---------------------------------------------------------------------------
# Conditional variance functions for 10-covariate variable selection DGPs (exp_4)
# Active covariates: X_1, X_3, X_4, X_5, X_9 (1-based)
# ---------------------------------------------------------------------------

def sigma2_10cov_varsel(
    model: int,
    eps_lag: NDArray[np.float64],
    x: NDArray[np.float64],
) -> float:
    """
    Compute sigma_t^2 for 10-covariate variable selection DGPs (exp_4).
    Active covariates: X_1, X_3, X_4, X_5, X_9 (1-based indexing).

    Parameters
    ----------
    model : int
        Model index (1-8).
    eps_lag : NDArray[np.float64]
        Lagged epsilon values.
    x : NDArray[np.float64]
        Covariate vector at time t-1, shape (10,).

    Returns
    -------
    float
        Conditional variance sigma_t^2.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    x6, x7, x8, x9, x10 = x[5], x[6], x[7], x[8], x[9]
    e1 = eps_lag[0] ** 2

    if model == 1:
        # alpha=0.3, p=1: m(x) = 2*(x1-2.5)^2+2*(x3-2.5)^2+2*(x4-2.5)^2+2*(x5-2.5)^2+2*(x9-2.5)^2
        val = (0.2 + 0.3 * e1
               + 2.0 * (x1 - 2.5) ** 2 + 2.0 * (x3 - 2.5) ** 2
               + 2.0 * (x4 - 2.5) ** 2 + 2.0 * (x5 - 2.5) ** 2
               + 2.0 * (x9 - 2.5) ** 2)
    elif model == 2:
        # alpha=0.3, p=1: m(x) = 0.5*(2*x1+2*x3+2*x4+2*x5+2*x9-25)^2 + 0.001
        val = (0.2 + 0.3 * e1
               + 0.5 * (2.0 * x1 + 2.0 * x3 + 2.0 * x4 + 2.0 * x5 + 2.0 * x9 - 25.0) ** 2
               + 0.001)
    elif model == 3:
        # alpha=(0.3,0.2), p=2: same m(x) as Model 1
        e2 = eps_lag[1] ** 2
        val = (0.2 + 0.3 * e1 + 0.2 * e2
               + 2.0 * (x1 - 2.5) ** 2 + 2.0 * (x3 - 2.5) ** 2
               + 2.0 * (x4 - 2.5) ** 2 + 2.0 * (x5 - 2.5) ** 2
               + 2.0 * (x9 - 2.5) ** 2)
    elif model == 4:
        # alpha=(0.3,0.2), p=2
        e2 = eps_lag[1] ** 2
        val = (0.2 + 0.3 * e1 + 0.2 * e2
               + 0.5 * (2.0 * x1 + 2.0 * x3 + 2.0 * x4 + 2.0 * x5 + 2.0 * x9 - 25.0) ** 2
               + 0.001)
    elif model == 5:
        # alpha=0.4, p=1: m(x) = 8*sin^2(pi*xi/2) for i in {1,3,4,5,9}
        val = (0.2 + 0.4 * e1
               + 8.0 * np.sin(np.pi * x1 / 2.0) ** 2
               + 8.0 * np.sin(np.pi * x3 / 2.0) ** 2
               + 8.0 * np.sin(np.pi * x4 / 2.0) ** 2
               + 8.0 * np.sin(np.pi * x5 / 2.0) ** 2
               + 8.0 * np.sin(np.pi * x9 / 2.0) ** 2)
    elif model == 6:
        # alpha=0.4, p=1: m(x) = 10*sin^2(pi*x1/2)*sin^2(pi*x3/2)
        #                        + 10*sin^2(pi*x4/2)*sin^2(pi*x5/2) + 10*sin^2(pi*x9/2)
        val = (0.2 + 0.4 * e1
               + 10.0 * np.sin(np.pi * x1 / 2.0) ** 2 * np.sin(np.pi * x3 / 2.0) ** 2
               + 10.0 * np.sin(np.pi * x4 / 2.0) ** 2 * np.sin(np.pi * x5 / 2.0) ** 2
               + 10.0 * np.sin(np.pi * x9 / 2.0) ** 2)
    elif model == 7:
        # alpha=0.4, p=1: m(x) = 8*x1+8*x3+8*x4+8*x5+8*x9
        val = 0.2 + 0.4 * e1 + 8.0 * (x1 + x3 + x4 + x5 + x9)
    elif model == 8:
        # alpha=0.4, p=1: m(x) = (x1+x4)*x3 + x5*x9
        val = 0.2 + 0.4 * e1 + (x1 + x4) * x3 + x5 * x9
    else:
        raise ValueError(f"Unknown model: {model}. Choose 1-8.")

    return max(val, 1e-8)


def get_p_10cov_varsel(model: int) -> int:
    """Return ARCH order p for 10-covariate variable selection model."""
    return 2 if model in (3, 4) else 1


# ---------------------------------------------------------------------------
# Generic ARCH-m(X) time series generator
# ---------------------------------------------------------------------------

def generate_arch_mx(
    n: int,
    X: NDArray[np.float64],
    sigma2_func,
    p: int,
    distribution: Literal["normal", "laplace", "t7"],
    rng: np.random.Generator,
    burnin: int = 500,
) -> NDArray[np.float64]:
    """
    Generate ARCH-m(X) time series eps_t = z_t * sqrt(sigma_t^2).

    Parameters
    ----------
    n : int
        Desired sample size (after burn-in).
    X : NDArray[np.float64]
        Covariate matrix of shape (n + burnin, d).
    sigma2_func : callable
        Function(eps_lag, x) -> float computing sigma_t^2.
    p : int
        ARCH order.
    distribution : str
        Innovation distribution.
    rng : np.random.Generator
        Random generator.
    burnin : int
        Number of burn-in observations to discard.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (n,) containing eps_t values.
    """
    n_total = n + burnin
    assert X.shape[0] >= n_total, "X must have at least n + burnin rows."

    z = sample_innovations(n_total, distribution, rng)
    eps = np.zeros(n_total)
    sigma2 = np.zeros(n_total)

    # Initial sigma^2: use a reasonable positive constant
    sigma2_init = 1.0

    for t in range(n_total):
        # Build eps_lag: [eps_{t-1}, eps_{t-2}, ..., eps_{t-p}]
        eps_lag = np.zeros(max(p, 2))
        for j in range(1, p + 1):
            if t - j >= 0:
                eps_lag[j - 1] = eps[t - j]

        # Covariate at t-1 (use t-1 index; for t=0 use X[0])
        x_tm1 = X[max(t - 1, 0)]

        s2 = sigma2_func(eps_lag, x_tm1)
        sigma2[t] = s2
        eps[t] = z[t] * np.sqrt(s2)

    # Discard burn-in
    return eps[burnin:]
