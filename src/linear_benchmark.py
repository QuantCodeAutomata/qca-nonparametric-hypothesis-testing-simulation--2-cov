"""
Linear QMLE Benchmark (Francq & Thieu, 2019) for ARCH-m(X) models.

Implements the parametric linear QMLE estimator:
sigma_t^2 = sum_j alpha_j * eps_{t-j}^2 + beta_0 + sum_l beta_l * x_{l,t-1}

Tests H_0: beta_l = 0 using Wald statistic.

Custom — Context7 found no library equivalent for this specific linear QMLE
for ARCH-m(X) models with Wald test (Francq & Thieu 2019 methodology).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm


def linear_qmle_objective(
    params: NDArray[np.float64],
    eps_sq: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> float:
    """
    QMLE objective: (1/n) * sum_t [eps_t^2/sigma_t^2 + ln(sigma_t^2)].

    Parameters
    ----------
    params : NDArray[np.float64]
        Parameter vector [alpha_1,...,alpha_p, beta_0, beta_1,...,beta_d].
    eps_sq : NDArray[np.float64]
        eps_t^2 for t=p+1,...,n, shape (n_eff,).
    Z : NDArray[np.float64]
        Design matrix [eps_{t-1}^2,...,eps_{t-p}^2, 1, x_{1,t-1},...,x_{d,t-1}],
        shape (n_eff, p+1+d).

    Returns
    -------
    float
        QMLE objective value.
    """
    sigma2 = Z @ params
    sigma2 = np.maximum(sigma2, 1e-8)
    return np.mean(eps_sq / sigma2 + np.log(sigma2))


def linear_qmle_gradient(
    params: NDArray[np.float64],
    eps_sq: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Gradient of QMLE objective.

    Parameters
    ----------
    params : NDArray[np.float64]
        Parameter vector.
    eps_sq : NDArray[np.float64]
        eps_t^2 values.
    Z : NDArray[np.float64]
        Design matrix.

    Returns
    -------
    NDArray[np.float64]
        Gradient vector.
    """
    n_eff = len(eps_sq)
    sigma2 = Z @ params
    sigma2 = np.maximum(sigma2, 1e-8)
    dsigma2 = (1.0 / sigma2 - eps_sq / sigma2 ** 2) / n_eff
    return Z.T @ dsigma2


def linear_qmle_hessian(
    params: NDArray[np.float64],
    eps_sq: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Approximate Hessian (information matrix) for standard error computation.

    H = (1/n) * sum_t (1/sigma_t^4) * z_t * z_t^T

    Parameters
    ----------
    params : NDArray[np.float64]
        Parameter vector.
    eps_sq : NDArray[np.float64]
        eps_t^2 values.
    Z : NDArray[np.float64]
        Design matrix.

    Returns
    -------
    NDArray[np.float64]
        Hessian matrix.
    """
    n_eff = len(eps_sq)
    sigma2 = Z @ params
    sigma2 = np.maximum(sigma2, 1e-8)
    weights = 1.0 / sigma2 ** 2  # shape (n_eff,)
    # H = (1/n) * Z^T * diag(weights) * Z
    return (Z.T * weights) @ Z / n_eff


def fit_linear_qmle(
    eps: NDArray[np.float64],
    X: NDArray[np.float64],
    p: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Fit linear QMLE model: sigma_t^2 = sum_j alpha_j*eps_{t-j}^2 + beta_0 + X_{t-1}^T*beta.

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    X : NDArray[np.float64]
        Covariate matrix, shape (n, d).
    p : int
        ARCH order.

    Returns
    -------
    tuple of:
        params_hat : NDArray[np.float64]
            Estimated parameters [alpha_1,...,alpha_p, beta_0, beta_1,...,beta_d].
        std_errors : NDArray[np.float64]
            Standard errors of parameters.
        Z_design : NDArray[np.float64]
            Design matrix used.
    """
    n = len(eps)
    n_eff = n - p
    d = X.shape[1]

    # Response
    eps_sq = eps[p:] ** 2  # shape (n_eff,)

    # Design matrix: [eps_{t-1}^2,...,eps_{t-p}^2, 1, x_{1,t-1},...,x_{d,t-1}]
    Z = np.zeros((n_eff, p + 1 + d))
    for j in range(1, p + 1):
        Z[:, j - 1] = eps[p - j: n - j] ** 2
    Z[:, p] = 1.0  # intercept
    Z[:, p + 1:] = X[p - 1: n - 1]  # x_{t-1} for t=p+1,...,n

    # Initial parameters
    alpha_init = np.full(p, 0.1)
    beta_init = np.zeros(1 + d)
    beta_init[0] = np.mean(eps_sq) * 0.5  # intercept
    params_init = np.concatenate([alpha_init, beta_init])

    # Bounds: alpha_j >= 0, beta_0 >= 0, beta_l unconstrained
    bounds = [(0.0, None)] * p + [(1e-6, None)] + [(None, None)] * d

    result = minimize(
        fun=linear_qmle_objective,
        x0=params_init,
        args=(eps_sq, Z),
        method="L-BFGS-B",
        jac=linear_qmle_gradient,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
    )

    params_hat = result.x

    # Compute standard errors via sandwich estimator
    # Hessian (information matrix)
    H = linear_qmle_hessian(params_hat, eps_sq, Z)

    # Outer product (score covariance)
    sigma2_hat = Z @ params_hat
    sigma2_hat = np.maximum(sigma2_hat, 1e-8)
    scores = ((eps_sq / sigma2_hat ** 2 - 1.0 / sigma2_hat)[:, None] * Z)  # (n_eff, k)
    J = scores.T @ scores / n_eff

    # Sandwich: V = H^{-1} J H^{-1} / n
    try:
        H_inv = np.linalg.inv(H)
        V = H_inv @ J @ H_inv / n_eff
        std_errors = np.sqrt(np.maximum(np.diag(V), 0.0))
    except np.linalg.LinAlgError:
        std_errors = np.full(len(params_hat), np.nan)

    return params_hat, std_errors, Z


def wald_test_covariate(
    eps: NDArray[np.float64],
    X: NDArray[np.float64],
    p: int,
    covariate_idx: int,
) -> tuple[float, float]:
    """
    Wald test for H_0: beta_l = 0 in the linear QMLE model.

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    X : NDArray[np.float64]
        Covariate matrix, shape (n, d).
    p : int
        ARCH order.
    covariate_idx : int
        0-based index of the covariate to test (0 = first covariate).

    Returns
    -------
    tuple of (z_stat, p_value):
        z_stat : float
            Wald z-statistic.
        p_value : float
            Two-sided p-value.
    """
    params_hat, std_errors, _ = fit_linear_qmle(eps, X, p)

    # beta_l is at position p + 1 + covariate_idx (after alpha's and intercept)
    param_idx = p + 1 + covariate_idx
    beta_l = params_hat[param_idx]
    se_l = std_errors[param_idx]

    if se_l < 1e-10 or np.isnan(se_l):
        return 0.0, 1.0

    z_stat = beta_l / se_l
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))  # two-sided

    return z_stat, p_value


def linear_benchmark_test(
    eps: NDArray[np.float64],
    X: NDArray[np.float64],
    p: int,
    covariate_idx: int,
    alpha_level: float = 0.05,
) -> tuple[bool, float]:
    """
    Linear QMLE benchmark test for covariate significance.

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    X : NDArray[np.float64]
        Full covariate matrix, shape (n, d).
    p : int
        ARCH order.
    covariate_idx : int
        0-based index of the covariate to test.
    alpha_level : float
        Significance level (default 0.05).

    Returns
    -------
    tuple of (rejected, p_value):
        rejected : bool
            True if H_0 is rejected.
        p_value : float
            Two-sided p-value.
    """
    _, p_value = wald_test_covariate(eps, X, p, covariate_idx)
    return p_value < alpha_level, p_value
