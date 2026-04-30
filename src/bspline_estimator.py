"""
B-Spline QMLE Estimator for semiparametric ARCH-m(X) models.

Implements tensor-product B-spline basis parameterization of the conditional
variance function m(x) in ARCH-m(X) models, estimated via QMLE with L-BFGS-B.

Uses scipy.interpolate.BSpline for basis construction.
# Using scipy.interpolate.BSpline — Context7 confirmed
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from typing import Optional


def build_bspline_basis_1d(
    x: NDArray[np.float64],
    n_internal_knots: int = 3,
    degree: int = 3,
) -> NDArray[np.float64]:
    """
    Build B-spline design matrix for 1D input.

    Places internal knots at quantiles of x. Total basis functions = n_internal_knots + degree + 1.

    Parameters
    ----------
    x : NDArray[np.float64]
        1D covariate values, shape (n,).
    n_internal_knots : int
        Number of internal knots (default 3).
    degree : int
        Spline degree (default 3 = cubic).

    Returns
    -------
    NDArray[np.float64]
        Design matrix of shape (n, K) where K = n_internal_knots + degree + 1.
    """
    n = len(x)
    # Place internal knots at quantiles
    quantile_positions = np.linspace(0, 100, n_internal_knots + 2)[1:-1]
    internal_knots = np.percentile(x, quantile_positions)

    # Augment with boundary knots (repeated degree+1 times)
    x_min, x_max = x.min(), x.max()
    # Add small buffer to avoid boundary issues
    x_min -= 1e-6
    x_max += 1e-6
    knots = np.concatenate([
        np.repeat(x_min, degree + 1),
        internal_knots,
        np.repeat(x_max, degree + 1),
    ])

    K = n_internal_knots + degree + 1  # number of basis functions
    B = np.zeros((n, K))

    for k in range(K):
        # k-th B-spline basis function
        c = np.zeros(K)
        c[k] = 1.0
        spl = BSpline(knots, c, degree)
        B[:, k] = spl(x)

    return B


def build_bspline_basis_2d(
    x: NDArray[np.float64],
    n_internal_knots: int = 3,
    degree: int = 3,
) -> NDArray[np.float64]:
    """
    Build tensor-product B-spline design matrix for 2D input.

    Total basis functions K = (n_internal_knots + degree + 1)^2.

    Parameters
    ----------
    x : NDArray[np.float64]
        2D covariate matrix, shape (n, 2).
    n_internal_knots : int
        Number of internal knots per dimension (default 3).
    degree : int
        Spline degree (default 3 = cubic).

    Returns
    -------
    NDArray[np.float64]
        Design matrix of shape (n, K) where K = (n_internal_knots + degree + 1)^2.
    """
    B1 = build_bspline_basis_1d(x[:, 0], n_internal_knots, degree)
    B2 = build_bspline_basis_1d(x[:, 1], n_internal_knots, degree)

    K1 = B1.shape[1]
    K2 = B2.shape[1]
    n = x.shape[0]

    # Tensor product: B[t, k1*K2 + k2] = B1[t, k1] * B2[t, k2]
    B = np.zeros((n, K1 * K2))
    for k1 in range(K1):
        for k2 in range(K2):
            B[:, k1 * K2 + k2] = B1[:, k1] * B2[:, k2]

    return B


def qmle_objective(
    params: NDArray[np.float64],
    eps_sq: NDArray[np.float64],
    eps_lags_sq: NDArray[np.float64],
    B: NDArray[np.float64],
    p: int,
) -> float:
    """
    QMLE objective function: (1/n) * sum_t [eps_t^2/sigma_t^2 + ln(sigma_t^2)].

    Parameters
    ----------
    params : NDArray[np.float64]
        Parameter vector [alpha_1,...,alpha_p, theta_1,...,theta_K].
    eps_sq : NDArray[np.float64]
        eps_t^2 for t=p+1,...,n, shape (n_eff,).
    eps_lags_sq : NDArray[np.float64]
        Lagged eps^2 matrix, shape (n_eff, p).
    B : NDArray[np.float64]
        B-spline design matrix, shape (n_eff, K).
    p : int
        ARCH order.

    Returns
    -------
    float
        QMLE objective value.
    """
    alpha = params[:p]
    theta = params[p:]

    # sigma_t^2 = sum_j alpha_j * eps_{t-j}^2 + theta^T * B(x_{t-1})
    sigma2 = eps_lags_sq @ alpha + B @ theta
    sigma2 = np.maximum(sigma2, 1e-8)

    obj = np.mean(eps_sq / sigma2 + np.log(sigma2))
    return obj


def qmle_gradient(
    params: NDArray[np.float64],
    eps_sq: NDArray[np.float64],
    eps_lags_sq: NDArray[np.float64],
    B: NDArray[np.float64],
    p: int,
) -> NDArray[np.float64]:
    """
    Gradient of QMLE objective with respect to params.

    Parameters
    ----------
    params : NDArray[np.float64]
        Parameter vector.
    eps_sq : NDArray[np.float64]
        eps_t^2 values.
    eps_lags_sq : NDArray[np.float64]
        Lagged eps^2 matrix.
    B : NDArray[np.float64]
        B-spline design matrix.
    p : int
        ARCH order.

    Returns
    -------
    NDArray[np.float64]
        Gradient vector.
    """
    alpha = params[:p]
    theta = params[p:]
    n_eff = len(eps_sq)

    sigma2 = eps_lags_sq @ alpha + B @ theta
    sigma2 = np.maximum(sigma2, 1e-8)

    # d/d(sigma2_t) [eps_t^2/sigma2_t + log(sigma2_t)] = -eps_t^2/sigma2_t^2 + 1/sigma2_t
    dsigma2 = (1.0 / sigma2 - eps_sq / sigma2 ** 2) / n_eff  # shape (n_eff,)

    # Gradient w.r.t. alpha: sum_t dsigma2_t * eps_{t-j}^2
    grad_alpha = eps_lags_sq.T @ dsigma2  # shape (p,)

    # Gradient w.r.t. theta: sum_t dsigma2_t * B_k(x_{t-1})
    grad_theta = B.T @ dsigma2  # shape (K,)

    return np.concatenate([grad_alpha, grad_theta])


def bspline_null_model_residuals(
    eps: NDArray[np.float64],
    x_reduced: NDArray[np.float64],
    p: int,
    n_internal_knots: int = 3,
    degree: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate null model residuals using B-spline QMLE.

    Null model: sigma_t^2 = sum_j alpha_j * eps_{t-j}^2 + theta^T * B(x^{-l}_{t-1})

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    x_reduced : NDArray[np.float64]
        Reduced covariate matrix, shape (n, d_reduced). d_reduced must be 1 or 2.
    p : int
        ARCH order.
    n_internal_knots : int
        Number of internal knots per dimension.
    degree : int
        Spline degree.

    Returns
    -------
    tuple of:
        v_hat : NDArray[np.float64]
            Null residuals, shape (n-p,).
        alpha_hat : NDArray[np.float64]
            Estimated ARCH coefficients, shape (p,).
    """
    n = len(eps)
    n_eff = n - p

    # Build B-spline basis
    x_tm1 = x_reduced[p - 1: n - 1]  # x_{t-1} for t=p+1,...,n, shape (n_eff, d)

    d_reduced = x_reduced.shape[1] if x_reduced.ndim > 1 else 1

    if d_reduced == 1:
        x1d = x_tm1[:, 0] if x_tm1.ndim > 1 else x_tm1
        B = build_bspline_basis_1d(x1d, n_internal_knots, degree)
    elif d_reduced == 2:
        B = build_bspline_basis_2d(x_tm1, n_internal_knots, degree)
    else:
        raise ValueError(f"B-spline estimator supports d_reduced <= 2, got {d_reduced}.")

    K = B.shape[1]

    # Response: eps_t^2 for t=p,...,n-1
    eps_sq = eps[p:] ** 2  # shape (n_eff,)

    # Lagged eps^2 matrix: shape (n_eff, p)
    eps_lags_sq = np.zeros((n_eff, p))
    for j in range(1, p + 1):
        eps_lags_sq[:, j - 1] = eps[p - j: n - j] ** 2

    # Initialize parameters
    alpha_init = np.full(p, 0.1)
    # Initialize theta so that B @ theta gives a positive constant
    # Use mean of eps_sq as target
    mean_eps_sq = np.mean(eps_sq)
    # Solve B @ theta_init ≈ mean_eps_sq * ones
    theta_init = np.linalg.lstsq(B, np.full(n_eff, mean_eps_sq * 0.5), rcond=None)[0]
    theta_init = np.maximum(theta_init, 0.01)

    params_init = np.concatenate([alpha_init, theta_init])

    # Bounds: alpha_j >= 0, theta_k >= 0 (to ensure sigma_t^2 > 0)
    bounds = [(0.0, None)] * p + [(0.0, None)] * K

    result = minimize(
        fun=qmle_objective,
        x0=params_init,
        args=(eps_sq, eps_lags_sq, B, p),
        method="L-BFGS-B",
        jac=qmle_gradient,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
    )

    params_hat = result.x
    alpha_hat = params_hat[:p]
    theta_hat = params_hat[p:]

    # Compute null residuals
    sigma2_hat = eps_lags_sq @ alpha_hat + B @ theta_hat
    sigma2_hat = np.maximum(sigma2_hat, 1e-8)

    v_hat = eps_sq - sigma2_hat

    return v_hat, alpha_hat
