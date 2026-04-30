"""
Kernel Smoothing QMLE Estimator for semiparametric ARCH-m(X) models.

Implements the nonparametric kernel-based estimation of the null model
(reduced model excluding the tested covariate) using product Gaussian kernels.

Custom — Context7 found no library equivalent for this specific ARCH-m(X)
kernel QMLE estimator (paper methodology).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def gaussian_kernel(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Standard Gaussian kernel K(u) = (1/sqrt(2*pi)) * exp(-u^2/2).

    Parameters
    ----------
    u : NDArray[np.float64]
        Standardized argument.

    Returns
    -------
    NDArray[np.float64]
        Kernel values.
    """
    return np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)


def silverman_bandwidth(x: NDArray[np.float64], d_eff: int) -> float:
    """
    Silverman rule-of-thumb bandwidth for d_eff-dimensional kernel.

    h = 1.06 * std(x) * n^{-1/(d_eff + 4)}

    Parameters
    ----------
    x : NDArray[np.float64]
        1D array of covariate values.
    d_eff : int
        Effective dimension for bandwidth selection.

    Returns
    -------
    float
        Bandwidth h.
    """
    n = len(x)
    std_x = np.std(x, ddof=1)
    if std_x < 1e-10:
        std_x = 1.0
    return 1.06 * std_x * n ** (-1.0 / (d_eff + 4.0))


def loo_cv_bandwidth(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    h_grid: Optional[NDArray[np.float64]] = None,
) -> float:
    """
    Leave-one-out cross-validation bandwidth selection for 1D kernel regression.

    Minimizes sum_t (y_t - m_hat_{-t}(x_t))^2 over a grid of h values.

    Parameters
    ----------
    x : NDArray[np.float64]
        1D covariate values, shape (n,).
    y : NDArray[np.float64]
        Response values, shape (n,).
    h_grid : NDArray[np.float64], optional
        Grid of bandwidth values to search. If None, uses 20 log-spaced
        values between 0.1*std(x) and 2*std(x).

    Returns
    -------
    float
        Optimal bandwidth minimizing LOO-CV criterion.
    """
    n = len(x)
    std_x = np.std(x, ddof=1)
    if std_x < 1e-10:
        std_x = 1.0

    if h_grid is None:
        h_grid = np.logspace(
            np.log10(0.1 * std_x),
            np.log10(2.0 * std_x),
            20,
        )

    best_h = h_grid[0]
    best_cv = np.inf

    for h in h_grid:
        # Compute full kernel matrix (n x n)
        diff = (x[:, None] - x[None, :]) / h  # (n, n)
        K = gaussian_kernel(diff)  # (n, n)
        np.fill_diagonal(K, 0.0)  # LOO: exclude self

        row_sums = K.sum(axis=1)  # (n,)
        # Avoid division by zero
        valid = row_sums > 1e-10
        m_loo = np.where(valid, (K @ y) / np.where(valid, row_sums, 1.0), y.mean())

        cv = np.mean((y - m_loo) ** 2)
        if cv < best_cv:
            best_cv = cv
            best_h = h

    return best_h


def compute_kernel_weights_1d(
    x_eval: NDArray[np.float64],
    x_ref: NDArray[np.float64],
    h: float,
) -> NDArray[np.float64]:
    """
    Compute Nadaraya-Watson kernel weight matrix for 1D kernel regression.

    W_{ts} = K_h(x_eval[t] - x_ref[s]) / sum_m K_h(x_eval[t] - x_ref[m])

    Parameters
    ----------
    x_eval : NDArray[np.float64]
        Evaluation points, shape (n_eval,).
    x_ref : NDArray[np.float64]
        Reference points, shape (n_ref,).
    h : float
        Bandwidth.

    Returns
    -------
    NDArray[np.float64]
        Weight matrix of shape (n_eval, n_ref).
    """
    diff = (x_eval[:, None] - x_ref[None, :]) / h  # (n_eval, n_ref)
    K = gaussian_kernel(diff)
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    return K / row_sums


def compute_kernel_weights_nd(
    x_eval: NDArray[np.float64],
    x_ref: NDArray[np.float64],
    h: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute product Gaussian kernel weight matrix for d-dimensional kernel regression.

    K_H(u) = prod_j (1/h_j) * phi(u_j/h_j)
    W_{ts} = K_H(x_eval[t] - x_ref[s]) / sum_m K_H(x_eval[t] - x_ref[m])

    Parameters
    ----------
    x_eval : NDArray[np.float64]
        Evaluation points, shape (n_eval, d).
    x_ref : NDArray[np.float64]
        Reference points, shape (n_ref, d).
    h : NDArray[np.float64]
        Bandwidth vector, shape (d,).

    Returns
    -------
    NDArray[np.float64]
        Weight matrix of shape (n_eval, n_ref).
    """
    n_eval = x_eval.shape[0]
    n_ref = x_ref.shape[0]
    d = x_eval.shape[1]

    # Compute log kernel values for numerical stability
    # log K_H(u) = sum_j log[(1/h_j)*phi(u_j/h_j)]
    # = sum_j [-log(h_j) - 0.5*(u_j/h_j)^2 - 0.5*log(2*pi)]
    log_K = np.zeros((n_eval, n_ref))
    for j in range(d):
        diff_j = (x_eval[:, j:j+1] - x_ref[None, :, j]) / h[j]  # (n_eval, n_ref)
        log_K += -0.5 * diff_j ** 2 - np.log(h[j]) - 0.5 * np.log(2.0 * np.pi)

    # Subtract row max for numerical stability before exp
    row_max = log_K.max(axis=1, keepdims=True)
    K = np.exp(log_K - row_max)

    row_sums = K.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    return K / row_sums


def compute_kernel_weights_nd_chunked(
    x_eval: NDArray[np.float64],
    x_ref: NDArray[np.float64],
    h: NDArray[np.float64],
    chunk_size: int = 500,
) -> NDArray[np.float64]:
    """
    Memory-efficient chunked computation of product Gaussian kernel weights.

    For large n, computes the weight matrix in chunks to avoid OOM errors.

    Parameters
    ----------
    x_eval : NDArray[np.float64]
        Evaluation points, shape (n_eval, d).
    x_ref : NDArray[np.float64]
        Reference points, shape (n_ref, d).
    h : NDArray[np.float64]
        Bandwidth vector, shape (d,).
    chunk_size : int
        Number of evaluation points per chunk.

    Returns
    -------
    NDArray[np.float64]
        Weight matrix of shape (n_eval, n_ref).
    """
    n_eval = x_eval.shape[0]
    W = np.zeros((n_eval, x_ref.shape[0]), dtype=np.float32)

    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        W[start:end] = compute_kernel_weights_nd(
            x_eval[start:end], x_ref, h
        ).astype(np.float32)

    return W.astype(np.float64)


def kernel_null_model_residuals(
    eps: NDArray[np.float64],
    x_reduced: NDArray[np.float64],
    p: int,
    d_reduced: int,
    use_loo_cv: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Estimate null model residuals using kernel smoothing.

    Null model: sigma_t^2 = sum_j alpha_j * eps_{t-j}^2 + m(x^{-l}_{t-1})
    where x^{-l} is the reduced covariate set (excluding tested covariate l).

    Steps:
    1. Compute kernel weight matrix W using x_reduced
    2. Partial out: Y_tilde = eps^2 - W*eps^2, X_tilde_j = eps_{t-j}^2 - W*eps_{t-j}^2
    3. Estimate alpha_hat by non-negative OLS
    4. Compute null residuals v_hat_t

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    x_reduced : NDArray[np.float64]
        Reduced covariate matrix (excluding tested covariate), shape (n, d_reduced).
        x_reduced[t] = x^{-l}_{t} (0-indexed, so x^{-l}_{t-1} = x_reduced[t-1]).
    p : int
        ARCH order.
    d_reduced : int
        Dimension of reduced covariate set.
    use_loo_cv : bool
        If True, use LOO-CV for bandwidth (1D only); else use Silverman rule.

    Returns
    -------
    tuple of:
        v_hat : NDArray[np.float64]
            Null residuals v_hat_t for t=p+1,...,n (0-indexed: t=p,...,n-1), shape (n-p,).
        alpha_hat : NDArray[np.float64]
            Estimated ARCH coefficients, shape (p,).
        h_used : float or NDArray
            Bandwidth(s) used.
    """
    n = len(eps)
    n_eff = n - p  # effective sample size

    # Indices: t runs from p to n-1 (0-indexed), so eps_t = eps[p:n]
    # x_{t-1} = x_reduced[p-1:n-1] for t=p,...,n-1
    x_eval = x_reduced[p - 1: n - 1]  # shape (n_eff, d_reduced) -- x_{t-1} for t=p+1,...,n
    x_ref = x_reduced[p - 1: n - 1]   # same for Nadaraya-Watson

    # Bandwidth selection
    if d_reduced == 1:
        x1d = x_eval[:, 0] if x_eval.ndim > 1 else x_eval
        if use_loo_cv:
            y_for_cv = eps[p:] ** 2
            h_val = loo_cv_bandwidth(x1d, y_for_cv)
        else:
            h_val = silverman_bandwidth(x1d, d_eff=1)
        h = np.array([h_val])
        h_used = h_val
    else:
        h = np.array([silverman_bandwidth(x_eval[:, j], d_eff=d_reduced)
                      for j in range(d_reduced)])
        h_used = h

    # Compute kernel weight matrix W: shape (n_eff, n_eff)
    if d_reduced == 1:
        W = compute_kernel_weights_1d(x_eval[:, 0], x_ref[:, 0], h[0])
    else:
        W = compute_kernel_weights_nd_chunked(x_eval, x_ref, h)

    # Response: eps_t^2 for t=p,...,n-1
    Y = eps[p:] ** 2  # shape (n_eff,)

    # Partialled-out response: Y_tilde = Y - W*Y
    WY = W @ Y  # shape (n_eff,)
    Y_tilde = Y - WY  # shape (n_eff,)

    # Partialled-out predictors: X_tilde_j for j=1,...,p
    # X_j = eps_{t-j}^2 for t=p,...,n-1 => eps[p-j:n-j]^2
    X_tilde = np.zeros((n_eff, p))
    for j in range(1, p + 1):
        Xj = eps[p - j: n - j] ** 2  # shape (n_eff,)
        WXj = W @ Xj
        X_tilde[:, j - 1] = Xj - WXj

    # Non-negative OLS: alpha_hat = (X_tilde'X_tilde + ridge*I)^{-1} X_tilde'Y_tilde
    XtX = X_tilde.T @ X_tilde
    cond_num = np.linalg.cond(XtX)
    ridge = 1e-8 if cond_num <= 1e12 else 1e-6
    XtX_reg = XtX + ridge * np.eye(p)
    XtY = X_tilde.T @ Y_tilde
    alpha_hat = np.linalg.solve(XtX_reg, XtY)
    alpha_hat = np.maximum(alpha_hat, 0.0)  # non-negativity constraint

    # Compute null residuals: v_hat_t = eps_t^2 - sum_j alpha_hat_j*eps_{t-j}^2 - m_hat(x^{-l}_{t-1})
    # m_hat(x^{-l}_{t-1}) = W * (eps_t^2 - sum_j alpha_hat_j*eps_{t-j}^2)
    arch_part = np.zeros(n_eff)
    for j in range(1, p + 1):
        arch_part += alpha_hat[j - 1] * eps[p - j: n - j] ** 2

    residual_for_m = Y - arch_part  # eps_t^2 - ARCH part
    m_hat = W @ residual_for_m  # kernel estimate of m

    v_hat = residual_for_m - m_hat  # null residuals

    return v_hat, alpha_hat, h_used
