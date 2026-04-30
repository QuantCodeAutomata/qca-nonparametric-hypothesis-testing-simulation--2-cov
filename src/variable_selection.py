"""
Variable Selection using Benjamini-Yekutieli FDR Control (ARM Procedure).

Implements the ARM (ARCH-m(X) with nonparametric testing and BY-FDR correction)
variable selection procedure for identifying relevant covariates in ARCH-m(X) models.

Also implements competing methods: Linear QMLE and LASSO.

Custom — Context7 found no library equivalent for this specific ARM variable
selection procedure with BY-FDR correction for ARCH-m(X) models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from .kernel_estimator import kernel_null_model_residuals
from .test_statistic import compute_test_statistic, select_window_size
from .linear_benchmark import fit_linear_qmle
from scipy.stats import norm


# ---------------------------------------------------------------------------
# BY-FDR Procedure
# ---------------------------------------------------------------------------

def benjamini_yekutieli_selection(
    p_values: NDArray[np.float64],
    q: float = 0.05,
) -> NDArray[np.bool_]:
    """
    Benjamini-Yekutieli (2001) FDR control procedure.

    Controls FDR at level q under arbitrary dependence using harmonic sum correction.

    BY threshold for rank k: threshold_k = (k/d) * (q / SUM_d)
    where SUM_d = sum_{l=1}^d 1/l (harmonic number).

    Parameters
    ----------
    p_values : NDArray[np.float64]
        Array of p-values, shape (d,).
    q : float
        FDR control level (default 0.05).

    Returns
    -------
    NDArray[np.bool_]
        Boolean array of shape (d,) indicating selected covariates.
    """
    d = len(p_values)
    # Harmonic sum
    SUM_d = np.sum(1.0 / np.arange(1, d + 1))

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]

    # BY thresholds: threshold_k = (k/d) * (q / SUM_d) for k=1,...,d
    thresholds = np.arange(1, d + 1) / d * (q / SUM_d)

    # Find k* = max{k: p_{(k)} <= threshold_k}
    k_star = 0
    for k in range(d, 0, -1):
        if sorted_pvals[k - 1] <= thresholds[k - 1]:
            k_star = k
            break

    # Reject first k_star hypotheses
    selected = np.zeros(d, dtype=bool)
    if k_star > 0:
        selected[sorted_idx[:k_star]] = True

    return selected


# ---------------------------------------------------------------------------
# ARM Procedure (full pipeline for one replication)
# ---------------------------------------------------------------------------

def arm_variable_selection(
    eps: NDArray[np.float64],
    X: NDArray[np.float64],
    p: int,
    q: float = 0.05,
    k_n: int | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """
    ARM variable selection procedure.

    For each covariate l=1,...,d:
    1. Estimate null model (excluding covariate l) via kernel smoothing
    2. Compute ANOVA test statistic T_n and p-value
    Apply BY-FDR correction to select relevant covariates.

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    X : NDArray[np.float64]
        Full covariate matrix, shape (n, d).
    p : int
        ARCH order.
    q : float
        FDR control level.
    k_n : int, optional
        Window size. If None, selected automatically based on n.

    Returns
    -------
    tuple of:
        selected : NDArray[np.bool_]
            Boolean array of shape (d,) indicating selected covariates.
        p_values : NDArray[np.float64]
            Raw p-values for each covariate, shape (d,).
    """
    n = len(eps)
    d = X.shape[1]

    if k_n is None:
        k_n = select_window_size(n)

    p_values = np.zeros(d)

    for l in range(d):
        # Reduced covariate set: all except covariate l
        cols = [j for j in range(d) if j != l]
        x_reduced = X[:, cols]  # shape (n, d-1)
        d_reduced = d - 1

        # Estimate null model residuals
        v_hat, _, _ = kernel_null_model_residuals(
            eps=eps,
            x_reduced=x_reduced,
            p=p,
            d_reduced=d_reduced,
            use_loo_cv=False,
        )

        # Test covariate l: sort by X_l values
        x_test = X[p:, l]  # X_{l,t-1} for t=p+1,...,n (0-indexed: X[p:, l])

        # Compute test statistic
        _, _, p_val = compute_test_statistic(v_hat, x_test, k_n, p)
        p_values[l] = p_val

    # BY-FDR selection
    selected = benjamini_yekutieli_selection(p_values, q)

    return selected, p_values


# ---------------------------------------------------------------------------
# Linear QMLE Variable Selection
# ---------------------------------------------------------------------------

def linear_qmle_variable_selection(
    eps: NDArray[np.float64],
    X: NDArray[np.float64],
    p: int,
    alpha_level: float = 0.05,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """
    Linear QMLE variable selection via individual Wald tests.

    No FDR correction applied (as per Francq & Thieu 2019 benchmark).

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    X : NDArray[np.float64]
        Covariate matrix, shape (n, d).
    p : int
        ARCH order.
    alpha_level : float
        Significance level for each individual test.

    Returns
    -------
    tuple of:
        selected : NDArray[np.bool_]
            Boolean array of shape (d,) indicating selected covariates.
        p_values : NDArray[np.float64]
            Two-sided p-values for each covariate.
    """
    d = X.shape[1]
    params_hat, std_errors, _ = fit_linear_qmle(eps, X, p)

    p_values = np.zeros(d)
    for l in range(d):
        param_idx = p + 1 + l
        beta_l = params_hat[param_idx]
        se_l = std_errors[param_idx]

        if se_l < 1e-10 or np.isnan(se_l):
            p_values[l] = 1.0
        else:
            z_stat = beta_l / se_l
            p_values[l] = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    selected = p_values < alpha_level
    return selected, p_values


# ---------------------------------------------------------------------------
# LASSO Variable Selection
# ---------------------------------------------------------------------------

def lasso_variable_selection(
    eps: NDArray[np.float64],
    X: NDArray[np.float64],
    p: int,
    cv_folds: int = 5,
) -> NDArray[np.bool_]:
    """
    LASSO variable selection for ARCH-m(X) models.

    Applies LASSO penalty only to x-related coefficients (beta_l), not ARCH
    coefficients (alpha_j). Uses 5-fold CV for lambda selection.

    Implementation:
    1. Response: Y_t = eps_t^2 for t=p+1,...,n
    2. Predictors: [eps_{t-1}^2,...,eps_{t-p}^2, x_{1,t-1},...,x_{d,t-1}]
    3. LASSO with penalty on x-columns only (partial LASSO via partialling out)

    Parameters
    ----------
    eps : NDArray[np.float64]
        Epsilon values, shape (n,).
    X : NDArray[np.float64]
        Covariate matrix, shape (n, d).
    p : int
        ARCH order.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    NDArray[np.bool_]
        Boolean array of shape (d,) indicating selected covariates.
    """
    # Using sklearn.linear_model.LassoCV — Context7 confirmed
    n = len(eps)
    d = X.shape[1]
    n_eff = n - p

    # Response
    Y = eps[p:] ** 2  # shape (n_eff,)

    # ARCH predictors (non-negative constrained)
    Z_arch = np.zeros((n_eff, p))
    for j in range(1, p + 1):
        Z_arch[:, j - 1] = eps[p - j: n - j] ** 2

    # Covariate predictors
    Z_x = X[p - 1: n - 1]  # x_{t-1} for t=p+1,...,n, shape (n_eff, d)

    # Partial out ARCH effects: project Y and Z_x onto orthogonal complement of Z_arch
    # Use non-negative OLS for ARCH part
    from scipy.optimize import nnls
    alpha_hat, _ = nnls(Z_arch, Y)
    Y_resid = Y - Z_arch @ alpha_hat

    # Project Z_x columns onto orthogonal complement of Z_arch
    Z_x_resid = np.zeros_like(Z_x)
    for l in range(d):
        alpha_l, _ = nnls(Z_arch, Z_x[:, l])
        Z_x_resid[:, l] = Z_x[:, l] - Z_arch @ alpha_l

    # Apply LASSO to (Y_resid, Z_x_resid)
    # Standardize predictors for LASSO
    scaler = StandardScaler()
    Z_x_scaled = scaler.fit_transform(Z_x_resid)

    lasso_cv = LassoCV(cv=cv_folds, fit_intercept=True, max_iter=2000, n_jobs=1)
    lasso_cv.fit(Z_x_scaled, Y_resid)

    # Select covariate l if |beta_hat_l| > 0 (after scaling back)
    coef_scaled = lasso_cv.coef_
    selected = np.abs(coef_scaled) > 1e-10

    return selected


# ---------------------------------------------------------------------------
# Selection Metrics
# ---------------------------------------------------------------------------

def compute_selection_metrics(
    selected: NDArray[np.bool_],
    true_active: list[int],
    d: int,
) -> dict[str, float]:
    """
    Compute variable selection metrics.

    Parameters
    ----------
    selected : NDArray[np.bool_]
        Boolean array of shape (d,) indicating selected covariates (0-based).
    true_active : list of int
        0-based indices of truly active covariates.
    d : int
        Total number of covariates.

    Returns
    -------
    dict with keys:
        CS : float
            Correctly selected (|I_hat ∩ active|).
        IS : float
            Incorrectly selected (|I_hat ∩ inactive|).
        CE : float
            Correctly excluded (|I_hat^c ∩ inactive|).
        IE : float
            Incorrectly excluded (|I_hat^c ∩ active|).
    """
    all_indices = set(range(d))
    active_set = set(true_active)
    inactive_set = all_indices - active_set
    selected_set = set(np.where(selected)[0])
    not_selected_set = all_indices - selected_set

    CS = len(selected_set & active_set)
    IS = len(selected_set & inactive_set)
    CE = len(not_selected_set & inactive_set)
    IE = len(not_selected_set & active_set)

    return {"CS": float(CS), "IS": float(IS), "CE": float(CE), "IE": float(IE)}
