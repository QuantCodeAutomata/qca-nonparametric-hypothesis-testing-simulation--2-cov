"""
ANOVA-Based Test Statistic T_n for nonparametric hypothesis testing
in semiparametric ARCH-m(X) models.

Implements the nonparametric ANOVA test statistic T_n = MST_n - MSE_n
and the standardized statistic Z_n for testing the significance of
a single exogenous covariate.

Custom — Context7 found no library equivalent for this specific ANOVA-based
test statistic for ARCH-m(X) models (paper methodology).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def rice_variance_estimator(
    v_hat: NDArray[np.float64],
    p: int,
) -> float:
    """
    Rice variance estimator for the null residuals.

    tau_hat = (1 / (2*(n-p-1))) * sum_{t=p+2}^{n} (v_hat_t - v_hat_{t-1})^2

    Note: v_hat is indexed from t=p+1 to n (0-indexed: 0 to n-p-1).
    The Rice estimator uses consecutive differences in original time order.

    Parameters
    ----------
    v_hat : NDArray[np.float64]
        Null residuals in original time order, shape (n-p,).
    p : int
        ARCH order.

    Returns
    -------
    float
        Rice variance estimate tau_hat.
    """
    n_eff = len(v_hat)  # n - p
    if n_eff < 2:
        return 1.0

    # Consecutive differences: v_hat[t] - v_hat[t-1] for t=1,...,n_eff-1
    diffs = np.diff(v_hat)  # shape (n_eff - 1,)
    tau_hat = np.sum(diffs ** 2) / (2.0 * (n_eff - 1))

    return max(tau_hat, 1e-10)


def construct_windows(
    x_test: NDArray[np.float64],
    k_n: int,
) -> list[list[int]]:
    """
    Construct k_n-windows for each observation sorted by x_test values.

    For each observation t (in sorted order by x_test):
    - Window W_t contains k_n observations with closest empirical CDF values
    - Edge correction: for bottom k_n/2 observations, use first k_n
    - Edge correction: for top k_n/2 observations, use last k_n

    Parameters
    ----------
    x_test : NDArray[np.float64]
        Values of the tested covariate X_l, shape (n_eff,).
    k_n : int
        Window size.

    Returns
    -------
    list of list of int
        windows[i] = list of k_n indices (in sorted order) for observation i.
        The outer list is indexed by sorted rank (0-based).
    """
    n_eff = len(x_test)
    sorted_idx = np.argsort(x_test)  # indices that sort x_test

    half = k_n // 2
    windows = []

    for r in range(n_eff):
        # Symmetric window centered at rank r
        lo = r - half
        hi = lo + k_n

        # Edge correction
        if lo < 0:
            lo = 0
            hi = k_n
        if hi > n_eff:
            hi = n_eff
            lo = n_eff - k_n

        lo = max(lo, 0)
        hi = min(hi, n_eff)

        # Window contains sorted indices lo:hi
        window = sorted_idx[lo:hi].tolist()
        windows.append(window)

    return windows


def compute_test_statistic(
    v_hat: NDArray[np.float64],
    x_test: NDArray[np.float64],
    k_n: int,
    p: int,
) -> tuple[float, float, float]:
    """
    Compute the ANOVA-based test statistic T_n and standardized Z_n.

    Steps:
    1. Rice variance estimator tau_hat (in original time order)
    2. Sort by x_test; construct k_n-windows
    3. Compute window means v_hat_{t.} and overall mean v_hat_{..}
    4. MST_n = [k_n/(n-p-1)] * sum_t (v_hat_{t.} - v_hat_{..})^2
    5. MSE_n = [1/((n-p)*(k_n-1))] * sum_t sum_{s in W_t} (v_hat_s - v_hat_{t.})^2
    6. T_n = MST_n - MSE_n
    7. Z_n = sqrt(n/k_n) * T_n / sqrt(4*tau_hat^2/3)

    Parameters
    ----------
    v_hat : NDArray[np.float64]
        Null residuals in original time order, shape (n-p,).
    x_test : NDArray[np.float64]
        Values of the tested covariate X_l for t=p+1,...,n, shape (n-p,).
    k_n : int
        Window size.
    p : int
        ARCH order.

    Returns
    -------
    tuple of (T_n, Z_n, p_value):
        T_n : float
            Raw test statistic.
        Z_n : float
            Standardized test statistic.
        p_value : float
            One-sided upper p-value = 1 - Phi(Z_n).
    """
    n_eff = len(v_hat)  # n - p
    n = n_eff + p

    # Step 1: Rice variance estimator (original time order)
    tau_hat = rice_variance_estimator(v_hat, p)

    # Step 2: Construct windows sorted by x_test
    windows = construct_windows(x_test, k_n)

    # Step 3: Compute window means
    v_hat_dot = np.array([np.mean(v_hat[w]) for w in windows])  # shape (n_eff,)
    v_hat_dotdot = np.mean(v_hat_dot)  # overall mean

    # Step 4: MST_n = [k_n/(n-p-1)] * sum_t (v_hat_{t.} - v_hat_{..})^2
    MST_n = (k_n / (n_eff - 1)) * np.sum((v_hat_dot - v_hat_dotdot) ** 2)

    # Step 5: MSE_n = [1/((n-p)*(k_n-1))] * sum_t sum_{s in W_t} (v_hat_s - v_hat_{t.})^2
    if k_n <= 1:
        MSE_n = 0.0
    else:
        sse = 0.0
        for i, w in enumerate(windows):
            sse += np.sum((v_hat[w] - v_hat_dot[i]) ** 2)
        MSE_n = sse / (n_eff * (k_n - 1))

    # Step 6: T_n = MST_n - MSE_n
    T_n = MST_n - MSE_n

    # Step 7: Z_n = sqrt(n/k_n) * T_n / sqrt(4*tau_hat^2/3)
    denom = np.sqrt(4.0 * tau_hat ** 2 / 3.0)
    if denom < 1e-10:
        Z_n = 0.0
    else:
        Z_n = np.sqrt(n / k_n) * T_n / denom

    # One-sided upper p-value
    p_value = 1.0 - norm.cdf(Z_n)

    return T_n, Z_n, p_value


def select_window_size(n: int) -> int:
    """
    Select window size k_n based on sample size n.

    k_n = 3 for n <= 1000
    k_n = 4 for 1000 < n <= 5000
    k_n = 5 for n > 5000

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    int
        Window size k_n.
    """
    if n <= 1000:
        return 3
    elif n <= 5000:
        return 4
    else:
        return 5
