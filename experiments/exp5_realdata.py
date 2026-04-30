"""
Experiment 5: Real Data Application - S&P 500 Volatility Variable Selection.

Applies the ARM variable selection procedure to identify which of 11 global
financial and commodity market variables significantly explain S&P 500 daily
return volatility using a semiparametric ARCH(1)-m(X) model.

Data: Daily log returns from Yahoo Finance (2009-present).
Covariates: China (SSE), Asia (AAXJ), Oil (CL=F), USD (DX-Y.NYB), Gold (GC=F),
            Copper (HG=F), Silver (SI=F), Steel (SLX), Rice (ZR=F),
            Wheat (ZW=F), Europe (^STOXX50E).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scipy.stats import norm

from src.kernel_estimator import (
    silverman_bandwidth,
    compute_kernel_weights_nd_chunked,
)
from src.test_statistic import rice_variance_estimator, construct_windows
from src.variable_selection import benjamini_yekutieli_selection

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Covariate definitions
# ---------------------------------------------------------------------------

COVARIATE_NAMES = [
    "China (SSE)",
    "Asia (AAXJ)",
    "Crude Oil (WTI)",
    "USD Index",
    "Gold",
    "Copper",
    "Silver",
    "Steel (SLX)",
    "Rice",
    "Wheat",
    "Europe (STOXX50)",
]

TICKERS = {
    "SP500": "^GSPC",
    "China": "000001.SS",
    "Asia": "AAXJ",
    "Oil": "CL=F",
    "USD": "DX-Y.NYB",
    "Gold": "GC=F",
    "Copper": "HG=F",
    "Silver": "SI=F",
    "Steel": "SLX",
    "Rice": "ZR=F",
    "Wheat": "ZW=F",
    "Europe": "^STOXX50E",
}

START_DATE = "2009-01-02"
END_DATE = "2023-12-31"


# ---------------------------------------------------------------------------
# Data download and preparation
# ---------------------------------------------------------------------------

def download_data() -> pd.DataFrame:
    """
    Download daily closing prices for S&P 500 and 11 covariates from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns, aligned to common trading dates.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    print("  Downloading data from Yahoo Finance...")
    all_returns = {}

    for name, ticker in TICKERS.items():
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True)
            if len(data) == 0:
                print(f"    WARNING: No data for {name} ({ticker})")
                continue
            # Handle both old (Series) and new (MultiIndex DataFrame) yfinance APIs
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]  # squeeze MultiIndex to Series
            prices = close.dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()
            all_returns[name] = log_returns
            print(f"    Downloaded {name}: {len(log_returns)} observations")
        except Exception as e:
            print(f"    WARNING: Failed to download {name} ({ticker}): {e}")

    if "SP500" not in all_returns:
        raise ValueError("Failed to download S&P 500 data.")

    # Align to common dates (inner join)
    df = pd.DataFrame(all_returns)
    df = df.dropna()

    print(f"  Common sample: {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare S&P 500 returns and covariate matrix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of log returns.

    Returns
    -------
    tuple of:
        eps : np.ndarray
            Demeaned S&P 500 log returns, shape (n,).
        X : np.ndarray
            Covariate matrix (lagged by 1), shape (n, 11).
        cov_names : list of str
            Covariate names.
    """
    sp500_returns = df["SP500"].values
    eps = sp500_returns - sp500_returns.mean()

    cov_cols = [c for c in df.columns if c != "SP500"]
    X_raw = df[cov_cols].values  # shape (n, d)

    # Covariates enter as lagged values X_{t-1}
    # We align: eps[1:] with X[:-1] (X_{t-1} for t=2,...,n)
    # But for ARCH(1) with p=1, we use eps[1:] and X[:-1]
    # Keep full alignment: eps and X are already aligned (same dates)
    # The kernel estimator handles the lag internally

    cov_names = cov_cols
    return eps, X_raw, cov_names


def adf_test(series: np.ndarray, name: str) -> bool:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns True if series is stationary at 1% level.
    """
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series, autolag="AIC")
    p_val = result[1]
    is_stationary = p_val < 0.01
    print(f"    ADF test {name}: p-value={p_val:.4f} -> {'Stationary' if is_stationary else 'NON-STATIONARY'}")
    return is_stationary


# ---------------------------------------------------------------------------
# Test statistic computation for real data
# ---------------------------------------------------------------------------

def compute_test_for_covariate(
    eps: np.ndarray,
    X: np.ndarray,
    l: int,
    p: int = 1,
    k_n: int | None = None,
) -> tuple[float, float, float]:
    """
    Compute ANOVA test statistic for covariate l in real data setting.

    Parameters
    ----------
    eps : np.ndarray
        Demeaned S&P 500 returns, shape (n,).
    X : np.ndarray
        Covariate matrix, shape (n, d).
    l : int
        0-based index of covariate to test.
    p : int
        ARCH order (default 1).
    k_n : int, optional
        Window size.

    Returns
    -------
    tuple of (Z_n, p_value, alpha_hat)
    """
    n = len(eps)
    d = X.shape[1]

    if k_n is None:
        # Use floor(n^{1/6}) as suggested in paper
        k_n = max(3, int(np.floor(n ** (1.0 / 6.0))))

    # Reduced covariate set: all except covariate l
    cols = [j for j in range(d) if j != l]
    x_reduced = X[:, cols]  # shape (n, d-1)
    d_reduced = d - 1

    # Bandwidth: Silverman rule for d_reduced=10 dimensions
    # h_j = 1.06 * std(x_j) * n^{-1/14}
    h = np.array([
        1.06 * np.std(x_reduced[:, j], ddof=1) * n ** (-1.0 / (d_reduced + 4.0))
        for j in range(d_reduced)
    ])

    # Compute kernel weights (streaming, chunked)
    x_eval = x_reduced[p - 1: n - 1]  # x^{-l}_{t-1} for t=p+1,...,n
    x_ref = x_reduced[p - 1: n - 1]

    W = compute_kernel_weights_nd_chunked(x_eval, x_ref, h, chunk_size=200)

    # Response: eps_t^2 for t=p,...,n-1
    Y = eps[p:] ** 2
    n_eff = len(Y)

    # Partialled-out response
    WY = W @ Y
    Y_tilde = Y - WY

    # Partialled-out predictor (p=1)
    Xj = eps[p - 1: n - 1] ** 2
    WXj = W @ Xj
    X_tilde = (Xj - WXj).reshape(-1, 1)

    # Estimate alpha_hat (non-negative)
    XtX = X_tilde.T @ X_tilde + 1e-8 * np.eye(p)
    XtY = X_tilde.T @ Y_tilde
    alpha_hat = np.linalg.solve(XtX, XtY).flatten()
    alpha_hat = np.clip(alpha_hat, 0.001, 0.999)  # ARCH stationarity

    # Null residuals
    arch_part = alpha_hat[0] * eps[p - 1: n - 1] ** 2
    residual_for_m = Y - arch_part
    m_hat = W @ residual_for_m
    v_hat = residual_for_m - m_hat

    # Rice variance estimator
    tau_hat = rice_variance_estimator(v_hat, p)

    # Window construction (sort by X_l values)
    x_test = X[p:, l]  # X_{l,t-1} for t=p+1,...,n
    windows = construct_windows(x_test, k_n)

    # Window means
    v_hat_dot = np.array([np.mean(v_hat[w]) for w in windows])
    v_hat_dotdot = np.mean(v_hat_dot)

    # MST_n and MSE_n
    MST_n = (k_n / (n_eff - 1)) * np.sum((v_hat_dot - v_hat_dotdot) ** 2)

    if k_n <= 1:
        MSE_n = 0.0
    else:
        sse = sum(np.sum((v_hat[w] - v_hat_dot[i]) ** 2) for i, w in enumerate(windows))
        MSE_n = sse / (n_eff * (k_n - 1))

    T_n = MST_n - MSE_n

    # Z_n statistic
    denom = np.sqrt(4.0 * tau_hat ** 2 / 3.0)
    Z_n = np.sqrt(n / k_n) * T_n / denom if denom > 1e-10 else 0.0

    p_value = 1.0 - norm.cdf(Z_n)

    return Z_n, p_value, alpha_hat[0]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_exp5() -> pd.DataFrame:
    """
    Run Experiment 5: Real data S&P 500 volatility variable selection.

    Returns
    -------
    pd.DataFrame
        Results table with Z_n, p-values, BY thresholds, and selection decisions.
    """
    # Download data
    df = download_data()

    # Prepare data
    eps, X, cov_names = prepare_data(df)
    n = len(eps)
    d = X.shape[1]

    print(f"\n  Sample size: n={n}")
    print(f"  Number of covariates: d={d}")

    # ADF tests
    print("\n  ADF stationarity tests:")
    adf_test(eps, "S&P 500 returns")
    for j, name in enumerate(cov_names):
        adf_test(X[:, j], name)

    # Window size
    k_n = max(3, int(np.floor(n ** (1.0 / 6.0))))
    print(f"\n  Window size k_n = {k_n}")

    # Test each covariate
    print("\n  Testing each covariate...")
    results = []
    for l, name in enumerate(cov_names):
        print(f"    Testing {name} (l={l+1}/{d})...", flush=True)
        try:
            Z_n, p_val, alpha_hat = compute_test_for_covariate(eps, X, l, p=1, k_n=k_n)
            results.append({
                "covariate": name,
                "l": l + 1,
                "Z_n": Z_n,
                "p_value": p_val,
                "alpha_hat": alpha_hat,
            })
            print(f"      Z_n={Z_n:.4f}, p-value={p_val:.6f}")
        except Exception as e:
            print(f"      ERROR: {e}")
            results.append({
                "covariate": name,
                "l": l + 1,
                "Z_n": np.nan,
                "p_value": 1.0,
                "alpha_hat": np.nan,
            })

    df_results = pd.DataFrame(results)

    # BY-FDR selection
    p_values = df_results["p_value"].values
    selected = benjamini_yekutieli_selection(p_values, q=0.05)
    df_results["selected"] = selected

    # Compute BY thresholds at each rank
    d_test = len(p_values)
    SUM_d = np.sum(1.0 / np.arange(1, d_test + 1))
    sorted_idx = np.argsort(p_values)
    ranks = np.empty(d_test, dtype=int)
    ranks[sorted_idx] = np.arange(1, d_test + 1)
    df_results["rank"] = ranks
    df_results["BY_threshold"] = ranks / d_test * (0.05 / SUM_d)

    # Sort by p-value
    df_results = df_results.sort_values("p_value").reset_index(drop=True)

    return df_results, n, df.index[0].date(), df.index[-1].date()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(df_results: pd.DataFrame, n: int, start_date, end_date, save_path: Path) -> None:
    """Plot Z_n statistics and p-values for all covariates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Z_n statistics
    ax1 = axes[0]
    colors = ["#d62728" if sel else "#1f77b4" for sel in df_results["selected"]]
    bars = ax1.barh(df_results["covariate"], df_results["Z_n"], color=colors, alpha=0.8)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Z_n statistic", fontsize=11)
    ax1.set_title("Test Statistics Z_n", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.8, label="Selected (BY-FDR)"),
        Patch(facecolor="#1f77b4", alpha=0.8, label="Not selected"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9)

    # Plot 2: p-values with BY thresholds
    ax2 = axes[1]
    ax2.scatter(range(len(df_results)), df_results["p_value"],
                c=colors, s=80, zorder=5, label="p-value")
    ax2.plot(range(len(df_results)), df_results["BY_threshold"],
             "k--", linewidth=1.5, label="BY threshold")
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(df_results)))
    ax2.set_xticklabels(df_results["covariate"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("p-value (log scale)", fontsize=11)
    ax2.set_title("p-values vs BY Thresholds", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exp 5: S&P 500 Volatility Variable Selection\n"
        f"n={n} obs ({start_date} to {end_date}), ARCH(1)-m(X), BY-FDR q=0.05",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 5: Real Data - S&P 500 Volatility")
    print("=" * 60)

    df_results, n, start_date, end_date = run_exp5()

    # Save results
    csv_path = RESULTS_DIR / "exp5_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Plot
    plot_results(df_results, n, start_date, end_date, RESULTS_DIR / "exp5_results.png")

    # Print results table
    print("\n--- Results Table (sorted by p-value) ---")
    print(df_results[["covariate", "Z_n", "p_value", "BY_threshold", "selected"]].to_string(index=False))

    selected_covs = df_results[df_results["selected"]]["covariate"].tolist()
    print(f"\nSelected covariates (BY-FDR q=0.05): {selected_covs}")
    print(f"Number selected: {len(selected_covs)} out of {len(df_results)}")

    print("\nExperiment 5 complete.")
