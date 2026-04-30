"""
Experiment 1: Nonparametric Hypothesis Testing Simulation for 2-Covariate ARCH-m(X) Models.

Simulates empirical size and power of the proposed ANOVA-based test statistic T_n
for testing the significance of X_2 in a 2-covariate semiparametric ARCH-m(X) model.

Compares:
- Proposed Kernel Smoothing estimator
- Proposed B-Spline QMLE estimator
- Linear QMLE Benchmark (Francq & Thieu, 2019)

Across 8 DGP models, 2 sample sizes, 3 error distributions, 2 covariate correlations,
and 11 signal strength levels.
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
from joblib import Parallel, delayed
import warnings
import json
from pathlib import Path

from src.dgp import (
    generate_covariates, generate_arch_mx, sigma2_2cov, get_p_2cov,
)
from src.kernel_estimator import kernel_null_model_residuals
from src.bspline_estimator import bspline_null_model_residuals
from src.test_statistic import compute_test_statistic, select_window_size
from src.linear_benchmark import linear_benchmark_test

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = list(range(1, 9))
N_VALUES = [500, 1000]
DISTRIBUTIONS = ["normal", "laplace", "t7"]
RHO_VALUES = [0.0, 0.5]
C_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPS = 50          # Reduced from 500 for feasibility; set to 500 for full experiment
BURNIN = 500
ALPHA_LEVEL = 0.05
D = 2                # Number of covariates
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Single replication function
# ---------------------------------------------------------------------------

def run_one_replication(
    model: int,
    n: int,
    distribution: str,
    rho: float,
    c: float,
    seed: int,
) -> dict[str, bool]:
    """
    Run one simulation replication for exp_1.

    Returns dict with rejection indicators for each method.
    """
    rng = np.random.default_rng(seed)
    p = get_p_2cov(model)
    n_total = n + BURNIN

    # Generate covariates
    X = generate_covariates(n_total, D, rho, rng)

    # Generate ARCH-m(X) process
    def sigma2_func(eps_lag, x):
        return sigma2_2cov(model, eps_lag, x, c)

    eps = generate_arch_mx(n, X, sigma2_func, p, distribution, rng, BURNIN)
    X_obs = X[BURNIN:]  # observed covariates (n x 2)

    k_n = select_window_size(n)

    # --- Kernel method ---
    try:
        # Null model: x_1 only (testing X_2)
        x_reduced_kernel = X_obs[:, 0:1]  # shape (n, 1)
        v_hat_kernel, _, _ = kernel_null_model_residuals(
            eps=eps,
            x_reduced=x_reduced_kernel,
            p=p,
            d_reduced=1,
            use_loo_cv=False,
        )
        x_test_kernel = X_obs[p:, 1]  # X_2 values for t=p+1,...,n
        _, _, pval_kernel = compute_test_statistic(v_hat_kernel, x_test_kernel, k_n, p)
        reject_kernel = pval_kernel < ALPHA_LEVEL
    except Exception:
        reject_kernel = False

    # --- B-Spline method ---
    try:
        x_reduced_bspline = X_obs[:, 0:1]  # x_1 only
        v_hat_bspline, _ = bspline_null_model_residuals(
            eps=eps,
            x_reduced=x_reduced_bspline,
            p=p,
            n_internal_knots=3,
            degree=3,
        )
        x_test_bspline = X_obs[p:, 1]
        _, _, pval_bspline = compute_test_statistic(v_hat_bspline, x_test_bspline, k_n, p)
        reject_bspline = pval_bspline < ALPHA_LEVEL
    except Exception:
        reject_bspline = False

    # --- Linear benchmark ---
    try:
        reject_linear, _ = linear_benchmark_test(
            eps=eps,
            X=X_obs,
            p=p,
            covariate_idx=1,  # test X_2 (0-based index 1)
            alpha_level=ALPHA_LEVEL,
        )
    except Exception:
        reject_linear = False

    return {
        "kernel": bool(reject_kernel),
        "bspline": bool(reject_bspline),
        "linear": bool(reject_linear),
    }


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_exp1(
    n_reps: int = N_REPS,
    n_jobs: int = -1,
    models: list[int] = MODELS,
    n_values: list[int] = N_VALUES,
    distributions: list[str] = DISTRIBUTIONS,
    rho_values: list[float] = RHO_VALUES,
    c_values: list[float] = C_VALUES,
) -> pd.DataFrame:
    """
    Run Experiment 1: 2-covariate ARCH-m(X) simulation.

    Returns DataFrame with empirical rejection rates.
    """
    records = []
    base_seed = 1000

    total_cells = len(models) * len(n_values) * len(distributions) * len(rho_values) * len(c_values)
    cell_idx = 0

    for model in models:
        for n in n_values:
            for dist in distributions:
                for rho in rho_values:
                    for c in c_values:
                        cell_idx += 1
                        print(f"  Cell {cell_idx}/{total_cells}: model={model}, n={n}, "
                              f"dist={dist}, rho={rho}, c={c:.1f}", flush=True)

                        seeds = [base_seed + cell_idx * 1000 + r for r in range(n_reps)]

                        results = Parallel(n_jobs=n_jobs, backend="loky")(
                            delayed(run_one_replication)(model, n, dist, rho, c, seed)
                            for seed in seeds
                        )

                        # Filter out None results
                        valid = [r for r in results if r is not None]
                        if len(valid) == 0:
                            continue

                        rate_kernel = np.mean([r["kernel"] for r in valid])
                        rate_bspline = np.mean([r["bspline"] for r in valid])
                        rate_linear = np.mean([r["linear"] for r in valid])

                        records.append({
                            "model": model,
                            "n": n,
                            "distribution": dist,
                            "rho": rho,
                            "c": c,
                            "rate_kernel": rate_kernel,
                            "rate_bspline": rate_bspline,
                            "rate_linear": rate_linear,
                            "n_valid": len(valid),
                        })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_power_curves(
    df: pd.DataFrame,
    n: int,
    distribution: str,
    rho: float,
    save_path: Path,
) -> None:
    """
    Plot power curves for all 8 models in a 2x4 grid.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    n : int
        Sample size to plot.
    distribution : str
        Error distribution to plot.
    rho : float
        Covariate correlation to plot.
    save_path : Path
        Path to save the figure.
    """
    subset = df[(df["n"] == n) & (df["distribution"] == distribution) & (df["rho"] == rho)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = {"rate_kernel": "#1f77b4", "rate_bspline": "#ff7f0e", "rate_linear": "#2ca02c"}
    labels = {"rate_kernel": "Kernel", "rate_bspline": "B-Spline", "rate_linear": "Linear"}
    linestyles = {"rate_kernel": "-", "rate_bspline": "--", "rate_linear": "-."}

    for i, model in enumerate(range(1, 9)):
        ax = axes[i]
        model_data = subset[subset["model"] == model].sort_values("c")

        if len(model_data) == 0:
            ax.set_title(f"Model {model}")
            continue

        for col in ["rate_kernel", "rate_bspline", "rate_linear"]:
            ax.plot(
                model_data["c"],
                model_data[col],
                color=colors[col],
                linestyle=linestyles[col],
                marker="o",
                markersize=4,
                label=labels[col],
            )

        # Nominal size line
        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7, label="Nominal 0.05")

        ax.set_title(f"Model {model}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Signal strength c", fontsize=9)
        ax.set_ylabel("Rejection rate", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Exp 1: 2-Covariate ARCH-m(X) Power Curves\n"
        f"n={n}, dist={distribution}, ρ={rho}",
        fontsize=13, fontweight="bold",
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
    print("Experiment 1: 2-Covariate ARCH-m(X) Simulation")
    print(f"  n_reps={N_REPS}, models={MODELS}")
    print("=" * 60)

    df = run_exp1(n_reps=N_REPS, n_jobs=-1)

    # Save results
    csv_path = RESULTS_DIR / "exp1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate plots for key configurations
    for n in N_VALUES:
        for dist in DISTRIBUTIONS:
            for rho in RHO_VALUES:
                plot_name = f"exp1_power_n{n}_{dist}_rho{int(rho*10)}.png"
                plot_power_curves(
                    df, n, dist, rho,
                    RESULTS_DIR / plot_name,
                )

    # Print summary table
    print("\n--- Empirical Size (c=0) Summary ---")
    size_df = df[df["c"] == 0.0].groupby(["model", "n", "distribution", "rho"])[
        ["rate_kernel", "rate_bspline", "rate_linear"]
    ].mean().round(3)
    print(size_df.to_string())

    print("\n--- Empirical Power (c=1.0) Summary ---")
    power_df = df[df["c"] == 1.0].groupby(["model", "n", "distribution", "rho"])[
        ["rate_kernel", "rate_bspline", "rate_linear"]
    ].mean().round(3)
    print(power_df.to_string())

    print("\nExperiment 1 complete.")
