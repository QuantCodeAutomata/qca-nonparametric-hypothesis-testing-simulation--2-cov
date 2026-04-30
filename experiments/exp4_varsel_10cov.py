"""
Experiment 4: Variable Selection Simulation for 10-Covariate ARCH-m(X) Models.

Evaluates the ARM variable selection procedure against Linear QMLE and LASSO
in a 10-covariate ARCH-m(X) setting.

True active covariates: X_1, X_3, X_4, X_5, X_9 (1-based, i.e., indices 0,2,3,4,8 in 0-based).
Irrelevant covariates: X_2, X_6, X_7, X_8, X_10 (indices 1,5,6,7,9 in 0-based).
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
import warnings
from joblib import Parallel, delayed
from pathlib import Path

from src.dgp import (
    generate_covariates, generate_arch_mx, sigma2_10cov_varsel, get_p_10cov_varsel,
)
from src.variable_selection import (
    arm_variable_selection,
    linear_qmle_variable_selection,
    lasso_variable_selection,
    compute_selection_metrics,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = list(range(1, 9))
N_VALUES = [500, 1000, 5000, 10000]
DISTRIBUTIONS = ["normal", "laplace", "t7"]
RHO_VALUES = [0.0, 0.5]
N_REPS = 20          # Reduced from 200 for feasibility
N_REPS_TABLE6 = 50   # Reduced from 500 for Table 6
BURNIN = 500
D = 10
TRUE_ACTIVE = [0, 2, 3, 4, 8]    # 0-based: X_1, X_3, X_4, X_5, X_9
TRUE_INACTIVE = [1, 5, 6, 7, 9]  # 0-based: X_2, X_6, X_7, X_8, X_10
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
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run one variable selection replication for exp_4."""
    rng = np.random.default_rng(seed)
    p = get_p_10cov_varsel(model)
    n_total = n + BURNIN

    X = generate_covariates(n_total, D, rho, rng)

    def sigma2_func(eps_lag, x):
        return sigma2_10cov_varsel(model, eps_lag, x)

    eps = generate_arch_mx(n, X, sigma2_func, p, distribution, rng, BURNIN)
    X_obs = X[BURNIN:]

    results = {}

    # --- ARM method ---
    try:
        selected_arm, _ = arm_variable_selection(eps, X_obs, p, q=0.05)
        metrics_arm = compute_selection_metrics(selected_arm, TRUE_ACTIVE, D)
        results["arm"] = metrics_arm
        results["arm_selected"] = selected_arm.tolist()
    except Exception:
        results["arm"] = {"CS": 0.0, "IS": 0.0, "CE": float(len(TRUE_INACTIVE)), "IE": float(len(TRUE_ACTIVE))}
        results["arm_selected"] = [False] * D

    # --- Linear QMLE method ---
    try:
        selected_linear, _ = linear_qmle_variable_selection(eps, X_obs, p, alpha_level=0.05)
        metrics_linear = compute_selection_metrics(selected_linear, TRUE_ACTIVE, D)
        results["linear"] = metrics_linear
        results["linear_selected"] = selected_linear.tolist()
    except Exception:
        results["linear"] = {"CS": 0.0, "IS": 0.0, "CE": float(len(TRUE_INACTIVE)), "IE": float(len(TRUE_ACTIVE))}
        results["linear_selected"] = [False] * D

    # --- LASSO method ---
    try:
        selected_lasso = lasso_variable_selection(eps, X_obs, p, cv_folds=5)
        metrics_lasso = compute_selection_metrics(selected_lasso, TRUE_ACTIVE, D)
        results["lasso"] = metrics_lasso
        results["lasso_selected"] = selected_lasso.tolist()
    except Exception:
        results["lasso"] = {"CS": 0.0, "IS": 0.0, "CE": float(len(TRUE_INACTIVE)), "IE": float(len(TRUE_ACTIVE))}
        results["lasso_selected"] = [False] * D

    return results


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_exp4(
    n_reps: int = N_REPS,
    n_jobs: int = -1,
    models: list[int] = MODELS,
    n_values: list[int] = N_VALUES,
    distributions: list[str] = DISTRIBUTIONS,
    rho_values: list[float] = RHO_VALUES,
) -> pd.DataFrame:
    """Run Experiment 4: Variable selection with 10 covariates."""
    records = []
    base_seed = 4000

    total_cells = len(models) * len(n_values) * len(distributions) * len(rho_values)
    cell_idx = 0

    for model in models:
        for n in n_values:
            for dist in distributions:
                for rho in rho_values:
                    cell_idx += 1
                    print(f"  Cell {cell_idx}/{total_cells}: model={model}, n={n}, "
                          f"dist={dist}, rho={rho}", flush=True)

                    seeds = [base_seed + cell_idx * 1000 + r for r in range(n_reps)]

                    results = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(run_one_replication)(model, n, dist, rho, seed)
                        for seed in seeds
                    )

                    valid = [r for r in results if r is not None]
                    if len(valid) == 0:
                        continue

                    for method in ["arm", "linear", "lasso"]:
                        cs = np.mean([r[method]["CS"] for r in valid])
                        is_ = np.mean([r[method]["IS"] for r in valid])
                        ce = np.mean([r[method]["CE"] for r in valid])
                        ie = np.mean([r[method]["IE"] for r in valid])

                        records.append({
                            "model": model,
                            "n": n,
                            "distribution": dist,
                            "rho": rho,
                            "method": method,
                            "CS": cs,
                            "IS": is_,
                            "CE": ce,
                            "IE": ie,
                            "n_valid": len(valid),
                        })

    return pd.DataFrame(records)


def run_table6(
    n_reps: int = N_REPS_TABLE6,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run Table 6: Covariate-wise selection proportions for Model 1, Normal errors."""
    records = []
    base_seed = 4500

    for rho in RHO_VALUES:
        for n in N_VALUES:
            print(f"  Table 6: n={n}, rho={rho}", flush=True)
            seeds = [base_seed + int(rho * 10) * 100000 + n * 10 + r for r in range(n_reps)]

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_one_replication)(1, n, "normal", rho, seed)
                for seed in seeds
            )

            valid = [r for r in results if r is not None]
            if len(valid) == 0:
                continue

            for method in ["arm", "linear", "lasso"]:
                key = f"{method}_selected"
                proportions = np.mean([r[key] for r in valid], axis=0)
                for l in range(D):
                    records.append({
                        "n": n,
                        "rho": rho,
                        "method": method,
                        "covariate": l + 1,
                        "proportion": proportions[l],
                        "is_active": l in TRUE_ACTIVE,
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_selection_metrics(
    df: pd.DataFrame,
    distribution: str,
    rho: float,
    save_path: Path,
) -> None:
    """Plot C.S. and I.S. metrics vs sample size for all models."""
    subset = df[(df["distribution"] == distribution) & (df["rho"] == rho)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = {"arm": "#1f77b4", "linear": "#2ca02c", "lasso": "#d62728"}
    markers = {"arm": "o", "linear": "s", "lasso": "^"}

    for i, model in enumerate(range(1, 9)):
        ax = axes[i]
        model_data = subset[subset["model"] == model]

        for method in ["arm", "linear", "lasso"]:
            mdata = model_data[model_data["method"] == method].sort_values("n")
            if len(mdata) == 0:
                continue
            ax.plot(np.log10(mdata["n"]), mdata["CS"],
                    color=colors[method], marker=markers[method], markersize=5,
                    label=f"{method.upper()} C.S.", linestyle="-")
            ax.plot(np.log10(mdata["n"]), mdata["IS"],
                    color=colors[method], marker=markers[method], markersize=5,
                    label=f"{method.upper()} I.S.", linestyle="--", alpha=0.6)

        ax.axhline(y=5.0, color="black", linestyle=":", alpha=0.5, label="Target C.S.=5")
        ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Model {model}", fontsize=11, fontweight="bold")
        ax.set_xlabel("log10(n)", fontsize=9)
        ax.set_ylabel("Mean count", fontsize=9)
        ax.set_xticks(np.log10(N_VALUES))
        ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=7)
        ax.set_ylim(-0.1, 5.5)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=6, loc="upper left", ncol=2)

    fig.suptitle(
        f"Exp 4: Variable Selection (d=10, Active: X1,X3,X4,X5,X9)\n"
        f"dist={distribution}, ρ={rho}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_table6(df_table6: pd.DataFrame, save_path: Path) -> None:
    """Plot covariate-wise selection proportions (Table 6)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    rho_n_combos = [(rho, n) for rho in RHO_VALUES for n in N_VALUES]
    colors = {"arm": "#1f77b4", "linear": "#2ca02c", "lasso": "#d62728"}

    for idx, (rho, n) in enumerate(rho_n_combos):
        if idx >= 8:
            break
        ax = axes[idx]
        subset = df_table6[(df_table6["rho"] == rho) & (df_table6["n"] == n)]

        x = np.arange(1, D + 1)
        width = 0.25
        for j, method in enumerate(["arm", "linear", "lasso"]):
            mdata = subset[subset["method"] == method].sort_values("covariate")
            if len(mdata) == 0:
                continue
            ax.bar(x + (j - 1) * width, mdata["proportion"].values,
                   width=width, color=colors[method], label=method.upper(), alpha=0.8)

        # Mark active covariates (1-based: 1, 3, 4, 5, 9)
        for l in [1, 3, 4, 5, 9]:
            ax.axvline(x=l, color="red", linestyle=":", alpha=0.4)

        ax.set_title(f"n={n}, ρ={rho}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Covariate", fontsize=9)
        ax.set_ylabel("Selection proportion", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"X{i}" for i in range(1, D + 1)], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Table 6: Covariate-wise Selection Proportions\n"
        "Model 1, Normal errors (Active: X1, X3, X4, X5, X9)",
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
    print("Experiment 4: Variable Selection (d=10)")
    print(f"  n_reps={N_REPS}, models={MODELS}")
    print("=" * 60)

    df = run_exp4(n_reps=N_REPS, n_jobs=-1)

    csv_path = RESULTS_DIR / "exp4_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    for dist in DISTRIBUTIONS:
        for rho in RHO_VALUES:
            plot_name = f"exp4_varsel_{dist}_rho{int(rho*10)}.png"
            plot_selection_metrics(df, dist, rho, RESULTS_DIR / plot_name)

    print("\nRunning Table 6 (covariate-wise proportions)...")
    df_table6 = run_table6(n_reps=N_REPS_TABLE6, n_jobs=-1)
    df_table6.to_csv(RESULTS_DIR / "exp4_table6.csv", index=False)
    plot_table6(df_table6, RESULTS_DIR / "exp4_table6.png")

    print("\n--- Variable Selection Summary (Normal, rho=0) ---")
    summary = df[(df["distribution"] == "normal") & (df["rho"] == 0.0)].groupby(
        ["model", "n", "method"]
    )[["CS", "IS", "CE", "IE"]].mean().round(3)
    print(summary.to_string())

    print("\nExperiment 4 complete.")
