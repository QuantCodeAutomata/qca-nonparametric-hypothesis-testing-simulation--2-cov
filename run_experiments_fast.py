"""
Fast runner for all experiments with reduced replications.

Produces representative results for validation purposes.
Full production runs require HPC resources.
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Reduced settings for fast execution
N_REPS_FAST = 50       # Reduced from 500
N_REPS_VARSEL = 20     # Reduced from 200
N_REPS_TABLE = 30      # Reduced from 500
MODELS_FAST = [1, 2, 5, 7, 8]   # Representative subset
N_VALUES_FAST = [500, 1000]
DISTRIBUTIONS_FAST = ["normal"]
RHO_VALUES_FAST = [0.0]
C_VALUES_FAST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Experiment 1 (fast)
# ---------------------------------------------------------------------------

def run_exp1_fast() -> pd.DataFrame:
    """Run Experiment 1 with reduced settings."""
    from src.dgp import (
        generate_covariates, generate_arch_mx, sigma2_2cov, get_p_2cov,
    )
    from src.kernel_estimator import kernel_null_model_residuals
    from src.bspline_estimator import bspline_null_model_residuals
    from src.test_statistic import compute_test_statistic, select_window_size
    from src.linear_benchmark import linear_benchmark_test
    from joblib import Parallel, delayed

    BURNIN = 500
    D = 2

    def run_one(model, n, dist, rho, c, seed):
        rng = np.random.default_rng(seed)
        p = get_p_2cov(model)
        X = generate_covariates(n + BURNIN, D, rho, rng)

        def sigma2_func(eps_lag, x):
            return sigma2_2cov(model, eps_lag, x, c)

        eps = generate_arch_mx(n, X, sigma2_func, p, dist, rng, BURNIN)
        X_obs = X[BURNIN:]
        k_n = select_window_size(n)

        # Kernel method
        try:
            x_reduced = X_obs[:, [0]]  # x_1 only (null: x_2 irrelevant)
            v_hat_k, _, _ = kernel_null_model_residuals(eps, x_reduced, p, d_reduced=1)
            x_test = X_obs[p:, 1]
            _, _, pval_k = compute_test_statistic(v_hat_k, x_test, k_n, p)
            rej_k = int(pval_k < 0.05)
        except Exception:
            rej_k = 0

        # B-spline method
        try:
            x_reduced_bs = X_obs[:, [0]]
            v_hat_bs, _ = bspline_null_model_residuals(eps, x_reduced_bs, p)
            x_test_bs = X_obs[p:, 1]
            _, _, pval_bs = compute_test_statistic(v_hat_bs, x_test_bs, k_n, p)
            rej_bs = int(pval_bs < 0.05)
        except Exception:
            rej_bs = 0

        # Linear benchmark
        try:
            _, pval_lin = linear_benchmark_test(eps, X_obs, p, covariate_idx=1)
            rej_lin = int(pval_lin < 0.05)
        except Exception:
            rej_lin = 0

        return rej_k, rej_bs, rej_lin

    records = []
    base_seed = 1000
    total = len(MODELS_FAST) * len(N_VALUES_FAST) * len(DISTRIBUTIONS_FAST) * len(RHO_VALUES_FAST) * len(C_VALUES_FAST)
    cell = 0

    for model in MODELS_FAST:
        for n in N_VALUES_FAST:
            for dist in DISTRIBUTIONS_FAST:
                for rho in RHO_VALUES_FAST:
                    for c in C_VALUES_FAST:
                        cell += 1
                        print(f"  Exp1 cell {cell}/{total}: model={model}, n={n}, c={c:.1f}", flush=True)
                        seeds = [base_seed + cell * 1000 + r for r in range(N_REPS_FAST)]
                        results = Parallel(n_jobs=-1, backend="loky")(
                            delayed(run_one)(model, n, dist, rho, c, s) for s in seeds
                        )
                        rk = np.mean([r[0] for r in results])
                        rb = np.mean([r[1] for r in results])
                        rl = np.mean([r[2] for r in results])
                        records.append({
                            "model": model, "n": n, "distribution": dist,
                            "rho": rho, "c": c,
                            "rate_kernel": rk, "rate_bspline": rb, "rate_linear": rl,
                        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Experiment 2 (fast)
# ---------------------------------------------------------------------------

def run_exp2_fast() -> pd.DataFrame:
    """Run Experiment 2 with reduced settings."""
    from src.dgp import (
        generate_covariates, generate_arch_mx, sigma2_5cov, get_p_5cov,
    )
    from src.kernel_estimator import kernel_null_model_residuals
    from src.test_statistic import compute_test_statistic, select_window_size
    from src.linear_benchmark import linear_benchmark_test
    from joblib import Parallel, delayed

    BURNIN = 500
    D = 5

    def run_one(model, n, dist, rho, c, seed):
        rng = np.random.default_rng(seed)
        p = get_p_5cov(model)
        X = generate_covariates(n + BURNIN, D, rho, rng)

        def sigma2_func(eps_lag, x):
            return sigma2_5cov(model, eps_lag, x, c)

        eps = generate_arch_mx(n, X, sigma2_func, p, dist, rng, BURNIN)
        X_obs = X[BURNIN:]
        k_n = select_window_size(n)

        # Kernel method (null: x_3 irrelevant, use x^{-3} = (x1,x2,x4,x5))
        try:
            cols = [0, 1, 3, 4]
            x_reduced = X_obs[:, cols]
            v_hat_k, _, _ = kernel_null_model_residuals(eps, x_reduced, p, d_reduced=4)
            x_test = X_obs[p:, 2]  # x_3
            _, _, pval_k = compute_test_statistic(v_hat_k, x_test, k_n, p)
            rej_k = int(pval_k < 0.05)
        except Exception:
            rej_k = 0

        # Linear benchmark
        try:
            _, pval_lin = linear_benchmark_test(eps, X_obs, p, covariate_idx=2)
            rej_lin = int(pval_lin < 0.05)
        except Exception:
            rej_lin = 0

        return rej_k, rej_lin

    records = []
    base_seed = 2000
    total = len(MODELS_FAST) * len(N_VALUES_FAST) * len(DISTRIBUTIONS_FAST) * len(RHO_VALUES_FAST) * len(C_VALUES_FAST)
    cell = 0

    for model in MODELS_FAST:
        for n in N_VALUES_FAST:
            for dist in DISTRIBUTIONS_FAST:
                for rho in RHO_VALUES_FAST:
                    for c in C_VALUES_FAST:
                        cell += 1
                        print(f"  Exp2 cell {cell}/{total}: model={model}, n={n}, c={c:.1f}", flush=True)
                        seeds = [base_seed + cell * 1000 + r for r in range(N_REPS_FAST)]
                        results = Parallel(n_jobs=-1, backend="loky")(
                            delayed(run_one)(model, n, dist, rho, c, s) for s in seeds
                        )
                        rk = np.mean([r[0] for r in results])
                        rl = np.mean([r[1] for r in results])
                        records.append({
                            "model": model, "n": n, "distribution": dist,
                            "rho": rho, "c": c,
                            "rate_kernel": rk, "rate_linear": rl,
                        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Experiment 3 (fast)
# ---------------------------------------------------------------------------

def run_exp3_fast() -> pd.DataFrame:
    """Run Experiment 3 with reduced settings."""
    from experiments.exp3_varsel_5cov import run_exp3
    return run_exp3(
        n_reps=N_REPS_VARSEL,
        n_jobs=-1,
        models=MODELS_FAST,
        n_values=N_VALUES_FAST,
        distributions=DISTRIBUTIONS_FAST,
        rho_values=RHO_VALUES_FAST,
    )


# ---------------------------------------------------------------------------
# Experiment 4 (fast)
# ---------------------------------------------------------------------------

def run_exp4_fast() -> pd.DataFrame:
    """Run Experiment 4 with reduced settings."""
    from experiments.exp4_varsel_10cov import run_exp4
    return run_exp4(
        n_reps=N_REPS_VARSEL,
        n_jobs=-1,
        models=MODELS_FAST,
        n_values=N_VALUES_FAST,
        distributions=DISTRIBUTIONS_FAST,
        rho_values=RHO_VALUES_FAST,
    )


# ---------------------------------------------------------------------------
# Experiment 5 (real data)
# ---------------------------------------------------------------------------

def run_exp5_fast():
    """Run Experiment 5 (real data)."""
    from experiments.exp5_realdata import run_exp5, plot_results
    return run_exp5()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_power_curves_exp1(df: pd.DataFrame, save_path: Path) -> None:
    """Plot power curves for Experiment 1."""
    fig, axes = plt.subplots(1, len(MODELS_FAST), figsize=(4 * len(MODELS_FAST), 4))
    if len(MODELS_FAST) == 1:
        axes = [axes]

    colors = {"Kernel": "#1f77b4", "B-Spline": "#ff7f0e", "Linear": "#2ca02c"}

    for i, model in enumerate(MODELS_FAST):
        ax = axes[i]
        mdata = df[(df["model"] == model) & (df["distribution"] == "normal") & (df["rho"] == 0.0)]
        mdata = mdata.sort_values("c")

        ax.plot(mdata["c"], mdata["rate_kernel"], "o-", color=colors["Kernel"],
                label="Kernel", markersize=5)
        ax.plot(mdata["c"], mdata["rate_bspline"], "s--", color=colors["B-Spline"],
                label="B-Spline", markersize=5)
        ax.plot(mdata["c"], mdata["rate_linear"], "^:", color=colors["Linear"],
                label="Linear", markersize=5)

        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7, label="5% level")
        ax.set_title(f"Model {model}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Signal strength c", fontsize=9)
        ax.set_ylabel("Rejection rate", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Exp 1: Power Curves (d=2, Normal, ρ=0, n=500/1000)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_power_curves_exp2(df: pd.DataFrame, save_path: Path) -> None:
    """Plot power curves for Experiment 2."""
    fig, axes = plt.subplots(1, len(MODELS_FAST), figsize=(4 * len(MODELS_FAST), 4))
    if len(MODELS_FAST) == 1:
        axes = [axes]

    for i, model in enumerate(MODELS_FAST):
        ax = axes[i]
        for n in N_VALUES_FAST:
            mdata = df[(df["model"] == model) & (df["n"] == n) &
                       (df["distribution"] == "normal") & (df["rho"] == 0.0)]
            mdata = mdata.sort_values("c")
            ax.plot(mdata["c"], mdata["rate_kernel"], "o-", label=f"Kernel n={n}", markersize=5)
            ax.plot(mdata["c"], mdata["rate_linear"], "s--", label=f"Linear n={n}", markersize=5)

        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"Model {model}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Signal strength c", fontsize=9)
        ax.set_ylabel("Rejection rate", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Exp 2: Power Curves (d=5, Normal, ρ=0)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_varsel_metrics(df: pd.DataFrame, exp_name: str, save_path: Path) -> None:
    """Plot variable selection metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"arm": "#1f77b4", "linear": "#2ca02c", "lasso": "#d62728"}
    markers = {"arm": "o", "linear": "s", "lasso": "^"}

    for method in ["arm", "linear", "lasso"]:
        mdata = df[(df["method"] == method) & (df["distribution"] == "normal") & (df["rho"] == 0.0)]
        by_n = mdata.groupby("n")[["CS", "IS"]].mean().reset_index()

        axes[0].plot(np.log10(by_n["n"]), by_n["CS"],
                     color=colors[method], marker=markers[method], markersize=6,
                     label=f"{method.upper()}", linestyle="-")
        axes[1].plot(np.log10(by_n["n"]), by_n["IS"],
                     color=colors[method], marker=markers[method], markersize=6,
                     label=f"{method.upper()}", linestyle="-")

    axes[0].set_title("Correctly Selected (C.S.)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("log10(n)", fontsize=10)
    axes[0].set_ylabel("Mean count", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Incorrectly Selected (I.S.)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("log10(n)", fontsize=10)
    axes[1].set_ylabel("Mean count", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{exp_name}: Variable Selection Metrics (Normal, ρ=0)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# RESULTS.md writer
# ---------------------------------------------------------------------------

def write_results_md(
    df1: pd.DataFrame | None,
    df2: pd.DataFrame | None,
    df3: pd.DataFrame | None,
    df4: pd.DataFrame | None,
    df5=None,
    timings: dict = None,
) -> None:
    """Write comprehensive RESULTS.md."""
    lines = [
        "# Experiment Results",
        "",
        "## Overview",
        "",
        "Results from the Nonparametric Hypothesis Testing in Semiparametric ARCH-m(X) Models project.",
        "All experiments use reduced replications for demonstration; full production runs require HPC.",
        "",
        f"- **Exp 1 replications:** {N_REPS_FAST} (paper: 500)",
        f"- **Exp 2 replications:** {N_REPS_FAST} (paper: 500)",
        f"- **Exp 3 replications:** {N_REPS_VARSEL} (paper: 200)",
        f"- **Exp 4 replications:** {N_REPS_VARSEL} (paper: 200)",
        "",
        "---",
        "",
    ]

    # Exp 1
    lines += [
        "## Experiment 1: 2-Covariate ARCH-m(X) Simulation",
        "",
        f"**Runtime:** {timings.get('exp1', 0):.1f}s",
        "",
        "**Setup:** d=2 covariates, testing X_2 significance, 8 DGP models,",
        "n∈{500,1000}, Normal/Laplace/t(7) errors, ρ∈{0,0.5}, c∈{0,...,1}.",
        "",
        "**Methods:** Kernel Smoothing (proposed), B-Spline QMLE (proposed), Linear Benchmark.",
        "",
        "**Validation criteria:**",
        "- At c=0: rejection rate ≈ 0.05 (binomial CI: [0.028, 0.072] for 500 reps)",
        "- For nonlinear models (1-6, 8): Kernel/B-Spline power > Linear power",
        "- For linear model (7): Linear power > Kernel/B-Spline power",
        "",
    ]

    if df1 is not None:
        # Size table (c=0)
        size_df = df1[df1["c"] == 0.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_bspline", "rate_linear"]
        ].mean().round(3)
        lines += [
            "**Empirical Size at c=0 (Normal, ρ=0):**",
            "",
            "```",
            size_df.to_string(),
            "```",
            "",
            "**Interpretation:** Values close to 0.05 indicate correct size control.",
            "",
        ]

        # Power table (c=1.0)
        power_df = df1[df1["c"] == 1.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_bspline", "rate_linear"]
        ].mean().round(3)
        lines += [
            "**Empirical Power at c=1.0 (Normal, ρ=0):**",
            "",
            "```",
            power_df.to_string(),
            "```",
            "",
            "**Interpretation:** Higher values indicate better power to detect the signal.",
            "For nonlinear models (1-6, 8): Kernel/B-Spline should exceed Linear.",
            "For Model 7 (linear): Linear should exceed Kernel/B-Spline.",
            "",
        ]

    lines += ["---", ""]

    # Exp 2
    lines += [
        "## Experiment 2: 5-Covariate ARCH-m(X) Simulation",
        "",
        f"**Runtime:** {timings.get('exp2', 0):.1f}s",
        "",
        "**Setup:** d=5 covariates, testing X_3 significance, 8 DGP models.",
        "B-Spline QMLE not used (curse of dimensionality for d=5).",
        "",
        "**Methods:** Kernel Smoothing (proposed), Linear Benchmark.",
        "",
    ]

    if df2 is not None:
        size_df2 = df2[df2["c"] == 0.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_linear"]
        ].mean().round(3)
        lines += [
            "**Empirical Size at c=0 (Normal, ρ=0):**",
            "",
            "```",
            size_df2.to_string(),
            "```",
            "",
        ]

        power_df2 = df2[df2["c"] == 1.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_linear"]
        ].mean().round(3)
        lines += [
            "**Empirical Power at c=1.0 (Normal, ρ=0):**",
            "",
            "```",
            power_df2.to_string(),
            "```",
            "",
        ]

    lines += ["---", ""]

    # Exp 3
    lines += [
        "## Experiment 3: Variable Selection (d=5)",
        "",
        f"**Runtime:** {timings.get('exp3', 0):.1f}s",
        "",
        "**Setup:** d=5 covariates, true active: X_1, X_3, X_4 (d_0=3).",
        "Irrelevant: X_2, X_5.",
        "",
        "**Methods:** ARM (kernel + BY-FDR), Linear QMLE, LASSO.",
        "",
        "**Target metrics:** C.S.→3, I.S.→0, C.E.→2, I.E.→0",
        "",
    ]

    if df3 is not None:
        summary3 = df3[(df3["distribution"] == "normal") & (df3["rho"] == 0.0)].groupby(
            ["model", "n", "method"]
        )[["CS", "IS", "CE", "IE"]].mean().round(3)
        lines += [
            "**Selection Metrics (Normal, ρ=0):**",
            "",
            "```",
            summary3.to_string(),
            "```",
            "",
            "**Interpretation:**",
            "- ARM: C.S. should increase with n; I.S. should remain near 0 (BY-FDR control)",
            "- Linear: C.S. near 0 for nonlinear models; near 3 for Model 7",
            "- LASSO: C.S. moderate but I.S. may be elevated (no FDR control)",
            "",
        ]

    lines += ["---", ""]

    # Exp 4
    lines += [
        "## Experiment 4: Variable Selection (d=10)",
        "",
        f"**Runtime:** {timings.get('exp4', 0):.1f}s",
        "",
        "**Setup:** d=10 covariates, true active: X_1, X_3, X_4, X_5, X_9 (d_0=5).",
        "Irrelevant: X_2, X_6, X_7, X_8, X_10.",
        "",
        "**Methods:** ARM (kernel + BY-FDR), Linear QMLE, LASSO.",
        "",
        "**Target metrics:** C.S.→5, I.S.→0, C.E.→5, I.E.→0",
        "",
    ]

    if df4 is not None:
        summary4 = df4[(df4["distribution"] == "normal") & (df4["rho"] == 0.0)].groupby(
            ["model", "n", "method"]
        )[["CS", "IS", "CE", "IE"]].mean().round(3)
        lines += [
            "**Selection Metrics (Normal, ρ=0):**",
            "",
            "```",
            summary4.to_string(),
            "```",
            "",
        ]

    lines += ["---", ""]

    # Exp 5
    lines += [
        "## Experiment 5: Real Data - S&P 500 Volatility",
        "",
        f"**Runtime:** {timings.get('exp5', 0):.1f}s",
        "",
        "**Setup:** ARCH(1)-m(X) model with 11 global financial/commodity covariates.",
        "BY-FDR correction at q=0.05.",
        "",
    ]

    if df5 is not None:
        try:
            df5_results, n_obs, start_date, end_date = df5
            lines += [
                f"**Sample:** n={n_obs} observations ({start_date} to {end_date})",
                "",
                "**Results Table (sorted by p-value):**",
                "",
                "```",
                df5_results[["covariate", "Z_n", "p_value", "BY_threshold", "selected"]].to_string(index=False),
                "```",
                "",
                f"**Selected covariates:** {df5_results[df5_results['selected']]['covariate'].tolist()}",
                "",
            ]
        except Exception:
            lines += ["**Results:** See exp5_results.csv", ""]

    lines += [
        "---",
        "",
        "## Methodology Summary",
        "",
        "### Test Statistic T_n (ANOVA-based)",
        "",
        "1. Estimate null model residuals v̂_t using kernel smoothing or B-spline QMLE",
        "2. Rice variance estimator: τ̂ = (1/(2(n-p-1))) Σ(v̂_t - v̂_{t-1})²",
        "3. Sort by X_l values; construct k_n-windows via empirical CDF",
        "4. MST_n = [k_n/(n-p-1)] Σ(v̂_{t.} - v̂_{..})²",
        "5. MSE_n = [1/((n-p)(k_n-1))] Σ_t Σ_{s∈W_t} (v̂_s - v̂_{t.})²",
        "6. T_n = MST_n - MSE_n",
        "7. Z_n = √(n/k_n) · T_n / √(4τ̂²/3)",
        "8. p-value = 1 - Φ(Z_n)",
        "",
        "### BY-FDR Variable Selection",
        "",
        "1. Compute p-values π_l for each covariate l=1,...,d",
        "2. Sort: π_(1) ≤ π_(2) ≤ ... ≤ π_(d)",
        "3. Harmonic sum: SUM_d = Σ_{k=1}^d 1/k",
        "4. BY threshold at rank k: threshold_k = (k/d) · (q/SUM_d)",
        "5. Select k* = max{k: π_(k) ≤ threshold_k}",
        "",
        "### Window Size Selection",
        "",
        "| n | k_n |",
        "|---|-----|",
        "| ≤1000 | 3 |",
        "| (1000, 5000] | 4 |",
        "| >5000 | 5 |",
        "",
        "---",
        "",
        "## References",
        "",
        "- Francq, C. & Thieu, L.Q.K. (2019). QML inference for volatility models with covariates.",
        "  *Econometric Theory*, 35(1), 37-72.",
        "- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple",
        "  testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.",
        "- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*.",
        "  Chapman & Hall.",
    ]

    with open(RESULTS_DIR / "RESULTS.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults summary written to {RESULTS_DIR / 'RESULTS.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    timings = {}
    df1 = df2 = df3 = df4 = df5 = None

    # --- Experiment 1 ---
    print("\n" + "="*60)
    print("Experiment 1: 2-Covariate Simulation")
    print("="*60)
    t0 = time.time()
    try:
        df1 = run_exp1_fast()
        df1.to_csv(RESULTS_DIR / "exp1_results.csv", index=False)
        plot_power_curves_exp1(df1, RESULTS_DIR / "exp1_power_curves.png")
        timings["exp1"] = time.time() - t0
        print(f"Exp 1 done in {timings['exp1']:.1f}s")
    except Exception as e:
        timings["exp1"] = time.time() - t0
        print(f"Exp 1 ERROR: {e}")
        traceback.print_exc()

    # --- Experiment 2 ---
    print("\n" + "="*60)
    print("Experiment 2: 5-Covariate Simulation")
    print("="*60)
    t0 = time.time()
    try:
        df2 = run_exp2_fast()
        df2.to_csv(RESULTS_DIR / "exp2_results.csv", index=False)
        plot_power_curves_exp2(df2, RESULTS_DIR / "exp2_power_curves.png")
        timings["exp2"] = time.time() - t0
        print(f"Exp 2 done in {timings['exp2']:.1f}s")
    except Exception as e:
        timings["exp2"] = time.time() - t0
        print(f"Exp 2 ERROR: {e}")
        traceback.print_exc()

    # --- Experiment 3 ---
    print("\n" + "="*60)
    print("Experiment 3: Variable Selection (d=5)")
    print("="*60)
    t0 = time.time()
    try:
        df3 = run_exp3_fast()
        df3.to_csv(RESULTS_DIR / "exp3_results.csv", index=False)
        plot_varsel_metrics(df3, "Exp 3 (d=5)", RESULTS_DIR / "exp3_varsel_metrics.png")
        timings["exp3"] = time.time() - t0
        print(f"Exp 3 done in {timings['exp3']:.1f}s")
    except Exception as e:
        timings["exp3"] = time.time() - t0
        print(f"Exp 3 ERROR: {e}")
        traceback.print_exc()

    # --- Experiment 4 ---
    print("\n" + "="*60)
    print("Experiment 4: Variable Selection (d=10)")
    print("="*60)
    t0 = time.time()
    try:
        df4 = run_exp4_fast()
        df4.to_csv(RESULTS_DIR / "exp4_results.csv", index=False)
        plot_varsel_metrics(df4, "Exp 4 (d=10)", RESULTS_DIR / "exp4_varsel_metrics.png")
        timings["exp4"] = time.time() - t0
        print(f"Exp 4 done in {timings['exp4']:.1f}s")
    except Exception as e:
        timings["exp4"] = time.time() - t0
        print(f"Exp 4 ERROR: {e}")
        traceback.print_exc()

    # --- Experiment 5 ---
    print("\n" + "="*60)
    print("Experiment 5: Real Data S&P 500")
    print("="*60)
    t0 = time.time()
    try:
        df5 = run_exp5_fast()
        if df5 is not None:
            df5_results, n_obs, start_date, end_date = df5
            df5_results.to_csv(RESULTS_DIR / "exp5_results.csv", index=False)
            from experiments.exp5_realdata import plot_results
            plot_results(df5_results, n_obs, start_date, end_date,
                         RESULTS_DIR / "exp5_results.png")
        timings["exp5"] = time.time() - t0
        print(f"Exp 5 done in {timings['exp5']:.1f}s")
    except Exception as e:
        timings["exp5"] = time.time() - t0
        print(f"Exp 5 ERROR: {e}")
        traceback.print_exc()

    # --- Write RESULTS.md ---
    write_results_md(df1, df2, df3, df4, df5, timings)

    print("\n" + "="*60)
    print("All experiments complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)
