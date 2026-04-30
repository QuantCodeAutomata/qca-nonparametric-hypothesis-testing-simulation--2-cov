"""
Main runner script for all experiments.

Runs all 5 experiments and saves results to the results/ directory.
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment(name: str, func, *args, **kwargs):
    """Run an experiment with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n{name} completed in {elapsed:.1f}s")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nERROR in {name}: {e}")
        traceback.print_exc()
        return None, elapsed


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")

    results_summary = {}

    # -----------------------------------------------------------------------
    # Experiment 1: 2-covariate simulation
    # -----------------------------------------------------------------------
    from experiments.exp1_2cov import run_exp1, plot_power_curves, N_VALUES, DISTRIBUTIONS, RHO_VALUES

    df_exp1, t1 = run_experiment("Experiment 1 (2-covariate)", run_exp1, n_reps=50, n_jobs=-1)

    if df_exp1 is not None:
        df_exp1.to_csv(RESULTS_DIR / "exp1_results.csv", index=False)
        for n in N_VALUES:
            for dist in DISTRIBUTIONS:
                for rho in RHO_VALUES:
                    plot_power_curves(df_exp1, n, dist, rho,
                                      RESULTS_DIR / f"exp1_power_n{n}_{dist}_rho{int(rho*10)}.png")

        size_df = df_exp1[df_exp1["c"] == 0.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_bspline", "rate_linear"]
        ].mean().round(3)
        results_summary["exp1_size"] = size_df

        power_df = df_exp1[df_exp1["c"] == 1.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_bspline", "rate_linear"]
        ].mean().round(3)
        results_summary["exp1_power"] = power_df

    # -----------------------------------------------------------------------
    # Experiment 2: 5-covariate simulation
    # -----------------------------------------------------------------------
    from experiments.exp2_5cov import run_exp2, plot_power_curves as plot_exp2

    df_exp2, t2 = run_experiment("Experiment 2 (5-covariate)", run_exp2, n_reps=50, n_jobs=-1)

    if df_exp2 is not None:
        df_exp2.to_csv(RESULTS_DIR / "exp2_results.csv", index=False)
        for n in N_VALUES:
            for dist in DISTRIBUTIONS:
                for rho in RHO_VALUES:
                    plot_exp2(df_exp2, n, dist, rho,
                              RESULTS_DIR / f"exp2_power_n{n}_{dist}_rho{int(rho*10)}.png")

        size_df2 = df_exp2[df_exp2["c"] == 0.0].groupby(["model", "n"])[
            ["rate_kernel", "rate_linear"]
        ].mean().round(3)
        results_summary["exp2_size"] = size_df2

    # -----------------------------------------------------------------------
    # Experiment 3: Variable selection d=5
    # -----------------------------------------------------------------------
    from experiments.exp3_varsel_5cov import (
        run_exp3, run_table5, plot_selection_metrics as plot_exp3,
        plot_table5, DISTRIBUTIONS as DIST3, RHO_VALUES as RHO3, N_VALUES as N3,
    )

    df_exp3, t3 = run_experiment("Experiment 3 (varsel d=5)", run_exp3, n_reps=20, n_jobs=-1)

    if df_exp3 is not None:
        df_exp3.to_csv(RESULTS_DIR / "exp3_results.csv", index=False)
        for dist in DIST3:
            for rho in RHO3:
                plot_exp3(df_exp3, dist, rho,
                          RESULTS_DIR / f"exp3_varsel_{dist}_rho{int(rho*10)}.png")

        df_table5, _ = run_experiment("Experiment 3 Table 5", run_table5, n_reps=50, n_jobs=-1)
        if df_table5 is not None:
            df_table5.to_csv(RESULTS_DIR / "exp3_table5.csv", index=False)
            plot_table5(df_table5, RESULTS_DIR / "exp3_table5.png")

        summary3 = df_exp3[(df_exp3["distribution"] == "normal") & (df_exp3["rho"] == 0.0)].groupby(
            ["model", "n", "method"]
        )[["CS", "IS", "CE", "IE"]].mean().round(3)
        results_summary["exp3_summary"] = summary3

    # -----------------------------------------------------------------------
    # Experiment 4: Variable selection d=10
    # -----------------------------------------------------------------------
    from experiments.exp4_varsel_10cov import (
        run_exp4, run_table6, plot_selection_metrics as plot_exp4,
        plot_table6, DISTRIBUTIONS as DIST4, RHO_VALUES as RHO4,
    )

    df_exp4, t4 = run_experiment("Experiment 4 (varsel d=10)", run_exp4, n_reps=20, n_jobs=-1)

    if df_exp4 is not None:
        df_exp4.to_csv(RESULTS_DIR / "exp4_results.csv", index=False)
        for dist in DIST4:
            for rho in RHO4:
                plot_exp4(df_exp4, dist, rho,
                          RESULTS_DIR / f"exp4_varsel_{dist}_rho{int(rho*10)}.png")

        df_table6, _ = run_experiment("Experiment 4 Table 6", run_table6, n_reps=50, n_jobs=-1)
        if df_table6 is not None:
            df_table6.to_csv(RESULTS_DIR / "exp4_table6.csv", index=False)
            plot_table6(df_table6, RESULTS_DIR / "exp4_table6.png")

        summary4 = df_exp4[(df_exp4["distribution"] == "normal") & (df_exp4["rho"] == 0.0)].groupby(
            ["model", "n", "method"]
        )[["CS", "IS", "CE", "IE"]].mean().round(3)
        results_summary["exp4_summary"] = summary4

    # -----------------------------------------------------------------------
    # Experiment 5: Real data
    # -----------------------------------------------------------------------
    from experiments.exp5_realdata import run_exp5, plot_results

    exp5_result, t5 = run_experiment("Experiment 5 (real data)", run_exp5)

    if exp5_result is not None:
        df_exp5, n_obs, start_date, end_date = exp5_result
        df_exp5.to_csv(RESULTS_DIR / "exp5_results.csv", index=False)
        plot_results(df_exp5, n_obs, start_date, end_date, RESULTS_DIR / "exp5_results.png")
        results_summary["exp5"] = df_exp5

    # -----------------------------------------------------------------------
    # Write RESULTS.md
    # -----------------------------------------------------------------------
    write_results_md(results_summary, t1, t2, t3, t4, t5)

    print("\n" + "="*60)
    print("All experiments complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)


def write_results_md(results_summary: dict, t1: float, t2: float, t3: float, t4: float, t5: float):
    """Write RESULTS.md with all experiment summaries."""
    import pandas as pd

    lines = [
        "# Experiment Results",
        "",
        "## Summary",
        "",
        "This file contains the results of all 5 experiments for the",
        "Nonparametric Hypothesis Testing in Semiparametric ARCH-m(X) Models project.",
        "",
        "---",
        "",
    ]

    # Experiment 1
    lines += [
        "## Experiment 1: 2-Covariate ARCH-m(X) Simulation",
        "",
        f"**Runtime:** {t1:.1f}s",
        "",
        "**Description:** Empirical size and power of the ANOVA-based test statistic T_n",
        "for testing X_2 significance in 2-covariate ARCH-m(X) models.",
        "",
        "**Methods compared:** Kernel Smoothing, B-Spline QMLE, Linear Benchmark",
        "",
        "**Key findings:**",
        "- At c=0 (null hypothesis): all methods should have rejection rate ≈ 0.05",
        "- For nonlinear models (1-6, 8): Kernel and B-Spline show higher power than Linear",
        "- For linear model (7): Linear benchmark shows highest power",
        "",
    ]

    if "exp1_size" in results_summary:
        lines += [
            "**Empirical Size (c=0, averaged over distributions and rho):**",
            "",
            "```",
            results_summary["exp1_size"].to_string(),
            "```",
            "",
        ]

    if "exp1_power" in results_summary:
        lines += [
            "**Empirical Power (c=1.0, averaged over distributions and rho):**",
            "",
            "```",
            results_summary["exp1_power"].to_string(),
            "```",
            "",
        ]

    lines += ["---", ""]

    # Experiment 2
    lines += [
        "## Experiment 2: 5-Covariate ARCH-m(X) Simulation",
        "",
        f"**Runtime:** {t2:.1f}s",
        "",
        "**Description:** Empirical size and power for testing X_3 significance",
        "in 5-covariate ARCH-m(X) models. B-Spline not used (curse of dimensionality).",
        "",
        "**Methods compared:** Kernel Smoothing, Linear Benchmark",
        "",
    ]

    if "exp2_size" in results_summary:
        lines += [
            "**Empirical Size (c=0):**",
            "",
            "```",
            results_summary["exp2_size"].to_string(),
            "```",
            "",
        ]

    lines += ["---", ""]

    # Experiment 3
    lines += [
        "## Experiment 3: Variable Selection (d=5)",
        "",
        f"**Runtime:** {t3:.1f}s",
        "",
        "**Description:** ARM variable selection with BY-FDR correction.",
        "True active covariates: X_1, X_3, X_4 (d_0=3 out of d=5).",
        "",
        "**Methods compared:** ARM (kernel + BY-FDR), Linear QMLE, LASSO",
        "",
        "**Target metrics:** C.S.→3, I.S.→0, C.E.→2, I.E.→0",
        "",
    ]

    if "exp3_summary" in results_summary:
        lines += [
            "**Selection Metrics (Normal errors, rho=0):**",
            "",
            "```",
            results_summary["exp3_summary"].to_string(),
            "```",
            "",
        ]

    lines += ["---", ""]

    # Experiment 4
    lines += [
        "## Experiment 4: Variable Selection (d=10)",
        "",
        f"**Runtime:** {t4:.1f}s",
        "",
        "**Description:** ARM variable selection with BY-FDR correction.",
        "True active covariates: X_1, X_3, X_4, X_5, X_9 (d_0=5 out of d=10).",
        "",
        "**Methods compared:** ARM (kernel + BY-FDR), Linear QMLE, LASSO",
        "",
        "**Target metrics:** C.S.→5, I.S.→0, C.E.→5, I.E.→0",
        "",
    ]

    if "exp4_summary" in results_summary:
        lines += [
            "**Selection Metrics (Normal errors, rho=0):**",
            "",
            "```",
            results_summary["exp4_summary"].to_string(),
            "```",
            "",
        ]

    lines += ["---", ""]

    # Experiment 5
    lines += [
        "## Experiment 5: Real Data - S&P 500 Volatility",
        "",
        f"**Runtime:** {t5:.1f}s",
        "",
        "**Description:** ARM variable selection applied to S&P 500 daily return volatility.",
        "11 candidate covariates: China, Asia, Oil, USD, Gold, Copper, Silver, Steel, Rice, Wheat, Europe.",
        "",
        "**Model:** ARCH(1)-m(X) with kernel smoothing and BY-FDR at q=0.05",
        "",
    ]

    if "exp5" in results_summary:
        df5 = results_summary["exp5"]
        lines += [
            "**Results Table (sorted by p-value):**",
            "",
            "```",
            df5[["covariate", "Z_n", "p_value", "BY_threshold", "selected"]].to_string(index=False),
            "```",
            "",
            f"**Selected covariates:** {df5[df5['selected']]['covariate'].tolist()}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Test statistic:** T_n = MST_n - MSE_n (ANOVA-based nonparametric test)",
        "- **Standardized statistic:** Z_n = sqrt(n/k_n) * T_n / sqrt(4*tau_hat^2/3)",
        "- **Kernel:** Product Gaussian kernel with Silverman bandwidth",
        "- **BY-FDR:** Benjamini-Yekutieli (2001) procedure with harmonic sum correction",
        "- **Window size:** k_n=3 (n≤1000), k_n=4 (1000<n≤5000), k_n=5 (n>5000)",
        "",
        "## References",
        "",
        "- Francq, C. & Thieu, L.Q.K. (2019). QML inference for volatility models with covariates.",
        "  *Econometric Theory*, 35(1), 37-72.",
        "- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple",
        "  testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.",
    ]

    with open(RESULTS_DIR / "RESULTS.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults summary written to {RESULTS_DIR / 'RESULTS.md'}")
