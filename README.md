# Nonparametric Hypothesis Testing in Semiparametric ARCH-m(X) Models

This repository implements the nonparametric ANOVA-based test statistic T_n for testing
the significance of exogenous covariates in semiparametric ARCH-m(X) models, as described
in the paper by Francq & Thieu (2019) and extensions.

## Overview

The repository contains five experiments:

- **Exp 1**: 2-covariate ARCH-m(X) simulation (kernel + B-spline vs linear benchmark)
- **Exp 2**: 5-covariate ARCH-m(X) simulation (kernel vs linear benchmark)
- **Exp 3**: Variable selection with 5 covariates using BY-FDR correction
- **Exp 4**: Variable selection with 10 covariates using BY-FDR correction
- **Exp 5**: Real data application to S&P 500 volatility

## Repository Structure

```
src/
  dgp.py                  # Data generating processes (8 DGP models)
  kernel_estimator.py     # Kernel smoothing QMLE estimator
  bspline_estimator.py    # B-spline QMLE estimator
  test_statistic.py       # ANOVA-based test statistic T_n
  linear_benchmark.py     # Linear QMLE benchmark (Francq & Thieu 2019)
  variable_selection.py   # BY-FDR variable selection (ARM procedure)
  simulation.py           # Simulation runner utilities
experiments/
  exp1_2cov.py            # Experiment 1: 2-covariate simulation
  exp2_5cov.py            # Experiment 2: 5-covariate simulation
  exp3_varsel_5cov.py     # Experiment 3: Variable selection d=5
  exp4_varsel_10cov.py    # Experiment 4: Variable selection d=10
  exp5_realdata.py        # Experiment 5: S&P 500 real data
tests/
  test_dgp.py
  test_kernel_estimator.py
  test_bspline_estimator.py
  test_test_statistic.py
  test_variable_selection.py
results/
  RESULTS.md              # Summary of all experiment results
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run all experiments
python experiments/exp1_2cov.py
python experiments/exp2_5cov.py
python experiments/exp3_varsel_5cov.py
python experiments/exp4_varsel_10cov.py
python experiments/exp5_realdata.py

# Run tests
pytest tests/ -v
```

## Key Methodology

### ANOVA-Based Test Statistic T_n

The test statistic is based on a nonparametric ANOVA decomposition:
- **MST_n**: Mean square between windows (sorted by tested covariate)
- **MSE_n**: Mean square within windows
- **T_n = MST_n - MSE_n**
- **Z_n = sqrt(n/k_n) * T_n / sqrt(4*tau_hat^2/3)**

### BY-FDR Variable Selection (ARM Procedure)

The Benjamini-Yekutieli (2001) procedure controls FDR under arbitrary dependence:
- Harmonic sum correction: SUM_d = sum_{l=1}^d 1/l
- BY threshold for rank k: threshold_k = (k/d) * (q / SUM_d)

## References

- Francq, C. & Thieu, L.Q.K. (2019). QML inference for volatility models with covariates.
  *Econometric Theory*, 35(1), 37-72.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple
  testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.
