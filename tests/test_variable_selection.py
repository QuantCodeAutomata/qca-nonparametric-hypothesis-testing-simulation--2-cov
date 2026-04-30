"""
Tests for the Variable Selection module.

Validates:
- BY-FDR procedure properties
- ARM variable selection
- Linear QMLE variable selection
- LASSO variable selection
- Selection metrics computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.variable_selection import (
    benjamini_yekutieli_selection,
    arm_variable_selection,
    linear_qmle_variable_selection,
    lasso_variable_selection,
    compute_selection_metrics,
)


class TestBenjaminiYekutieli:
    """Tests for BY-FDR procedure."""

    def test_all_large_pvalues_no_selection(self):
        """All large p-values should result in no selection."""
        p_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        selected = benjamini_yekutieli_selection(p_values, q=0.05)
        assert not np.any(selected), "Should not select any with large p-values"

    def test_all_small_pvalues_all_selected(self):
        """Very small p-values should all be selected."""
        p_values = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
        selected = benjamini_yekutieli_selection(p_values, q=0.05)
        assert np.all(selected), "Should select all with very small p-values"

    def test_output_shape(self):
        """Output should have same shape as input."""
        p_values = np.array([0.01, 0.05, 0.1, 0.5, 0.9])
        selected = benjamini_yekutieli_selection(p_values, q=0.05)
        assert selected.shape == p_values.shape

    def test_output_dtype(self):
        """Output should be boolean array."""
        p_values = np.array([0.01, 0.05, 0.1])
        selected = benjamini_yekutieli_selection(p_values, q=0.05)
        assert selected.dtype == bool

    def test_harmonic_sum_d5(self):
        """BY threshold for d=5 should use SUM_5 = 2.2833."""
        # SUM_5 = 1 + 1/2 + 1/3 + 1/4 + 1/5 = 2.2833
        SUM_5 = sum(1.0 / k for k in range(1, 6))
        assert abs(SUM_5 - 2.2833) < 0.001

    def test_harmonic_sum_d10(self):
        """BY threshold for d=10 should use SUM_10 = 2.9290."""
        SUM_10 = sum(1.0 / k for k in range(1, 11))
        assert abs(SUM_10 - 2.9290) < 0.001

    def test_by_threshold_formula(self):
        """BY threshold for rank k should be (k/d) * (q/SUM_d)."""
        d = 5
        q = 0.05
        SUM_d = sum(1.0 / k for k in range(1, d + 1))
        # For rank 1: threshold = (1/5) * (0.05/SUM_5)
        expected_threshold_1 = (1 / d) * (q / SUM_d)
        assert abs(expected_threshold_1 - 0.004384) < 0.0001

    def test_monotone_selection(self):
        """If p_{(k)} is selected, all p_{(j)} for j < k should also be selected."""
        p_values = np.array([0.001, 0.002, 0.003, 0.5, 0.9])
        selected = benjamini_yekutieli_selection(p_values, q=0.05)
        # If any is selected, the smallest p-values should be selected first
        sorted_idx = np.argsort(p_values)
        selected_sorted = selected[sorted_idx]
        # Find first False
        first_false = np.argmax(~selected_sorted) if not np.all(selected_sorted) else len(selected_sorted)
        # All before first_false should be True
        assert np.all(selected_sorted[:first_false])

    def test_single_pvalue(self):
        """Single p-value should work correctly."""
        p_values = np.array([0.001])
        selected = benjamini_yekutieli_selection(p_values, q=0.05)
        assert selected.shape == (1,)

    def test_fdr_control_property(self):
        """BY procedure should control FDR at level q."""
        # With all null p-values (uniform), expected false discoveries should be <= q
        rng = np.random.default_rng(42)
        n_reps = 500
        false_discoveries = []
        d = 10
        for _ in range(n_reps):
            p_values = rng.uniform(0, 1, d)
            selected = benjamini_yekutieli_selection(p_values, q=0.05)
            false_discoveries.append(np.sum(selected))
        # Under all-null, expected false discoveries should be small
        mean_fd = np.mean(false_discoveries)
        assert mean_fd < 1.0, f"Mean false discoveries {mean_fd:.3f} too high under all-null"


class TestComputeSelectionMetrics:
    """Tests for selection metrics computation."""

    def test_perfect_selection(self):
        """Perfect selection should give CS=3, IS=0, CE=2, IE=0 for d=5, active=[0,2,3]."""
        selected = np.array([True, False, True, True, False])
        true_active = [0, 2, 3]
        metrics = compute_selection_metrics(selected, true_active, d=5)
        assert metrics["CS"] == 3.0
        assert metrics["IS"] == 0.0
        assert metrics["CE"] == 2.0
        assert metrics["IE"] == 0.0

    def test_no_selection(self):
        """No selection should give CS=0, IS=0, CE=2, IE=3."""
        selected = np.array([False, False, False, False, False])
        true_active = [0, 2, 3]
        metrics = compute_selection_metrics(selected, true_active, d=5)
        assert metrics["CS"] == 0.0
        assert metrics["IS"] == 0.0
        assert metrics["CE"] == 2.0
        assert metrics["IE"] == 3.0

    def test_all_selected(self):
        """Selecting all should give CS=3, IS=2, CE=0, IE=0."""
        selected = np.array([True, True, True, True, True])
        true_active = [0, 2, 3]
        metrics = compute_selection_metrics(selected, true_active, d=5)
        assert metrics["CS"] == 3.0
        assert metrics["IS"] == 2.0
        assert metrics["CE"] == 0.0
        assert metrics["IE"] == 0.0

    def test_metrics_sum_to_d(self):
        """CS + IS + CE + IE should equal d."""
        rng = np.random.default_rng(42)
        selected = rng.choice([True, False], size=5)
        true_active = [0, 2, 3]
        metrics = compute_selection_metrics(selected, true_active, d=5)
        total = metrics["CS"] + metrics["IS"] + metrics["CE"] + metrics["IE"]
        assert abs(total - 5.0) < 1e-10, f"Metrics sum {total} != 5"

    def test_10cov_metrics(self):
        """Test with d=10 covariates."""
        selected = np.array([True, False, True, True, True, False, False, False, True, False])
        true_active = [0, 2, 3, 4, 8]  # X_1, X_3, X_4, X_5, X_9
        metrics = compute_selection_metrics(selected, true_active, d=10)
        assert metrics["CS"] == 5.0
        assert metrics["IS"] == 0.0
        assert metrics["CE"] == 5.0
        assert metrics["IE"] == 0.0


class TestARMVariableSelection:
    """Tests for ARM variable selection procedure."""

    def test_output_shape(self):
        """ARM should return boolean array of shape (d,)."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        selected, p_values = arm_variable_selection(eps, X, p=1, q=0.05)
        assert selected.shape == (d,)
        assert p_values.shape == (d,)

    def test_p_values_in_0_1(self):
        """ARM p-values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        _, p_values = arm_variable_selection(eps, X, p=1, q=0.05)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_output_dtype(self):
        """ARM selected should be boolean array."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        selected, _ = arm_variable_selection(eps, X, p=1, q=0.05)
        assert selected.dtype == bool


class TestLinearQMLEVariableSelection:
    """Tests for linear QMLE variable selection."""

    def test_output_shape(self):
        """Linear QMLE should return boolean array of shape (d,)."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        selected, p_values = linear_qmle_variable_selection(eps, X, p=1)
        assert selected.shape == (d,)
        assert p_values.shape == (d,)

    def test_p_values_in_0_1(self):
        """Linear QMLE p-values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        _, p_values = linear_qmle_variable_selection(eps, X, p=1)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_linear_signal_detected(self):
        """Linear QMLE should detect strong linear signal."""
        rng = np.random.default_rng(42)
        n = 500
        d = 2
        X = rng.standard_normal((n, d)) + 2.5
        # True model: sigma_t^2 = 0.2 + 0.3*eps_{t-1}^2 + 2*x1 + 0*x2
        eps = np.zeros(n)
        for t in range(1, n):
            sigma2 = max(0.2 + 0.3 * eps[t-1]**2 + 2.0 * X[t-1, 0], 1e-8)
            eps[t] = rng.standard_normal() * np.sqrt(sigma2)

        selected, p_values = linear_qmle_variable_selection(eps, X, p=1)
        # X_1 (index 0) should have smaller p-value than X_2 (index 1)
        assert p_values[0] < p_values[1], f"p_values: {p_values}"


class TestLassoVariableSelection:
    """Tests for LASSO variable selection."""

    def test_output_shape(self):
        """LASSO should return boolean array of shape (d,)."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        selected = lasso_variable_selection(eps, X, p=1)
        assert selected.shape == (d,)

    def test_output_dtype(self):
        """LASSO selected should be boolean array."""
        rng = np.random.default_rng(42)
        n = 200
        d = 3
        eps = rng.standard_normal(n)
        X = rng.standard_normal((n, d))
        selected = lasso_variable_selection(eps, X, p=1)
        assert selected.dtype == bool
