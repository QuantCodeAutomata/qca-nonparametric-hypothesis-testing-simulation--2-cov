"""
Tests for the ANOVA-based test statistic module.

Validates:
- Rice variance estimator properties
- Window construction correctness
- Test statistic computation
- p-value properties
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.test_statistic import (
    rice_variance_estimator,
    construct_windows,
    compute_test_statistic,
    select_window_size,
)


class TestRiceVarianceEstimator:
    """Tests for Rice variance estimator."""

    def test_positive_estimate(self):
        """Rice estimator should return positive value."""
        rng = np.random.default_rng(42)
        v_hat = rng.standard_normal(100)
        tau = rice_variance_estimator(v_hat, p=1)
        assert tau > 0, f"Rice estimator {tau} <= 0"

    def test_iid_normal_estimate(self):
        """For iid N(0,1) residuals, Rice estimator should be close to 1."""
        rng = np.random.default_rng(42)
        v_hat = rng.standard_normal(10000)
        tau = rice_variance_estimator(v_hat, p=1)
        # E[tau] = E[(v_t - v_{t-1})^2 / 2] = E[v_t^2] = 1 for iid N(0,1)
        assert abs(tau - 1.0) < 0.1, f"Rice estimator {tau:.4f} far from 1.0"

    def test_formula_correctness(self):
        """Rice estimator should match manual computation."""
        v_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # tau = (1/(2*4)) * sum((v[t] - v[t-1])^2) for t=1,...,4
        # diffs = [1, 1, 1, 1], sum of squares = 4
        # tau = 4 / 8 = 0.5
        tau = rice_variance_estimator(v_hat, p=1)
        assert abs(tau - 0.5) < 1e-10, f"Rice estimator {tau} != 0.5"

    def test_single_element(self):
        """Single element should return default value."""
        v_hat = np.array([1.0])
        tau = rice_variance_estimator(v_hat, p=1)
        assert tau > 0

    def test_two_elements(self):
        """Two elements should work correctly."""
        v_hat = np.array([1.0, 3.0])
        # tau = (1/(2*1)) * (3-1)^2 = 4/2 = 2
        tau = rice_variance_estimator(v_hat, p=1)
        assert abs(tau - 2.0) < 1e-10


class TestConstructWindows:
    """Tests for window construction."""

    def test_window_size(self):
        """Each window should have exactly k_n elements."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        k_n = 3
        windows = construct_windows(x, k_n)
        for i, w in enumerate(windows):
            assert len(w) == k_n, f"Window {i} has {len(w)} elements, expected {k_n}"

    def test_number_of_windows(self):
        """Number of windows should equal n_eff."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        windows = construct_windows(x, k_n=3)
        assert len(windows) == 50

    def test_edge_correction_first(self):
        """First window should use first k_n observations (sorted)."""
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])  # sorted: [1,2,3,4,5] at indices [1,3,2,4,0]
        k_n = 3
        windows = construct_windows(x, k_n)
        # First window (rank 0 in sorted order) should have 3 elements
        assert len(windows[0]) == k_n

    def test_edge_correction_last(self):
        """Last window should use last k_n observations (sorted)."""
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        k_n = 3
        windows = construct_windows(x, k_n)
        assert len(windows[-1]) == k_n

    def test_k_n_5(self):
        """Window size k_n=5 should work correctly."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        windows = construct_windows(x, k_n=5)
        for w in windows:
            assert len(w) == 5


class TestComputeTestStatistic:
    """Tests for ANOVA test statistic computation."""

    def test_output_types(self):
        """Test statistic should return three floats."""
        rng = np.random.default_rng(42)
        v_hat = rng.standard_normal(99)
        x_test = rng.standard_normal(99)
        T_n, Z_n, p_val = compute_test_statistic(v_hat, x_test, k_n=3, p=1)
        assert isinstance(T_n, float)
        assert isinstance(Z_n, float)
        assert isinstance(p_val, float)

    def test_p_value_in_0_1(self):
        """p-value should be in [0, 1]."""
        rng = np.random.default_rng(42)
        v_hat = rng.standard_normal(99)
        x_test = rng.standard_normal(99)
        _, _, p_val = compute_test_statistic(v_hat, x_test, k_n=3, p=1)
        assert 0.0 <= p_val <= 1.0, f"p-value {p_val} not in [0, 1]"

    def test_finite_statistics(self):
        """Test statistics should be finite."""
        rng = np.random.default_rng(42)
        v_hat = rng.standard_normal(99)
        x_test = rng.standard_normal(99)
        T_n, Z_n, p_val = compute_test_statistic(v_hat, x_test, k_n=3, p=1)
        assert np.isfinite(T_n)
        assert np.isfinite(Z_n)
        assert np.isfinite(p_val)

    def test_null_hypothesis_size(self):
        """Under null (iid residuals), rejection rate should be near 0.05."""
        rng = np.random.default_rng(42)
        n_reps = 200
        rejections = 0
        for _ in range(n_reps):
            v_hat = rng.standard_normal(499)
            x_test = rng.standard_normal(499)
            _, _, p_val = compute_test_statistic(v_hat, x_test, k_n=3, p=1)
            if p_val < 0.05:
                rejections += 1
        rate = rejections / n_reps
        # Should be close to 0.05; allow wide tolerance for small n_reps
        assert 0.01 <= rate <= 0.20, f"Null rejection rate {rate:.3f} far from 0.05"

    def test_power_under_alternative(self):
        """Under alternative (x_test correlated with v_hat), should have higher rejection rate."""
        rng = np.random.default_rng(42)
        n_reps = 100
        rejections = 0
        for _ in range(n_reps):
            x_test = rng.standard_normal(499)
            # v_hat depends on x_test (alternative hypothesis)
            v_hat = 2.0 * x_test ** 2 + rng.standard_normal(499) * 0.5
            _, _, p_val = compute_test_statistic(v_hat, x_test, k_n=3, p=1)
            if p_val < 0.05:
                rejections += 1
        rate = rejections / n_reps
        # Should have higher power than 0.05
        assert rate > 0.1, f"Power {rate:.3f} too low under alternative"

    def test_p2_model(self):
        """p=2 model should work correctly."""
        rng = np.random.default_rng(42)
        v_hat = rng.standard_normal(98)  # n-p = 100-2 = 98
        x_test = rng.standard_normal(98)
        T_n, Z_n, p_val = compute_test_statistic(v_hat, x_test, k_n=3, p=2)
        assert np.isfinite(T_n)
        assert 0.0 <= p_val <= 1.0


class TestSelectWindowSize:
    """Tests for window size selection."""

    def test_n_500(self):
        """n=500 should give k_n=3."""
        assert select_window_size(500) == 3

    def test_n_1000(self):
        """n=1000 should give k_n=3."""
        assert select_window_size(1000) == 3

    def test_n_2000(self):
        """n=2000 should give k_n=4."""
        assert select_window_size(2000) == 4

    def test_n_5000(self):
        """n=5000 should give k_n=4."""
        assert select_window_size(5000) == 4

    def test_n_10000(self):
        """n=10000 should give k_n=5."""
        assert select_window_size(10000) == 5
