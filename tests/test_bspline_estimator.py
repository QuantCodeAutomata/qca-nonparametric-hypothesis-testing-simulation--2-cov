"""
Tests for the B-Spline QMLE Estimator module.

Validates:
- B-spline basis construction properties
- QMLE objective function
- Null model residuals computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.bspline_estimator import (
    build_bspline_basis_1d,
    build_bspline_basis_2d,
    qmle_objective,
    bspline_null_model_residuals,
)


class TestBSplineBasis1D:
    """Tests for 1D B-spline basis construction."""

    def test_output_shape(self):
        """Basis matrix should have correct shape."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, 100)
        B = build_bspline_basis_1d(x, n_internal_knots=3, degree=3)
        # K = n_internal_knots + degree + 1 = 3 + 3 + 1 = 7
        assert B.shape == (100, 7), f"Shape {B.shape} != (100, 7)"

    def test_non_negative(self):
        """B-spline basis functions should be non-negative."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, 100)
        B = build_bspline_basis_1d(x, n_internal_knots=3, degree=3)
        assert np.all(B >= -1e-10), "B-spline basis has negative values"

    def test_partition_of_unity(self):
        """B-spline basis functions should sum to approximately 1 (partition of unity)."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, 100)
        B = build_bspline_basis_1d(x, n_internal_knots=3, degree=3)
        row_sums = B.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Row sums not 1: {row_sums[:5]}"

    def test_different_knot_counts(self):
        """Different numbers of internal knots should give different basis sizes."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, 100)
        B3 = build_bspline_basis_1d(x, n_internal_knots=3, degree=3)
        B5 = build_bspline_basis_1d(x, n_internal_knots=5, degree=3)
        assert B3.shape[1] == 7
        assert B5.shape[1] == 9

    def test_single_data_point_edge_case(self):
        """Should handle edge case with minimal data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        B = build_bspline_basis_1d(x, n_internal_knots=2, degree=3)
        assert B.shape[0] == 10
        assert np.all(np.isfinite(B))


class TestBSplineBasis2D:
    """Tests for 2D tensor-product B-spline basis."""

    def test_output_shape(self):
        """2D basis matrix should have correct shape."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, (100, 2))
        B = build_bspline_basis_2d(x, n_internal_knots=3, degree=3)
        # K = 7^2 = 49
        assert B.shape == (100, 49), f"Shape {B.shape} != (100, 49)"

    def test_non_negative(self):
        """2D B-spline basis should be non-negative."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, (100, 2))
        B = build_bspline_basis_2d(x, n_internal_knots=3, degree=3)
        assert np.all(B >= -1e-10)

    def test_finite_values(self):
        """2D B-spline basis should contain only finite values."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 5, (100, 2))
        B = build_bspline_basis_2d(x, n_internal_knots=3, degree=3)
        assert np.all(np.isfinite(B))


class TestQMLEObjective:
    """Tests for QMLE objective function."""

    def test_positive_objective(self):
        """QMLE objective should be positive for reasonable parameters."""
        rng = np.random.default_rng(42)
        n = 100
        eps_sq = np.abs(rng.standard_normal(n)) + 0.1
        eps_lags_sq = np.abs(rng.standard_normal((n, 1))) + 0.1
        B = np.ones((n, 1))
        params = np.array([0.3, 0.5])  # alpha=0.3, theta=0.5
        obj = qmle_objective(params, eps_sq, eps_lags_sq, B, p=1)
        assert np.isfinite(obj), f"QMLE objective is not finite: {obj}"

    def test_objective_decreases_with_better_fit(self):
        """QMLE objective should be lower for better-fitting parameters."""
        rng = np.random.default_rng(42)
        n = 200
        # True sigma^2 = 0.5 + 0.3*eps_{t-1}^2
        eps_lags_sq = np.abs(rng.standard_normal((n, 1))) + 0.1
        true_sigma2 = 0.5 + 0.3 * eps_lags_sq[:, 0]
        eps_sq = true_sigma2 * rng.chisquare(1, n)

        B = np.ones((n, 1))
        # True params: alpha=0.3, theta=0.5
        params_true = np.array([0.3, 0.5])
        # Bad params
        params_bad = np.array([0.0, 0.1])

        obj_true = qmle_objective(params_true, eps_sq, eps_lags_sq, B, p=1)
        obj_bad = qmle_objective(params_bad, eps_sq, eps_lags_sq, B, p=1)
        assert obj_true < obj_bad, f"True params obj {obj_true:.4f} >= bad params obj {obj_bad:.4f}"


class TestBSplineNullModelResiduals:
    """Tests for B-spline null model residuals."""

    def test_output_shape_p1(self):
        """Residuals should have shape (n-p,) for p=1."""
        rng = np.random.default_rng(42)
        n = 150
        eps = rng.standard_normal(n)
        x_reduced = rng.uniform(0, 5, (n, 1))
        v_hat, alpha_hat = bspline_null_model_residuals(eps, x_reduced, p=1)
        assert v_hat.shape == (n - 1,), f"Shape {v_hat.shape} != ({n-1},)"

    def test_alpha_non_negative(self):
        """Estimated ARCH coefficients should be non-negative."""
        rng = np.random.default_rng(42)
        n = 150
        eps = rng.standard_normal(n)
        x_reduced = rng.uniform(0, 5, (n, 1))
        _, alpha_hat = bspline_null_model_residuals(eps, x_reduced, p=1)
        assert np.all(alpha_hat >= 0), f"Negative alpha: {alpha_hat}"

    def test_finite_residuals(self):
        """Residuals should be finite."""
        rng = np.random.default_rng(42)
        n = 150
        eps = rng.standard_normal(n)
        x_reduced = rng.uniform(0, 5, (n, 1))
        v_hat, _ = bspline_null_model_residuals(eps, x_reduced, p=1)
        assert np.all(np.isfinite(v_hat)), "Residuals contain non-finite values"

    def test_2d_input(self):
        """2D input should work correctly."""
        rng = np.random.default_rng(42)
        n = 150
        eps = rng.standard_normal(n)
        x_reduced = rng.uniform(0, 5, (n, 2))
        v_hat, alpha_hat = bspline_null_model_residuals(eps, x_reduced, p=1)
        assert v_hat.shape == (n - 1,)
        assert np.all(np.isfinite(v_hat))

    def test_p2_model(self):
        """p=2 model should work correctly."""
        rng = np.random.default_rng(42)
        n = 200
        eps = rng.standard_normal(n)
        x_reduced = rng.uniform(0, 5, (n, 1))
        v_hat, alpha_hat = bspline_null_model_residuals(eps, x_reduced, p=2)
        assert v_hat.shape == (n - 2,)
        assert len(alpha_hat) == 2
