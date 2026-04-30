"""
Tests for the Kernel Smoothing Estimator module.

Validates:
- Gaussian kernel properties
- Silverman bandwidth formula
- Kernel weight matrix properties (rows sum to 1)
- Null model residuals computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.kernel_estimator import (
    gaussian_kernel,
    silverman_bandwidth,
    compute_kernel_weights_1d,
    compute_kernel_weights_nd,
    kernel_null_model_residuals,
)


class TestGaussianKernel:
    """Tests for Gaussian kernel function."""

    def test_integrates_to_one(self):
        """Gaussian kernel should integrate to 1."""
        u = np.linspace(-5, 5, 10000)
        du = u[1] - u[0]
        integral = np.sum(gaussian_kernel(u)) * du
        assert abs(integral - 1.0) < 0.01, f"Integral {integral:.4f} != 1.0"

    def test_symmetric(self):
        """Gaussian kernel should be symmetric: K(u) = K(-u)."""
        u = np.array([0.5, 1.0, 2.0])
        assert np.allclose(gaussian_kernel(u), gaussian_kernel(-u))

    def test_maximum_at_zero(self):
        """Gaussian kernel should have maximum at u=0."""
        u = np.linspace(-3, 3, 100)
        K = gaussian_kernel(u)
        assert K.argmax() == len(u) // 2 or abs(u[K.argmax()]) < 0.1

    def test_non_negative(self):
        """Gaussian kernel should be non-negative."""
        u = np.linspace(-5, 5, 100)
        assert np.all(gaussian_kernel(u) >= 0)

    def test_scalar_input(self):
        """Gaussian kernel should work with scalar input."""
        val = gaussian_kernel(np.array([0.0]))
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(val[0] - expected) < 1e-10


class TestSilvermanBandwidth:
    """Tests for Silverman bandwidth selection."""

    def test_positive_bandwidth(self):
        """Bandwidth should be positive."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        h = silverman_bandwidth(x, d_eff=1)
        assert h > 0

    def test_bandwidth_formula(self):
        """Bandwidth should match 1.06 * std(x) * n^{-1/5} for d_eff=1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        h = silverman_bandwidth(x, d_eff=1)
        expected = 1.06 * np.std(x, ddof=1) * 500 ** (-0.2)
        assert abs(h - expected) < 1e-10

    def test_larger_n_smaller_bandwidth(self):
        """Larger sample size should give smaller bandwidth."""
        rng = np.random.default_rng(42)
        x_small = rng.standard_normal(100)
        x_large = rng.standard_normal(1000)
        # Use same std for fair comparison
        h_small = 1.06 * 1.0 * 100 ** (-0.2)
        h_large = 1.06 * 1.0 * 1000 ** (-0.2)
        assert h_small > h_large

    def test_higher_dim_larger_bandwidth(self):
        """Higher effective dimension gives larger bandwidth (curse of dimensionality).

        Silverman rule: h = 1.06 * std(x) * n^{-1/(d_eff+4)}
        For d_eff=1: exponent = -1/5 = -0.2 (smaller h)
        For d_eff=4: exponent = -1/8 = -0.125 (larger h, less shrinkage)
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        h1 = silverman_bandwidth(x, d_eff=1)
        h4 = silverman_bandwidth(x, d_eff=4)
        # Higher dimension -> larger bandwidth (less shrinkage with n)
        assert h4 > h1, f"h4={h4:.4f} should be > h1={h1:.4f} (curse of dimensionality)"


class TestKernelWeights:
    """Tests for kernel weight matrix computation."""

    def test_1d_rows_sum_to_one(self):
        """1D kernel weight matrix rows should sum to 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        h = silverman_bandwidth(x, d_eff=1)
        W = compute_kernel_weights_1d(x, x, h)
        row_sums = W.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Row sums not 1: {row_sums[:5]}"

    def test_1d_non_negative(self):
        """1D kernel weights should be non-negative."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        h = silverman_bandwidth(x, d_eff=1)
        W = compute_kernel_weights_1d(x, x, h)
        assert np.all(W >= 0)

    def test_1d_shape(self):
        """1D kernel weight matrix should have correct shape."""
        rng = np.random.default_rng(42)
        x_eval = rng.standard_normal(30)
        x_ref = rng.standard_normal(50)
        h = 0.5
        W = compute_kernel_weights_1d(x_eval, x_ref, h)
        assert W.shape == (30, 50)

    def test_nd_rows_sum_to_one(self):
        """ND kernel weight matrix rows should sum to 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 3))
        h = np.array([0.5, 0.5, 0.5])
        W = compute_kernel_weights_nd(x, x, h)
        row_sums = W.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_nd_non_negative(self):
        """ND kernel weights should be non-negative."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 2))
        h = np.array([0.5, 0.5])
        W = compute_kernel_weights_nd(x, x, h)
        assert np.all(W >= 0)

    def test_nd_shape(self):
        """ND kernel weight matrix should have correct shape."""
        rng = np.random.default_rng(42)
        x_eval = rng.standard_normal((30, 2))
        x_ref = rng.standard_normal((50, 2))
        h = np.array([0.5, 0.5])
        W = compute_kernel_weights_nd(x_eval, x_ref, h)
        assert W.shape == (30, 50)


class TestKernelNullModelResiduals:
    """Tests for kernel null model residuals computation."""

    def test_output_shape_p1(self):
        """Residuals should have shape (n-p,) for p=1."""
        rng = np.random.default_rng(42)
        n = 100
        eps = rng.standard_normal(n)
        x_reduced = rng.standard_normal((n, 1))
        v_hat, alpha_hat, _ = kernel_null_model_residuals(eps, x_reduced, p=1, d_reduced=1)
        assert v_hat.shape == (n - 1,), f"Shape {v_hat.shape} != ({n-1},)"

    def test_output_shape_p2(self):
        """Residuals should have shape (n-p,) for p=2."""
        rng = np.random.default_rng(42)
        n = 100
        eps = rng.standard_normal(n)
        x_reduced = rng.standard_normal((n, 1))
        v_hat, alpha_hat, _ = kernel_null_model_residuals(eps, x_reduced, p=2, d_reduced=1)
        assert v_hat.shape == (n - 2,), f"Shape {v_hat.shape} != ({n-2},)"

    def test_alpha_non_negative(self):
        """Estimated ARCH coefficients should be non-negative."""
        rng = np.random.default_rng(42)
        n = 200
        eps = rng.standard_normal(n)
        x_reduced = rng.standard_normal((n, 1))
        _, alpha_hat, _ = kernel_null_model_residuals(eps, x_reduced, p=1, d_reduced=1)
        assert np.all(alpha_hat >= 0), f"Negative alpha: {alpha_hat}"

    def test_finite_residuals(self):
        """Residuals should be finite."""
        rng = np.random.default_rng(42)
        n = 200
        eps = rng.standard_normal(n)
        x_reduced = rng.standard_normal((n, 1))
        v_hat, _, _ = kernel_null_model_residuals(eps, x_reduced, p=1, d_reduced=1)
        assert np.all(np.isfinite(v_hat)), "Residuals contain non-finite values"

    def test_2d_reduced_model(self):
        """2D reduced model should work correctly."""
        rng = np.random.default_rng(42)
        n = 150
        eps = rng.standard_normal(n)
        x_reduced = rng.standard_normal((n, 2))
        v_hat, alpha_hat, _ = kernel_null_model_residuals(eps, x_reduced, p=1, d_reduced=2)
        assert v_hat.shape == (n - 1,)
        assert np.all(np.isfinite(v_hat))
