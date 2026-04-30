"""
Tests for the Data Generating Process (DGP) module.

Validates:
- Innovation sampling distributions have correct moments
- Covariate generation has correct mean and covariance structure
- DGP models produce positive conditional variances
- ARCH-m(X) time series generation is correct
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.dgp import (
    sample_innovations,
    generate_covariates,
    sigma2_2cov,
    sigma2_5cov,
    sigma2_5cov_varsel,
    sigma2_10cov_varsel,
    get_p_2cov,
    get_p_5cov,
    get_p_5cov_varsel,
    get_p_10cov_varsel,
    generate_arch_mx,
)


# ---------------------------------------------------------------------------
# Tests for innovation sampling
# ---------------------------------------------------------------------------

class TestSampleInnovations:
    """Tests for innovation distribution sampling."""

    def test_normal_unit_variance(self):
        """Normal innovations should have unit variance."""
        rng = np.random.default_rng(42)
        z = sample_innovations(10000, "normal", rng)
        assert abs(np.var(z) - 1.0) < 0.05, f"Normal variance {np.var(z):.4f} != 1.0"

    def test_normal_zero_mean(self):
        """Normal innovations should have zero mean."""
        rng = np.random.default_rng(42)
        z = sample_innovations(10000, "normal", rng)
        assert abs(np.mean(z)) < 0.05, f"Normal mean {np.mean(z):.4f} != 0"

    def test_laplace_unit_variance(self):
        """Laplace innovations should have unit variance after standardization."""
        rng = np.random.default_rng(42)
        z = sample_innovations(10000, "laplace", rng)
        assert abs(np.var(z) - 1.0) < 0.05, f"Laplace variance {np.var(z):.4f} != 1.0"

    def test_laplace_zero_mean(self):
        """Laplace innovations should have zero mean."""
        rng = np.random.default_rng(42)
        z = sample_innovations(10000, "laplace", rng)
        assert abs(np.mean(z)) < 0.05

    def test_t7_unit_variance(self):
        """t(7) innovations should have unit variance after standardization."""
        rng = np.random.default_rng(42)
        z = sample_innovations(10000, "t7", rng)
        assert abs(np.var(z) - 1.0) < 0.05, f"t7 variance {np.var(z):.4f} != 1.0"

    def test_t7_zero_mean(self):
        """t(7) innovations should have zero mean."""
        rng = np.random.default_rng(42)
        z = sample_innovations(10000, "t7", rng)
        assert abs(np.mean(z)) < 0.05

    def test_invalid_distribution(self):
        """Invalid distribution should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            sample_innovations(100, "invalid_dist", rng)

    def test_output_shape(self):
        """Output should have correct shape."""
        rng = np.random.default_rng(42)
        for dist in ["normal", "laplace", "t7"]:
            z = sample_innovations(500, dist, rng)
            assert z.shape == (500,), f"Shape mismatch for {dist}"


# ---------------------------------------------------------------------------
# Tests for covariate generation
# ---------------------------------------------------------------------------

class TestGenerateCovariates:
    """Tests for covariate generation."""

    def test_mean_close_to_2_5(self):
        """Covariates should have mean close to 2.5."""
        rng = np.random.default_rng(42)
        X = generate_covariates(10000, d=2, rho=0.0, rng=rng)
        assert np.allclose(X.mean(axis=0), 2.5, atol=0.1), f"Mean {X.mean(axis=0)} != 2.5"

    def test_identity_covariance_rho0(self):
        """With rho=0, covariance should be identity."""
        rng = np.random.default_rng(42)
        X = generate_covariates(10000, d=3, rho=0.0, rng=rng)
        cov = np.cov(X.T)
        assert np.allclose(cov, np.eye(3), atol=0.1), f"Covariance not identity: {cov}"

    def test_correlation_structure_rho05(self):
        """With rho=0.5, off-diagonal covariance should be ~0.5."""
        rng = np.random.default_rng(42)
        X = generate_covariates(10000, d=2, rho=0.5, rng=rng)
        cov = np.cov(X.T)
        assert abs(cov[0, 1] - 0.5) < 0.1, f"Off-diagonal {cov[0,1]} != 0.5"

    def test_output_shape(self):
        """Output should have correct shape."""
        rng = np.random.default_rng(42)
        X = generate_covariates(1000, d=5, rho=0.0, rng=rng)
        assert X.shape == (1000, 5)

    def test_5d_correlation_structure(self):
        """5D covariate with rho=0.5 should have Sigma_{ij} = 0.5^|i-j|."""
        rng = np.random.default_rng(42)
        X = generate_covariates(20000, d=5, rho=0.5, rng=rng)
        cov = np.cov(X.T)
        # Check Sigma_{1,3} = 0.5^2 = 0.25
        assert abs(cov[0, 2] - 0.25) < 0.1, f"Sigma_13 = {cov[0,2]:.3f} != 0.25"


# ---------------------------------------------------------------------------
# Tests for DGP models
# ---------------------------------------------------------------------------

class TestSigma2Models:
    """Tests for conditional variance functions."""

    def test_2cov_all_models_positive(self):
        """All 2-covariate DGP models should produce positive sigma^2."""
        rng = np.random.default_rng(42)
        x = np.array([2.5, 2.5])
        eps_lag = np.array([0.1, 0.05])

        for model in range(1, 9):
            for c in [0.0, 0.5, 1.0]:
                val = sigma2_2cov(model, eps_lag, x, c)
                assert val > 0, f"Model {model}, c={c}: sigma^2={val} <= 0"

    def test_2cov_model1_null_formula(self):
        """Model 1 at c=0 should match formula: 0.2 + 0.3*e1 + 0.5*(x1-2.5)^2."""
        x = np.array([3.0, 2.5])
        eps_lag = np.array([0.5, 0.0])
        c = 0.0
        expected = 0.2 + 0.3 * 0.25 + 0.5 * (3.0 - 2.5) ** 2
        result = sigma2_2cov(1, eps_lag, x, c)
        assert abs(result - expected) < 1e-10, f"Model 1 formula mismatch: {result} != {expected}"

    def test_2cov_model7_linear(self):
        """Model 7 should be linear in x1 and x2."""
        x = np.array([3.0, 2.0])
        eps_lag = np.array([0.5, 0.0])
        c = 0.5
        expected = 0.2 + 0.4 * 0.25 + 0.25 * 3.0 + 0.5 * 2.0
        result = sigma2_2cov(7, eps_lag, x, c)
        assert abs(result - expected) < 1e-10

    def test_5cov_all_models_positive(self):
        """All 5-covariate DGP models should produce positive sigma^2."""
        x = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        eps_lag = np.array([0.1, 0.05])

        for model in range(1, 9):
            for c in [0.0, 0.5, 1.0]:
                val = sigma2_5cov(model, eps_lag, x, c)
                assert val > 0, f"5cov Model {model}, c={c}: sigma^2={val} <= 0"

    def test_5cov_varsel_all_models_positive(self):
        """All 5-covariate variable selection DGP models should produce positive sigma^2."""
        x = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        eps_lag = np.array([0.1, 0.05])

        for model in range(1, 9):
            val = sigma2_5cov_varsel(model, eps_lag, x)
            assert val > 0, f"5cov varsel Model {model}: sigma^2={val} <= 0"

    def test_10cov_varsel_all_models_positive(self):
        """All 10-covariate variable selection DGP models should produce positive sigma^2."""
        x = np.full(10, 2.5)
        eps_lag = np.array([0.1, 0.05])

        for model in range(1, 9):
            val = sigma2_10cov_varsel(model, eps_lag, x)
            assert val > 0, f"10cov varsel Model {model}: sigma^2={val} <= 0"

    def test_invalid_model_raises(self):
        """Invalid model index should raise ValueError."""
        x = np.array([2.5, 2.5])
        eps_lag = np.array([0.1])
        with pytest.raises(ValueError):
            sigma2_2cov(9, eps_lag, x, 0.5)

    def test_p_values_2cov(self):
        """ARCH order p should be correct for 2-covariate models."""
        assert get_p_2cov(1) == 1
        assert get_p_2cov(3) == 2
        assert get_p_2cov(4) == 2
        assert get_p_2cov(5) == 1

    def test_p_values_5cov(self):
        """ARCH order p should be correct for 5-covariate models."""
        assert get_p_5cov(1) == 1
        assert get_p_5cov(3) == 2
        assert get_p_5cov(4) == 2


# ---------------------------------------------------------------------------
# Tests for ARCH-m(X) time series generation
# ---------------------------------------------------------------------------

class TestGenerateArchMX:
    """Tests for ARCH-m(X) time series generation."""

    def test_output_shape(self):
        """Output should have correct shape after burn-in."""
        rng = np.random.default_rng(42)
        n = 200
        burnin = 100
        X = generate_covariates(n + burnin, d=2, rho=0.0, rng=rng)

        def sigma2_func(eps_lag, x):
            return sigma2_2cov(1, eps_lag, x, c=0.5)

        eps = generate_arch_mx(n, X, sigma2_func, p=1, distribution="normal",
                               rng=rng, burnin=burnin)
        assert eps.shape == (n,), f"Shape {eps.shape} != ({n},)"

    def test_finite_values(self):
        """Generated series should contain only finite values."""
        rng = np.random.default_rng(42)
        n = 500
        burnin = 100
        X = generate_covariates(n + burnin, d=2, rho=0.0, rng=rng)

        def sigma2_func(eps_lag, x):
            return sigma2_2cov(1, eps_lag, x, c=0.0)

        eps = generate_arch_mx(n, X, sigma2_func, p=1, distribution="normal",
                               rng=rng, burnin=burnin)
        assert np.all(np.isfinite(eps)), "Generated series contains non-finite values"

    def test_all_distributions(self):
        """Generation should work for all three distributions."""
        rng = np.random.default_rng(42)
        n = 200
        burnin = 50
        X = generate_covariates(n + burnin, d=2, rho=0.0, rng=rng)

        def sigma2_func(eps_lag, x):
            return sigma2_2cov(1, eps_lag, x, c=0.0)

        for dist in ["normal", "laplace", "t7"]:
            eps = generate_arch_mx(n, X, sigma2_func, p=1, distribution=dist,
                                   rng=rng, burnin=burnin)
            assert eps.shape == (n,)
            assert np.all(np.isfinite(eps))

    def test_p2_model(self):
        """ARCH(2) model should work correctly."""
        rng = np.random.default_rng(42)
        n = 300
        burnin = 100
        X = generate_covariates(n + burnin, d=2, rho=0.0, rng=rng)

        def sigma2_func(eps_lag, x):
            return sigma2_2cov(3, eps_lag, x, c=0.5)

        eps = generate_arch_mx(n, X, sigma2_func, p=2, distribution="normal",
                               rng=rng, burnin=burnin)
        assert eps.shape == (n,)
        assert np.all(np.isfinite(eps))
