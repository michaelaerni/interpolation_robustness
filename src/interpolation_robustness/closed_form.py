import typing

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
import scipy.special


def linear_regression_st_risks(
        l2_lambda: typing.Union[float, jnp.ndarray],
        gamma: float,
        sigma_sq: float,
        epsilon_test: float
) -> typing.Tuple[typing.Union[float, jnp.ndarray], typing.Union[float, jnp.ndarray]]:
    orig_l2_lambda = l2_lambda
    if isinstance(orig_l2_lambda, float):
        l2_lambda = jnp.asarray([l2_lambda])

    if jnp.any(l2_lambda < 0):
        raise ValueError(f'Ridge parameters must be non-negative but are {l2_lambda}')
    if gamma <= 0:
        raise ValueError(f'd/n ratio must be positive but is {gamma}')
    if epsilon_test < 0:
        raise ValueError(f'Test attack epsilon must be non-negative but is {epsilon_test}')
    if sigma_sq < 0:
        raise ValueError(f'Noise variance must be non-negative but is {sigma_sq}')

    # Write m(z) = n(z) / z to avoid div by zero for lambda = 0
    def n(z: typing.Union[jnp.ndarray, float]) -> jnp.ndarray:
        t1 = 1.0 - gamma - z
        return (t1 - jnp.sqrt(jnp.square(t1) - 4.0 * gamma * z)) / (2.0 * gamma)
    n_prime = jax.vmap(jax.grad(n))

    bias = -l2_lambda * n_prime(-l2_lambda) - n(-l2_lambda)
    variance = sigma_sq * gamma * n_prime(-l2_lambda)
    proj_sq = bias + variance + 2.0 * n(-l2_lambda) + 1.0 - jnp.square(1.0 + n(-l2_lambda))

    standard_risk = bias + variance
    robust_risk = bias + variance \
        + 2.0 * jnp.sqrt(2.0 / jnp.pi) * epsilon_test * jnp.sqrt(proj_sq) * jnp.sqrt(standard_risk) \
        + jnp.square(epsilon_test) * proj_sq

    # Convert to float/numpy arrays
    if isinstance(orig_l2_lambda, float):
        standard_risk = float(standard_risk)
        robust_risk = float(robust_risk)

    return standard_risk, robust_risk


def logistic_regression_asymptotic_risks_linf(
        weight_norm_l2_orthogonal: float,
        weight_norm_l2_parallel: float,
        weight_norm_l1_projected: float,
        epsilon: float
) -> typing.Tuple[float, float]:
    # NB: This assumes that the ground truth has unit length!

    weight_norm_l2 = np.sqrt(np.square(weight_norm_l2_parallel) + np.square(weight_norm_l2_orthogonal))

    # If weight norm is zero, ground truth angle is ignored
    with np.errstate(divide='ignore'):
        ground_truth_cos_angle = weight_norm_l2_parallel / weight_norm_l2

    return logistic_regression_risks(
        weight_norm_l2=weight_norm_l2,
        ground_truth_cos_angle=ground_truth_cos_angle,
        weight_norm_lq_projected=weight_norm_l1_projected,
        epsilon=epsilon
    )


def logistic_regression_risks(
        weight_norm_l2: float,
        ground_truth_cos_angle: float,
        weight_norm_lq_projected: float,  # FIXME: This is more general, the name is a bit misleading
        epsilon: float
) -> typing.Tuple[float, float]:
    # Risks are simple if weights are zero
    if weight_norm_l2 == 0:
        true_std_risk = 0.5
        true_robust_risk = 1.0
    else:
        # Standard risk (1 - standard accuracy)
        true_std_risk = np.arccos(ground_truth_cos_angle) / np.pi

        # Robust risk
        assert ground_truth_cos_angle != 0  # else things break down, but this should not happen
        xi = float(epsilon * weight_norm_lq_projected / weight_norm_l2)
        t_term = np.abs(ground_truth_cos_angle) \
            / np.sqrt(1.0 - np.square(ground_truth_cos_angle))
        t_term = float(t_term)  # need a Python float for numerical integration later

        # Solve integral in risk numerically
        def integration_function(x: float) -> float:
            return 0.5 * scipy.special.erf(x / np.sqrt(2.0) * t_term) \
                   / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * np.square(x))

        integral_approximation, *_ = scipy.integrate.quad(
            integration_function, a=0.0, b=xi
        )
        integral_term = 2.0 * integral_approximation
        true_robust_risk = true_std_risk + 0.5 * scipy.special.erf(xi / np.sqrt(2)) + integral_term

    return true_std_risk, true_robust_risk


def linear_regression_st_noiseless_risk_decomposition(
        l2_lambda: typing.Union[float, jnp.ndarray],
        gamma: float
) -> typing.Tuple[typing.Union[float, jnp.ndarray], typing.Union[float, jnp.ndarray]]:
    orig_l2_lambda = l2_lambda
    if isinstance(orig_l2_lambda, float):
        l2_lambda = jnp.asarray([l2_lambda])

    if jnp.any(l2_lambda < 0):
        raise ValueError(f'Ridge parameters must be non-negative but are {l2_lambda}')
    if gamma <= 0:
        raise ValueError(f'd/n ratio must be positive but is {gamma}')

    # Write m(z) = n(z) / z to avoid div by zero for lambda = 0
    def n(z: typing.Union[jnp.ndarray, float]) -> jnp.ndarray:
        t1 = 1.0 - gamma - z
        return (t1 - jnp.sqrt(jnp.square(t1) - 4.0 * gamma * z)) / (2.0 * gamma)
    n_prime = jax.vmap(jax.grad(n))

    bias = -l2_lambda * n_prime(-l2_lambda) - n(-l2_lambda)
    # Avoid negative norms due to numerical imprecision
    parallel_diff_norm_sq = jnp.maximum(0.0, jnp.square(1.0 + n(-l2_lambda)) - 2.0 * n(-l2_lambda) - 1.0)
    orthogonal_norm_sq = jnp.maximum(0.0, bias - parallel_diff_norm_sq)

    # Handle special case gamma = 1 and lambda = 0
    if gamma == 1:
        parallel_diff_norm_sq = jnp.where(l2_lambda == 0, 0.0, parallel_diff_norm_sq)
        orthogonal_norm_sq = jnp.where(l2_lambda == 0, 0.0, orthogonal_norm_sq)

    # Convert to float/numpy arrays
    if isinstance(orig_l2_lambda, float):
        parallel_diff_norm_sq = float(parallel_diff_norm_sq)
        orthogonal_norm_sq = float(orthogonal_norm_sq)
    return parallel_diff_norm_sq, orthogonal_norm_sq
