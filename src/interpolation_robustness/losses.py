import typing

import jax.numpy as jnp
import jax.scipy.special

import interpolation_robustness as ir


def logistic_with_logits_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the logistic loss directly on the logits.
    This uses the more numerically stable equivalent formulation from
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    :param logits: Predicted logits
    :param labels: True labels in [0, 1]
    :return: Logistic loss
    """

    assert labels.ndim == 2 and labels.shape[1] == 1

    positive_logits = jnp.where(logits >= 0, logits, 0)
    negative_abs_logits = jnp.where(logits < 0, logits, -logits)
    return jnp.mean(positive_logits - labels * logits + jnp.log(1 + jnp.exp(negative_abs_logits)))


def sparse_softmax_cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the softmax cross-entropy loss directly on logits for numerical stability.
    This method expects sparse labels in [0, num_classes) for efficiency.
    :param logits: Predicted unscaled logits
    :param labels: True labels in [0, num_classes) as a shape 1 array
    :return: Softmax cross-entropy loss
    """

    log_predictions = jax.nn.log_softmax(logits, axis=-1)
    return jnp.mean(-log_predictions[jnp.arange(logits.shape[0]), labels])


def closed_form_adversarial_logistic_loss(
        xs: jnp.ndarray,
        labels: jnp.ndarray,
        weights: jnp.ndarray,
        epsilon: float,
        attack_q: typing.Union[float, int, str],
        clamp_range: typing.Optional[typing.Tuple[float, float]],
        ground_truth_normalized: typing.Optional[jnp.ndarray] = None
) -> jnp.ndarray:

    # NB: Expects labels to be in {-1, +1} range, NOT {0, 1}!
    loss_input = ir.attacks.linear_closed_form_attack(
        xs, labels, weights, epsilon, attack_q, clamp_range, ground_truth_normalized
    )
    per_sample_losses = jnp.log(1.0 + jnp.exp(-loss_input))

    return jnp.mean(per_sample_losses)


def quadratic_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    per_sample_losses = 0.5 * jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)


def closed_form_adversarial_quadratic_loss(
        predictions: jnp.ndarray,
        ys: jnp.ndarray,
        weights: jnp.ndarray,
        epsilon: float,
        attack_q: typing.Union[str, float, int],
        consistent_attack_gt: typing.Optional[jnp.ndarray]
) -> jnp.ndarray:
    abs_residuals = jnp.abs(ys - predictions)

    if consistent_attack_gt is not None:
        # NB: ground truth is assumed to be normalized!
        # Project weights orthogonal to ground truth for consistency if provided
        weights = weights - jnp.dot(weights, consistent_attack_gt) * consistent_attack_gt

    # Make sure to not take sqrt of zero when weights are zero, else derivative is undefined
    assert weights.ndim == 1
    weight_norm = jax.lax.cond(
        jnp.any(weights != 0),
        lambda _: jnp.linalg.norm(weights, ord=attack_q),
        lambda _: 0.0,
        operand=None
    )

    per_sample_losses = 0.5 * jnp.square(abs_residuals + epsilon * weight_norm)

    return jnp.mean(per_sample_losses)
