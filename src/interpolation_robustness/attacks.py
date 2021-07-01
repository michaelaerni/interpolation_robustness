import typing

import jax
import jax.numpy as jnp

ClampFunction = typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
LossFunction = typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
AttackFunction = typing.Callable[[jnp.ndarray, jnp.ndarray, LossFunction, ClampFunction, jnp.ndarray], jnp.ndarray]

_PGDInitFunction = typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
_ClipFunction = typing.Callable[[jnp.ndarray], jnp.ndarray]
_PGDStepFunction = typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


@jax.partial(jax.jit, static_argnums=(3, 4, 5))
def linear_closed_form_attack(
        xs: jnp.ndarray,
        labels: jnp.ndarray,
        weights: jnp.ndarray,
        epsilon: float,
        attack_q: typing.Union[float, int, str],
        clamp_range: typing.Optional[typing.Tuple[float, float]],
        ground_truth_normalized: typing.Optional[jnp.ndarray] = None
) -> jnp.ndarray:

    # Work with vectors instead of kx1 matrices
    assert labels.ndim == 1
    assert weights.ndim == 1

    # NB: Expects labels to be in {-1, +1} range, NOT {0, 1}!
    if clamp_range is None:
        natural_logits = jnp.dot(xs, weights)

        # If ground truth given, perturbations are restricted to null space of ground truth
        # This can be achieved by projecting the weights onto the null space
        # by the properties of orthogonal projections.
        if ground_truth_normalized is not None:
            weights = weights - ground_truth_normalized * jnp.dot(weights, ground_truth_normalized)

        weight_norm = jnp.linalg.norm(weights, ord=attack_q)
        adversarial_offset = epsilon * weight_norm
        return jnp.multiply(labels, natural_logits) - adversarial_offset
    else:
        assert ground_truth_normalized is None

        # Clamping implicitly assumes l_inf perturbations!
        assert attack_q == 1
        clamp_min, clamp_max = clamp_range
        optimal_delta_unrestricted = (- epsilon) * jnp.sign(jnp.outer(labels, weights))
        xs_adversarial = xs + optimal_delta_unrestricted
        # Clipping like this is optimal for l_inf norm, but incorrect for other p
        xs_adversarial = jnp.clip(
            xs_adversarial,
            a_min=clamp_min,
            a_max=clamp_max
        )
        return jnp.multiply(labels, jnp.dot(xs_adversarial, weights))


def make_ffgsm_attacker(
        num_restarts: int,
        epsilon: float,
        p_norm: typing.Union[int, float, str],
        step_size: float
) -> AttackFunction:
    """
    Convenience method for creating a fast FGSM attacker.
    See make_pgd_attacker for parameter documentation.
    """
    return make_pgd_attacker(
        num_restarts=num_restarts,
        num_inner_iters=1,
        epsilon=epsilon,
        p_norm=p_norm,
        step_size=step_size,
        rand_init=True
    )


def make_pgd_attacker(
        num_restarts: int,
        num_inner_iters: int,
        epsilon: float,
        p_norm: typing.Union[int, float, str],
        step_size: float,
        rand_init: bool
) -> AttackFunction:
    """
    Creates a PGD adversarial attacker according to (Madry et al. 2017).

    Setting rand_init = False, num_inner_iters = 1, and step_size = epsilon results in
    the FGSM method by (Goodfellow et al. 2014).
    Setting rand_init = True and num_inner_iters = 1 results in
    the FFGSM method by (Wong et al. 2020).

    Currently, only l_2 and l_inf norm attacks are supported.
    Whether attacks are targeted or untargeted is determined by the loss function given to the attack function.

    Paper link (Goodfellow et al. 2014): https://arxiv.org/abs/1412.6572
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    Paper link (Wong et al. 2020): https://arxiv.org/abs/2001.03994

    :param num_restarts: How many times PGD is re-initialized and executed. Only 1 restart is currently supported.
    :param num_inner_iters: How many iterations of PGD are performed per restart.
    :param epsilon: Radius of the l_p ball around any input to which the attack is constrained.
    :param p_norm: l_p norm which the attack uses. Currently supported values are 2 and 'inf'.
    :param step_size: PGD step size in the space corresponding to the l_p norm.
    :param rand_init: If True, the starting perturbations are randomly initialized.
    :return: A function which tries to find an adversarial example for a given input and loss via PGD.
    """
    if num_restarts > 1:
        raise NotImplementedError(
            'Multiple restarts are not implemented yet since they are currently not needed '
            'and might break vectorization later on.'
        )
    if num_restarts > 1 and not rand_init:
        raise ValueError('Multiple restarts do not have any effect if random initialization is disabled')

    if epsilon <= 0:
        raise ValueError(f'Attack radius epsilon must be > 0, but is {epsilon}')

    if step_size > 2.0 * epsilon:
        raise ValueError(f'step_size must be at most the diameter of the epsilon ball, but is {step_size}')

    if p_norm in [1, '1']:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop step for PGD when norm=1, "
            "because norm=1 FGM changes only one pixel at a time. "
            "We need to rigorously test a strong norm=1 PGD before enabling this feature."
        )

    # Construct initializer, clipping, and gradient step functions
    init_fn: _PGDInitFunction = _make_zero_init()  # default initialization
    if p_norm in [2, '2']:
        if rand_init:
            init_fn = _make_l2_init(epsilon)
        clip_fn = _make_l2_clip(epsilon)
        step_fn = _make_l2_step(step_size)
    elif p_norm in [float('inf'), 'inf']:
        if rand_init:
            init_fn = _make_linf_init(epsilon)
        clip_fn = _make_linf_clip(epsilon)
        step_fn = _make_linf_step(step_size)
    else:
        raise ValueError('Norm order must be either inf or 2')
    assert init_fn is not None
    assert clip_fn is not None
    assert step_fn is not None

    # Always JIT the individual functions for performance
    init_fn = jax.jit(init_fn)
    clip_fn = jax.jit(clip_fn)
    step_fn = jax.jit(step_fn)

    def attack_fn(
            xs: jnp.ndarray,
            ys: jnp.ndarray,
            loss_fn: LossFunction,
            clamp_fn: ClampFunction,
            rng_key: jnp.ndarray
    ) -> jnp.ndarray:
        # NB: clamp_fn clamps x to the input space, clip_fn clips delta to the epsilon-ball
        # xs is expected to have shape (batch_size, dim)
        # ys can be the true or target labels, depending on the attack defined by the given loss_fn

        # Initialize perturbation
        rng_key, init_key = jax.random.split(rng_key)
        deltas = init_fn(xs, init_key)
        deltas = clamp_fn(xs, clip_fn(deltas))

        def delta_loss_fn(delta_t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return loss_fn(x + delta_t, y)
        grad_fn = jax.vmap(jax.grad(delta_loss_fn, argnums=0), in_axes=(0, 0, 0), out_axes=0)

        # Perform PGD
        # This returns the *last* perturbation, not necessarily the *best* one!
        # This allows for JIT compilation but might make the attacker weaker!
        # Some tests showed that the difference between last and best loss is quite negligible,
        # but still keep this in mind for potentially later!
        for _ in range(num_inner_iters):
            grads = grad_fn(deltas, xs, ys)
            deltas = step_fn(deltas, grads)
            deltas = clamp_fn(xs, clip_fn(deltas))

        # FIXME: Adding x and delta might result in an adversarial example that is slightly further from x
        #  than epsilon. It's not a big problem but could be fixed at some point.
        return xs + deltas

    return attack_fn


def _make_zero_init() -> _PGDInitFunction:
    def init_fn(x: jnp.ndarray, rng_key: typing.Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return jnp.zeros_like(x)

    return init_fn


def _make_l2_init(epsilon: float) -> _PGDInitFunction:
    def init_fn(x: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
        # Uniformly within hypersphere
        surface_key, radius_key = jax.random.split(rng_key)
        radius = jax.random.uniform(radius_key, shape=(x.shape[0], 1), minval=0.0, maxval=epsilon)
        delta = jax.random.normal(surface_key, shape=x.shape)
        norm = jnp.maximum(jnp.linalg.norm(delta, axis=-1, keepdims=True), jnp.finfo(jnp.float32).eps)
        delta = delta * (radius / norm)
        return delta

    return init_fn


def _make_l2_clip(epsilon: float) -> _ClipFunction:
    def clip_fn(delta: jnp.ndarray) -> jnp.ndarray:
        # Project back to hypersphere if necessary
        norm = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        # NB: zero-norm vectors will never be projected back to the sphere with radius epsilon > 0
        # However, due to the way autodiff works, this still needs to add an epsilon offset to avoid NaNs
        projected = jnp.where(
            norm > epsilon,
            delta * (epsilon / jnp.maximum(norm, jnp.finfo(jnp.float32).eps)),
            delta
        )
        return projected

    return clip_fn


def _make_l2_step(step_size: float) -> _PGDStepFunction:
    def step_fn(delta: jnp.ndarray, grads: jnp.ndarray) -> jnp.ndarray:
        # avoid division by 0
        grad_norm = jnp.maximum(jnp.linalg.norm(grads, axis=-1, keepdims=True), jnp.finfo(jnp.float32).eps)
        return delta + step_size * (grads / grad_norm)

    return step_fn


def _make_linf_init(epsilon: float) -> _PGDInitFunction:
    def init_fn(x: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
        # Uniformly within hypercube
        return jax.random.uniform(rng_key, shape=x.shape, minval=-epsilon, maxval=epsilon)

    return init_fn


def _make_linf_clip(epsilon: float) -> _ClipFunction:
    def clip_fn(delta: jnp.ndarray) -> jnp.ndarray:
        # Project back to hypercube if necessary
        return jnp.clip(delta, a_min=-epsilon, a_max=epsilon)

    return clip_fn


def _make_linf_step(step_size: float) -> _PGDStepFunction:
    def step_fn(delta: jnp.ndarray, grads: jnp.ndarray) -> jnp.ndarray:
        return delta + step_size * jnp.sign(grads)

    return step_fn
