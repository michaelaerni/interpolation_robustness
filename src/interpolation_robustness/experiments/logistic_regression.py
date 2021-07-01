import argparse
import contextlib
import datetime
import logging
import os
import typing

import dotenv
import flax
import flax.training.lr_schedule
import flax.traverse_util
import jax
import jax.numpy as jnp
import jax.ops
import jax.scipy
import numpy as np

import interpolation_robustness as ir

EXPERIMENT_TAG = 'logistic_regression'


class GMMDatasetConfig(typing.NamedTuple):
    sigma: float
    one_dim: bool
    scale_mean: bool


class SingleGaussianDatasetConfig(typing.NamedTuple):
    single_entry_ground_truth: bool
    logspace_covariance: bool
    logits_noise_variance: float


class BinaryMNISTDatasetConfig(typing.NamedTuple):
    class_0: int
    class_1: int


class ExperimentConfig(typing.NamedTuple):
    epochs: int
    eval_every_epochs: int
    save_every_epochs: int
    log_every_epochs: int
    initial_learning_rate: float
    learning_rate_decay_epochs: typing.Tuple[int]
    learning_rate_decay_step: float
    learning_rate_warmup: int
    momentum: float
    nesterov: bool
    label_noise: float
    adversarial_training: bool
    train_consistent_attacks: bool
    train_attack_epsilon: float
    test_attack_epsilon: float
    attack_p: typing.Union[float, str]
    data_num_train_samples: int
    data_dim: int
    dataset_config: typing.Union[GMMDatasetConfig, BinaryMNISTDatasetConfig, SingleGaussianDatasetConfig]
    root_log_dir: typing.Optional[str]
    l2_lambda: float
    l1_lambda: float


class DataLogger(object):
    def __init__(self, log_dir: typing.Optional[str], save_every: int, num_epochs: int):
        self._log_dir = log_dir
        self._save_every = save_every
        self._num_epochs = num_epochs
        self._all_weights = None
        self._log = logging.getLogger(__name__)

    def prepare(self):
        if self._log_dir is None:
            return

        self._log.info('Using log directory %s', self._log_dir)
        os.makedirs(self._log_dir, exist_ok=True)

    def log_2gmm_dataset(
            self,
            train_xs: np.ndarray,
            train_ys: np.ndarray,
            test_xs: np.ndarray,
            test_ys: np.ndarray,
            train_flip_indices: np.ndarray,
            train_noise_indices: np.ndarray,
            test_noise_indices: np.ndarray,
            mu: np.ndarray,
            sigma: float
    ):
        if self._log_dir is None:
            return

        target_file = os.path.join(self._log_dir, 'dataset.npz')
        np.savez_compressed(
            target_file,
            train_xs=train_xs,
            train_ys=train_ys,
            test_xs=test_xs,
            test_ys=test_ys,
            train_flip_indices=train_flip_indices,
            train_noise_indices=train_noise_indices,
            test_noise_indices=test_noise_indices,
            mu=mu,
            sigma=np.asarray((sigma,))
        )
        self._log.info('Saved dataset to %s', target_file)

    def log_weights(self, weights: jnp.ndarray, epoch: int):
        if self._log_dir is None:
            return

        assert weights.ndim == 1

        # Create weights storage array if not yet done
        if self._all_weights is None:
            self._all_weights = np.zeros((self._num_epochs, weights.shape[0]), weights.dtype)

        # Record weights
        self._all_weights[epoch] = np.asarray(weights)

        # Save weights if frequency hit or last epoch
        if epoch % self._save_every == 0 or epoch == self._num_epochs - 1:
            target_file = os.path.join(self._log_dir, f'weights.npy')
            np.save(target_file, self._all_weights)

    def log_single_gaussian_dataset(
            self,
            train_xs: np.ndarray,
            train_ys: np.ndarray,
            test_xs: np.ndarray,
            test_ys: np.ndarray,
            train_flip_indices: np.ndarray,
            train_noise_indices: np.ndarray,
            test_noise_indices: np.ndarray,
            ground_truth: np.ndarray,
            x_covariance_diagonal: np.ndarray,
            logits_noise_variance: float
    ):
        if self._log_dir is None:
            return

        target_file = os.path.join(self._log_dir, 'dataset.npz')
        np.savez_compressed(
            target_file,
            train_xs=train_xs,
            train_ys=train_ys,
            test_xs=test_xs,
            test_ys=test_ys,
            train_flip_indices=train_flip_indices,
            train_noise_indices=train_noise_indices,
            test_noise_indices=test_noise_indices,
            ground_truth=ground_truth,
            x_covariance_diagonal=x_covariance_diagonal,
            logits_noise_variance=np.asarray((logits_noise_variance,))
        )
        self._log.info('Saved dataset to %s', target_file)


def main():
    # Load environment variables
    dotenv.load_dotenv()

    # Setup logging
    ir.util.setup_logging()

    # Determine config
    args = _parse_args()
    config = build_config(args)

    with setup_experiment(config, args.tag) as run:
        run_experiment(config, run.info.run_uuid)


def run_experiment(config: ExperimentConfig, run_id: str):
    rng_key = jax.random.PRNGKey(seed=1)
    log = logging.getLogger(__name__)

    log.info('Starting experiment with config %s', config._asdict())
    log.info('MLFlow run id is %s', run_id)

    # Prepare weight etc logging
    log_dir = os.path.join(config.root_log_dir, run_id) if config.root_log_dir is not None else None
    data_logger = DataLogger(log_dir, config.save_every_epochs, config.epochs)
    data_logger.prepare()
    metric_logger = ir.mlflow.DelayedMetricLogger(run_id, config.log_every_epochs, config.epochs)

    # Build dataset
    rng_key, dataset_key = jax.random.split(rng_key)
    (train_xs, train_ys), (test_xs, test_ys), train_flip_indices, covariate_noise_indices, data_aux = make_dataset(
        config,
        dataset_key,
        data_logger
    )
    train_covariate_noise_indices, test_covariate_noise_indices = covariate_noise_indices
    train_ys_rescaled = jnp.where(train_ys == 1, train_ys, -jnp.ones_like(train_ys))  # {-1, +1} instead of {0, 1}
    train_ys = jnp.expand_dims(train_ys, axis=-1)
    test_ys_rescaled = jnp.where(test_ys == 1, test_ys, -jnp.ones_like(test_ys))  # {-1, +1} instead of {0, 1}
    test_flip_indices = jnp.zeros((0,))  # No noise on the test set
    num_train_samples, data_dim = train_xs.shape
    num_test_samples = test_xs.shape[0]
    num_noise_flip_samples, = train_flip_indices.shape
    num_train_covariate_noise_samples, = train_covariate_noise_indices.shape
    num_test_covariate_noise_samples, = test_covariate_noise_indices.shape
    log.info(
        'Built data set with %d training samples (%d with flipped label, %d mislabeled due to noise), %d test samples (%d mislabeled due to noise)',
        num_train_samples,
        num_noise_flip_samples,
        num_train_covariate_noise_samples,
        num_test_samples,
        num_test_covariate_noise_samples
    )

    # Build model
    model = ir.models.jax.LinearModel(
        num_out=1,
        weight_init=flax.linen.initializers.zeros
    )
    rng_key, init_key = jax.random.split(rng_key)
    params = model.init(init_key, jnp.ones((1, data_dim)))
    log.info('Built and initialized model')

    # Save initial output weights for tracking later
    def get_output_weights(current_params: flax.core.FrozenDict) -> jnp.ndarray:
        path = ('params', 'dense', 'kernel')
        flattened_params_dict = flax.traverse_util.flatten_dict(flax.core.unfreeze(current_params))
        return flattened_params_dict[path][:, 0]
    initial_output_weights = get_output_weights(params)

    # Build optimizer
    optimizer_def = flax.optim.Momentum(
        learning_rate=config.initial_learning_rate,
        beta=config.momentum,
        nesterov=config.nesterov
    )
    optimizer = optimizer_def.create(params)

    # Build learning rate schedule
    lr_schedule_fn = flax.training.lr_schedule.create_stepped_learning_rate_schedule(
        base_learning_rate=config.initial_learning_rate,
        steps_per_epoch=1,  # full-batch gradient descent
        lr_sched_steps=[
            (decay_epoch, config.learning_rate_decay_step ** (epoch_idx + 1))
            for epoch_idx, decay_epoch in enumerate(config.learning_rate_decay_epochs)
        ],
        warmup_length=config.learning_rate_warmup  # Passing 0 (the default) disables warmup
    )

    # Prepare l-inf clamping for MNIST
    if isinstance(config.dataset_config, BinaryMNISTDatasetConfig):
        assert config.attack_p == 'inf'
        clamp_range = (0.0, 1.0)
    else:
        clamp_range = None

    if config.attack_p in ('inf', float('inf')):
        attack_q = 1
    elif config.attack_p == 1.0:
        attack_q = jnp.inf
    else:
        assert config.attack_p > 1.0
        attack_q = config.attack_p / (config.attack_p - 1.0)

    # Setup adversarial training if enabled
    if config.adversarial_training:
        # Prepare consistent attacks if required
        # Ground truth must be normalized
        if config.train_consistent_attacks:
            if isinstance(config.dataset_config, GMMDatasetConfig):
                ground_truth = data_aux[0]
            elif isinstance(config.dataset_config, SingleGaussianDatasetConfig):
                ground_truth = data_aux[0]
            else:
                assert False
            assert ground_truth.ndim == 1
            ground_truth = ground_truth / jnp.linalg.norm(ground_truth)
        else:
            ground_truth = None

        def train_loss_fn(model_params):
            weights = get_output_weights(model_params)
            loss = ir.losses.closed_form_adversarial_logistic_loss(
                train_xs, train_ys_rescaled, weights, config.train_attack_epsilon, attack_q, clamp_range, ground_truth
            )
            if config.l2_lambda > 0:
                loss = loss + config.l2_lambda * jnp.sum(jnp.square(weights))
            if config.l1_lambda > 0:
                loss = loss + config.l1_lambda * jnp.sum(jnp.abs(weights))
            return loss
    else:
        def train_loss_fn(model_params):
            logits = model.apply(model_params, train_xs)
            weights = get_output_weights(model_params)
            loss = ir.losses.logistic_with_logits_loss(logits, train_ys)
            if config.l2_lambda > 0:
                loss = loss + config.l2_lambda * jnp.sum(jnp.square(weights))
            if config.l1_lambda > 0:
                loss = loss + config.l1_lambda * jnp.sum(jnp.abs(weights))
            return loss
    grad_fn = jax.value_and_grad(train_loss_fn, has_aux=False)
    # For very large data dimensions, jit-ing the gradient function leads to OOM issues
    if config.data_dim <= 60000:
        grad_fn = jax.jit(grad_fn)

    log.info('Starting training')
    for epoch in range(config.epochs):
        # Full batch GD
        # Adversarial attack (if enabled) happens in closed-form in training loss function directly
        current_loss, grad = grad_fn(optimizer.target)
        current_lr = lr_schedule_fn(optimizer.state.step)
        optimizer = optimizer.apply_gradient(grad, learning_rate=current_lr)

        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs - 1:
            metric_logger.add_metrics({'training_loss': float(current_loss)}, step=epoch)

        # Model saving, automatically checks whether it should save or not
        current_weights = get_output_weights(optimizer.target)
        data_logger.log_weights(current_weights, epoch)

        # Evaluation
        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs - 1:
            eval_metrics = dict()
            log_metrics = []

            # Always evaluate on training set
            rng_key, eval_train_key = jax.random.split(rng_key)
            num_std_train_correct, num_adv_train_correct, num_std_train_flip_correct, num_adv_train_flip_correct, num_std_train_noise_correct, num_adv_train_noise_correct = eval_dataset(
                train_xs,
                train_ys_rescaled,
                current_weights,
                train_flip_indices,
                train_covariate_noise_indices,
                config.test_attack_epsilon,
                attack_q,
                clamp_range
            )
            log_metrics += ['train_std_accuracy', 'train_robust_accuracy']
            eval_metrics['train_std_accuracy'] = float(num_std_train_correct) / num_train_samples
            eval_metrics['train_robust_accuracy'] = float(num_adv_train_correct) / num_train_samples

            # Evaluate on test set if there is a test set
            if num_test_samples > 0:
                num_std_test_correct, num_adv_test_correct, _, _, num_std_test_noise_correct, num_adv_test_noise_correct = eval_dataset(
                    test_xs,
                    test_ys_rescaled,
                    current_weights,
                    test_flip_indices,
                    test_covariate_noise_indices,
                    config.test_attack_epsilon,
                    attack_q,
                    clamp_range
                )
                log_metrics += ['test_std_accuracy', 'test_robust_accuracy']
                eval_metrics['test_std_accuracy'] = float(num_std_test_correct) / num_test_samples
                eval_metrics['test_robust_accuracy'] = float(num_adv_test_correct) / num_test_samples

                if num_test_covariate_noise_samples > 0:
                    log_metrics += ['fraction_std_test_noise_fitted', 'fraction_adv_test_noise_fitted']
                    eval_metrics['num_std_test_noise_fitted'] = num_std_test_noise_correct
                    eval_metrics['fraction_std_test_noise_fitted'] = float(num_std_test_noise_correct) / num_test_covariate_noise_samples
                    eval_metrics['num_adv_test_noise_fitted'] = num_adv_test_noise_correct
                    eval_metrics['fraction_adv_test_noise_fitted'] = float(num_adv_test_noise_correct) / num_test_covariate_noise_samples

            # Calculate closed-form accuracies if available
            if isinstance(config.dataset_config, GMMDatasetConfig):
                mu, sigma = data_aux

                sigma_weights_norm = jnp.linalg.norm(sigma * current_weights)
                mu_weight_dot = jnp.dot(mu, current_weights)
                assert config.attack_p in (2, 'inf')
                q_ord = 1 if config.attack_p == 'inf' else 2
                true_std_accuracy = jax.scipy.stats.norm.cdf(
                    mu_weight_dot / sigma_weights_norm
                )
                true_robust_accuracy = jax.scipy.stats.norm.cdf(
                    (mu_weight_dot - config.test_attack_epsilon * jnp.linalg.norm(current_weights, ord=q_ord))
                    / sigma_weights_norm
                )
                mu_weights_cos_angle = mu_weight_dot / (jnp.linalg.norm(mu) * jnp.linalg.norm(current_weights))

                log_metrics += ['true_std_accuracy', 'true_robust_accuracy', 'mu_weights_cos_angle']
                eval_metrics['true_std_accuracy'] = float(true_std_accuracy)
                eval_metrics['true_robust_accuracy'] = float(true_robust_accuracy)
                eval_metrics['mu_weights_cos_angle'] = float(mu_weights_cos_angle)
            elif isinstance(config.dataset_config, SingleGaussianDatasetConfig):
                # Calculate closed-form risks on NOISELESS test data (i.e. sigma_0 = 0.0)
                ground_truth, x_covariance_diagonal = data_aux

                # Currently closed-form evaluation is only implemented for a subset of the data model
                if not (np.all(x_covariance_diagonal == 1.0) and np.all(ground_truth[1:] == 0.0)):
                    raise NotImplementedError(
                        'Closed-form risks for single Gaussian data model are only implemented'
                        ' for sparse ground truth and identity covariance'
                    )

                # Closed-form solutions
                weight_norm_l2 = np.linalg.norm(current_weights)
                weight_norm_lq_projected = np.linalg.norm(current_weights[1:], ord=attack_q)

                # If weight norm is zero, ground truth angle is ignored
                with np.errstate(divide='ignore'):
                    ground_truth_cos_angle = np.dot(ground_truth, current_weights) / weight_norm_l2

                true_std_risk, true_robust_risk = ir.closed_form.logistic_regression_risks(
                    weight_norm_l2=weight_norm_l2,
                    ground_truth_cos_angle=ground_truth_cos_angle,
                    weight_norm_lq_projected=weight_norm_lq_projected,
                    epsilon=config.test_attack_epsilon
                )

                # Also track weight norm when projected onto subspace orthogonal to ground truth
                projected_weights = np.copy(current_weights)
                projected_weights[0] = 0.0  # this is hardcoded due to the assertion further up
                projected_weight_norm = jnp.linalg.norm(projected_weights, ord=2)
                projected_dual_weight_norm = jnp.linalg.norm(projected_weights, ord=attack_q)
                projected_dual_primal_norm_ratio = projected_dual_weight_norm / projected_weight_norm
                projected_dual_normal_primal_norm_ratio = projected_dual_weight_norm / weight_norm_l2
                eval_metrics['projected_weight_norm_l2'] = float(projected_weight_norm)
                eval_metrics['projected_weight_norm_lq'] = float(projected_dual_weight_norm)
                eval_metrics['projected_weight_lq_l2_norm_ratio'] = float(projected_dual_primal_norm_ratio)
                eval_metrics['projected_weight_lq_normal_l2_norm_ratio'] = float(projected_dual_normal_primal_norm_ratio)

                # Track alignment of gradient and weights with and ground truth
                # NB: ground truth already has norm 1
                gradient_vector = grad['params']['dense']['kernel'][:, 0]
                assert gradient_vector.shape == ground_truth.shape
                gt_gradient_cos_angle = jnp.dot(ground_truth, gradient_vector) / jnp.linalg.norm(gradient_vector)

                log_metrics += ['true_std_risk', 'true_robust_risk', 'gt_weights_cos_angle']
                eval_metrics['true_std_risk'] = float(true_std_risk)
                eval_metrics['true_robust_risk'] = float(true_robust_risk)
                eval_metrics['gt_weights_cos_angle'] = float(ground_truth_cos_angle)
                eval_metrics['gt_gradient_cos_angle'] = float(gt_gradient_cos_angle)

            # Calculate fraction of noise (flipped labels) samples fitted if available
            if num_noise_flip_samples > 0:
                log_metrics += ['fraction_std_flip_fitted', 'fraction_adv_flip_fitted']
                eval_metrics['num_std_flip_fitted'] = num_std_train_flip_correct
                eval_metrics['fraction_std_flip_fitted'] = float(num_std_train_flip_correct) / num_noise_flip_samples
                eval_metrics['num_adv_flip_fitted'] = num_adv_train_flip_correct
                eval_metrics['fraction_adv_flip_fitted'] = float(num_adv_train_flip_correct) / num_noise_flip_samples

            # Calculate fraction of noise (covariates) samples fitted if available
            if num_train_covariate_noise_samples > 0:
                log_metrics += ['fraction_std_train_noise_fitted', 'fraction_adv_train_noise_fitted']
                eval_metrics['num_std_train_noise_fitted'] = num_std_train_noise_correct
                eval_metrics['fraction_std_train_noise_fitted'] = float(num_std_train_noise_correct) / num_train_covariate_noise_samples
                eval_metrics['num_adv_train_noise_fitted'] = num_adv_train_noise_correct
                eval_metrics['fraction_adv_train_noise_fitted'] = float(num_adv_train_noise_correct) / num_train_covariate_noise_samples

            # Log weight norm and distance from initialization
            current_weight_norm = jnp.sqrt(jnp.sum(jnp.square(current_weights)))
            current_dual_weight_norm = jnp.linalg.norm(current_weights, ord=attack_q)
            current_dual_primal_norm_ratio = current_dual_weight_norm / current_weight_norm
            current_weight_init_distance = jnp.sqrt(jnp.sum(jnp.square(current_weights - initial_output_weights)))
            eval_metrics['output_weight_norm_l2'] = float(current_weight_norm)
            eval_metrics['output_weight_norm_lq'] = float(current_dual_weight_norm)
            eval_metrics['output_lq_l2_norm_ratio'] = float(current_dual_primal_norm_ratio)
            eval_metrics['output_weight_init_dist'] = float(current_weight_init_distance)

            # Margin (actual and per-sample)
            std_train_predictions = train_ys_rescaled * np.dot(train_xs, current_weights)
            margin = np.min(std_train_predictions)
            mean_margin = np.mean(std_train_predictions)
            adversarial_train_offset = config.test_attack_epsilon * current_dual_weight_norm
            adversarial_train_predictions = std_train_predictions - adversarial_train_offset
            robust_margin = np.min(adversarial_train_predictions)
            mean_robust_margin = np.mean(adversarial_train_predictions)
            eval_metrics['std_margin'] = float(margin)
            eval_metrics['std_margin_normalized'] = float(margin / current_weight_norm)
            eval_metrics['std_mean_margin'] = float(mean_margin)
            eval_metrics['std_mean_margin_normalized'] = float(mean_margin / current_weight_norm)
            eval_metrics['robust_margin'] = float(robust_margin)
            eval_metrics['robust_margin_normalized'] = float(robust_margin / current_weight_norm)
            eval_metrics['robust_mean_margin'] = float(mean_robust_margin)
            eval_metrics['robust_mean_margin_normalized'] = float(mean_robust_margin / current_weight_norm)

            # Log weight sparsity
            current_weight_sparsity = jnp.linalg.norm(current_weights, ord=0)
            eval_metrics['output_weight_nonzero_coefficients'] = float(current_weight_sparsity)

            metric_logger.add_metrics(eval_metrics, step=epoch)
            log.info(
                'Epoch %04d: loss=%.4f, %s',
                epoch + 1,
                float(current_loss),
                ', '.join('{0}={1:.4f}'.format(metric_key, eval_metrics[metric_key]) for metric_key in log_metrics)
            )

        # Checks automatically whether metrics should be submitted this epoch or not
        metric_logger.submit(epoch)

    log.info('Finished training')


def eval_dataset(
        xs: jnp.ndarray,
        ys_scaled: jnp.ndarray,
        weights: jnp.ndarray,
        flip_indices: jnp.ndarray,
        covariate_noise_indices: jnp.ndarray,
        epsilon: float,
        attack_q: typing.Union[float, int, str],
        clamp_range: typing.Optional[typing.Tuple[float, float]]
) -> typing.Tuple[int, int, int, int, int, int]:
    # NB: Expects ys in {-1, +1}, not {0, 1}

    assert weights.ndim == 1
    assert ys_scaled.ndim == 1

    # Calculate standard accuracy
    natural_raw_predictions = jnp.multiply(ys_scaled, jnp.dot(xs, weights))
    std_correct = (natural_raw_predictions > 0)
    num_std_correct = int(jnp.sum(jnp.int32(std_correct)))

    # Calculate robust accuracy, only necessary to use correctly classified samples
    adversarial_raw_predictions = ir.attacks.linear_closed_form_attack(
        xs,
        ys_scaled,
        weights,
        epsilon,
        attack_q,
        clamp_range
    )
    adv_correct = (adversarial_raw_predictions > 0)
    num_adv_correct = int(jnp.sum(jnp.int32(adv_correct)))

    if flip_indices.shape[0] > 0:
        num_std_flip_correct = int(jnp.sum(jnp.int32(std_correct[flip_indices])))
        num_adv_flip_correct = int(jnp.sum(jnp.int32(adv_correct[flip_indices])))
    else:
        num_std_flip_correct = 0
        num_adv_flip_correct = 0

    if covariate_noise_indices.shape[0] > 0:
        noise_mask = jnp.zeros_like(std_correct)
        noise_mask = jax.ops.index_update(
            x=noise_mask, idx=covariate_noise_indices, y=True, indices_are_sorted=True, unique_indices=True
        )
        std_correct_noise = noise_mask[std_correct]
        num_std_noise_correct = int(jnp.sum(jnp.int32(std_correct[covariate_noise_indices])))
        num_adv_noise_correct = int(jnp.sum(jnp.int32(adv_correct[std_correct_noise])))
    else:
        num_std_noise_correct = 0
        num_adv_noise_correct = 0

    return num_std_correct, num_adv_correct, num_std_flip_correct, num_adv_flip_correct, num_std_noise_correct, num_adv_noise_correct


def build_config(args) -> ExperimentConfig:
    # Determine attack epsilons, depending on arguments configuration
    test_attack_epsilon = args.attack_epsilon
    train_attack_epsilon = args.train_attack_epsilon if args.train_attack_epsilon is not None else test_attack_epsilon

    # Parse perturbation radius
    attack_p = 'inf' if args.attack_p == 'inf' else float(args.attack_p)

    if args.attack_train_consistent and args.dataset not in ('single_gauss', '2gmm'):
        raise ValueError('Consistent training attacks are only supported for datasets with known ground truth')

    # Determine learning rate decay epochs
    if args.learning_rate_decay_epochs is None:
        learning_rate_decay_epochs = tuple()
    else:
        learning_rate_decay_epochs = tuple(args.learning_rate_decay_epochs)

    # Massive hack, fix this as soon as time permits
    if args.dataset == '2gmm':
        one_dim = args.gmm_mean_singledim
        if attack_p == 'inf':
            scale_mean = False
            sigma = 1.0 if one_dim else 1.0 * float(jnp.sqrt(args.data_dim))
        elif attack_p == 2:
            scale_mean = True
            sigma = 1.0
        else:
            raise NotImplementedError('GMM only supported for l_inf and l_2 attacks')

        dataset_config = GMMDatasetConfig(
            sigma=sigma,
            one_dim=one_dim,
            scale_mean=scale_mean
        )
    elif args.dataset == 'mnist':
        if attack_p != 'inf':
            raise ValueError('Since MNIST requires clipping to a hypercube, only l_inf perturbations are supported')

        if args.mnist_class_0 is None or args.mnist_class_1 is None:
            raise ValueError('Using a binary MNIST dataset requires specification of the two classes')

        dataset_config = BinaryMNISTDatasetConfig(
            class_0=int(args.mnist_class_0),
            class_1=int(args.mnist_class_1)
        )
    elif args.dataset == 'single_gauss':
        if args.singlegauss_ground_truth is None:
            raise ValueError('Ground truth needs to be specified for single Gaussian data model')
        assert args.singlegauss_ground_truth in ('single_entry', 'scaled_full')
        single_entry_ground_truth = args.singlegauss_ground_truth == 'single_entry'

        if args.singlegauss_covariance is None:
            raise ValueError('Covariance diagonal needs to be specified for single Gaussian data model')
        assert args.singlegauss_covariance in ('identity', 'logspace')
        logspace_covariance = args.singlegauss_covariance == 'logspace'

        dataset_config = SingleGaussianDatasetConfig(
            single_entry_ground_truth=single_entry_ground_truth,
            logspace_covariance=logspace_covariance,
            logits_noise_variance=args.singlegauss_logits_noise_variance
        )
    else:
        assert False

    return ExperimentConfig(
        epochs=args.epochs,
        eval_every_epochs=args.eval_every_epochs,
        save_every_epochs=args.save_every_epochs,
        log_every_epochs=args.log_every_epochs,
        initial_learning_rate=args.learning_rate,
        learning_rate_decay_epochs=learning_rate_decay_epochs,
        learning_rate_decay_step=args.learning_rate_decay_step,
        learning_rate_warmup=args.learning_rate_warmup,
        momentum=args.momentum,
        nesterov=args.nesterov,
        label_noise=args.label_noise,
        adversarial_training=args.adversarial_training,
        train_consistent_attacks=args.attack_train_consistent,
        train_attack_epsilon=train_attack_epsilon,
        test_attack_epsilon=test_attack_epsilon,
        attack_p=attack_p,
        data_num_train_samples=args.training_samples,
        data_dim=args.data_dim,
        dataset_config=dataset_config,
        root_log_dir=args.logdir,
        l2_lambda=args.l2,
        l1_lambda=args.l1
    )


def make_dataset(config: ExperimentConfig, rng_key: jnp.ndarray, logger: DataLogger) -> typing.Tuple[
    typing.Tuple[jnp.ndarray, jnp.ndarray],  # training data
    typing.Tuple[jnp.ndarray, jnp.ndarray],  # test data
    jnp.ndarray,  # label noise indices
    typing.Tuple[jnp.ndarray, jnp.ndarray],  # covariate noise indices (on train and test)
    typing.Optional[typing.Any]
]:
    rng_key, seed_key = jax.random.split(rng_key)
    seed = int(jax.random.randint(seed_key, (), 0, jnp.iinfo(jnp.int32).max))

    if isinstance(config.dataset_config, GMMDatasetConfig):
        if config.dataset_config.one_dim:
            mu = np.zeros(config.data_dim)
            mu[0] = 8.0
        else:
            mu = 8.0 * np.ones(config.data_dim)
            if config.dataset_config.scale_mean:
                mu = mu / np.sqrt(config.data_dim)

        (train_xs, train_ys), (test_xs, test_ys), train_label_noise, covariate_noise = ir.data.make_2gmm(
            mu=mu,
            sigma=config.dataset_config.sigma,
            train_samples=config.data_num_train_samples,
            test_samples=0,
            train_label_noise=config.label_noise,
            seed=seed
        )
        aux = (mu, config.dataset_config.sigma)

        # Make sure covariate noise is sorted
        covariate_noise = (jnp.sort(covariate_noise[0]), jnp.sort(covariate_noise[1]))

        logger.log_2gmm_dataset(
            train_xs,
            train_ys,
            test_xs,
            test_ys,
            train_label_noise,
            covariate_noise[0],
            covariate_noise[1],
            mu,
            config.dataset_config.sigma
        )
    elif isinstance(config.dataset_config, SingleGaussianDatasetConfig):
        if config.dataset_config.logspace_covariance:
            x_covariance_diagonal = np.logspace(0, -3, num=config.data_dim)
        else:
            x_covariance_diagonal = np.ones((config.data_dim,))

        if config.dataset_config.single_entry_ground_truth:
            ground_truth = np.zeros_like(x_covariance_diagonal)
            ground_truth[0] = 1.0
        else:
            ground_truth = (1.0 / np.sqrt(config.data_dim)) * np.ones_like(x_covariance_diagonal)

        (train_xs, train_ys), (test_xs, test_ys), train_label_noise, covariate_noise = ir.data.make_gaussian_logistic(
            train_samples=config.data_num_train_samples,
            test_samples=0,
            x_covariance_diagonal=x_covariance_diagonal,
            logits_noise_variance=config.dataset_config.logits_noise_variance,
            ground_truth=ground_truth,
            label_noise_fraction=config.label_noise,
            min_decision_boundary_distance=0.0,  # don't throw away samples which are too close to the db
            seed=seed
        )

        aux = (ground_truth, x_covariance_diagonal)

        logger.log_single_gaussian_dataset(
            train_xs,
            train_ys,
            test_xs,
            test_ys,
            train_label_noise,
            covariate_noise[0],
            covariate_noise[1],
            ground_truth,
            x_covariance_diagonal,
            config.dataset_config.logits_noise_variance
        )
    elif isinstance(config.dataset_config, BinaryMNISTDatasetConfig):
        (train_xs, train_ys), (test_xs, test_ys), train_label_noise = ir.data.make_image_dataset(
            dataset=ir.data.Dataset.MNIST,
            train_samples=config.data_num_train_samples,
            binarized_classes=(config.dataset_config.class_0, config.dataset_config.class_1),
            train_label_noise=config.label_noise,
            seed=seed
        )

        # Flatten image data
        train_xs = np.reshape(train_xs, (train_xs.shape[0], -1))
        test_xs = np.reshape(test_xs, (test_xs.shape[0], -1))

        covariate_noise = (jnp.zeros((0,)), jnp.zeros((0,)))
        aux = None
    else:
        assert False

    # Make sure data is on the correct device (e.g. GPU) and handled by JAX
    train_xs, train_ys = jax.device_put(train_xs), jax.device_put(train_ys)
    test_xs, test_ys = jax.device_put(test_xs), jax.device_put(test_ys)

    return (train_xs, train_ys), (test_xs, test_ys), train_label_noise, covariate_noise, aux


@contextlib.contextmanager
def setup_experiment(config: ExperimentConfig, experiment_name: str):
    run_name = '{timestamp:%Y%m%d-%H%M%S}_{training_method}_dn{overparam_ratio}'.format(
        timestamp=datetime.datetime.utcnow(),
        training_method='at' if config.adversarial_training else 'st',
        overparam_ratio=float(config.data_dim) / config.data_num_train_samples
    )

    ir.mlflow.set_experiment(experiment_name)
    with ir.mlflow.start_run(run_name=run_name) as run:
        ir.mlflow.set_tag('base_experiment', EXPERIMENT_TAG)
        ir.mlflow.log_params(config._asdict())
        yield run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument('--adversarial-training', action='store_true', help='Use adversarial training')
    parser.add_argument(
        '--attack-train-consistent', action='store_true',
        help='Use consistent attacks during training (only applicable in some settings)'
    )
    parser.add_argument('--attack-epsilon', type=float, required=True, help='Radius of adversarial perturbation ball')
    parser.add_argument(
        '--train-attack-epsilon', type=float,
        help='Radius of adversarial perturbation ball for training if different from the one for testing'
    )
    parser.add_argument('--attack-p', choices=('2', 'inf'), type=str, default='inf', help='p of epsilon-ball norm for adversarial attacks')
    parser.add_argument('--epochs', type=int, default=500000, help='Number of training epochs')
    parser.add_argument('--eval-every-epochs', type=int, default=100, help='How many epochs happen between model evaluations')
    parser.add_argument('--save-every-epochs', type=int, default=1000, help='How many epochs happen between weight saving')
    parser.add_argument('--log-every-epochs', type=int, default=10000, help='How many epochs happen between metric logging')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Target learning rate to use for training')
    parser.add_argument('--learning-rate-warmup', type=int, default=0, help='Number of epochs to linearly interpolate from 0 to target learning rate')
    parser.add_argument('--learning-rate-decay-epochs', type=int, nargs='*', help='Decay learning rate at those epochs')
    parser.add_argument('--learning-rate-decay-step', type=float, default=1.0, help='Amount the learning rate is decayed at each decay step')
    parser.add_argument('--momentum', type=float, default=0.0, help='Optimizer momentum for training')
    parser.add_argument('--nesterov', action='store_true', help='Use nesterov momentum for training')
    parser.add_argument('--data-dim', type=int, default=8000, help='Dimensionality of the data for synthetic datasets')
    parser.add_argument('--training-samples', type=int, default=1000,
        help='Number of training samples. Synthetic datasets generate this many samples, real-world datasets are subsampled')
    parser.add_argument('--label-noise', type=float, default=0.0, help='Label flip probability')
    parser.add_argument('--dataset', type=str, required=True, choices=('2gmm', 'mnist', 'single_gauss'), help='Dataset to use')
    parser.add_argument('--mnist-class-0', type=str, help='First class for binary MNIST')
    parser.add_argument('--mnist-class-1', type=str, help='Second class for binary MNIST')
    parser.add_argument('--gmm-mean-singledim', action='store_true', help='Use mu = (1, 0, ..., 0) for GMM')
    parser.add_argument(
        '--singlegauss-ground-truth', type=str,
        choices='(single_entry, scaled_full)',
        help='Which ground truth to use for single Gaussian data model'
    )
    parser.add_argument(
        '--singlegauss-covariance', type=str,
        choices='(identity, logspace)',
        help='Which x covariance diagonal to use for single Gaussian data model'
    )
    parser.add_argument(
        '--singlegauss-logits-noise-variance', type=float, default=0.0,
        help='Which logit noise variance to use for single Gaussian data model'
    )
    parser.add_argument('--logdir', type=str, help='If set, weights and the dataset are stored in {logdir}/{run_id}/')
    parser.add_argument(
        '--tag', type=str, required=True, help='Name of the current set of experiment runs, used for grouping'
    )
    parser.add_argument(
        '--l2', type=float, default=0.0,  help='L2 penalty weight'
    )
    parser.add_argument(
        '--l1', type=float, default=0.0,  help='L1 penalty weight'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
