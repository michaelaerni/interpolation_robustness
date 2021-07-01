import argparse
import contextlib
import datetime
import enum
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

EXPERIMENT_TAG = 'linear_regression'


@enum.unique
class GroundTruth(enum.Enum):
    SingleEntry = 'single_entry'
    ScaledFull = 'scaled_full'


@enum.unique
class CovarianceDiagonal(enum.Enum):
    Identity = 'identity'
    Logspace = 'logspace'


class ExperimentConfig(typing.NamedTuple):
    epochs: int
    eval_every_epochs: int
    save_every_epochs: int
    log_every_epochs: int
    learning_rate: float
    learning_rate_warmup: int
    noise_fraction: float
    adversarial_training: bool
    train_consistent_attacks: bool
    train_attack_epsilon: float
    test_attack_epsilon: float
    attack_p: typing.Union[float, str]
    num_train_samples: int
    data_dim: int
    data_ground_truth: GroundTruth
    data_covariance_diagonal: CovarianceDiagonal
    data_gaussian_noise_variance: float
    data_rademacher_noise_scale: float
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

    def log_dataset(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            rademacher_noise_indices: np.ndarray,
            x_covariance_diagonal: np.ndarray,
            y_noise_variance: float,
            radmeacher_noise_fraction: float,
            rademacher_noise_scale: float,
            ground_truth: np.ndarray
    ):
        if self._log_dir is None:
            return

        target_file = os.path.join(self._log_dir, 'dataset.npz')
        np.savez_compressed(
            target_file,
            xs=xs,
            ys=ys,
            rademacher_noise_indices=rademacher_noise_indices,
            x_covariance_diagonal=x_covariance_diagonal,
            y_noise_variance=np.asarray((y_noise_variance,)),
            radmeacher_noise_fraction=np.asarray((radmeacher_noise_fraction,)),
            rademacher_noise_scale=np.asarray((rademacher_noise_scale,)),
            ground_truth=ground_truth
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

    # Prepare weight and metric logging
    log_dir = os.path.join(config.root_log_dir, run_id) if config.root_log_dir is not None else None
    data_logger = DataLogger(log_dir, config.save_every_epochs, config.epochs)
    data_logger.prepare()
    metric_logger = ir.mlflow.DelayedMetricLogger(run_id, config.log_every_epochs, config.epochs)

    # Build dataset
    rng_key, dataset_key = jax.random.split(rng_key)
    train_xs, train_ys, train_sample_noise_indices, x_covariance_diagonal, ground_truth, y_noise_variance = make_dataset(
        config,
        dataset_key,
        data_logger
    )
    train_ys = jnp.expand_dims(train_ys, axis=-1)
    num_train_samples, data_dim = train_xs.shape
    num_noise_samples, = train_sample_noise_indices.shape
    log.info(
        'Built data set with %d training samples (%d with Rademacher noise)',
        num_train_samples,
        num_noise_samples
    )

    # Build model
    model = ir.models.jax.LinearModel(
        num_out=1,
        weight_init=flax.linen.initializers.zeros
    )
    rng_key, init_key = jax.random.split(rng_key)
    params = model.init(init_key, jnp.ones((1, data_dim)))
    log.info('Built and initialized model')

    # Get initial weights for tracking later
    def get_weights(current_params: flax.core.FrozenDict) -> jnp.ndarray:
        path = ('params', 'dense', 'kernel')
        flattened_params_dict = flax.traverse_util.flatten_dict(flax.core.unfreeze(current_params))
        return jnp.squeeze(flattened_params_dict[path])
    initial_weights = get_weights(params)

    # Build optimizer
    optimizer_def = flax.optim.GradientDescent(learning_rate=config.learning_rate)
    optimizer = optimizer_def.create(params)

    # Build learning rate schedule
    lr_schedule_fn = flax.training.lr_schedule.create_stepped_learning_rate_schedule(
        base_learning_rate=config.learning_rate,
        steps_per_epoch=1,  # full-batch gradient descent
        warmup_length=config.learning_rate_warmup,  # passing 0 (the default) disables warmup,
        lr_sched_steps=[]
    )

    # Setup adversarial training if enabled
    if config.attack_p in ('inf', float('inf')):
        attack_q = 1
    elif config.attack_p == 1.0:
        attack_q = jnp.inf
    else:
        assert config.attack_p > 1.0
        attack_q = config.attack_p / (config.attack_p - 1.0)
    if config.train_consistent_attacks:
        consistent_attack_gt = ground_truth / jnp.linalg.norm(ground_truth)
    else:
        consistent_attack_gt = None
    if config.adversarial_training:
        def train_loss_fn(model_params):
            predictions = model.apply(model_params, train_xs)
            weights = get_weights(model_params)
            loss = ir.losses.closed_form_adversarial_quadratic_loss(
                predictions=predictions,
                ys=train_ys,
                weights=weights,
                epsilon=config.train_attack_epsilon,
                attack_q=attack_q,
                consistent_attack_gt=consistent_attack_gt
            )
            if config.l2_lambda > 0:
                loss = loss + config.l2_lambda * jnp.sum(jnp.square(weights))
            if config.l1_lambda > 0:
                loss = loss + config.l1_lambda * jnp.sum(jnp.abs(weights))
            return loss
    else:
        def train_loss_fn(model_params):
            predictions = model.apply(model_params, train_xs)
            loss = ir.losses.quadratic_loss(predictions, train_ys)
            weights = get_weights(model_params)
            if config.l2_lambda > 0:
                loss = loss + config.l2_lambda * jnp.sum(jnp.square(weights))
            if config.l1_lambda > 0:
                loss = loss + config.l1_lambda * jnp.sum(jnp.abs(weights))
            return loss

    grad_fn = jax.value_and_grad(train_loss_fn, has_aux=False)
    # For very large data dimensions, jit-ing the gradient function leads to OOM issues
    if config.data_dim <= 60000:
        grad_fn = jax.jit(grad_fn)

    # Calculate loss for theta=0, useful for normalization later
    zero_weights_loss = 0.5 * jnp.mean(jnp.square(train_ys))
    zero_weights_risk_empirical = 2.0 * zero_weights_loss
    zero_weights_risk_population = (
        jnp.sum(x_covariance_diagonal * jnp.square(ground_truth))
    )

    log.info('Starting training')
    for epoch in range(config.epochs):
        # Full batch GD
        # Adversarial attack (if enabled) happens in closed-form in training loss function directly
        current_loss, grad = grad_fn(optimizer.target)
        current_lr = lr_schedule_fn(optimizer.state.step)
        optimizer = optimizer.apply_gradient(grad, learning_rate=current_lr)

        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs - 1:
            metric_logger.add_metrics({
                    'training_loss': float(current_loss),
                    'training_loss_normalized': float(current_loss / zero_weights_loss)
                },
                step=epoch
            )

        # Model saving, automatically checks whether it should save or not
        current_weights = get_weights(optimizer.target)
        data_logger.log_weights(current_weights, epoch)

        # Evaluation
        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs - 1:
            eval_metrics = dict()
            log_metrics = []

            # Calculate empirical standard risk on the training data
            current_predictions = model.apply(optimizer.target, train_xs)[:, 0]
            current_residuals = train_ys[:, 0] - current_predictions
            train_std_risk = jnp.mean(jnp.square(current_residuals))

            # Calculate empirical robust risk on the training data
            squared_weight_norm = jnp.sum(jnp.square(current_weights))
            weight_norm = jnp.sqrt(squared_weight_norm)
            weight_norm_lq = jnp.linalg.norm(current_weights, ord=attack_q)
            # NB: Like the training loss, this uses inconsistent perturbations and the training epsilon
            train_robust_risk = jnp.mean(
                np.square(jnp.abs(current_residuals) + config.train_attack_epsilon * weight_norm_lq)
            )
            train_robust_risk_test_attack = jnp.mean(
                jnp.square(jnp.abs(current_residuals) + config.test_attack_epsilon * weight_norm_lq)
            )

            log_metrics += ['train_std_risk', 'train_robust_risk', 'train_robust_risk_test_attack']
            eval_metrics['train_std_risk'] = float(train_std_risk)
            eval_metrics['train_std_risk_normalized'] = float(train_std_risk / zero_weights_risk_empirical)
            eval_metrics['train_robust_risk'] = float(train_robust_risk)
            eval_metrics['train_robust_risk_normalized'] = float(train_robust_risk / zero_weights_risk_empirical)
            eval_metrics['train_robust_risk_test_attack'] = float(train_robust_risk_test_attack)

            # Calculate closed-form standard risk
            weighted_normalized_param_distance = jnp.sum(
                x_covariance_diagonal * jnp.square(current_weights - ground_truth)
            )
            true_std_risk = weighted_normalized_param_distance

            # Calculate closed-form adversarial risk (consistent and inconsistent)
            squared_weight_norm_lq = jnp.square(weight_norm_lq)
            projected_weights = current_weights \
                - jnp.dot(ground_truth, current_weights) / jnp.sum(jnp.square(ground_truth)) * ground_truth
            projected_weight_norm_lq = jnp.linalg.norm(projected_weights, ord=attack_q)
            projected_squared_weight_norm_lq = jnp.square(projected_weight_norm_lq)
            true_robust_risk = true_std_risk \
                + jnp.square(config.test_attack_epsilon) * projected_squared_weight_norm_lq \
                + 2.0 * jnp.sqrt(2.0 / jnp.pi) \
                * config.test_attack_epsilon * projected_weight_norm_lq * jnp.sqrt(true_std_risk)
            true_robust_risk_inconsistent = true_std_risk \
                + jnp.square(config.test_attack_epsilon) * squared_weight_norm_lq \
                + 2.0 * jnp.sqrt(2.0 / jnp.pi) \
                * config.test_attack_epsilon * weight_norm_lq * jnp.sqrt(true_std_risk)

            log_metrics += ['true_std_risk', 'true_robust_risk']
            eval_metrics['true_std_risk'] = float(true_std_risk)
            eval_metrics['true_std_risk_normalized'] = float(true_std_risk / zero_weights_risk_population)
            eval_metrics['true_robust_risk'] = float(true_robust_risk)
            eval_metrics['true_robust_risk_normalized'] = float(true_robust_risk / zero_weights_risk_population)
            eval_metrics['true_robust_risk_inconsistent'] = float(true_robust_risk_inconsistent)
            eval_metrics['true_robust_risk_inconsistent_normalized'] = float(true_robust_risk_inconsistent / zero_weights_risk_population)

            # Log weight norm and distance from initialization
            current_weight_init_distance = jnp.linalg.norm(current_weights - initial_weights)
            eval_metrics['output_weight_norm_l2'] = float(weight_norm)
            eval_metrics['output_weight_norm_lq'] = float(weight_norm_lq)
            eval_metrics['output_weight_norm_proj_lq'] = float(projected_weight_norm_lq)
            eval_metrics['output_weight_init_dist_l2'] = float(current_weight_init_distance)

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


def build_config(args) -> ExperimentConfig:
    # Determine attack epsilons, depending on arguments configuration
    test_attack_epsilon = args.attack_epsilon
    train_attack_epsilon = args.train_attack_epsilon if args.train_attack_epsilon is not None else test_attack_epsilon
    attack_p = 'inf' if args.attack_p == 'inf' else float(args.attack_p)

    return ExperimentConfig(
        epochs=args.epochs,
        eval_every_epochs=args.eval_every_epochs,
        save_every_epochs=args.save_every_epochs,
        log_every_epochs=args.log_every_epochs,
        learning_rate=args.learning_rate,
        learning_rate_warmup=args.learning_rate_warmup,
        noise_fraction=args.noise_fraction,
        adversarial_training=args.adversarial_training,
        train_consistent_attacks=args.attack_train_consistent,
        train_attack_epsilon=train_attack_epsilon,
        test_attack_epsilon=test_attack_epsilon,
        attack_p=attack_p,
        num_train_samples=args.training_samples,
        data_dim=args.data_dim,
        data_ground_truth=GroundTruth(args.data_ground_truth),
        data_covariance_diagonal=CovarianceDiagonal(args.data_covariance_diagonal),
        data_gaussian_noise_variance=args.data_noise_variance,
        data_rademacher_noise_scale=0.0,  # no noise
        root_log_dir=args.logdir,
        l2_lambda=args.l2,
        l1_lambda=args.l1
    )


def make_dataset(config: ExperimentConfig, rng_key: jnp.ndarray, logger: DataLogger) -> typing.Tuple[
    jnp.ndarray,  # training xs
    jnp.ndarray,  # training ys
    jnp.ndarray,  # noise training sample indices
    jnp.ndarray,  # sample covariance diagonal
    jnp.ndarray,  # ground truth
    float  # i.i.d. Gaussian noise variance
]:
    rng_key, seed_key = jax.random.split(rng_key)
    seed = int(jax.random.randint(seed_key, (), 0, jnp.iinfo(jnp.int32).max))

    # All the parameters of the data model are specified here.

    # Diagonal entries of the covariance matrix for the x_i (\Sigma).
    # The dimensionality of the data is automatically determined by the dimension of this vector.
    if config.data_covariance_diagonal == CovarianceDiagonal.Identity:
        x_covariance_diagonal = np.ones((config.data_dim,))
    elif config.data_covariance_diagonal == CovarianceDiagonal.Logspace:
        x_covariance_diagonal = np.logspace(0, -3, num=config.data_dim)
    else:
        assert False

    # Variance of the i.i.d. Gaussian noise added to y_i (\sigma_0^2) here.
    y_noise_variance = config.data_gaussian_noise_variance

    # Scale of the Rademacher noise (\gamma) here
    rademacher_noise_scale = config.data_rademacher_noise_scale

    # Ground truth (\theta_0), dimension must match the one of the covariance diagonal vector.
    if config.data_ground_truth == GroundTruth.SingleEntry:
        ground_truth = np.zeros_like(x_covariance_diagonal)
        ground_truth[0] = 1.0
    elif config.data_ground_truth == GroundTruth.ScaledFull:
        ground_truth = (1.0 / np.sqrt(config.data_dim)) * np.ones_like(x_covariance_diagonal)
    else:
        assert False

    assert x_covariance_diagonal.shape == ground_truth.shape

    train_xs, train_ys, train_sample_noise_indices = ir.data.make_gaussian_linear(
        num_samples=config.num_train_samples,
        x_covariance_diagonal=x_covariance_diagonal,
        y_noise_variance=y_noise_variance,
        ground_truth=ground_truth,
        rademacher_noise_fraction=config.noise_fraction,
        rademacher_noise_scale=rademacher_noise_scale,
        seed=seed
    )

    # Make sure noise indices are sorted!
    train_sample_noise_indices = jnp.sort(train_sample_noise_indices)

    # Log dataset
    logger.log_dataset(
        train_xs,
        train_ys,
        train_sample_noise_indices,
        x_covariance_diagonal,
        y_noise_variance,
        config.noise_fraction,
        rademacher_noise_scale,
        ground_truth
    )

    # Make sure data is on the correct device (e.g. GPU) and handled by JAX
    train_xs, train_ys = jax.device_put(train_xs), jax.device_put(train_ys)
    train_sample_noise_indices = jax.device_put(train_sample_noise_indices)
    x_covariance_diagonal = jax.device_put(x_covariance_diagonal)
    ground_truth = jax.device_put(ground_truth)

    return train_xs, train_ys, train_sample_noise_indices, x_covariance_diagonal, ground_truth, y_noise_variance


@contextlib.contextmanager
def setup_experiment(config: ExperimentConfig, experiment_name: str):
    run_name = '{timestamp:%Y%m%d-%H%M%S}_{training_method}_n{num_samples}_d{data_dim}'.format(
        timestamp=datetime.datetime.utcnow(),
        experiment_name=experiment_name,
        training_method='at' if config.adversarial_training else 'st',
        num_samples=config.num_train_samples,
        data_dim=config.data_dim
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
        help='Use consistent attacks during training'
    )
    parser.add_argument('--attack-epsilon', type=float, default=0.4, help='Radius of adversarial perturbation ball')
    parser.add_argument(
        '--train-attack-epsilon', type=float,
        help='Radius of adversarial perturbation ball for training if different from the one for testing'
    )
    parser.add_argument('--attack-p', choices=('2', 'inf'), type=str, default='2', help='p of epsilon-ball norm for adversarial attacks')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--eval-every-epochs', type=int, default=1, help='How many epochs happen between model evaluations')
    parser.add_argument('--save-every-epochs', type=int, default=1, help='How many epochs happen between weight saving')
    parser.add_argument('--log-every-epochs', type=int, default=1, help='How many epochs happen between metric logging')
    parser.add_argument('--learning-rate', type=float, required=True, help='Target learning rate to use for training')
    parser.add_argument('--learning-rate-warmup', type=int, default=250, help='Number of epochs to linearly interpolate from 0 to target learning rate')
    parser.add_argument('--training-samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--noise-fraction', type=float, default=0.0, help='Fraction of samples to which Rademacher noise is added')
    parser.add_argument('--data-dim', type=int, required=True, help='Dimensionality of the data')
    parser.add_argument(
        '--data-ground-truth', type=str, default=GroundTruth.SingleEntry.value,
        choices=[ground_truth.value for ground_truth in GroundTruth],
        help='Ground truth to use'
    )
    parser.add_argument(
        '--data-covariance-diagonal', type=str, default=CovarianceDiagonal.Identity.value,
        choices=[covariance_diagonal.value for covariance_diagonal in CovarianceDiagonal],
        help='Covariance diagonal stucture to use for training sample generation'
    )
    parser.add_argument('--data-noise-variance', type=float, default=0.0, help='Variance of iid Gaussian noise added to targets')
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
