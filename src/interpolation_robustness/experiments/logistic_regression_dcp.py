import argparse
import contextlib
import datetime
import logging
import os
import typing

import cvxpy as cp
import dotenv
import numpy as np

import interpolation_robustness as ir

EXPERIMENT_TAG = 'logistic_regression_dcp'


class ExperimentConfig(typing.NamedTuple):
    label_noise: float
    train_consistent_attacks: bool
    train_attack_epsilon: float
    test_attack_epsilon: float
    attack_p: typing.Union[float, str]
    data_num_train_samples: int
    data_dim: int
    data_logits_noise_variance: float
    data_min_decision_boundary_distance: float
    root_log_dir: typing.Optional[str]
    solvers: typing.Tuple[str]
    verbose: bool
    l2_lambdas: typing.Tuple[float, ...]
    l1_lambdas: typing.Tuple[float, ...]
    seed: int


class DataLogger(object):
    def __init__(self, log_dir: typing.Optional[str]):
        self._log_dir = log_dir
        self._log = logging.getLogger(__name__)

    def prepare(self):
        if self._log_dir is None:
            return

        self._log.info('Using log directory %s', self._log_dir)
        os.makedirs(self._log_dir, exist_ok=True)

    def log_weights(self, weights: np.ndarray, run_id: str):
        if self._log_dir is None:
            return

        assert weights.ndim == 1

        target_file = os.path.join(self._log_dir, f'weights_{run_id}.npy')
        np.save(target_file, np.asarray(weights))

    def log_single_gaussian_dataset(
            self,
            train_xs: np.ndarray,
            train_ys: np.ndarray,
            train_flip_indices: np.ndarray,
            train_noise_indices: np.ndarray,
            ground_truth: np.ndarray,
            x_covariance_diagonal: np.ndarray,
            logits_noise_variance: float
    ):
        if self._log_dir is None:
            return

        num_samples, data_dim = train_xs.shape
        target_file = os.path.join(self._log_dir, f'dataset_d{data_dim}_n{num_samples}.npz')
        np.savez_compressed(
            target_file,
            train_xs=train_xs,
            train_ys=train_ys,
            train_flip_indices=train_flip_indices,
            train_noise_indices=train_noise_indices,
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

    ir.mlflow.set_experiment(args.tag)
    run_experiment(config)


def run_experiment(config: ExperimentConfig):
    rng = np.random.default_rng(seed=config.seed)
    log = logging.getLogger(__name__)

    log.info('Starting experiment with config %s', config._asdict())

    # Prepare weight etc logging
    data_logger = DataLogger(config.root_log_dir)
    data_logger.prepare()

    # Build dataset
    (train_xs, train_ys), train_flip_indices, (ground_truth, x_covariance_diagonal) = make_dataset(
        config,
        rng,
        data_logger
    )
    assert np.all(x_covariance_diagonal == 1.0), 'Closed-form evaluation only implemented for identity covariance'
    train_ys_rescaled = np.where(train_ys == 1, train_ys, -np.ones_like(train_ys))  # {-1, +1} instead of {0, 1}
    num_train_samples, data_dim = train_xs.shape
    num_noise_flip_samples, = train_flip_indices.shape
    log.info(
        'Built data set with %d training samples (%d with flipped label)',
        num_train_samples,
        num_noise_flip_samples
    )

    # Prepare various things for adversarial attacks
    if config.attack_p in ('inf', float('inf')):
        attack_q = 1
    elif config.attack_p == 1.0:
        attack_q = np.inf
    else:
        assert config.attack_p > 1.0
        attack_q = config.attack_p / (config.attack_p - 1.0)

    assert np.allclose(np.linalg.norm(ground_truth), 1.0)

    # Build DCP instance
    weight_var = cp.Variable(data_dim)
    l2_lambda_param = cp.Parameter(nonneg=True)
    l1_lambda_param = cp.Parameter(nonneg=True)
    standard_loss_input = cp.multiply(train_ys_rescaled, train_xs @ weight_var)
    # For train epsilon 0 we do ST, hence can directly use the standard loss input
    if config.train_attack_epsilon == 0:
        loss_inputs = standard_loss_input
    else:
        if config.train_consistent_attacks:
            # TODO: Generally we would have to project the weights onto the space orthogonal to the ground truth.
            #  However, since there is only one hardcoded ground truth at this time, simply exploiting that
            #  fact is faster and leads to less other issues.
            #  The projection of x onto the space orthogonal to the hardcoded ground truth is obtained by setting x[0] = 0.
            assert np.all(x_covariance_diagonal == 1.0) and np.all(ground_truth[1:] == 0.0)
            perturbation_weight_norm = cp.norm(weight_var[1:], p=attack_q)
        else:
            perturbation_weight_norm = cp.norm(weight_var, p=attack_q)
        loss_inputs = standard_loss_input - config.train_attack_epsilon * perturbation_weight_norm
    at_log_loss = cp.sum(cp.logistic(-loss_inputs)) / num_train_samples
    objective = at_log_loss
    if any(l2_lambda > 0 for l2_lambda in config.l2_lambdas):
        objective = objective + l2_lambda_param * cp.sum_squares(weight_var)
    if any(l1_lambda > 0 for l1_lambda in config.l1_lambdas):
        objective = objective + l1_lambda_param * cp.sum(cp.abs(weight_var))

    problem = cp.Problem(cp.Minimize(objective))
    log.info('Built DCP instance. Validation: DCP=%s, DPP=%s', problem.is_dcp(dpp=False), problem.is_dcp(dpp=True))

    # The problem is DPP, meaning that only changing the parameter value might result
    # in much quicker optimization runs.
    assert problem.is_dcp(dpp=True)

    for l2_lambda in config.l2_lambdas:
        for l1_lambda in config.l1_lambdas:
            try:
                with setup_run(config, l2_lambda, l1_lambda) as run:
                    log.info(
                        'Evaluating L2 lambda %f, L1 lambda %f, MLFlow run id is %s',
                        l2_lambda,
                        l1_lambda,
                        run.info.run_uuid
                    )
                    l2_lambda_param.value = l2_lambda
                    l1_lambda_param.value = l1_lambda
                    run_single(
                        config=config,
                        weight_var=weight_var,
                        problem=problem,
                        train_xs=train_xs,
                        train_ys_rescaled=train_ys_rescaled,
                        ground_truth=ground_truth,
                        train_flip_indices=train_flip_indices,
                        attack_q=attack_q,
                        max_margin=(l2_lambda == 0 and l1_lambda == 0),
                        data_logger=data_logger,
                        log=log,
                        run_id=run.info.run_uuid
                    )
            except:
                log.exception(
                    'Failed to solve the optimization problem for L2 lambda %f, L1 lambda %f',
                    l2_lambda,
                    l1_lambda
                )

    log.info('Experiment finished')


def run_single(
        config: ExperimentConfig,
        weight_var: cp.Variable,
        problem: cp.Problem,
        train_xs: np.ndarray,
        train_ys_rescaled: np.ndarray,
        ground_truth: np.ndarray,
        train_flip_indices: np.ndarray,
        attack_q: typing.Union[int, str, float],
        max_margin: bool,
        data_logger: DataLogger,
        log: logging.Logger,
        run_id: str
):
    num_train_samples, data_dim = train_xs.shape
    num_noise_flip_samples, = train_flip_indices.shape

    # Optimize for current parameter
    weights, stats_dict, objective = solve_single(problem, weight_var, config, log, allow_infeasible=False)

    # If lambda = 0 we need to explicitly optimize for the max margin solution since that is
    # what gradient descent obtains. Otherwise we would get an arbitrary solution.
    # For any lambda > 0, the problem is well determined.
    if max_margin:
        # The max margin solution objective is unbounded if it exists,
        # and the optimization problem here uses the traiditional formulation which leads to
        # a different loss than for lambda > 0.
        # However, in the interpolating regime, the original optimization problem has the same loss
        # as the max margin solution, hence we can just take that.
        weights, stats_dict, _ = handle_max_margin(
            weights,
            stats_dict,
            objective,
            train_xs,
            train_ys_rescaled,
            attack_q,
            config,
            log
        )

    # Store optimization result
    ir.mlflow.set_tags(stats_dict)

    # Obtain and save weights
    assert isinstance(weights, np.ndarray) and weights.ndim == 1 and weights.shape[0] == data_dim
    data_logger.log_weights(weights, run_id)

    # Evaluation
    eval_metrics = dict()
    eval_metrics['training_loss'] = objective

    # Evaluate on training set
    num_std_train_correct, num_adv_train_correct, num_std_train_flip_correct, num_adv_train_flip_correct = eval_dataset(
        train_xs,
        train_ys_rescaled,
        weights,
        train_flip_indices,
        config.test_attack_epsilon,
        attack_q
    )
    eval_metrics['train_std_accuracy'] = float(num_std_train_correct) / num_train_samples
    eval_metrics['train_robust_accuracy'] = float(num_adv_train_correct) / num_train_samples

    # Calculate fraction of noise (flipped labels) samples fitted if available
    if num_noise_flip_samples > 0:
        eval_metrics['num_std_flip_fitted'] = num_std_train_flip_correct
        eval_metrics['fraction_std_flip_fitted'] = float(num_std_train_flip_correct) / num_noise_flip_samples
        eval_metrics['num_adv_flip_fitted'] = num_adv_train_flip_correct
        eval_metrics['fraction_adv_flip_fitted'] = float(num_adv_train_flip_correct) / num_noise_flip_samples

    # Closed-form solutions
    # Currently closed-form evaluation is only implemented for a subset of the data model
    assert np.all(ground_truth[1:] == 0.0)
    weight_norm_l2 = np.linalg.norm(weights)
    weight_norm_lq_projected = np.linalg.norm(weights[1:], ord=attack_q)

    # If weight norm is zero, ground truth angle is ignored
    with np.errstate(divide='ignore'):
        ground_truth_cos_angle = np.dot(ground_truth, weights) / weight_norm_l2

    true_std_risk, true_robust_risk = ir.closed_form.logistic_regression_risks(
        weight_norm_l2=weight_norm_l2,
        ground_truth_cos_angle=ground_truth_cos_angle,
        weight_norm_lq_projected=weight_norm_lq_projected,
        epsilon=config.test_attack_epsilon
    )

    eval_metrics['true_std_risk'] = float(true_std_risk)
    eval_metrics['true_robust_risk'] = float(true_robust_risk)

    # Margin (actual and per-sample)
    std_train_predictions = train_ys_rescaled * np.dot(train_xs, weights)
    margin = np.min(std_train_predictions)
    mean_margin = np.mean(std_train_predictions)
    adversarial_train_offset = config.test_attack_epsilon * np.linalg.norm(weights, ord=attack_q)
    adversarial_train_predictions = std_train_predictions - adversarial_train_offset
    robust_margin = np.min(adversarial_train_predictions)
    mean_robust_margin = np.mean(adversarial_train_predictions)
    eval_metrics['std_margin'] = float(margin)
    eval_metrics['std_margin_normalized'] = float(margin / weight_norm_l2)
    eval_metrics['std_mean_margin'] = float(mean_margin)
    eval_metrics['std_mean_margin_normalized'] = float(mean_margin / weight_norm_l2)
    eval_metrics['robust_margin'] = float(robust_margin)
    eval_metrics['robust_margin_normalized'] = float(robust_margin / weight_norm_l2)
    eval_metrics['robust_mean_margin'] = float(mean_robust_margin)
    eval_metrics['robust_mean_margin_normalized'] = float(mean_robust_margin / weight_norm_l2)

    # General weight-related stuff
    dual_weight_norm = np.linalg.norm(weights, ord=attack_q)
    dual_normal_norm_ratio = dual_weight_norm / weight_norm_l2
    projected_weights = np.copy(weights)
    projected_weights[0] = 0.0  # this is hardcoded due to the assertion further up
    projected_weight_norm = np.linalg.norm(projected_weights, ord=2)
    projected_dual_weight_norm = np.linalg.norm(projected_weights, ord=attack_q)
    projected_dual_primal_norm_ratio = projected_dual_weight_norm / projected_weight_norm
    projected_dual_normal_primal_norm_ratio = projected_dual_weight_norm / weight_norm_l2
    eval_metrics['weight_norm_l2'] = float(weight_norm_l2)
    eval_metrics['weight_norm_lq'] = float(dual_weight_norm)
    eval_metrics['weight_lq_l2_norm_ratio'] = float(dual_normal_norm_ratio)
    eval_metrics['projected_weight_norm_l2'] = float(projected_weight_norm)
    eval_metrics['projected_weight_norm_lq'] = float(projected_dual_weight_norm)
    eval_metrics['projected_weight_lq_l2_norm_ratio'] = float(projected_dual_primal_norm_ratio)
    eval_metrics['projected_weight_lq_normal_l2_norm_ratio'] = float(projected_dual_normal_primal_norm_ratio)

    # Track alignment of weights and ground truth
    gt_weights_cos_angle = ground_truth_cos_angle  # NB: ground truth has norm 1
    eval_metrics['gt_weights_cos_angle'] = float(gt_weights_cos_angle)

    # Log weight sparsity
    current_weight_sparsity = np.linalg.norm(weights, ord=0)
    eval_metrics['weight_nonzero_coefficients_fraction'] = float(current_weight_sparsity) / data_dim

    ir.mlflow.log_metrics(eval_metrics)


def handle_max_margin(
        weights: np.ndarray,
        stats_dict: typing.Dict[str, typing.Any],
        objective_value: float,
        train_xs: np.ndarray,
        train_ys_rescaled: np.ndarray,
        attack_q: typing.Union[float, int, str],
        config: ExperimentConfig,
        log: logging.Logger
) -> typing.Tuple[
    np.ndarray,
    typing.Dict[str, typing.Any],
    float
]:
    assert train_ys_rescaled.ndim == 1
    assert weights.ndim == 1

    # Build DCP instance
    num_samples, data_dim = train_xs.shape
    weight_var = cp.Variable(data_dim)
    objective = cp.norm(weight_var, p=2)

    # Initialize weight variable with known weights to potentially speed up optimization
    weight_var.value = weights

    per_sample_margins = cp.multiply(train_ys_rescaled, train_xs @ weight_var)
    if config.train_attack_epsilon > 0:
        if config.train_consistent_attacks:
            # TODO: As in the general case, this is hardcoded for one specific ground truth.
            perturbation_weight_norm = cp.norm(weight_var[1:], p=attack_q)
        else:
            perturbation_weight_norm = cp.norm(weight_var, p=attack_q)
        per_sample_margins = per_sample_margins - config.train_attack_epsilon * perturbation_weight_norm

    per_sample_margins = per_sample_margins - 1.0

    constraints = [per_sample_margins >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints=constraints)
    log.info('Built DCP max margin instance. Validation: DCP=%s, DPP=%s', problem.is_dcp(dpp=False), problem.is_dcp(dpp=True))
    assert problem.is_dcp(dpp=False)

    # Solve problem
    max_margin_result = solve_single(problem, weight_var, config, log, allow_infeasible=True)

    if max_margin_result is None:
        log.info('Max margin solution does not exist, original solution is unique')
        return weights, stats_dict, objective_value
    else:
        log.info('Max margin solution exists, returning it')
        return max_margin_result


def solve_single(
        problem: cp.Problem,
        weight_var: cp.Variable,
        config: ExperimentConfig,
        log: logging.Logger,
        allow_infeasible: bool
) -> typing.Optional[typing.Tuple[
    np.ndarray,
    typing.Dict[str, typing.Any],
    float
]]:
    # Try out solvers iteratively until one works
    weights = None
    best_objective = float('inf')
    best_stats_dict = dict()
    for current_solver in config.solvers:
        try:
            problem.solve(verbose=config.verbose, enforce_dpp=True, solver=current_solver, warm_start=True)
            log.info('Problem solved with solver %s, status: %s', current_solver, problem.status)

            if problem.status == cp.INFEASIBLE:
                if allow_infeasible:
                    return None
                else:
                    raise RuntimeError('Solver reported problem to be infeasible')

            current_objective = problem.value

            if problem.status == cp.OPTIMAL or best_objective is None or current_objective < best_objective:
                log.info(
                    'New minimal objective value %f found with solver %s',
                    current_objective,
                    current_solver
                )
                best_objective = current_objective
                weights = weight_var.value
                best_stats_dict = {
                    'solver.status': problem.status,
                    'solver.name': problem.solver_stats.solver_name,
                    'solver.solve_time': problem.solver_stats.solve_time,
                    'solver.setup_time': problem.solver_stats.setup_time,
                    'solver.num_iters': problem.solver_stats.num_iters,
                    'solver.extra_stats': problem.solver_stats.extra_stats
                }

                if problem.status == cp.OPTIMAL:
                    log.info('Found an optimal solution, stopping search')
                    break
        except:
            log.exception('Solver %s failed to find a solution', current_solver)

    # Check whether any solution at all was found
    if weights is None:
        raise RuntimeError(
            f'None of the solvers found an optimal solution'
        )

    # Check whether the best found solution is inaccurate
    if best_stats_dict['solver.status'] == cp.OPTIMAL_INACCURATE:
        log.warning('Found an optimal solution but all solutions were inaccurate')

    assert isinstance(weights, np.ndarray)
    return weights, best_stats_dict, best_objective


def eval_dataset(
        xs: np.ndarray,
        ys_scaled: np.ndarray,
        weights: np.ndarray,
        flip_indices: np.ndarray,
        epsilon: float,
        attack_q: typing.Union[float, int, str]
) -> typing.Tuple[int, int, int, int]:
    # NB: Expects ys in {-1, +1}, not {0, 1}

    assert weights.ndim == 1
    assert ys_scaled.ndim == 1

    # Calculate standard accuracy
    natural_logits = np.dot(xs, weights)
    natural_raw_predictions = np.multiply(ys_scaled, natural_logits)
    std_correct = (natural_raw_predictions > 0)
    num_std_correct = int(np.sum(np.int32(std_correct)))

    # Calculate robust accuracy
    # This implicitly assume the single entry ground truth!
    adversarial_offset = epsilon * np.linalg.norm(weights[1:], ord=attack_q)
    adversarial_raw_predictions = natural_raw_predictions - adversarial_offset
    adv_correct = (adversarial_raw_predictions > 0)
    num_adv_correct = int(np.sum(np.int32(adv_correct)))

    if flip_indices.shape[0] > 0:
        num_std_flip_correct = int(np.sum(np.int32(std_correct[flip_indices])))
        num_adv_flip_correct = int(np.sum(np.int32(adv_correct[flip_indices])))
    else:
        num_std_flip_correct = 0
        num_adv_flip_correct = 0

    return num_std_correct, num_adv_correct, num_std_flip_correct, num_adv_flip_correct


def build_config(args) -> ExperimentConfig:
    # Determine attack epsilons, depending on arguments configuration
    test_attack_epsilon = args.attack_epsilon
    train_attack_epsilon = args.train_attack_epsilon if args.train_attack_epsilon is not None else test_attack_epsilon

    # Parse perturbation radius
    attack_p = 'inf' if args.attack_p == 'inf' else float(args.attack_p)

    solvers = tuple(args.solvers)
    if len(solvers) == 0:
        raise ValueError('At least one solver must be specified')

    return ExperimentConfig(
        label_noise=args.label_noise,
        train_consistent_attacks=args.attack_train_consistent,
        train_attack_epsilon=train_attack_epsilon,
        test_attack_epsilon=test_attack_epsilon,
        attack_p=attack_p,
        data_num_train_samples=args.training_samples,
        data_dim=args.data_dim,
        data_logits_noise_variance=args.data_logits_noise_variance,
        data_min_decision_boundary_distance=args.min_decision_boundary_distance,
        root_log_dir=args.logdir,
        solvers=solvers,
        verbose=args.verbose,
        l2_lambdas=tuple(args.l2),
        l1_lambdas=tuple(args.l1),
        seed=args.seed
    )


def make_dataset(config: ExperimentConfig, rng: np.random.Generator, logger: DataLogger) -> typing.Tuple[
    typing.Tuple[np.ndarray, np.ndarray],  # training data
    np.ndarray,  # label noise indices
    typing.Tuple[np.ndarray, np.ndarray]  # ground truth and feature covariance diagonal
]:
    # Dataset parameters, fixed
    x_covariance_diagonal = np.ones((config.data_dim,))
    ground_truth = np.zeros_like(x_covariance_diagonal)
    ground_truth[0] = 1.0

    (train_xs, train_ys), _, train_label_noise, (train_covariate_noise, _) = ir.data.make_gaussian_logistic(
        train_samples=config.data_num_train_samples,
        test_samples=0,
        x_covariance_diagonal=x_covariance_diagonal,
        logits_noise_variance=config.data_logits_noise_variance,
        ground_truth=ground_truth,
        label_noise_fraction=config.label_noise,
        min_decision_boundary_distance=config.data_min_decision_boundary_distance,
        seed=rng
    )

    logger.log_single_gaussian_dataset(
        train_xs,
        train_ys,
        train_label_noise,
        train_covariate_noise,
        ground_truth,
        x_covariance_diagonal,
        config.data_logits_noise_variance,
    )

    return (train_xs, train_ys), train_label_noise, (ground_truth, x_covariance_diagonal)


@contextlib.contextmanager
def setup_run(config: ExperimentConfig, l2_lambda: float, l1_lambda: float):
    run_name = '{timestamp:%Y%m%d-%H%M%S}_L2{l2_lambda}_L1{l1_lambda}'.format(
        timestamp=datetime.datetime.utcnow(),
        l2_lambda=l2_lambda,
        l1_lambda=l1_lambda
    )

    with ir.mlflow.start_run(run_name=run_name) as run:
        ir.mlflow.set_tag('base_experiment', EXPERIMENT_TAG)
        config_dict = config._asdict()

        # Make sure that too many lambda values doesn't crash MLFlow
        if len(str(config_dict['l2_lambdas'])) > 250:
            config_dict['l2_lambdas'] = str(config_dict['l2_lambdas'])[:250-3] + '...'
        if len(str(config_dict['l1_lambdas'])) > 250:
            config_dict['l1_lambdas'] = str(config_dict['l1_lambdas'])[:250-3] + '...'

        config_dict['l2_lambda'] = l2_lambda
        config_dict['l1_lambda'] = l1_lambda
        ir.mlflow.log_params(config_dict)
        yield run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # General args
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
    parser.add_argument('--data-dim', type=int, default=8000, help='Dimensionality of the data')
    parser.add_argument('--training-samples', type=int, default=1000, help='Number of training samples to generate')
    parser.add_argument('--label-noise', type=float, default=0.0, help='Label flip probability')
    parser.add_argument(
        '--data-logits-noise-variance', type=float, default=0.0,
        help='Which logit noise variance to use for single Gaussian data model'
    )
    parser.add_argument('--logdir', type=str, help='If set, weights and the dataset are stored in {logdir}/')
    parser.add_argument('--min-decision-boundary-distance', type=float, default=0.0,
        help='Use a data set consisting only of samples which at least this distance from the decision boundary'
    )
    parser.add_argument('--solvers', type=str, nargs='+', default=(cp.SCS, cp.ECOS),
        help='DCP solvers to use, in the order they should be applied until one works'
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose CVXPY loggging')
    parser.add_argument(
        '--tag', type=str, required=True, help='Name of the current set of experiment runs, used for grouping'
    )
    parser.add_argument(
        '--l2', type=float, nargs='+', default=(0,), help='Range of L2 penalty weights to use'
    )
    parser.add_argument(
        '--l1', type=float, nargs='+', default=(0,), help='Range of L1 penalty weights to use'
    )
    parser.add_argument('--seed', type=int, default=1, help='Seed to initialize rngs')

    return parser.parse_args()


if __name__ == '__main__':
    main()
