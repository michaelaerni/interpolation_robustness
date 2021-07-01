import argparse
import contextlib
import datetime
import enum
import logging
import math
import os
import typing

import art.attacks.evasion
import art.estimators.classification
import dotenv
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional
import tqdm

import interpolation_robustness as ir

EXPERIMENT_TAG = 'neural_networks'


@enum.unique
class ModelType(enum.Enum):
    PreActResNet18 = 'preact_resnet18'
    PreActResNet34 = 'preact_resnet34'
    PreActResNet50 = 'preact_resnet50'
    MLP = 'mlp'
    RandomFeatureMLP = 'rf_mlp'


@enum.unique
class MLPActivation(enum.Enum):
    ReLU = 'relu'
    Linear = 'linear'


@enum.unique
class AttackMethod(enum.Enum):
    FFGSM = 'ffgsm'
    PGD = 'pgd'


@enum.unique
class DataAugmentation(enum.Enum):
    FlipHorizontal = 'flip_horizontal'


class ExperimentConfig(typing.NamedTuple):
    train_batch_size: int
    test_batch_size: int
    epochs: int
    eval_every_epochs: int
    save_every_epochs: int
    initial_learning_rate: float
    learning_rate_decay_epochs: int
    learning_rate_decay_step: float
    momentum: float
    weight_decay: float
    label_noise: float
    adversarial_training: bool
    train_attack_epsilon: float
    test_attack_epsilon: float
    attack_p: typing.Union[float, str]
    train_attack_method: typing.Optional[AttackMethod]
    train_attack_steps: int
    train_attack_step_size: float
    test_attack_method: AttackMethod
    test_attack_steps: int
    test_attack_step_size: float
    num_train_samples: typing.Optional[int]
    num_validation_samples: typing.Optional[int]
    dataset: ir.data.Dataset
    data_split: ir.data.DataSplit
    data_augmentations: typing.Tuple[DataAugmentation, ...]
    data_binary_classes: typing.Optional[typing.Tuple[int, int]]
    train_discard_indices: typing.Optional[typing.Tuple[int, ...]]
    model_type: ModelType
    cnn: bool
    mlp_units: typing.Optional[typing.Tuple[int]]
    mlp_activation: typing.Optional[MLPActivation]
    seed: int
    display_progressbar: bool
    root_log_dir: typing.Optional[str]
    fit_times_dir: typing.Optional[str]


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
    # Fix randomness as much as possible
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    log = logging.getLogger(__name__)

    log.info('Starting experiment with config %s', config._asdict())
    log.info('MLFlow run id is %s', run_id)

    device = torch.device('cuda')

    # Build dataset
    train_data, train_eval_data, test_eval_data, normalization_constants, num_classes, train_noise_indices = \
        make_datasets(config, rng=rng)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_data, batch_size=config.test_batch_size, shuffle=False)
    test_eval_loader = torch.utils.data.DataLoader(test_eval_data, batch_size=config.test_batch_size, shuffle=False)
    sample_shape = next(iter(train_loader))[0].size()[1:]  # need to look at an output, cannot look at original data
    num_train_samples = len(train_data)
    num_test_samples = len(test_eval_data)
    num_noisy_samples = train_noise_indices.shape[0]
    num_batches_per_epoch = math.ceil(num_train_samples / config.train_batch_size)

    # Normalizer for inputs, depending on the data set
    normalizer_mean, normalizer_std = normalization_constants
    if not config.cnn:
        normalizer_target_shape = (1,)
    else:
        normalizer_target_shape = (1, sample_shape[0]) + (1,) * (len(sample_shape) - 1)
    normalizer_mean = torch.from_numpy(normalizer_mean).view(normalizer_target_shape).to(device)
    normalizer_std = torch.from_numpy(normalizer_std).view(normalizer_target_shape).to(device)

    def normalizer(x: torch.Tensor) -> torch.Tensor:
        return (x - normalizer_mean) / normalizer_std

    log.info(
        'Built data set for %d classes, with %d training samples (%d with flipped labels), %d test samples, split strategy %s',
        num_classes,
        num_train_samples,
        num_noisy_samples,
        num_test_samples,
        config.data_split.value
    )

    # Build and initialize model
    model = make_model(config, num_classes, sample_shape)
    model.to(device)
    log.info('Built model')

    # Build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.initial_learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    loss_function = torch.nn.CrossEntropyLoss()

    # Build learning rate schedule
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.learning_rate_decay_epochs,
        gamma=config.learning_rate_decay_step
    )

    # Setup adversarial attacks
    adversarial_classifier_wrapper = art.estimators.classification.PyTorchClassifier(
        model=model,
        loss=loss_function,
        input_shape=sample_shape,
        nb_classes=num_classes,
        channels_first=True,
        clip_values=(0, 1),
        device_type=device.type,
        preprocessing=normalization_constants  # will automatically do (x - mean) / std
    )
    train_attack, test_attack = make_attacks(adversarial_classifier_wrapper, config)

    # Prepare checkpoint saving if enabled
    if config.root_log_dir is not None:
        checkpoint_dir = os.path.join(config.root_log_dir, run_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        log.info('Storing checkpoints into directory %s', checkpoint_dir)
    else:
        checkpoint_dir = None

    # Prepare fit times saving if enabled
    fit_time_tracker = FitTimeTracker(config.fit_times_dir, num_train_samples, config.epochs)
    fit_time_tracker.prepare()

    log.info('Starting training')
    for epoch in tqdm.trange(0, config.epochs, unit='epoch', disable=not config.display_progressbar):
        # Standard training
        epoch_loss = 0

        # FIXME: Refactor this into smaller chunks

        # Train
        model.train()
        for batch_xs, batch_ys in tqdm.tqdm(train_loader, desc='Training', unit='batch', disable=not config.display_progressbar):
            # Find adversarial examples for batch if AT is enabled
            if config.adversarial_training:
                model.eval()
                # Don't normalize batch, will be done explicitly
                batch_xs = train_attack.generate(batch_xs, batch_ys)
                batch_xs = torch.from_numpy(batch_xs)
                model.train()

            # Note that the AT library requires the batch to lie on the CPU
            # hence moving things to the target device after potentially attacking the model makes sense
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)

            # Perform gradient descent step
            optimizer.zero_grad()
            logits = model(normalizer(batch_xs))
            current_loss = loss_function(logits, batch_ys)
            current_loss.backward()
            optimizer.step()

            epoch_loss += current_loss.item()

        lr_scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches_per_epoch
        ir.mlflow.log_metric('avg_epoch_loss', avg_epoch_loss, step=epoch)

        # Evaluation
        model.eval()
        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs - 1:
            eval_metrics = dict()
            log_metrics = []

            # Evaluate on test set
            num_std_test_correct, num_adv_test_correct, _, _ = eval_dataset(
                name='test',
                epoch=epoch,
                data=test_eval_loader,
                model=model,
                normalizer=normalizer,
                attack=test_attack,
                device=device,
                fit_time_tracker=None,  # we don't care about fit times on the test set, only training
                display_progressbar=config.display_progressbar
            )
            log_metrics += ['test_std_accuracy', 'test_robust_accuracy']
            eval_metrics['test_std_accuracy'] = float(num_std_test_correct) / num_test_samples
            eval_metrics['test_robust_accuracy'] = float(num_adv_test_correct) / num_test_samples

            # Evaluate on training set
            num_std_train_correct, num_adv_train_correct, num_std_train_noise_correct, num_adv_train_noise_correct = eval_dataset(
                name='train',
                epoch=epoch,
                data=train_eval_loader,
                model=model,
                normalizer=normalizer,
                attack=test_attack,
                device=device,
                fit_time_tracker=fit_time_tracker,
                display_progressbar=config.display_progressbar
            )
            log_metrics += ['train_std_accuracy', 'train_robust_accuracy']
            eval_metrics['train_std_accuracy'] = float(num_std_train_correct) / num_train_samples
            eval_metrics['train_robust_accuracy'] = float(num_adv_train_correct) / num_train_samples

            # Calculate fraction of noisy samples fitted if available
            if num_noisy_samples > 0:
                log_metrics += ['fraction_std_noise_fitted', 'fraction_adv_noise_fitted']
                eval_metrics['num_std_noise_fitted'] = num_std_train_noise_correct
                eval_metrics['fraction_std_noise_fitted'] = float(num_std_train_noise_correct) / num_noisy_samples
                eval_metrics['num_adv_noise_fitted'] = num_adv_train_noise_correct
                eval_metrics['fraction_adv_noise_fitted'] = float(num_adv_train_noise_correct) / num_noisy_samples

            ir.mlflow.log_metrics(eval_metrics, step=epoch)
            log.info(
                'Epoch %04d: avg_loss=%.4f, %s',
                epoch + 1,
                float(avg_epoch_loss),
                ', '.join('{0}={1:.4f}'.format(metric_key, eval_metrics[metric_key]) for metric_key in log_metrics)
            )

        # Save checkpoint if enabled
        if checkpoint_dir is not None and (epoch % config.save_every_epochs == 0 or epoch == config.epochs - 1):
            checkpoint_file = os.path.join(checkpoint_dir, f'model-{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'run_id': run_id
            }, checkpoint_file)
            log.debug('Saved checkpoint of epoch %d to %s', epoch, checkpoint_file)

    # Save fit times if enabled
    fit_time_tracker.save(run_id)

    log.info('Finished training')


def eval_dataset(
        name: str,
        epoch: int,
        data: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        normalizer: typing.Callable[[torch.Tensor], torch.Tensor],
        attack: art.attacks.EvasionAttack,
        device: torch.device,
        fit_time_tracker: typing.Optional['FitTimeTracker'],
        display_progressbar: bool
) -> typing.Tuple[int, int, int, int]:

    num_std_correct = 0
    num_adv_correct = 0
    num_std_noise_correct = 0
    num_adv_noise_correct = 0

    fit_mask_raw = []
    for batch_idx, (batch_xs, batch_ys, batch_noise_mask) in enumerate(tqdm.tqdm(data, desc=f'Eval {name}', unit='batch', disable=not display_progressbar)):
        batch_xs = batch_xs.to(device)
        batch_ys = batch_ys.to(device)
        batch_noise_mask = batch_noise_mask.to(device)

        # Calculate standard accuracy
        standard_predictions = torch.argmax(model(normalizer(batch_xs)), dim=-1)
        std_correct = torch.eq(standard_predictions, batch_ys)
        fit_mask_raw.append(std_correct.detach().cpu().numpy())
        std_correct_xs = batch_xs[std_correct]
        std_correct_ys = batch_ys[std_correct]
        num_std_correct += std_correct_xs.size()[0]

        # Calculate how much noise was fit
        std_correct_noise = batch_noise_mask[std_correct]
        num_std_noise_correct += torch.sum(std_correct_noise.int()).item()

        # Calculate robust accuracy, only necessary to use correctly classified samples
        evaluate_adversarial = (std_correct_xs.size()[0] > 0)
        if evaluate_adversarial:
            # Samples need to be copied to CPU before they get copied to GPU again due to the way
            # the adversarial attacks library works...
            adversarial_xs = attack.generate(std_correct_xs.cpu(), std_correct_ys.cpu())
            adversarial_xs = torch.from_numpy(adversarial_xs).to(device)
            adversarial_predictions = torch.argmax(model(normalizer(adversarial_xs)), dim=-1)
            adv_correct = torch.eq(adversarial_predictions, std_correct_ys)
            num_adv_correct += torch.sum(adv_correct.int()).item()

            # Calculate how much noise was fit robustly
            num_adv_noise_correct += torch.sum(adv_correct[std_correct_noise].int()).item()

    if fit_time_tracker is not None:
        fit_mask = np.concatenate(fit_mask_raw, axis=0)
        fit_time_tracker.update(fit_mask, epoch)

    return num_std_correct, num_adv_correct, num_std_noise_correct, num_adv_noise_correct


def make_datasets(config: ExperimentConfig, rng: np.random.Generator) -> typing.Tuple[
    torch.utils.data.Dataset,  # training data
    torch.utils.data.Dataset,  # training evaluation data
    torch.utils.data.Dataset,  # test evaluation data
    typing.Tuple[np.ndarray, np.ndarray],  # mean and stddev for additional normalization
    int,  # number of classes
    np.ndarray  # label noise indices
]:
    train_transformations = []
    test_transformations = []

    (train_xs, train_ys), (test_xs, test_ys), train_noise_indices = ir.data.make_image_dataset(
        dataset=config.dataset,
        train_samples=config.num_train_samples,
        data_split=config.data_split,
        validation_samples=config.num_validation_samples,
        binarized_classes=config.data_binary_classes,
        train_discard_indices=config.train_discard_indices,
        train_label_noise=config.label_noise,
        seed=rng
    )
    num_classes = np.max(test_ys) + 1

    # Determine normalization constants for datasets
    if config.dataset == ir.data.Dataset.CIFAR10:
        # Normalize wrt CIFAR10 mean and std over all training data
        normalization_constants = np.mean(train_xs, axis=(0, 1, 2)), np.std(train_xs, axis=(0, 1, 2))
        normalization_constants_flat = np.mean(train_xs), np.std(train_xs)
    else:
        normalization_constants = normalization_constants_flat = (0.0, 1.0)

    data_dtype = train_xs.dtype

    train_noise_mask = torch.zeros((train_xs.shape[0],), dtype=torch.bool)
    train_noise_mask[train_noise_indices] = True
    test_noise_mask = torch.zeros((test_xs.shape[0],), dtype=torch.bool)

    # Switch from NHWC to NCHW
    train_xs = np.transpose(train_xs, (0, 3, 1, 2))
    test_xs = np.transpose(test_xs, (0, 3, 1, 2))

    # Convert data to tensors
    train_xs = torch.from_numpy(train_xs)
    train_ys = torch.from_numpy(train_ys)
    test_xs = torch.from_numpy(test_xs)
    test_ys = torch.from_numpy(test_ys)

    # Add data augmentation if specified
    augmentation_mapping = {
        DataAugmentation.FlipHorizontal: _random_flip_lr
    }
    for augmentation in config.data_augmentations:
        train_transformations.append(augmentation_mapping[augmentation])

    # Flatten images for MLPs
    if not config.cnn:
        def flatten(tensors: typing.Tuple[torch.Tensor, ...]) -> typing.Tuple[torch.Tensor, ...]:
            image = tensors[0]
            image = image.reshape(-1)
            return image, *tensors[1:]
        train_transformations.append(flatten)
        test_transformations.append(flatten)

        # Use normalization constants for flattened data
        normalization_constants = normalization_constants_flat

    # Make sure normalization constants are numpy arrays
    normalization_mean, normalization_std = normalization_constants
    if not isinstance(normalization_mean, np.ndarray):
        normalization_mean = np.asarray(normalization_mean, dtype=data_dtype)
    if not isinstance(normalization_std, np.ndarray):
        normalization_std = np.asarray(normalization_std, dtype=data_dtype)
    normalization_constants = normalization_mean, normalization_std

    # Build datasets
    train_data = ImageDataset(train_xs, train_ys, transformations=train_transformations)
    train_eval_data = ImageDataset(train_xs, train_ys, train_noise_mask, transformations=test_transformations)
    test_eval_data = ImageDataset(test_xs, test_ys, test_noise_mask, transformations=test_transformations)

    return train_data, train_eval_data, test_eval_data, normalization_constants, num_classes, train_noise_indices


def make_model(
        config: ExperimentConfig,
        num_classes: int,
        sample_shape: torch.Size
) -> torch.nn.Module:
    if config.model_type == ModelType.PreActResNet18:
        return ir.models.pytorch.PreActResNet18(num_classes=num_classes, image_channels=sample_shape[0])
    elif config.model_type == ModelType.PreActResNet34:
        return ir.models.pytorch.PreActResNet34(num_classes=num_classes, image_channels=sample_shape[0])
    elif config.model_type == ModelType.PreActResNet50:
        return ir.models.pytorch.PreActResNet50(num_classes=num_classes, image_channels=sample_shape[0])
    elif config.model_type in (ModelType.MLP, ModelType.RandomFeatureMLP):
        assert config.mlp_units is not None and len(config.mlp_units) > 0

        if config.mlp_activation == MLPActivation.ReLU:
            activation = torch.nn.ReLU()
        elif config.mlp_activation == MLPActivation.Linear:
            activation = None
        else:
            assert False

        num_features_flat = sum(sample_shape)
        model = ir.models.pytorch.MLP(
            in_features=num_features_flat,
            out_features=num_classes,
            num_hidden=config.mlp_units,
            activation=activation
        )

        # Freeze layers for RFM
        if config.model_type == ModelType.RandomFeatureMLP:
            for layer in model.layers[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False

        return model
    else:
        assert False


def make_attacks(
        classifier: art.estimators.classification.PyTorchClassifier,
        config: ExperimentConfig
) -> typing.Tuple[typing.Optional[art.attacks.EvasionAttack], art.attacks.EvasionAttack]:
    train_attack = None
    if config.adversarial_training:
        assert config.train_attack_method == AttackMethod.FFGSM
        train_attack = art.attacks.evasion.ProjectedGradientDescentPyTorch(
            classifier,
            norm=config.attack_p,
            eps=config.train_attack_epsilon,
            eps_step=config.train_attack_step_size,
            max_iter=1,
            targeted=False,
            num_random_init=1,
            batch_size=config.train_batch_size,
            verbose=False
        )

    if config.test_attack_method == AttackMethod.FFGSM:
        test_attack = art.attacks.evasion.ProjectedGradientDescentPyTorch(
            classifier,
            norm=config.attack_p,
            eps=config.test_attack_epsilon,
            eps_step=config.test_attack_step_size,
            max_iter=1,
            targeted=False,
            num_random_init=1,
            batch_size=config.test_batch_size,
            verbose=False
        )
    elif config.test_attack_method == AttackMethod.PGD:
        test_attack = art.attacks.evasion.ProjectedGradientDescentPyTorch(
            classifier,
            norm=config.attack_p,
            eps=config.test_attack_epsilon,
            eps_step=config.test_attack_step_size,
            max_iter=config.test_attack_steps,
            targeted=False,
            num_random_init=1,
            batch_size=config.test_batch_size,
            verbose=False
        )
    else:
        assert False

    # Disable the *IDIOTICALLY VERBOSE* logger of the adversarial attacks package
    logging.getLogger(
        'art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch'
    ).setLevel(level=logging.FATAL)

    return train_attack, test_attack


@contextlib.contextmanager
def setup_experiment(config: ExperimentConfig, experiment_name: str):
    run_name = '{timestamp:%Y%m%d-%H%M%S}_{training_method}_{dataset}_{model}'.format(
        timestamp=datetime.datetime.utcnow(),
        training_method='at' if config.adversarial_training else 'st',
        dataset=config.dataset.value,
        model=config.model_type.value
    )

    ir.mlflow.set_experiment(experiment_name)
    with ir.mlflow.start_run(run_name=run_name) as run:
        ir.mlflow.set_tag('base_experiment', EXPERIMENT_TAG)

        # Remove potentially large params from config dict for logging
        config_dict = config._asdict()
        del config_dict['train_discard_indices']

        ir.mlflow.log_params(config_dict)
        yield run


def build_config(args) -> ExperimentConfig:
    attack_p = 'inf' if args.attack_p == 'inf' else float(args.attack_p)
    if not (0.0 <= args.label_noise < 1.0):
        raise ValueError(f'Label noise fraction must be in [0, 1) but was {args.label_noise}')
    if args.test_attack_epsilon < 0.0:
        raise ValueError(f'Adversarial perturbation radius epsilon for evaluation must be non-negative but is {args.test_attack_epsilon}')
    if args.train_attack_epsilon is not None and args.train_attack_epsilon < 0.0:
        raise ValueError(f'Adversarial perturbation radius epsilon for training must be non-negative but is {args.train_attack_epsilon}')
    # FIXME: Allow for different training attack methods?
    train_attack_method = AttackMethod.FFGSM if args.adversarial_training else None
    test_attack_method = AttackMethod(args.test_attack_method)
    train_attack_epsilon = args.train_attack_epsilon if args.train_attack_epsilon is not None else args.test_attack_epsilon
    dataset = ir.data.Dataset(args.dataset)
    data_split = ir.data.DataSplit(args.data_split)
    num_validation_samples = 0 if data_split == ir.data.DataSplit.NoSplit else 10000
    if args.data_augmentations is not None:
        data_augmentations = tuple(DataAugmentation(augmentation) for augmentation in args.data_augmentations)
    else:
        data_augmentations = tuple()
    model_type = ModelType(args.model)
    if model_type in (ModelType.MLP, ModelType.RandomFeatureMLP):
        if args.mlp_units is None or len(args.mlp_units) == 0:
            raise ValueError('At least one layer width has to be specified for MLP models')
        mlp_activation = MLPActivation.ReLU if args.mlp_activation is None else MLPActivation(args.mlp_activation)
        cnn = False
    else:
        mlp_activation = None
        cnn = True
    test_batch_size = args.test_batch_size if args.test_batch_size is not None else args.train_batch_size
    save_every_epochs = args.save_every_epochs if args.save_every_epochs is not None else args.eval_every_epochs
    if args.binary_classes is not None:
        assert len(args.binary_classes) == 2
        data_binary_classes = tuple(args.binary_classes)
    else:
        data_binary_classes = None

    return ExperimentConfig(
        train_batch_size=args.train_batch_size,
        test_batch_size=test_batch_size,
        epochs=args.epochs,
        eval_every_epochs=args.eval_every_epochs,
        save_every_epochs=save_every_epochs,
        initial_learning_rate=args.learning_rate,
        learning_rate_decay_epochs=args.learning_rate_decay_epochs,
        learning_rate_decay_step=args.learning_rate_decay_step,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_noise=args.label_noise,
        adversarial_training=args.adversarial_training,
        train_attack_epsilon=train_attack_epsilon,
        test_attack_epsilon=args.test_attack_epsilon,
        attack_p=attack_p,
        train_attack_method=train_attack_method,
        train_attack_steps=1,
        train_attack_step_size=args.train_attack_step_size,
        test_attack_method=test_attack_method,
        test_attack_steps=args.test_attack_steps,
        test_attack_step_size=args.test_attack_step_size,
        num_train_samples=args.training_samples,
        num_validation_samples=num_validation_samples,
        dataset=dataset,
        data_split=data_split,
        data_augmentations=data_augmentations,
        data_binary_classes=data_binary_classes,
        train_discard_indices=args.training_discard_indices,
        model_type=model_type,
        cnn=cnn,
        mlp_units=args.mlp_units,
        mlp_activation=mlp_activation,
        seed=args.seed,
        display_progressbar=args.progressbar,
        root_log_dir=args.logdir,
        fit_times_dir=args.save_fit_times_to
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-batch-size', type=int, default=200, help='Batch size for training')
    parser.add_argument(
        '--test-batch-size', type=int, default=None, help='Batch size for evaluation, defaults to training batch size'
    )
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--eval-every-epochs', type=int, default=1, help='How many epochs happen between model evaluations')
    parser.add_argument(
        '--save-every-epochs', type=int,
        help='How many epochs happen between checkpoints if enabled, defaults to model evaluation epochs'
    )
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Initial learning rate for training')
    parser.add_argument('--learning-rate-decay-epochs', type=int, help='Decay learning rate every epochs')
    parser.add_argument('--learning-rate-decay-step', type=float, default=0.1, help='Amount the learning rate is decayed at each decay step')
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum for training')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for training')
    parser.add_argument('--label-noise', type=float, default=0.0, help='Fraction of training labels to corrupt')
    parser.add_argument('--adversarial-training', action='store_true', help='Use adversarial training')
    parser.add_argument('--train-attack-epsilon', type=float, help='Radius of epsilon-ball for adversarial attacks during adversarial training')
    parser.add_argument('--test-attack-epsilon', type=float, default=0.1, help='Radius of epsilon-ball for adversarial attacks during evaluation')
    parser.add_argument('--attack-p', choices=('2', 'inf'), type=str, default='inf', help='p of epsilon-ball norm for adversarial attacks')
    parser.add_argument('--train-attack-step-size', type=float, default=0.125, help='FFGSM step size for training')
    parser.add_argument(
        '--test-attack-method', type=str, choices=[method.value for method in AttackMethod],
        default=AttackMethod.PGD.value,
        help='Adversarial attack method for model evaluation'
    )
    parser.add_argument('--test-attack-steps', type=int, default=10, help='Number of steps for PGD attack on test set')
    parser.add_argument('--test-attack-step-size', type=float, default=0.02, help='Attack step size for test')
    parser.add_argument(
        '--training-samples', type=int, default=None,
        help='If set, the number of training samples will be subsampled to the given value'
    )
    parser.add_argument(
        '--dataset', type=str, required=True, choices=[dataset.value for dataset in ir.data.Dataset],
        help='Dataset to use'
    )
    parser.add_argument(
        '--data-split', type=str,
        default=ir.data.DataSplit.NoSplit.value,
        choices=[split.value for split in ir.data.DataSplit],
        help='Split-off a validation set, and if yes, whether to use it or not'
    )
    parser.add_argument(
        '--data-augmentations', type=str, nargs='*', choices=[augmentation.value for augmentation in DataAugmentation],
        help='List and order of data augmentations to apply during training'
    )
    parser.add_argument(
        '--binary-classes', type=int, nargs=2, help='If provided, the dataset will be binarized to those two classes'
    )
    parser.add_argument(
        '--training-discard-indices', type=int, nargs='*',
        help='Indices of training samples to discard a priori.'
            ' Indices are wrt a potentially binarized training set but before sampling and consistent with the seed set.'
    )
    parser.add_argument(
        '--model', type=str, required=True, choices=[model.value for model in ModelType], help='Type of model to use'
    )
    parser.add_argument(
        '--mlp-units', type=int, nargs='+',
        help='Number of units per hidden MLP layer (excluding output) if the model is "MLP"'
    )
    parser.add_argument(
        '--mlp-activation', type=str, choices=[activation.value for activation in MLPActivation],
        help='Activation to use in MLP models'
    )
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--progressbar', action='store_true', help='Display a progress bar during training and evaluation')
    parser.add_argument('--logdir', type=str, help='If set, checkpoints are stored in {logdir}/{run_id}/')
    parser.add_argument('--save-fit-times-to', type=str, help='If set, the time when each test sample was last fit is stored into this directory')
    parser.add_argument(
        '--tag', type=str, required=True, help='Name of the current set of experiment runs, used for grouping'
    )

    return parser.parse_args()


class ImageDataset(torch.utils.data.TensorDataset):
    def __init__(
            self,
            *tensors: torch.Tensor,
            transformations: typing.List[typing.Callable[[typing.Tuple[torch.Tensor, ...]], typing.Tuple[torch.Tensor, ...]]] = None
    ):
        super(ImageDataset, self).__init__(*tensors)
        if transformations is None:
            transformations = []
        self.transformations = transformations

    def __getitem__(self, index):
        tensors = super(ImageDataset, self).__getitem__(index)
        for transformation in self.transformations:
            tensors = transformation(tensors)
        return tensors


def _random_flip_lr(tensors: typing.Tuple[torch.Tensor, ...]) -> typing.Tuple[torch.Tensor, ...]:
    image = tensors[0]
    if torch.rand(1) < 0.5:
        image = torchvision.transforms.functional.hflip(image)
    return image, *tensors[1:]


class FitTimeTracker(object):
    def __init__(self, output_dir: typing.Optional[str], num_train_samples: int, epochs: int):
        self._output_dir = output_dir

        # Only initialize tracking arrays if we actually track
        if self._output_dir is not None:
            self._fit_times = np.ones(num_train_samples, dtype=np.int) * epochs
            self._last_fit = np.zeros(num_train_samples, dtype=np.bool)
            self._last_epoch = -1

    def prepare(self):
        if self._output_dir is None:
            return

        os.makedirs(self._output_dir, exist_ok=True)

    def update(self, current_fit: np.ndarray, epoch: int):
        if self._output_dir is None:
            return

        if epoch <= self._last_epoch:
            raise ValueError(f'Given epoch {epoch} must be larger than the last epoch {self._last_epoch}')
        self._last_epoch = epoch

        assert current_fit.dtype == np.bool and current_fit.shape == self._last_fit.shape

        newly_fit_samples_mask = (~self._last_fit) & current_fit
        self._fit_times[newly_fit_samples_mask] = epoch

        self._last_fit = current_fit.copy()

    def save(self, run_id: str):
        if self._output_dir is None:
            return

        np.save(os.path.join(self._output_dir, f'fit_times_{run_id}.npy'), self._fit_times)


if __name__ == '__main__':
    main()
