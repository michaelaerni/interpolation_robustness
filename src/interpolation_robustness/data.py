import enum
import typing

import numpy as np
import scipy.stats
import tensorflow_datasets as tfds

import interpolation_robustness as ir


@enum.unique
class Dataset(enum.Enum):
    MNIST = 'mnist'
    CIFAR10 = 'cifar10'
    FashionMNIST = 'fashion_mnist'


@enum.unique
class DataSplit(enum.Enum):
    NoSplit = 'no_split'  # do not use a validation set
    SplitEvalTest = 'split_eval_test'  # use a validation set, but evaluate on test set
    SplitEvalValidation = 'split_eval_validation'  # use a validation set and evaluate on it


def make_2gmm_linf(
        train_samples: int,
        test_samples: int,
        distance: float,
        distance_prob: float,
        stddevs: np.ndarray,
        train_label_noise: float,
        class1_prob: float = 0.5,
        seed=42
) -> typing.Tuple[
        typing.Tuple[np.ndarray, np.ndarray],  # training data
        typing.Tuple[np.ndarray, np.ndarray],  # test data
        np.ndarray,  # label noise indices
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]  # mean, std, extents of mixture components
]:
    rng = np.random.default_rng(seed=seed)
    num_samples = train_samples + test_samples

    if len(stddevs.shape) != 2 or stddevs.shape[0] != 2:
        raise ValueError(f'stddevs must be a (2 x dim) matrix but has shape {stddevs.shape}')
    dim = stddevs.shape[1]

    # Means are calculated such that
    # 1. in each dimension samples for component 1 are at least distance apart
    #    from samples from component 2 with probability distance_prob,
    #    i.e. the l_inf distance between two samples from different components
    #    is at least distance with distance_prob,
    # 2. and the means are symmetric around 0 (per dimension).
    means = np.zeros((2, dim))
    extents = np.zeros_like(means)
    extents[0, :] = scipy.stats.norm.ppf(distance_prob) * stddevs[0, :]
    extents[1, :] = -scipy.stats.norm.ppf(distance_prob) * stddevs[1, :]
    means[1, :] = (distance + extents[0, :] - extents[1, :]) / 2.0
    means[0, :] = -means[1, :]

    # Sample labels
    # 0 = inner, 1 = outer
    labels = (rng.uniform(size=(num_samples,)) < class1_prob).astype(np.int)

    # Sample points according to their assigned component
    points = rng.normal(means[labels], stddevs[labels])

    # Apply label noise to training data
    if train_label_noise > 0:
        assert train_label_noise < 1
        flip_indices = rng.permutation(train_samples)[:int(train_samples * train_label_noise)]
        labels[flip_indices] = 1 - labels[flip_indices]
    else:
        flip_indices = np.zeros((0,))

    return \
        (points[:train_samples], labels[:train_samples]),\
        (points[train_samples:], labels[train_samples:]),\
        flip_indices,\
        (means, stddevs, extents)


def make_2gmm(
        mu: np.ndarray,
        sigma: float,
        train_samples: int,
        test_samples: int,
        seed,
        train_label_noise: float = 0.0,
        class1_prob: float = 0.5
) -> typing.Tuple[
    typing.Tuple[np.ndarray, np.ndarray],  # training data
    typing.Tuple[np.ndarray, np.ndarray],  # test data
    np.ndarray,  # label noise indices (flipped)
    typing.Tuple[np.ndarray, np.ndarray]  # label noise indices (covariate noise), train and test
]:
    rng = np.random.default_rng(seed=seed)
    num_samples = train_samples + test_samples

    if sigma <= 0:
        raise ValueError(f'Standard deviation sigma must be positive but is {sigma}')

    # Sample labels from {0, 1} with given probability of class 1
    labels = (rng.uniform(size=(num_samples,)) < class1_prob).astype(np.int)
    pm_labels = np.where(labels == 1, 1.0, -1.0)

    # Sample points according to their assigned component
    means = np.tile(mu, (num_samples, 1)) * np.expand_dims(pm_labels, axis=-1)
    points = rng.normal(means, sigma)

    # Apply label noise to training data
    if train_label_noise > 0:
        assert train_label_noise < 1
        flip_indices = rng.permutation(train_samples)[:int(train_samples * train_label_noise)]
        flip_indices = np.sort(flip_indices)
        labels[flip_indices] = 1 - labels[flip_indices]
    else:
        flip_indices = np.zeros((0,))

    # Determine indices of mislabeled samples due to covariate noise
    train_noise_indices = _2gmm_noise_indices(
        points,
        pm_labels,
        noise_candidate_indices=np.delete(np.arange(train_samples, dtype=np.int), flip_indices),
        mu=mu
    )
    test_noise_indices = _2gmm_noise_indices(
        points,
        pm_labels,
        noise_candidate_indices=np.arange(num_samples, dtype=np.int)[train_samples:],
        mu=mu
    )

    return \
        (points[:train_samples], labels[:train_samples]),\
        (points[train_samples:], labels[train_samples:]),\
        flip_indices, \
        (train_noise_indices, test_noise_indices)


def _2gmm_noise_indices(
        points: np.ndarray,
        pm_labels: np.ndarray,
        noise_candidate_indices: np.ndarray,
        mu: np.ndarray
) -> np.ndarray:
    noise_candidate_predictions = pm_labels[noise_candidate_indices] * np.matmul(points[noise_candidate_indices], mu)
    noise_indices = noise_candidate_indices[noise_candidate_predictions <= 0]
    return np.sort(noise_indices)


def make_gaussian_logistic(
        train_samples: int,
        test_samples: int,
        x_covariance_diagonal: np.ndarray,
        logits_noise_variance: float,
        ground_truth: np.ndarray,
        label_noise_fraction: float,
        min_decision_boundary_distance: float,
        seed
) -> typing.Tuple[
    typing.Tuple[np.ndarray, np.ndarray],  # training data
    typing.Tuple[np.ndarray, np.ndarray],  # test data
    np.ndarray,  # label noise indices
    typing.Tuple[np.ndarray, np.ndarray]  # train and test covariate noise indices
]:
    if np.any(x_covariance_diagonal < 0):
        raise ValueError('Diagonal entries of the covariance matrix for the x must all be non-negative')

    if x_covariance_diagonal.ndim != 1:
        raise ValueError(
            f'Diagonal entries of the covariance matrix must be a vector but has shape {x_covariance_diagonal.shape}'
        )

    if logits_noise_variance < 0:
        raise ValueError(
            f'Variance of Gaussian noise in logits (sigma_0) must be non-negative but was {logits_noise_variance}'
        )

    if ground_truth.shape != x_covariance_diagonal.shape:
        raise ValueError(
            f'Ground truth must have same shape as diagonal entries of the covariance matrix'
            f' but shapes are {ground_truth.shape} and {x_covariance_diagonal.shape} respectively'
        )

    if not (0 <= label_noise_fraction < 1.0):
        raise ValueError(f'Label noise fraction must be in [0, 1) but is {label_noise_fraction}')

    if min_decision_boundary_distance < 0:
        raise ValueError(f'Min distance to decision boundary must be non-negative but is {min_decision_boundary_distance}')

    rng = np.random.default_rng(seed=seed)
    dim, = ground_truth.shape

    # Sample xs from a zero-mean Gaussian with given diagonal covariance
    num_samples = train_samples + test_samples
    # FIXME: Could fix this generally for diagonal covariance, is quite slow
    if np.all(x_covariance_diagonal == 1.0):
        # In the case of identity covariance, sampling from a standard normal directly is
        # much more computationally efficient.
        xs = rng.normal(loc=0.0, scale=1.0, size=(num_samples, dim))
    else:
        x_covariance = np.diag(x_covariance_diagonal)
        xs = rng.multivariate_normal(
            mean=np.zeros((dim,)),
            cov=x_covariance,
            size=num_samples
        )

    # Resample as long as there are samples too close to the ground truth decision boundary
    if min_decision_boundary_distance > 0.0:
        # FIXME: This code only works for identity covariance right now
        assert np.all(x_covariance_diagonal == 1.0), 'Min distance to db only implemented for identity covariance'
        ground_truth_normalized = ground_truth / np.linalg.norm(ground_truth)
        too_close_indices, = np.nonzero(np.abs(np.dot(xs, ground_truth_normalized)) < min_decision_boundary_distance)
        while too_close_indices.shape[0] > 0:
            # Sample new data points
            resampled_xs = rng.normal(loc=0.0, scale=1.0, size=(len(too_close_indices), dim))
            xs[too_close_indices] = resampled_xs
            too_close_indices, = np.nonzero(np.abs(np.dot(xs, ground_truth_normalized)) < min_decision_boundary_distance)

        assert np.all(np.abs(np.dot(xs, ground_truth_normalized)) >= min_decision_boundary_distance)

    # Calculate unnormalized predictions
    true_logits = np.dot(xs, ground_truth)
    logits_noise = rng.normal(loc=0.0, scale=np.sqrt(label_noise_fraction), size=(num_samples,))
    logits = true_logits + logits_noise

    # Calculate labels
    ys = (logits >= 0).astype(np.int)
    assert ys.ndim == 1

    # Split data into training and test
    train_xs, train_ys = xs[:train_samples], ys[:train_samples]
    test_xs, test_ys = xs[train_samples:], ys[train_samples:]

    # Determine covariante noise indices, i.e. where the logits noise flipped the label
    train_covariate_noise_indices = np.arange(train_samples)[np.sign(true_logits[:train_samples]) != np.sign(logits[:train_samples])]
    test_covariate_noise_indices = np.arange(test_samples)[np.sign(true_logits[train_samples:]) != np.sign(logits[train_samples:])]

    # Flip given fraction of training labels
    if label_noise_fraction > 0:
        # Determine to which samples to flip and make sure indices are sorted
        assert label_noise_fraction < 1
        flip_indices = rng.permutation(train_samples)[:int(train_samples * label_noise_fraction)]
        flip_indices = np.sort(flip_indices)

        # and flip
        train_ys[flip_indices] = 1 - train_ys[flip_indices]
    else:
        flip_indices = np.zeros((0,))

    return (train_xs, train_ys), (test_xs, test_ys), flip_indices, (train_covariate_noise_indices, test_covariate_noise_indices)


def make_gaussian_linear(
        num_samples: int,
        x_covariance_diagonal: np.ndarray,
        y_noise_variance: float,
        ground_truth: np.ndarray,
        rademacher_noise_fraction: float,
        seed,
        rademacher_noise_scale: float = 0.0
) -> typing.Tuple[
    np.ndarray,  # xs
    np.ndarray,  # ys
    np.ndarray  # sample indices with Rademacher noise
]:
    if np.any(x_covariance_diagonal < 0):
        raise ValueError('Diagonal entries of the covariance matrix for the x must all be non-negative')

    if x_covariance_diagonal.ndim != 1:
        raise ValueError(
            f'Diagonal entries of the covariance matrix must be a vector but has shape {x_covariance_diagonal.shape}'
        )

    if y_noise_variance < 0:
        raise ValueError(f'Variance of Gaussian noise in y (sigma_0) must be non-negative but was {y_noise_variance}')

    if ground_truth.shape != x_covariance_diagonal.shape:
        raise ValueError(
            f'Ground truth must have same shape as diagonal entries of the covariance matrix'
            f' but shapes are {ground_truth.shape} and {x_covariance_diagonal.shape} respectively'
        )

    if not (0 <= rademacher_noise_fraction < 1.0):
        raise ValueError(f'Rademacher noise fraction must be in [0, 1) but is {rademacher_noise_fraction}')

    if rademacher_noise_fraction > 0 and rademacher_noise_scale <= 0:
        raise ValueError(f'Rademacher noise scale must be positive but is {rademacher_noise_scale}')

    rng = np.random.default_rng(seed=seed)
    dim, = ground_truth.shape

    # Sample xs from a zero-mean Gaussian with given diagonal covariance
    # FIXME: Could fix this generally for diagonal covariance, is quite slow
    if np.all(x_covariance_diagonal == 1.0):
        # In the case of identity covariance, sampling from a standard normal directly is
        # much more computationally efficient.
        xs = rng.normal(loc=0.0, scale=1.0, size=(num_samples, dim))
    else:
        x_covariance = np.diag(x_covariance_diagonal)
        xs = rng.multivariate_normal(
            mean=np.zeros((dim,)),
            cov=x_covariance,
            size=num_samples
        )
    assert xs.shape == (num_samples, dim)

    # Calculate targets
    ys = np.dot(xs, ground_truth) + rng.normal(loc=0.0, scale=np.sqrt(y_noise_variance), size=(num_samples,))

    # Add Rademacher noise to the given fraction of targets
    if rademacher_noise_fraction > 0:
        # Determine to which samples to add noise
        assert rademacher_noise_fraction < 1
        rademacher_noise_indices = rng.permutation(num_samples)[:int(num_samples * rademacher_noise_fraction)]
        rademacher_noise_indices = np.sort(rademacher_noise_indices)

        # Add noise
        rademacher_noise = rng.integers(0, 1, size=(rademacher_noise_indices.shape[0]), endpoint=True).astype(np.float)
        rademacher_noise = (rademacher_noise - 0.5) * 2.0
        assert np.all((rademacher_noise == 1.0) | (rademacher_noise == -1.0))
        ys[rademacher_noise_indices] += rademacher_noise * rademacher_noise_scale
    else:
        rademacher_noise_indices = np.zeros((0,))

    return xs, ys, rademacher_noise_indices


def make_image_dataset(
        dataset: Dataset,
        train_samples: typing.Optional[int] = None,
        data_split: DataSplit = DataSplit.NoSplit,
        validation_samples: typing.Optional[int] = None,
        binarized_classes: typing.Optional[typing.Tuple[int, int]] = None,
        train_discard_indices: typing.Optional[typing.Tuple[int, ...]] = None,
        train_label_noise: float = 0.0,
        seed: typing.Union[int, np.random.Generator] = 1
) -> typing.Tuple[
        typing.Tuple[np.ndarray, np.ndarray],  # training data
        typing.Tuple[np.ndarray, np.ndarray],  # test data
        np.ndarray  # label noise indices
]:
    rng = np.random.default_rng(seed=seed)

    # Load raw data
    (xs_train, ys_train), (xs_test, ys_test), num_classes = _load_raw_image_dataset(dataset.name, binarized_classes)

    # Discard training samples a priori if given
    if train_discard_indices is not None and len(train_discard_indices) > 0:
        xs_train = np.delete(xs_train, train_discard_indices, axis=0)
        ys_train = np.delete(ys_train, train_discard_indices, axis=0)

    # Handle train/test/val split if specified
    if data_split == DataSplit.NoSplit:
        num_validation_samples = 0
    else:
        if validation_samples is None or validation_samples < 0:
            raise ValueError(
                f'If a train/test/val split is used, number of validation samples '
                f'must be specified and non-negative but is {validation_samples}'
            )
        num_validation_samples = validation_samples

    # Check whether enough samples in total are available
    if num_validation_samples > xs_train.shape[0]:
        raise ValueError(
            f'Requested {num_validation_samples} validation samples but only have {xs_train.shape[0]} total samples'
        )
    if train_samples is not None and train_samples + num_validation_samples > xs_train.shape[0]:
        raise ValueError(
            f'Requested {train_samples} + {num_validation_samples} samples but only have {xs_train.shape[0]}'
        )
    num_train_samples = train_samples if train_samples is not None else xs_train.shape[0] - num_validation_samples
    assert num_train_samples >= 0
    num_total_samples = num_train_samples + num_validation_samples

    # Subsample training set uniformly if necessary
    if xs_train.shape[0] > num_total_samples:
        subsampled_indices = rng.permutation(xs_train.shape[0])[:num_total_samples]
        xs_train = xs_train[subsampled_indices]
        ys_train = ys_train[subsampled_indices]

    # Split training set into training and validation if required, and adjust the returned test set accordingly
    if data_split != DataSplit.NoSplit:
        shuffled_indices = rng.permutation(num_total_samples)
        train_indices = shuffled_indices[:num_train_samples]
        validation_indices = shuffled_indices[num_train_samples:]

        # Adjust returned test set if validation set should be used for evaluation
        if data_split == DataSplit.SplitEvalValidation:
            xs_test, ys_test = xs_train[validation_indices], ys_train[validation_indices]
        xs_train, ys_train = xs_train[train_indices], ys_train[train_indices]

    # Apply label noise to training data
    if train_label_noise > 0:
        assert train_label_noise < 1
        flip_indices = rng.permutation(xs_train.shape[0])[:int(xs_train.shape[0] * train_label_noise)]
        true_labels = ys_train[flip_indices]
        new_label_indices = rng.integers(low=0, high=num_classes - 1, size=(true_labels.shape[0]))
        new_labels = np.where(new_label_indices < true_labels, new_label_indices, new_label_indices + 1)
        assert np.all(true_labels != new_labels)
        ys_train[flip_indices] = new_labels
    else:
        flip_indices = np.zeros((0,))

    # Sort flipped indices for faster access later
    flip_indices = np.sort(flip_indices)

    return (xs_train, ys_train), (xs_test, ys_test), flip_indices


def _load_raw_image_dataset(
        dataset_name: str,
        binarized_classes: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Tuple[
        typing.Tuple[np.ndarray, np.ndarray],  # training data
        typing.Tuple[np.ndarray, np.ndarray],  # test data
        int  # num classes
]:
    train_ds, dataset_info = tfds.load(dataset_name, split=tfds.Split.TRAIN, data_dir=ir.util.DATA_DIR, with_info=True)
    test_ds = tfds.load(dataset_name, split=tfds.Split.TEST, data_dir=ir.util.DATA_DIR, with_info=False)
    original_num_classes = dataset_info.features['label'].num_classes

    # Check for valid class indices for binarized datasets
    if binarized_classes is not None:
        class_0, class_1 = binarized_classes
        if not (0 <= class_0 < original_num_classes) or not (0 <= class_1 < original_num_classes):
            raise ValueError(
                f'Binary class indices must be in [0, {original_num_classes - 1}] but were {binarized_classes}'
            )
        num_classes = 2
    else:
        num_classes = original_num_classes

    def process_data(all_data: typing.Dict[str, np.ndarray]):
        xs = all_data['image']
        ys = all_data['label']

        # Make sure dtypes are correct
        xs = xs.astype(np.float32)
        ys = ys.astype(np.int)

        # Binarize if necessary
        if binarized_classes is not None:
            # Filter by classes
            class_0, class_1 = binarized_classes
            sample_indices = (ys == class_0) | (ys == class_1)
            xs = xs[sample_indices]
            ys = ys[sample_indices]

            # Convert label to 0/1
            ys = np.where(ys == class_0, np.zeros_like(ys), np.ones_like(ys))

        # Convert images to [0, 1] range
        xs = xs / 255.0

        return xs, ys

    train_data, = tfds.as_numpy(train_ds.batch(dataset_info.splits['train'].num_examples))
    test_data, = tfds.as_numpy(test_ds.batch(dataset_info.splits['test'].num_examples))

    xs_train, ys_train = process_data(train_data)
    xs_test, ys_test = process_data(test_data)

    return (xs_train, ys_train), (xs_test, ys_test), num_classes
