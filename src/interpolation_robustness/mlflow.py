import logging
import random
import time
import typing

import mlflow
import mlflow.entities
import numpy as np
import pandas as pd

MetricsMap = typing.Dict[str, typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]]


def load_metrics(
        runs: pd.DataFrame,
        metrics: typing.Tuple[str],
        client: mlflow.tracking.MlflowClient
) -> MetricsMap:
    """
    Load the specified set of metrics from MLFlow and return as a dictionary
    where the keys are run ids, the values are tuples of a numpy array and a dict (epochs, metrics).
    Epochs contains the epochs at which the metrics were recorded, metrics maps from metric names
    to a numpy array containing the values.
    Note that this method requires all metrics to be recorded at the same epochs.
    :param runs: DataFrame containing all run information as provided by MLFlow.
    :param metrics: Metrics to load without the `metric.` prefix.
    :param client: MLFlow client instance to use.
    :return: Dictionary mapping from run ids to tuples (epochs, {metric: values}).
    """

    if len(metrics) == 0:
        raise ValueError('Expected at least one metric to fetch but received none')

    result = dict()
    for run_id in runs['run_id']:
        current_metrics = dict()
        epochs = None

        for metric in metrics:
            raw_data = client.get_metric_history(run_id, metric)
            current_epochs = np.zeros(len(raw_data), dtype=np.int)
            current_values = np.zeros_like(current_epochs, dtype=np.float)
            for idx, current_record in enumerate(sorted(raw_data, key=lambda record: record.step)):
                current_epochs[idx] = current_record.step
                current_values[idx] = current_record.value

            if epochs is None:
                epochs = current_epochs
            else:
                if not np.all(epochs == current_epochs):
                    raise ValueError(f'Some of the metrics {metrics} of run {run_id} were recorded at different steps')

            current_metrics[metric] = current_values

        assert epochs is not None
        result[run_id] = epochs, current_metrics

    return result


def set_experiment(experiment_name):
    retry(lambda: mlflow.set_experiment(experiment_name))


def start_run(run_id=None, experiment_id=None, run_name=None, nested=False, tags=None):
    run = retry(lambda: mlflow.start_run(run_id, experiment_id, run_name, nested, tags))

    # Monkey patch the exit method to also retry if necessary
    def patched_exit(self, exc_type, exc_val, exc_tb):
        return retry(lambda: self.__exit__(exc_type, exc_val, exc_tb))
    run.__exit__ = patched_exit

    return run


def set_tag(key, value):
    retry(lambda: mlflow.set_tag(key, value))


def log_params(params):
    retry(lambda: mlflow.log_params(params))


def log_metric(key, value, step=None):
    retry(lambda: mlflow.log_metric(key, value, step))


def log_metrics(metrics, step=None):
    retry(lambda: mlflow.log_metrics(metrics, step))


def set_tags(tags: typing.Dict[str, typing.Any]):
    retry(lambda: mlflow.set_tags(tags))


class DelayedMetricLogger(object):
    _MAX_METRICS_PER_BATCH = 1000

    def __init__(self, run_id: str, save_every: int, num_epochs: int):
        self._run_id = run_id
        self._save_every = save_every
        self._num_epochs = num_epochs
        self._stored_metrics = []

    def add_metrics(self, metrics: typing.Dict[str, float], step: int):
        timestamp = int(time.time() * 1000)
        self._stored_metrics.extend(
            mlflow.entities.Metric(key, value, timestamp, step) for key, value in metrics.items()
        )

    def submit(self, epoch: int):
        # Submit metrics only if epoch criteria are met
        if epoch % self._save_every == 0 or epoch == self._num_epochs - 1:
            for current_batch_idx in range(0, len(self._stored_metrics), self._MAX_METRICS_PER_BATCH):
                retry(
                    lambda: mlflow.tracking.MlflowClient().log_batch(
                        run_id=self._run_id,
                        metrics=self._stored_metrics[current_batch_idx:current_batch_idx+self._MAX_METRICS_PER_BATCH]
                    )
                )
            self._stored_metrics = []


# FIXME: This is super ugly, hacky, and also a bit dangerous:
#  currently, this catches *all* exceptions, not only server-related exceptions.
#  Hence, user-error might actually result in a method looping forever!
def retry(
        fn: typing.Callable[[], typing.Any],
        min_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 120.0,
        backoff_jitter: float = 0.2
) -> typing.Any:
    current_sleep = min_backoff
    while True:
        # noinspection PyBroadException
        try:
            return fn()
        except:
            logging.getLogger(__name__).warning(
                'Caught exception in retry method, sleeping for %f seconds',
                current_sleep,
                exc_info=True
            )
            time.sleep(current_sleep)
            current_sleep = min(current_sleep * backoff_factor, max_backoff)
            # Apply some random jitter to avoid all processes retrying at the same time
            current_sleep += random.uniform(-backoff_jitter, backoff_jitter) * current_sleep
            current_sleep = max(min_backoff, current_sleep)
