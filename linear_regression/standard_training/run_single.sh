#!/bin/bash

EXPERIMENT_NAME='linear_regression_standard_training'

# Parameters given
DATA_DIM=$1
GAUSSIAN_NOISE_VARIANCE=$2

# Parameters to sweep over
L2_PENALTIES=( 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 4.0 6.0 8.0 10.0 15.0 20.0 30.0 40.0 50.0 75.0 )

# Constant experiment parameters
NUM_SAMPLES=1000
NUM_EPOCHS=2000
LEARNING_RATE_WARMUP=250
EVAL_EVERY_EPOCHS=500
LOG_EVERY_EPOCHS=1
GROUND_TRUTH='single_entry'
COVARIANCE_DIAGONAL='identity'
TEST_ATTACK_EPSILON=0.4
BASE_LEARNING_RATE=1.0

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  l2_penalty=$1

  learning_rate=$(python -c "import math; print(${BASE_LEARNING_RATE} * math.sqrt(1.0 / ${DATA_DIM}))")

  python -m interpolation_robustness.experiments.linear_regression \
    --tag "${EXPERIMENT_NAME}" \
    --data-dim $DATA_DIM \
    --training-samples $NUM_SAMPLES \
    --l2 $l2_penalty \
    --data-ground-truth "${GROUND_TRUTH}" \
    --data-covariance-diagonal "${COVARIANCE_DIAGONAL}" \
    --attack-epsilon $TEST_ATTACK_EPSILON \
    --epochs $NUM_EPOCHS \
    --learning-rate $learning_rate \
    --learning-rate-warmup $LEARNING_RATE_WARMUP \
    --data-noise-variance $GAUSSIAN_NOISE_VARIANCE \
    --eval-every-epochs $EVAL_EVERY_EPOCHS \
    --log-every-epochs $LOG_EVERY_EPOCHS
}

for l2_penalty in "${L2_PENALTIES[@]}"; do
  run_configuration "${l2_penalty}"
done
