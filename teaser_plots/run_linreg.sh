#!/bin/bash

EXPERIMENT_NAME='teaser_plot_linreg'

# Parameter ranges
L2_LAMBDAS=( 75.0 50.0 10.0 8.0 6.0 4.0 2.0 1.0 0.8 0.6 0.4 0.3 0.2 0.1 0.05 0.03 0.01 0.005 0.003 0.001 0.0001 0.00001 0.0 )
DATA_DIMENSIONS=( 10000 8000 6000 4000 3000 2000 1500 1000 500 250 100 ) # n=1000
NOISE_VARIANCES=( 0.0 0.2 )

# Constant experiment parameters
NUM_SAMPLES=1000
NUM_EPOCHS=2000
LEARNING_RATE_WARMUP=250
GROUND_TRUTH='single_entry'
COVARIANCE_DIAGONAL='identity'
BASE_LEARNING_RATE=1.0
TRAIN_ATTACK_EPSILON=0.0
TEST_ATTACK_EPSILON=0.4
EVAL_EVERY_EPOCHS=500
LOG_EVERY_EPOCHS=500

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../)"
MODULE_DIR="${REPOSITORY_DIR}/src"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

if [ "${TRAIN_ATTACK_EPSILON}" != "0.0" ]; then
  AT_FLAG='--adversarial-training'
else
  AT_FLAG=''
fi

function run_configuration {
  data_dim=$1
  l2_lambda=$2
  gaussian_noise_variance=$3
  learning_rate=$(python -c "import math; print(${BASE_LEARNING_RATE} * math.sqrt(1.0 / ${data_dim}))")

  python -m interpolation_robustness.experiments.linear_regression \
    --tag "${EXPERIMENT_NAME}" \
    --data-dim $data_dim \
    --training-samples $NUM_SAMPLES \
    --l2 $l2_lambda \
    --data-ground-truth "${GROUND_TRUTH}" \
    --data-covariance-diagonal "${COVARIANCE_DIAGONAL}" \
    --attack-epsilon $TEST_ATTACK_EPSILON \
    $AT_FLAG \
    --train-attack-epsilon $TRAIN_ATTACK_EPSILON \
    --epochs $NUM_EPOCHS \
    --learning-rate $learning_rate \
    --learning-rate-warmup $LEARNING_RATE_WARMUP \
    --data-noise-variance $gaussian_noise_variance \
    --eval-every-epochs $EVAL_EVERY_EPOCHS \
    --log-every-epochs $LOG_EVERY_EPOCHS
}

for gaussian_noise_variance in "${NOISE_VARIANCES[@]}"; do
  for data_dimension in "${DATA_DIMENSIONS[@]}"; do
    for l2_penalty in "${L2_LAMBDAS[@]}"; do
      run_configuration "${data_dimension}" "${l2_penalty}" "${gaussian_noise_variance}"
    done
  done
done
