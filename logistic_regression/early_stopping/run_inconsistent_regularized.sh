#!/bin/bash

EXPERIMENT_NAME='logistic_regression_early_stopping_inconsistent_regularized'

# Parameter ranges
L2_LAMBDAS=( 1000000.0 100000.0 10000.0 5000.0 1000.0 500.0 100.0 50.0 10.0 9.0 8.0 7.0 6.0 5.0 4.0 3.0 3.5 3.0 2.5 2.0 1.9 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 0.008 0.005 0.003 0.001 0.0005 0.0001 0.00005 0.00001 0.000001 0.0 )

# Constant experiment parameters
TEST_ATTACK_EPSILON=0.1
TRAIN_ATTACK_EPSILON=0.1
ATTACK_P='inf'
NUM_SAMPLES=1000
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0
DATA_DIMENSION=8000 # n=1000 => d/n = 8
LABEL_NOISE=0.0
CONSISTENT_ATTACK=false

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

if $CONSISTENT_ATTACK; then
  consistent_attack_flag='--attack-train-consistent'
else
  consistent_attack_flag=''
fi

mkdir -p "${LOG_DIR}"

python -m interpolation_robustness.experiments.logistic_regression_dcp \
  --attack-epsilon $TEST_ATTACK_EPSILON \
  $consistent_attack_flag \
  --train-attack-epsilon $TRAIN_ATTACK_EPSILON \
  --attack-p "${ATTACK_P}" \
  --data-dim $DATA_DIMENSION \
  --training-samples $NUM_SAMPLES \
  --verbose \
  --tag "${EXPERIMENT_NAME}" \
  --l2 ${L2_LAMBDAS[*]} \
  --solvers ${SOLVERS[*]} \
  --label-noise $LABEL_NOISE \
  --data-logits-noise-variance $GAUSS_NOISE_VARIANCE \
  --logdir "${LOG_DIR}"
