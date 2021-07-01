#!/bin/bash

EXPERIMENT_NAME='logistic_regression_explain_ro'

# Parameter ranges
L2_LAMBDAS=( 10000.0 5000.0 1000.0 500.0 100.0 50.0 10.0 5.0 3.0 1.0 0.8 0.5 0.1 0.05 0.03 0.01 0.005 0.003 0.001 0.0005 0.0001 0.00005 0.00001 0.0 )

# Constant experiment parameters
TEST_ATTACK_EPSILON=0.1
TRAIN_ATTACK_EPSILON=0.1
ATTACK_P='inf'
NUM_SAMPLES=1000
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0
DATA_DIMENSION=8000 # n=1000 => d/n = 8
LABEL_NOISE=0.0
CONSISTENT_ATTACK=true

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
