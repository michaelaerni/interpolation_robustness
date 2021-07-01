#!/bin/bash

EXPERIMENT_NAME='logistic_regression_noise_increase'

# Parameter ranges
LABEL_NOISES=( 0.0 0.005 0.008 0.014 0.024 0.038 0.064 0.1 0.18 0.3 0.5 )
SEEDS=( 1 2 3 4 5 )

# Constant experiment parameters
TEST_ATTACK_EPSILON=0.1
TRAIN_ATTACK_EPSILON=0.1
ATTACK_P='inf'
NUM_SAMPLES=1000
DATA_DIMENSION=8000
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0
CONSISTENT_ATTACK=true
L2_LAMBDA=0.0

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  label_noise=$1
  seed=$2

  if $CONSISTENT_ATTACK; then
    consistent_attack_flag='--attack-train-consistent'
    consistent_attack_tag='consistent'
  else
    consistent_attack_flag=''
    consistent_attack_tag='inconsistent'
  fi

  python -m interpolation_robustness.experiments.logistic_regression_dcp \
    --attack-epsilon $TEST_ATTACK_EPSILON \
    $consistent_attack_flag \
    --train-attack-epsilon $TRAIN_ATTACK_EPSILON \
    --attack-p "${ATTACK_P}" \
    --data-dim $DATA_DIMENSION \
    --training-samples $NUM_SAMPLES \
    --verbose \
    --tag "${EXPERIMENT_NAME}" \
    --l2 $L2_LAMBDA \
    --solvers ${SOLVERS[*]} \
    --label-noise $label_noise \
    --data-logits-noise-variance $GAUSS_NOISE_VARIANCE \
    --seed $seed
}

mkdir -p "${LOG_DIR}"

for seed in "${SEEDS[@]}"; do
  for label_noise in "${LABEL_NOISES[@]}"; do
    run_configuration "${label_noise}" "${seed}"
  done
done
