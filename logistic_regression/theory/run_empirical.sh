#!/bin/bash

EXPERIMENT_NAME='logistic_regression_theory_empirical'

# Parameter ranges
L2_LAMBDAS=( 1.0 0.0 )
DATA_DIMENSIONS=( 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 ) # n=1k
TEST_ATTACK_EPSILONS=( 0.05 0.02 )

# Constant experiment parameters
ATTACK_P='inf'
NUM_SAMPLES=1000
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0
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

for data_dimension in "${DATA_DIMENSIONS[@]}"; do
  for test_attack_epsilon in "${TEST_ATTACK_EPSILONS[@]}"; do
    train_attack_epsilon=$test_attack_epsilon

    python -m interpolation_robustness.experiments.logistic_regression_dcp \
      --attack-epsilon $test_attack_epsilon \
      $consistent_attack_flag \
      --train-attack-epsilon $train_attack_epsilon \
      --attack-p "${ATTACK_P}" \
      --data-dim $data_dimension \
      --training-samples $NUM_SAMPLES \
      --verbose \
      --tag "${EXPERIMENT_NAME}" \
      --l2 ${L2_LAMBDAS[*]} \
      --solvers ${SOLVERS[*]} \
      --label-noise $LABEL_NOISE \
      --data-logits-noise-variance $GAUSS_NOISE_VARIANCE \
      --logdir "${LOG_DIR}"
  done
done
