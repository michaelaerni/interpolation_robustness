#!/bin/bash

EXPERIMENT_NAME='logistic_regression_noise_increase_reg'

# Parameter ranges
LABEL_NOISES=( 0.0 0.005 0.015 0.024 0.04 0.064 0.1 0.18 0.3 0.4 0.5 )
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
L2_LAMBDAS=( 1000.0 500.0 100.0 50.0 10.0 9.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.03 0.01 0.005 0.003 0.001 0.0005 0.0001 0.00005 0.00001 0.0 )

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
    --l2 ${L2_LAMBDAS[*]} \
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
