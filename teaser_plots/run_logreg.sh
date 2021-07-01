#!/bin/bash

EXPERIMENT_NAME='teaser_plot_logreg'

# Parameter ranges
L2_LAMBDAS=( 10000.0 5000.0 1000.0 500.0 100.0 50.0 10.0 5.0 3.0 1.0 0.8 0.5 0.1 0.05 0.03 0.01 0.005 0.003 0.001 0.0005 0.0001 0.00005 0.00001 0.0 )
DATA_DIMENSIONS=( 10000 8000 6000 4000 3000 2000 1500 1000 500 250 100 ) # n=1000
LABEL_NOISES=( 0.0 0.02 )

# Constant experiment parameters
TEST_ATTACK_EPSILON=0.1
TRAIN_ATTACK_EPSILON=0.1
CONSISTENT_ATTACKS=true
ATTACK_P='inf'
NUM_SAMPLES=1000
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  data_dimension=$1
  label_noise=$2

  if $CONSISTENT_ATTACKS; then
    consistent_attack_flag='--attack-train-consistent'
  else
    consistent_attack_flag=''
  fi

  if [ "${TRAIN_ATTACK_EPSILON}" != "0.0" ]; then
    training_tag='at'
  else
    training_tag='st'
  fi

  python -m interpolation_robustness.experiments.logistic_regression_dcp \
    --attack-epsilon $TEST_ATTACK_EPSILON \
    $consistent_attack_flag \
    --train-attack-epsilon $TRAIN_ATTACK_EPSILON \
    --attack-p "${ATTACK_P}" \
    --data-dim $data_dimension \
    --training-samples $NUM_SAMPLES \
    --verbose \
    --tag "${EXPERIMENT_NAME}" \
    --l2 ${L2_LAMBDAS[*]} \
    --solvers ${SOLVERS[*]} \
    --label-noise $label_noise \
    --data-logits-noise-variance $GAUSS_NOISE_VARIANCE \
    --logdir "${LOG_DIR}"
}

mkdir -p "${LOG_DIR}"

for label_noise in "${LABEL_NOISES[@]}"; do
  for data_dimension in "${DATA_DIMENSIONS[@]}"; do
    run_configuration "${data_dimension}" "${label_noise}"
  done
done
