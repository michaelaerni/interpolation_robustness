#!/bin/bash

EXPERIMENT_NAME='logistic_regression_noise_levels'

# Parameter ranges
L2_LAMBDAS=( 1000.0 500.0 100.0 50.0 10.0 9.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.03 0.01 0.005 0.003 0.001 0.0005 0.0001 0.00005 0.00001 0.0 )
DATA_DIMENSIONS=( 8000 7000 6000 5000 4000 3000 2000 1500 1000 500 250 100 ) # n=1000
LABEL_NOISES=( 0.0 0.1 )
SEEDS=( 1 2 3 4 5 )

# Constant experiment parameters
TEST_ATTACK_EPSILON=0.1
TRAIN_ATTACK_EPSILON=0.1
ATTACK_P='inf'
NUM_SAMPLES=1000
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0
CONSISTENT_ATTACK=true

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  data_dimension=$1
  train_attack_epsilon=$2
  label_noise=$3
  seed=$4

  if $CONSISTENT_ATTACK; then
    consistent_attack_flag='--attack-train-consistent'
    consistent_attack_tag='consistent'
  else
    consistent_attack_flag=''
    consistent_attack_tag='inconsistent'
  fi

  if [ "${train_attack_epsilon}" != "0.0" ]; then
    training_tag='at'
  else
    training_tag='st'
  fi

  python -m interpolation_robustness.experiments.logistic_regression_dcp \
    --attack-epsilon $TEST_ATTACK_EPSILON \
    $consistent_attack_flag \
    --train-attack-epsilon $train_attack_epsilon \
    --attack-p "${ATTACK_P}" \
    --data-dim $data_dimension \
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
  for data_dimension in "${DATA_DIMENSIONS[@]}"; do
    for label_noise in "${LABEL_NOISES[@]}"; do
      run_configuration "${data_dimension}" $TRAIN_ATTACK_EPSILON "${label_noise}" $seed
    done
  done
done
