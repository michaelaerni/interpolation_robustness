#!/bin/bash

EXPERIMENT_NAME='logistic_regression_inconsistent_consistent_increase_epsilon'

# Parameter ranges
SEEDS=( 1 2 3 4 5 )
ATTACK_EPSILONS=( 0.0 0.05 0.1 0.15 0.2 0.25 0.3 )
SAMPLE_COUNTS=( 200 1000 )

# Constant experiment parameters
ATTACK_P='inf'
SOLVERS=( 'MOSEK' 'SCS' 'ECOS' )
GAUSS_NOISE_VARIANCE=0.0
LABEL_NOISE=0.0
L2_LAMBDA=0.0
DATA_DIMENSION=500

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  attack_epsilon=$1
  num_samples=$2
  consistent_attack=$3
  seed=$4

  if $consistent_attack; then
    consistent_attack_flag='--attack-train-consistent'
    consistent_attack_tag='consistent'
  else
    consistent_attack_flag=''
    consistent_attack_tag='inconsistent'
  fi

  if [ "${attack_epsilon}" != "0.0" ]; then
    training_tag='at'
  else
    training_tag='st'
  fi

  python -m interpolation_robustness.experiments.logistic_regression_dcp \
    --attack-epsilon $attack_epsilon \
    $consistent_attack_flag \
    --train-attack-epsilon $attack_epsilon \
    --attack-p "${ATTACK_P}" \
    --data-dim $DATA_DIMENSION \
    --training-samples $num_samples \
    --verbose \
    --tag "${EXPERIMENT_NAME}" \
    --l2 $L2_LAMBDA \
    --solvers ${SOLVERS[*]} \
    --label-noise $LABEL_NOISE \
    --data-logits-noise-variance $GAUSS_NOISE_VARIANCE \
    --seed $seed
}

mkdir -p "${LOG_DIR}"

for attack_epsilon in "${ATTACK_EPSILONS[@]}"; do
  for num_samples in "${SAMPLE_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # Run consistent and inconsistent
      run_configuration $attack_epsilon $num_samples true $seed
      run_configuration $attack_epsilon $num_samples false $seed
    done
  done
done
