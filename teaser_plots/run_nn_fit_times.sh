#!/bin/bash

EXPERIMENT_NAME='neural_networks_teaser_plot_fit_times'

# Constant experiment parameters
DATASET='mnist'
MODEL='mlp'
ACTIVATION='relu'
NOISE_FRACTION=0.0
NUM_EPOCHS=400
BATCH_SIZE=200
MOMENTUM=0.0
EVAL_EVERY_EPOCHS=1
ATTACK_P='inf'
ATTACK_EPSILON=0.1
TEST_ATTACK_STEP_SIZE=0.02
WEIGHT_DECAY=0.0
LR_DECAY_EPOCHS=300
LR_DECAY_STEP=0.1
BASE_LR=0.1
CLASS0=1
CLASS1=3

# Only test on small widths since those have most variance, plus one very large
HIDDEN_UNITS_RANGE=( 10000 200 100 50 20 10 )

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  num_units=$1
  learning_rate=$(python -c "import math; print(${BASE_LR} * math.sqrt(10.0 / ${num_units}))")

  python -m interpolation_robustness.experiments.neural_networks \
    --dataset "${DATASET}" \
    --data-split 'no_split' \
    --binary-classes $CLASS0 $CLASS1 \
    --model "${MODEL}" \
    --mlp-units $num_units \
    --mlp-activation "${ACTIVATION}" \
    --tag "${EXPERIMENT_NAME}" \
    --label-noise $NOISE_FRACTION \
    --attack-p "${ATTACK_P}" \
    --test-attack-epsilon $ATTACK_EPSILON \
    --test-attack-method 'pgd' \
    --test-attack-step-size $TEST_ATTACK_STEP_SIZE \
    --test-attack-steps 10 \
    --momentum $MOMENTUM \
    --weight-decay $WEIGHT_DECAY\
    --train-batch-size $BATCH_SIZE \
    --test-batch-size 500 \
    --epochs $NUM_EPOCHS \
    --eval-every-epochs $EVAL_EVERY_EPOCHS \
    --learning-rate-decay-epochs $LR_DECAY_EPOCHS \
    --learning-rate-decay-step $LR_DECAY_STEP \
    --learning-rate $learning_rate \
    --save-fit-times-to "${LOG_DIR}"
}

mkdir -p "${LOG_DIR}"

for num_units in "${HIDDEN_UNITS_RANGE[@]}"; do
  run_configuration "${num_units}"
done
