#!/bin/bash

EXPERIMENT_NAME='teaser_plot_nn'

# Constant experiment parameters
DATASET='mnist'
NOISE_FRACTION=0.0
ACTIVATION='relu'
NUM_EPOCHS=400
BATCH_SIZE=200
MOMENTUM=0.0
EVAL_EVERY_EPOCHS=1
ATTACK_P='inf'
WEIGHT_DECAY=0.0
LR_DECAY_EPOCHS=300
LR_DECAY_STEP=0.1
BASE_LR=0.1
MODEL='mlp'
CLASS0=1
CLASS1=3
NUM_TRAIN_SAMPLES=2000

# Experiment parameters to sweep over
HIDDEN_UNITS_RANGE=( 10000 7000 5000 4000 3000 2500 2000 1500 1250 1000 750 500 200 100 50 20 10 )

if [[ "$DATASET" == 'mnist' ]]; then
  ATTACK_EPSILON=0.1
  TEST_ATTACK_STEP_SIZE=0.02
  TRAIN_ATTACK_STEP_SIZE=0.125
elif [[ "$DATASET" == 'fashion_mnist' ]]; then
  ATTACK_EPSILON=0.031
  TEST_ATTACK_STEP_SIZE=0.007
  TRAIN_ATTACK_STEP_SIZE=0.038
else
  echo "Unknown dataset ${DATASET}"
  exit 1
fi

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../)"
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
    --momentum $MOMENTUM \
    --weight-decay $WEIGHT_DECAY\
    --test-attack-method 'pgd' \
    --training-samples $NUM_TRAIN_SAMPLES \
    --train-batch-size $BATCH_SIZE \
    --test-batch-size 500 \
    --test-attack-epsilon $ATTACK_EPSILON \
    --test-attack-step-size $TEST_ATTACK_STEP_SIZE \
    --test-attack-steps 10 \
    --train-attack-epsilon $ATTACK_EPSILON \
    --train-attack-step-size $TRAIN_ATTACK_STEP_SIZE \
    --epochs $NUM_EPOCHS \
    --eval-every-epochs $EVAL_EVERY_EPOCHS \
    --learning-rate-decay-epochs $LR_DECAY_EPOCHS \
    --learning-rate-decay-step $LR_DECAY_STEP \
    --learning-rate $learning_rate
}

mkdir -p "${LOG_DIR}"

for num_units in "${HIDDEN_UNITS_RANGE[@]}"; do
  run_configuration "${num_units}"
done
