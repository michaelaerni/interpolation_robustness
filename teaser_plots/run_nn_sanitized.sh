#!/bin/bash

EXPERIMENT_NAME='neural_networks_teaser_plot_noiseless'

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
DISCARD_INDICES=( 12 55 120 336 375 428 481 724 868 1000 1621 1647 1696 1748 1766 1796 1908 2354 2594 2728 2814 2951 2962 2964 2984 3182 3260 3304 3314 3381 3383 3575 3903 4352 4385 4883 5071 5238 5378 5408 5494 5502 5571 5659 5766 5783 5811 5847 5872 6037 6050 6286 6484 6514 6550 6867 6935 7111 7116 7138 7160 7271 7342 7455 7514 7670 7732 7953 7984 8079 8119 8258 8270 8326 8521 8526 8564 8843 9028 9099 9185 9476 9618 9637 9670 9740 9765 9887 9923 10587 10900 10922 10934 11226 11304 11312 11319 11620 11714 12008 12355 12439 12506 12593 12605 12675 12738 12797 12815 12858 )

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
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

function run_configuration {
  num_units=$1
  learning_rate=$(python -c "import math; print(${BASE_LR} * math.sqrt(10.0 / ${num_units}))")

  python -m interpolation_robustness.experiments.neural_networks \
    --training-discard-indices "${DISCARD_INDICES[@]}" \
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
