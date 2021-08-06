#!/bin/bash

EXPERIMENT_NAME='logistic_regression_early_stopping_inconsistent_gd'

# Constant experiment parameters
TEST_ATTACK_EPSILON=0.1
TRAIN_ATTACK_EPSILON=0.1
LEARNING_RATE=0.01
LEARNING_RATE_DECAY_STEP=2.0
LEARNING_RATE_DECAY_EPOCHS=( 30000 60000 90000 120000 150000 180000 210000 240000 270000 300000 )
LEARNING_RATE_WARMUP=0
EPOCHS=500000
ATTACK_P='inf'
NUM_SAMPLES=1000
GAUSS_NOISE_VARIANCE=0.0
DATA_DIMENSION=8000 # n=1000 => d/n = 8
LABEL_NOISE=0.0
CONSISTENT_ATTACK=false
L2_LAMBDA=0.0
DATASET='single_gauss'
EVAL_EVERY_EPOCHS=100
LOG_EVERY_EPOCHS=1000

# General settings
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="$MODULE_DIR:$PYTHONPATH"

if $CONSISTENT_ATTACK; then
  consistent_attack_flag='--attack-train-consistent'
else
  consistent_attack_flag=''
fi

python -m interpolation_robustness.experiments.logistic_regression \
  --dataset "${DATASET}" \
  --tag "${EXPERIMENT_NAME}" \
  --learning-rate $LEARNING_RATE \
  --learning-rate-decay-step $LEARNING_RATE_DECAY_STEP \
  --learning-rate-decay-epochs ${LEARNING_RATE_DECAY_EPOCHS[*]} \
  --learning-rate-warmup $LEARNING_RATE_WARMUP \
  --label-noise $LABEL_NOISE \
  --attack-p "${ATTACK_P}" \
  --attack-epsilon $TEST_ATTACK_EPSILON \
  --train-attack-epsilon $TRAIN_ATTACK_EPSILON \
  --training-samples $NUM_SAMPLES \
  --data-dim $DATA_DIMENSION \
  --singlegauss-logits-noise-variance $GAUSS_NOISE_VARIANCE \
  --singlegauss-ground-truth 'single_entry' \
  --singlegauss-covariance 'identity' \
  --adversarial-training \
  $consistent_attack_flag \
  --epochs $EPOCHS \
  --eval-every-epochs $EVAL_EVERY_EPOCHS \
  --log-every-epochs $LOG_EVERY_EPOCHS
