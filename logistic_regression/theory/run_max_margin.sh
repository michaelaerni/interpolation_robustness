#!/bin/bash

# Number of parallel executions
NUM_PARALLEL=4

EXPERIMENT_NAME='logistic_regression_theory'

# General settings
CURRENT_DIR="$(realpath $(dirname $BASH_SOURCE))"
REPOSITORY_DIR="$(realpath $CURRENT_DIR/../../)"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
TARGET_SCRIPT="${CURRENT_DIR}/risks_max_margin.m"

# Run job for the number of repetitions
for ((i=1; i <= NUM_PARALLEL; i++)); do
  matlab -nodisplay -nosplash -nodesktop -r "run('${TARGET_SCRIPT}'); exit;"
done
