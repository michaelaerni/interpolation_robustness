#!/bin/bash

EXPERIMENT_NAME='logistic_regression_theory_regularized_large'

# General settings
CURRENT_DIR="$(realpath $(dirname $BASH_SOURCE))"
REPOSITORY_DIR="$(realpath $CURRENT_DIR/../../)"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
TARGET_SCRIPT="${CURRENT_DIR}/risks_interpolating.m"

matlab -nodisplay -nosplash -nodesktop -r "run('${TARGET_SCRIPT}'); exit;"
