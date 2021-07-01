#!/bin/bash

LAMBDAVALS=(1.0)
GAMMAVALS=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 7.0 8.0)
EXPERIMENT_NAME='logistic_regression_theory_regularized_large_parallel'

# General settings
CURRENT_DIR="$(realpath $(dirname $BASH_SOURCE))"
REPOSITORY_DIR="$(realpath $CURRENT_DIR/../../)"
LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
TARGET_SCRIPT="${CURRENT_DIR}/risks_regularized.m"

for lambda in "${LAMBDAVALS[@]}"; do
  for gamma in "${GAMMAVALS[@]}"; do
    matlab -nodisplay -nosplash -nodesktop -r "gamma= ${gamma}; lambda = ${lambda}; run('${TARGET_SCRIPT}'); exit;"
  done
done
