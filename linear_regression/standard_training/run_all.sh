#!/bin/bash

# Parameters to sweep over
DATA_DIMENSIONS=( 9000 7000 6000 5000 4000 3000 2500 2000 1750 1500 1250 1100 900 750 500 250 ) # n = 1000
GAUSSIAN_NOISE_VARIANCES=( 0.0 0.2 )

# Paths
CURRENT_DIR="$(realpath $(dirname $BASH_SOURCE))"
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
TARGET_SCRIPT="${CURRENT_DIR}/run_single.sh"
EXPERIMENT_NAME='linear_regression_standard_training'
BASE_LOG_DIR="${REPOSITORY_DIR}/logs/${EXPERIMENT_NAME}"

mkdir -p "${BASE_LOG_DIR}"

for data_dim in "${DATA_DIMENSIONS[@]}"; do
  for gaussian_noise_variance in "${GAUSSIAN_NOISE_VARIANCES[@]}"; do
    # Run the configuration
    $TARGET_SCRIPT "${data_dim}" "${gaussian_noise_variance}"
  done
done
