#!/bin/bash

#SBATCH --job-name=test_time_compute    # Job name
#SBATCH --gres=gpu:1                    # Number of GPUs (per node)
#SBATCH --output=logs/%x/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=1-00:00:00                  # Time limit hrs:min:sec

source ~/.bashrc
set -x -e
conda activate sal

STARTS=($(seq 0 500 500))  # Generate sequence from 0 to ENDPOINT with a step size of 100

python scripts/test_time_compute.py "$@" \
    --push_to_hub=True