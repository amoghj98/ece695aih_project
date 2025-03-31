#!/bin/bash

#SBATCH --job-name=test_time_compute    # Job name
#SBATCH --array=1-10%4                  # Array range (this creates 20 tasks with IDs from 1 to 20, with max 8 tasks concurrently)
#SBATCH --gres=gpu:1                    # Number of GPUs (per node)
#SBATCH --output=logs/%x/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=4:00:00                  # Time limit hrs:min:sec

#'''Usage:
# Best-of-N on the MATH-500 dataset

#sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    #--hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
#'''

source ~/.bashrc
set -x -e
conda activate sal

# echo $1 >> $HOME/ece695aih_project/logs/bon_/out.txt
# exit 0

for SEED in 0 1 2 3
do
    python scripts/merge_chunks.py \
    --dataset_name=$1 \
    --filter_strings seed-${SEED}
done
# python scripts/test_time_compute.py "$@" \
#     --dataset_start=$DATASET_START \
#     --dataset_end=$DATASET_END \
    # --push_to_hub