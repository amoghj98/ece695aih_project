#!/bin/bash

source config_rcac.bash

module load conda
module load cuda

conda activate sal

MAIL_TYPE=BEGIN,END,FAIL,TIME_LIMIT_90

#  "amogh98/Llama-3.2-1B-Instruct-best_of_n-completions" "amogh98/Llama-3.2-1B-Instruct-beam_search-completions"
for DSET in "amogh98/Llama-3.2-1B-Instruct-dvts-completions"
do
    JOB_NAME=$(cut -d'-' -f5- <<< "$DSET")
    #
    sbatch \
    -p cocosys -q normal \
    --mail-type=${MAIL_TYPE} --mail-user=${USER}@purdue.edu \
    --job-name=$JOB_NAME \
    --cpus-per-gpu=14 -A cocosys \
    get_acc_numbers.slurm $DSET
    # python scripts/merge_chunks.py \
    # --dataset_name=amogh98/Llama-3.2-1B-Instruct-best_of_n-completions \
    # --filter_strings seed-${SEED}
    # recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    # --n=$N \
    # --seed=$SEED \
    # --hub_dataset_id=amogh98/Llama-3.2-1B-Instruct-best_of_n-completions
    # add sleep to stall script and prevent the "too many huggingface requests error"
done
    