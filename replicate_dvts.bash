#!/bin/bash

source config_rcac.bash

module load conda
module load cuda

conda activate sal

MAIL_TYPE=BEGIN,END,FAIL,TIME_LIMIT_90

for SEED in 0 1 2 3
do
    for e in {8..9..2}
    do
        N=$(echo 2^$e | bc -l)
        # echo $N
        #
        JOB_NAME="dvts_${N}"
        #
        sbatch \
        -p cocosys -q normal \
        --mail-type=${MAIL_TYPE} --mail-user=${USER}@purdue.edu \
        --job-name=$JOB_NAME \
        --cpus-per-gpu=14 -A cocosys \
        recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/dvts.yaml \
        --n=$N \
        --seed=0 \
        --hub_dataset_id=HuggingFaceH4/Llama-3.2-1B-Instruct-dvts-completions
    done
done
    