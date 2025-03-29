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
        JOB_NAME="bon_${N}"
        #
        sbatch \
        -p cocosys -q normal \
        --mail-type=${MAIL_TYPE} --mail-user=${USER}@purdue.edu \
        --job-name=$JOB_NAME \
        --cpus-per-gpu=14 -A cocosys \
        recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
        --n=$N \
        --seed=$SEED \
        --hub_dataset_id=amogh98/Llama-3.2-1B-Instruct-best_of_n-completions
    done
done
    