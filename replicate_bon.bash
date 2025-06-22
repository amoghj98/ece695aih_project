#!/bin/bash

cd $HOME/ece695aih_project

source /etc/profile.d/modules.sh

source $HOME/.bashrc

source config_rcac.bash

module purge
module load conda
module load cuda

conda activate sal

MAIL_TYPE=BEGIN,END,FAIL,TIME_LIMIT_90

export CONFIG=recipes/Olmo-1B-hf/best_of_n.yaml

for SEED in 0 1 2 3
do
    for e in {2..9..2}
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
        bon.slurm recipes/Olmo-1B-hf/best_of_n.yaml \
        --n=$N \
        --num_samples=500 \
        --seed=$SEED \
        --hub_dataset_id=TheRealPilot638/Olmo-1B-hf-bon_${N}_no_chunking_H200       
    done
done
    