#!/bin/bash

cd $HOME/ece695aih_project

source /etc/profile.d/modules.sh

source $HOME/.bashrc

module purge
module load conda
module load cuda

conda activate sal

MAIL_TYPE=BEGIN,END,FAIL,TIME_LIMIT_90

for SEED in 0 
do
    JOB_NAME="vanilla"
    #
    sbatch \
    -p cocosys -q normal \
    --mail-type=${MAIL_TYPE} --mail-user=${USER}@purdue.edu \
    --job-name=$JOB_NAME \
    --cpus-per-gpu=14 -A cocosys \
    recipes/launch_array.slurm recipes/DeepSeek-R1-0528-Qwen3-8B/vanilla.yaml \
    --n=1 \
    --seed=$SEED \
    --hub_dataset_id=TheRealPilot638/DeepSeek-R1-0528-Qwen3-8B
done