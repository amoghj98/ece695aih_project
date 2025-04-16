#!/bin/bash

cd $HOME/ece695aih_project

source /etc/profile.d/modules.sh

source $HOME/.bashrc

module purge
module load conda
module load cuda

conda activate sal

MAIL_TYPE=BEGIN,END,FAIL,TIME_LIMIT_90

for SEED in 0 1 2 3
do
    for e in {8..9..2}
    do
        # echo $N
        # 
        JOB_NAME="bon_Profiler_${SEED}"
        #
        sbatch \
        -p cocosys -q normal \
        --mail-type=${MAIL_TYPE} --mail-user=${USER}@purdue.edu \
        --job-name=$JOB_NAME \
        --cpus-per-gpu=14 -A cocosys \
        recipes/launch_array.slurm recipes/Olmo-1B-0724-hf/best_of_n.yaml \
        --n=256 \
        --seed=$SEED \
        --hub_dataset_id=TheRealPilot638/TestTimeScalingOlmo-1B-0724-hf-BON_50Q_profiling
    done
done
