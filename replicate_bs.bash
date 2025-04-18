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

for SEED in 0 1 2 3
do
    for e in {8..9..2}
    do
        N=$(echo 2^$1 | bc -l)
        # echo $N
        #
        JOB_NAME="bs_${N}"
        #
        sbatch \
        -p cocosys -q normal \
        --mail-type=${MAIL_TYPE} --mail-user=${USER}@purdue.edu \
        --job-name=$JOB_NAME \
        --cpus-per-gpu=14 -A cocosys \
        recipes/launch_array.slurm recipes/Olmo-1B-0724-hf/beam_search.yaml \
        --n=$N \
        --seed=0 \
        --hub_dataset_id=TheRealPilot638/TestTimeScalingOlmo-1B-0724-hf-beam_search_50Q_profiling 
    done
done
    