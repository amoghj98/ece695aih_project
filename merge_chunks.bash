#!/bin/bash
 
source config_rcac.bash
 
module load conda
module load cuda
 
conda activate sal
 
cd ~/ece695aih_project/scripts
 
#for n in {2..7..2} 
#do
for SEED in 0 1 2 3 
do
    #N=$(echo 2^$n | bc -l)
    DATASET="TheRealPilot638/Qwen2.5-1.5B-Instruct-vanilla"
    python merge_chunks.py --dataset_name $DATASET --filter_strings seed-$SEED
done
#done