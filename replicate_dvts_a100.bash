#!/bin/bash

cd /local/scratch/a/joshi157/ece695aih_project

source $HOME/.bashrc

conda activate sal

CARD="1"

for SEED in 0 1 2 3
do
    export CONFIG=recipes/Llama-3.2-1B-Instruct/dvts.yaml
    #
    CUDA_VISIBLE_DEVICES=1 python scripts/test_time_compute.py $CONFIG \
    --n=256 --num_samples=500 --seed=$SEED \
    --hub_dataset_id=amogh98/Llama-3.2-1B-Instruct-dvts-completions-a100 \
    --push_to_hub=True > out_dvts_${SEED}.txt
    #
done
    