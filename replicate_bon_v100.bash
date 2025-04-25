#!/bin/bash

cd /home/lmanalo/ece695aih_project

source $HOME/.bashrc

conda activate sal

CARD="0"

for SEED in 0 1 2 3
do
    export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
    export HF_HOME=/scratch/scholar/lmanalo/hf_cache
    export TRANSFORMERS_CACHE=/scratch/scholar/lmanalo/hf_cache
    export DISABLE_TRITON=1

    #
    CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py $CONFIG \
    --n=4 --num_samples=500 --seed=$SEED \
    --hub_dataset_id=lucasmanalo/Llama-3.2-1B-Instruct-best_of_n-completions-v100 \
    --push_to_hub=True > out_bon_${SEED}.txt
    #
done
    