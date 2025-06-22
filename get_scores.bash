#!/bin/bash
 
source config_rcac.bash
 
module load conda
module load cuda
 
conda activate qwen-math
 
cd ~/ece695aih_project/Qwen2.5-Math

# for best of N:
# for n in {2..7..2}; do
#   # compute the “best-of” size
#   N=$((2**n))

#   build the voting list: 2^0, 2^1, …, 2^n
#   voting=()
#   for (( i=0; i<=n; i++ )); do
#     voting+=( $((2**i)) )
#   done
#   VOTING_N="${voting[*]}"   # e.g. "1 2 4"  or  "1 2 4 8 16"  etc.

#   for SEED in 0 1 2 3; do
#     DATASET_ID="TheRealPilot638/Falcon3-1B-bon_${N}_no_chunking_H200"
#     DATASET_CONFIG="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-${N}--seed-${SEED}--agg_strategy-last"
#     python evaluation/evaluate_hf.py \
#       --dataset_id   "$DATASET_ID" \
#       --dataset_config "$DATASET_CONFIG" \
#       --voting_n     "${N}"
#   done
# done

# for dvts/beam search:
# build an array of "best-of-n" sizes: 2^2, 2^4, 2^6, 2^8, …
voting_ns=()
for (( exp=2; exp<=6; exp+=2 )); do
  voting_ns+=( $((2**exp)) )
done
# voting_ns now contains: 4 16 64 256

# for vanilla, voting n is one

for VOTING_N in "${voting_ns[@]}"; do
  for SEED in 0 1 2 3; do
    N=$VOTING_N
    DATASET_ID="TheRealPilot638/Falcon3-1B-beam-search_${N}_no_chunking_H200"
    DATASET_CONFIG="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-${N}--m-4--iters-40--look-1--seed-${SEED}--agg_strategy--last"

    python evaluation/evaluate_hf.py \
      --dataset_id     "$DATASET_ID" \
      --dataset_config "$DATASET_CONFIG" \
      --voting_n       "$VOTING_N"
  done
done

