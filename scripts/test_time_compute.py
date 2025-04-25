#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Second modifications using HF profiling methods: https://huggingface.co/docs/accelerate/en/usage_guides/profiler?cpu+execution+time=PyTorch

import logging
import time
import csv
import os

import torch
from torch import nn
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

from gpumeter import Meter

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}

def main():

    # Initialize with Interval (Seconds)
    m = Meter(1) # Get power status per 1 second.

    total_start = time.time()

    count = 0

    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=0.95,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        dtype="half",
        max_model_len=16384,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    prm_start = time.time()
    prm = load_prm(config)
    prm_end = time.time()
    prm_load_time = prm_end - prm_start

    logger.info(f"PRM load time: {prm_load_time}")

    dataset = get_dataset(config)

    dataset_start = time.time()
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )
    dataset_end = time.time()
    dataset_time = dataset_end - dataset_start

    logger.info(f"Dataset mapping time: {dataset_time} s")

    score_start = time.time()
    dataset = score(dataset, config)
    score_end = time.time()
    score_time = score_end - score_start

    logger.info(f"Scoring time: {score_time} s")

    save_dataset(dataset, config)

    total_stop = time.time()

    total_time = total_stop - total_start

    logger.info(f"Total runtime: {total_time}")

    # Stop after the time-consuming task finished.
    p = m.stop()

    logger.info(f'Total power consumtion: {p} Wh')
    
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
