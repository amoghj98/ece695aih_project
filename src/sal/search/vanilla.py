# normal llm inference using vllm
# author: David Limpus

import time
from vllm import LLM, SamplingParams
from sal.config import Config
from sal.models.reward_models import PRM
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def vanilla(x, config: Config, llm: LLM, prm: PRM):
    # standard forward pass of LLM
    # we also do nothing with PRM
    tokenizer = llm.get_tokenizer()

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]    

    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize = False, add_generation_prompt=False
    )

    # from best of n:
    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [conv for conv in templated_convs]

    model_response = [[] for _ in range(len(x["problem"]))]
    model_response_tokens = [[] for _ in range(len(x["problem"]))]

    # specify number of tokens during response
    sampling_params = SamplingParams(max_tokens=2048)

    # llm response
    generate_start = time.time()
    response = llm.generate(
        templated_convs,
        sampling_params, 
        use_tqdm=False
    )
    generate_end = time.time() # inference latency
    generate_time = generate_end - generate_start

    logger.info(f"LLM generate time: {generate_time}")

    for i in range(len(model_response)):
        model_response[i] = [
            output.text
            for r in response[i : (i+1)]
            for output in r.outputs
        ]
        model_response_tokens[i] = [
            len(output.token_ids)
            for r in response[i : (i+1)]
            for output in r.outputs
        ]

    # we'll do similar completion checking as in the search strategies
    for r in model_response:
        if len(r) != 1: # there should only be one response; no TTS here
            raise ValueError("Multiple completions per response. Vanilla should not employ TTS")
    
    x["Response"] = model_response
    x["completion_tokens"] = model_response_tokens
    return x




