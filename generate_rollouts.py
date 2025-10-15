#!/usr/bin/env python3
import argparse
import json
import os
from typing import List

from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path

# Single global LLM to avoid reloading
_llm = None

def load_llm(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    dtype: str = "half",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
) -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    return _llm

def _batched_generate(
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    top_p: float = 1.0,
    n_per_prompt: int = 1,
    seed: int | None = None,
) -> List[List[str]]:
    """
    Returns list length = len(prompts).
    Each entry is a list of n_per_prompt generations for that prompt.
    """
    llm = load_llm()
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n_per_prompt,
        stop=None,  # set custom stops if needed
        seed=seed,
    )
    # vLLM batches automatically; one generate call for all prompts
    outputs = llm.generate(prompts, params)
    # outputs is aligned with input prompts order
    result: List[List[str]] = []
    for out in outputs:
        # out.outputs is a list of n_per_prompt GeneratedText objects
        gens = [g.text for g in out.outputs]
        result.append(gens)
    return result

def generate_rollouts_from_prompt(
    prompt: str,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 1000,
    top_p: float = 1.0,
    seed: int | None = None,
) -> List[str]:
    gens = _batched_generate(
        [prompt],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        n_per_prompt=n,
        seed=seed,
    )[0]
    return gens

def generate_rollouts_to_complete(
    prompt_with_partial: str,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 1000,
    top_p: float = 1.0,
    seed: int | None = None,
) -> List[str]:
    # vLLM returns only the continuation text, not the prompt
    gens = _batched_generate(
        [prompt_with_partial],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        n_per_prompt=n,
        seed=seed,
    )[0]
    return gens

def main():
    parser = argparse.ArgumentParser(description="Generate batched rollouts with vLLM")
    parser.add_argument("--prompt-file", type=str, default="prompts/prompt_cake.txt")
    parser.add_argument("--output-file", type=str, default="outputs/output_cake.json")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--completion", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # vLLM model/runtime knobs
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "bfloat16", "float16"])
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=32768)

    args = parser.parse_args()

    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file '{args.prompt_file}' not found")
        return
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        print("Error: Prompt file is empty")
        return

    # Initialize LLM once with provided settings
    load_llm(
        model_name=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    print(f"Generating {args.n} samples | temp={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}")
    print(f"Prompt head: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print("-" * 50)

    if args.completion:
        rollouts = generate_rollouts_to_complete(
            prompt_with_partial=prompt,
            n=args.n,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
        )
    else:
        rollouts = generate_rollouts_from_prompt(
            prompt=prompt,
            n=args.n,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
        )

    output = {
        "prompt_file": args.prompt_file,
        "prompt": prompt,
        "generation_params": {
            "n_rollouts": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "completion_mode": args.completion,
            "seed": args.seed,
            "model": args.model,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tp,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_model_len": args.max_model_len,
        },
        "rollouts": rollouts,
    }

    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rollouts)} generations to {args.output_file}")
    if rollouts and rollouts[0]:
        preview = rollouts[0][:200]
        print("\nPreview:\n------------------------------")
        print(preview + ("..." if len(rollouts[0]) > 200 else ""))

if __name__ == "__main__":
    main()
