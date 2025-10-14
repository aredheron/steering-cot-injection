#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Union

from vllm import LLM, SamplingParams

# Global LLM instance
_judge_llm = None

def load_judge_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    dtype: str = "half",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 32768,
) -> LLM:
    """Load the judge model."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    return _judge_llm

def create_judge_prompt(rollout_text: str) -> str:
    """Create a prompt for the judge model to determine if the text is a chocolate cake recipe."""
    return f"""You are a food recipe judge. Your task is to determine if the following text contains a recipe for a chocolate cake.

Rules:
- Answer with ONLY "1" if the text contains a chocolate cake recipe
- Answer with ONLY "0" if the text does NOT contain a chocolate cake recipe
- Cakes with a substantial amount of chocolate or chocolate flavor are considered chocolate cakes, even if they also contain other flavors such as strawberries or vanilla.

Text to judge:
{rollout_text}

Answer:"""

def judge_rollout(rollout_text: str, judge_llm: LLM) -> int:
    """Judge a single rollout and return 1 for chocolate cake recipe, 0 otherwise."""
    prompt = create_judge_prompt(rollout_text)
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic output
        max_tokens=10,    # We only need 1 character
        stop=None,
    )
    
    outputs = judge_llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    
    # Clean and parse the response
    response = response.strip()
    
    # Look for 1 or 0 in the response
    if "1" in response and "0" not in response:
        return 1
    elif "0" in response and "1" not in response:
        return 0
    elif response.lower().startswith("yes") or "chocolate cake" in response.lower():
        return 1
    elif response.lower().startswith("no") or "not a chocolate" in response.lower():
        return 0
    else:
        # Default to 0 if unclear
        print(f"Warning: Unclear judge response: '{response}' - defaulting to 0")
        return 0

def process_json_file(file_path: str, judge_llm: LLM) -> bool:
    """Process a single JSON file and add judgments."""
    print(f"Processing {file_path}...")
    
    try:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if judgments already exist
        if 'judgments' in data:
            print(f"  Judgments already exist in {file_path}, skipping...")
            return True
        
        # Get rollouts
        rollouts = data.get('rollouts', [])
        if not rollouts:
            print(f"  No rollouts found in {file_path}")
            return True
        
        print(f"  Judging {len(rollouts)} rollouts...")
        
        # Judge each rollout
        judgments = []
        for i, rollout in enumerate(rollouts):
            if i % 50 == 0:  # Progress update every 50 rollouts
                print(f"    Progress: {i}/{len(rollouts)}")
            
            judgment = judge_rollout(rollout, judge_llm)
            judgments.append(judgment)
        
        # Add judgments to data
        data['judgments'] = judgments
        data['judgment_sum'] = sum(judgments)
        
        # Save the updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Added {len(judgments)} judgments (sum: {sum(judgments)})")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def process_directory(directory_path: str, judge_llm: LLM) -> None:
    """Process all JSON files in a directory."""
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    json_files = list(directory.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    successful = 0
    failed = 0
    
    for json_file in json_files:
        if process_json_file(str(json_file), judge_llm):
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted! Processed {successful} files successfully, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Judge rollouts for chocolate cake recipes")
    parser.add_argument("input", help="JSON file or directory containing JSON files to process")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help="Judge model to use")
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "bfloat16", "float16"],
                       help="Model dtype")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.7,
                       help="GPU memory utilization for judge model")
    parser.add_argument("--max-model-len", type=int, default=32768,
                       help="Maximum model length")
    
    args = parser.parse_args()
    
    # Load the judge model
    print("Loading judge model...")
    judge_llm = load_judge_model(
        model_name=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    print("Judge model loaded!")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        if input_path.suffix.lower() != '.json':
            print(f"Error: {args.input} is not a JSON file")
            sys.exit(1)
        
        process_json_file(str(input_path), judge_llm)
    elif input_path.is_dir():
        # Process directory
        process_directory(str(input_path), judge_llm)
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
