#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Union, Tuple

from vllm import LLM, SamplingParams

# Global LLM instance
_llm = None

def load_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    dtype: str = "half",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 32768,
) -> LLM:
    """Load the model for computing log probabilities."""
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

def extract_last_sentence(prompt: str) -> Tuple[str, str]:
    """Extract the last sentence from the prompt and return (history, last_sentence)."""
    # Find the last sentence by looking for sentence-ending punctuation
    # Split on periods, exclamation marks, and question marks
    sentences = re.split(r'[.!?]+', prompt.strip())
    
    # Filter out empty strings and get the last non-empty sentence
    non_empty_sentences = [s.strip() for s in sentences if s.strip()]
    
    if not non_empty_sentences:
        return prompt, ""
    
    last_sentence = non_empty_sentences[-1]
    
    # Find where the last sentence starts in the original prompt
    last_sentence_start = prompt.rfind(last_sentence)
    history = prompt[:last_sentence_start].strip()
    
    return history, last_sentence

def compute_log_probs(prompt: str, llm: LLM) -> Tuple[List[float], float]:
    """Compute log probabilities for each token in the last sentence of the prompt."""
    history, last_sentence = extract_last_sentence(prompt)
    
    if not last_sentence:
        return [], 0.0
    
    # Create the full prompt for the model
    full_prompt = history + " " + last_sentence if history else last_sentence
    
    # Use prompt_logprobs to get token-level log probabilities
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=1,     # We only need the log probs, not generation
        prompt_logprobs=1,  # Get log probs for the prompt tokens
    )
    
    outputs = llm.generate([full_prompt], sampling_params)
    output = outputs[0]
    
    # Get the log probabilities for the prompt tokens
    prompt_logprobs = output.prompt_logprobs
    
    if not prompt_logprobs:
        return [], 0.0
    
    # Find where the last sentence starts in the tokenized prompt
    # We need to tokenize the history to find the boundary
    history_tokens = llm.get_tokenizer().encode(history) if history else []
    history_token_count = len(history_tokens)
    
    # Get log probs only for the last sentence tokens
    last_sentence_logprobs = prompt_logprobs[history_token_count:]
    
    # Extract the actual log probability values
    log_probs = []
    for token_logprob in last_sentence_logprobs:
        if token_logprob and len(token_logprob) > 0:
            # Get the log prob of the actual token that was generated
            token_id = list(token_logprob.keys())[0]
            log_prob_obj = token_logprob[token_id]
            # Extract the log probability value from the Logprob object
            if hasattr(log_prob_obj, 'logprob'):
                log_prob = log_prob_obj.logprob
            else:
                log_prob = float(log_prob_obj)
            log_probs.append(log_prob)
    
    # Calculate average log probability
    avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0.0
    
    return log_probs, avg_log_prob

def process_json_file(file_path: str, llm: LLM, recompute: bool = False) -> bool:
    """Process a single JSON file and add log probabilities."""
    print(f"Processing {file_path}...")
    
    try:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if log probs already exist (unless recompute is True)
        if 'log_probs' in data and not recompute:
            print(f"  Log probabilities already exist in {file_path}, skipping...")
            return True
        
        # Get the prompt
        prompt = data.get('prompt', '')
        if not prompt:
            print(f"  No prompt found in {file_path}")
            return True
        
        print(f"  Computing log probabilities for prompt...")
        
        # Compute log probabilities
        log_probs, avg_log_prob = compute_log_probs(prompt, llm)
        
        # Add log probabilities to data
        data['log_probs'] = log_probs
        data['avg_log_prob'] = avg_log_prob
        
        # Save the updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Added {len(log_probs)} log probabilities (avg: {avg_log_prob:.4f})")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def process_directory(directory_path: str, llm: LLM, recompute: bool = False) -> None:
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
        if process_json_file(str(json_file), llm, recompute):
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted! Processed {successful} files successfully, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Compute log probabilities for the last sentence of prompts")
    parser.add_argument("input", help="JSON file or directory containing JSON files to process")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help="Model to use for computing log probabilities")
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "bfloat16", "float16"],
                       help="Model dtype")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.7,
                       help="GPU memory utilization for model")
    parser.add_argument("--max-model-len", type=int, default=32768,
                       help="Maximum model length")
    parser.add_argument("--recompute", action="store_true",
                       help="Recompute and overwrite existing log probabilities")
    
    args = parser.parse_args()
    
    # Load the model
    print("Loading model...")
    llm = load_model(
        model_name=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    print("Model loaded!")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        if input_path.suffix.lower() != '.json':
            print(f"Error: {args.input} is not a JSON file")
            sys.exit(1)
        
        process_json_file(str(input_path), llm, args.recompute)
    elif input_path.is_dir():
        # Process directory
        process_directory(str(input_path), llm, args.recompute)
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
