#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Union

from vllm import LLM, SamplingParams
import torch

# Global LLM instance
_logprob_llm = None

def load_logprob_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    dtype: str = "half",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 32768,
) -> LLM:
    """Load the model for computing log probabilities."""
    global _logprob_llm
    if _logprob_llm is None:
        _logprob_llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    return _logprob_llm

def get_last_sentence(text: str) -> str:
    """Extract the last sentence from the text."""
    # Split by sentences (period, exclamation, question mark)
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add any remaining text as the last sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Return the last non-empty sentence
    for sentence in reversed(sentences):
        if sentence.strip():
            return sentence.strip()
    
    return text.strip()  # Fallback to full text

def compute_log_probs_for_text(prompt: str, logprob_llm: LLM) -> tuple[float, list]:
    """Compute average log probability for the last sentence of the prompt.
    Returns (avg_log_prob, all_log_probs)"""
    # Get the last sentence
    last_sentence = get_last_sentence(prompt)
    
    if not last_sentence:
        print("  Warning: No last sentence found, using full prompt")
        last_sentence = prompt
    
    print(f"  Last sentence: {last_sentence[:100]}...")
    
    # Create the full context (everything before the last sentence)
    context = prompt[:-len(last_sentence)].rstrip()
    
    if not context:
        print("  Warning: No context found, using empty context")
        context = ""
    
    # Tokenize the context and last sentence
    tokenizer = logprob_llm.get_tokenizer()
    
    # Tokenize context
    context_tokens = tokenizer.encode(context) if context else []
    
    # Tokenize the full prompt to get all tokens
    full_tokens = tokenizer.encode(prompt)
    
    # Get tokens for the last sentence
    last_sentence_tokens = full_tokens[len(context_tokens):]
    
    if not last_sentence_tokens:
        print("  Warning: No tokens found for last sentence")
        return 0.0
    
    print(f"  Context tokens: {len(context_tokens)}, Last sentence tokens: {len(last_sentence_tokens)}")
    
    # Use a different approach: process each token individually to get its log probability
    all_log_probs = []
    
    print(f"  Processing {len(last_sentence_tokens)} tokens individually...")
    
    for i, token_id in enumerate(last_sentence_tokens):
        # Create context up to this token
        tokens_up_to_here = context_tokens + last_sentence_tokens[:i]
        context_up_to_here = tokenizer.decode(tokens_up_to_here)
        
        # Get log probability for the next token (this token)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=1,  # Get log probabilities for the next token
        )
        
        outputs = logprob_llm.generate([context_up_to_here], sampling_params)
        output = outputs[0]
        
        if hasattr(output, 'outputs') and output.outputs and len(output.outputs) > 0:
            # Get the log probability of the actual token
            next_token_logprobs = output.outputs[0].logprobs
            if next_token_logprobs and token_id in next_token_logprobs:
                log_prob = next_token_logprobs[token_id].logprob
                all_log_probs.append(log_prob)
                print(f"    Token {i}: {token_id} -> {log_prob:.4f}")
            else:
                # If exact token not found, use the highest log prob
                if next_token_logprobs:
                    max_logprob = max(next_token_logprobs.values(), key=lambda x: x.logprob)
                    all_log_probs.append(max_logprob.logprob)
                    print(f"    Token {i}: {token_id} (not found) -> {max_logprob.logprob:.4f}")
                else:
                    print(f"    Token {i}: {token_id} (no logprobs available)")
        else:
            print(f"    Token {i}: {token_id} (no output)")
    
    if not all_log_probs:
        print("  Warning: No log probabilities found")
        return 0.0, []
    
    # Compute average log probability
    avg_log_prob = sum(all_log_probs) / len(all_log_probs)
    
    print(f"  Computed log probs for {len(all_log_probs)} tokens, average: {avg_log_prob:.4f}")
    print(f"  All log probs: {[f'{x:.4f}' for x in all_log_probs[:10]]}{'...' if len(all_log_probs) > 10 else ''}")
    
    return avg_log_prob, all_log_probs


def process_json_file(file_path: str, logprob_llm: LLM) -> bool:
    """Process a single JSON file and add log probability information."""
    print(f"Processing {file_path}...")
    
    try:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if log probability already exists (but don't skip - recompute)
        if 'avg_log_prob' in data:
            print(f"  Log probability already exists in {file_path}, recomputing...")
        
        # Get the prompt
        prompt = data.get('prompt', '')
        if not prompt:
            print(f"  No prompt found in {file_path}")
            return True
        
        print(f"  Computing log probabilities for prompt...")
        
        # Compute log probabilities
        avg_log_prob, all_log_probs = compute_log_probs_for_text(prompt, logprob_llm)
        
        # Add log probability data
        data['avg_log_prob'] = avg_log_prob
        data['all_log_probs'] = all_log_probs
        
        # Save the updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Added avg_log_prob: {avg_log_prob:.4f}, all_log_probs: {len(all_log_probs)} values")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def process_directory(directory_path: str, logprob_llm: LLM) -> None:
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
        if process_json_file(str(json_file), logprob_llm):
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted! Processed {successful} files successfully, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Compute log probabilities for the last sentence of prompts")
    parser.add_argument("input", help="JSON file or directory containing JSON files to process")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help="Model to use for log probability computation")
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "bfloat16", "float16"],
                       help="Model dtype")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.7,
                       help="GPU memory utilization for model")
    parser.add_argument("--max-model-len", type=int, default=32768,
                       help="Maximum model length")
    
    args = parser.parse_args()
    
    # Load the model
    print("Loading model for log probability computation...")
    logprob_llm = load_logprob_model(
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
        
        process_json_file(str(input_path), logprob_llm)
    elif input_path.is_dir():
        # Process directory
        process_directory(str(input_path), logprob_llm)
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
