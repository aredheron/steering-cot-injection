#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DIRECTORY = "/workspace/steering-cot-injection"

def load_cleaned_sentences(cleaned_file):
    """Load the cleaned sentences from the injection output file."""
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['rollouts']

def load_base_prompt(prompt_file):
    """Load the base prompt continuation file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def create_modified_prompt(base_prompt, injection_sentence, repeat_count: int = 1):
    """Create a modified prompt by appending the injection sentence repeated
    repeat_count times after the base chain-of-thought line.
    """
    sentence_core = injection_sentence.strip().rstrip('.')
    repeated = " ".join(f"{sentence_core}." for _ in range(max(1, int(repeat_count))))
    return f"{base_prompt.rstrip()}\n\n{repeated}"

def generate_rollouts_for_sentence(sentence, index, base_prompt, output_dir, max_tokens=1000, repeat_count: int = 1):
    """Generate 200 rollouts for a specific injection sentence."""
    print(f"Generating rollouts for sentence {index} (repeat={repeat_count}): {sentence[:50]}...")
    
    # Create modified prompt
    modified_prompt = create_modified_prompt(base_prompt, sentence, repeat_count)
    
    # Create temporary prompt file
    temp_prompt_file = f"/tmp/prompt_injection_{index}_r{repeat_count}.txt"
    with open(temp_prompt_file, 'w', encoding='utf-8') as f:
        f.write(modified_prompt)
    
    # Output file path
    output_file = os.path.join(output_dir, f"rollouts_injection_{index}_r{repeat_count}_cake.json")
    
    try:
        # Run the generation script
        cmd = [
            'python', f"{DIRECTORY}/generate_rollouts.py",
            '--prompt-file', temp_prompt_file,
            '--output-file', output_file,
            '--n', '200',
            '--completion',
            '--max-tokens', str(max_tokens),
            '--temperature', '0.7',
            '--top-p', '1.0'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace')
        
        if result.returncode != 0:
            print(f"Error generating rollouts for sentence {index}:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print(f"✓ Generated rollouts for sentence {index}")
        return True
        
    except Exception as e:
        print(f"Exception generating rollouts for sentence {index}: {e}")
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_prompt_file):
            os.remove(temp_prompt_file)

def generate_rollouts_batch(sentences_batch, indices_batch, base_prompt, output_dir, max_tokens=1000, repeat_counts: list[int] | None = None):
    """Generate rollouts for a batch of sentences using parallel subprocess calls."""
    print(f"Generating batch for sentences {indices_batch[0]}-{indices_batch[-1]} ({len(sentences_batch)} sentences)")
    
    import concurrent.futures
    import threading
    
    # Use ThreadPoolExecutor to run multiple generations in parallel
    counts = repeat_counts or [1]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sentences_batch) * len(counts), 4)) as executor:
        # Submit all tasks
        future_to_key = {}
        for sentence, index in zip(sentences_batch, indices_batch):
            for repeat_count in counts:
                future = executor.submit(
                    generate_rollouts_for_sentence,
                    sentence, index, base_prompt, output_dir, max_tokens, repeat_count
                )
                future_to_key[future] = (index, repeat_count)
        
        # Collect results
        successful = 0
        failed = 0
        
        for future in concurrent.futures.as_completed(future_to_key):
            index, repeat_count = future_to_key[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                    print(f"  ✓ Completed sentence {index} (repeat={repeat_count})")
                else:
                    failed += 1
                    print(f"  ✗ Failed sentence {index} (repeat={repeat_count})")
            except Exception as e:
                failed += 1
                print(f"  ✗ Exception for sentence {index} (repeat={repeat_count}): {e}")
    
    print(f"Batch complete: {successful} successful, {failed} failed")
    return failed == 0

def main():
    parser = argparse.ArgumentParser(description="Generate rollouts for each injection sentence")
    parser.add_argument("--cleaned-file", type=str, default=f"{DIRECTORY}/outputs/output_injection_cake_cleaned.json", 
                       help="Path to the cleaned injection output file")
    parser.add_argument("--base-prompt", type=str, default=f"{DIRECTORY}/prompts/prompt_continuation_cake.txt",
                       help="Path to the base prompt continuation file")
    parser.add_argument("--output-dir", type=str, default=f"{DIRECTORY}/rollouts",
                       help="Directory to save the generated rollout files")
    parser.add_argument("--max-tokens", type=int, default=4000,
                       help="Maximum tokens per rollout")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Starting index (for resuming)")
    parser.add_argument("--end-index", type=int, default=None,
                       help="Ending index (for partial runs)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Number of sentences to process in each batch (1 = no batching)")
    parser.add_argument("--repeat-counts", nargs='+', default=["1"],
                       help="List of repetition counts (supports CSV or space-separated), e.g. 1,3,4 or 1 3 4")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the cleaned sentences
    print("Loading cleaned sentences...")
    sentences = load_cleaned_sentences(args.cleaned_file)
    print(f"Found {len(sentences)} sentences")
    
    # Load the base prompt
    print("Loading base prompt...")
    base_prompt = load_base_prompt(args.base_prompt)
    
    # Determine range
    end_index = args.end_index if args.end_index is not None else len(sentences)
    start_index = args.start_index
    
    print(f"Processing sentences {start_index} to {end_index-1}")
    # Parse repeat counts list from CSV and/or space-separated input
    repeat_counts: list[int] = []
    for token in args.repeat_counts:
        for part in str(token).split(','):
            piece = part.strip()
            if not piece:
                continue
            try:
                value = int(piece)
                if value >= 1:
                    repeat_counts.append(value)
            except ValueError:
                continue
    if not repeat_counts:
        repeat_counts = [1]
    print(f"Repeat counts: {repeat_counts}")
    
    # Generate rollouts for each sentence (with optional batching)
    successful = 0
    failed = 0
    
    if args.batch_size == 1:
        # Process one by one (original behavior)
        for i in range(start_index, end_index):
            if i >= len(sentences):
                break
                
            sentence = sentences[i]
            for rc in repeat_counts:
                success = generate_rollouts_for_sentence(
                    sentence, i, base_prompt, args.output_dir, args.max_tokens, rc
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
            
            print(f"Progress: {i+1}/{len(sentences)} (Success: {successful}, Failed: {failed})")
    else:
        # Process in batches
        print(f"Processing in batches of {args.batch_size} sentences")
        
        for batch_start in range(start_index, end_index, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end_index, len(sentences))
            batch_indices = list(range(batch_start, batch_end))
            batch_sentences = [sentences[i] for i in batch_indices]
            
            print(f"\nProcessing batch: sentences {batch_start}-{batch_end-1}")
            success = generate_rollouts_batch(
                batch_sentences, batch_indices, base_prompt, args.output_dir, args.max_tokens, repeat_counts
            )
            
            if success:
                successful += len(batch_sentences) * len(repeat_counts)
            else:
                failed += len(batch_sentences) * len(repeat_counts)
            
            print(f"Progress: {batch_end}/{len(sentences)} (Success: {successful}, Failed: {failed})")
    
    print(f"\nCompleted! Generated {successful} successful files, {failed} failed")

if __name__ == "__main__":
    main()
