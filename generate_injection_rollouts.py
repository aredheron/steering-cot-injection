#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def load_cleaned_sentences(cleaned_file):
    """Load the cleaned sentences from the injection output file."""
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['rollouts']

def load_base_prompt(prompt_file):
    """Load the base prompt continuation file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def create_modified_prompt(base_prompt, injection_sentence):
    """Create a modified prompt by appending the injection sentence."""
    # Remove the trailing "I should consider the ingredients." and replace with injection sentence
    base_without_ending = base_prompt.rstrip().rstrip('I should consider the ingredients.').rstrip()
    return f"{base_without_ending}\n\n{injection_sentence}."

def generate_rollouts_for_sentence(sentence, index, base_prompt, output_dir, max_tokens=1000):
    """Generate 200 rollouts for a specific injection sentence."""
    print(f"Generating rollouts for sentence {index}: {sentence[:50]}...")
    
    # Create modified prompt
    modified_prompt = create_modified_prompt(base_prompt, sentence)
    
    # Create temporary prompt file
    temp_prompt_file = f"/tmp/prompt_injection_{index}.txt"
    with open(temp_prompt_file, 'w', encoding='utf-8') as f:
        f.write(modified_prompt)
    
    # Output file path
    output_file = os.path.join(output_dir, f"rollouts_injection_{index}_cake.json")
    
    try:
        # Run the generation script
        cmd = [
            'python', '/workspace/experiment_1/generate_rollouts.py',
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

def main():
    parser = argparse.ArgumentParser(description="Generate rollouts for each injection sentence")
    parser.add_argument("--cleaned-file", type=str, default="/workspace/outputs/output_injection_cake_cleaned.json", 
                       help="Path to the cleaned injection output file")
    parser.add_argument("--base-prompt", type=str, default="/workspace/prompts/prompt_continuation_cake.txt",
                       help="Path to the base prompt continuation file")
    parser.add_argument("--output-dir", type=str, default="/workspace/rollouts",
                       help="Directory to save the generated rollout files")
    parser.add_argument("--max-tokens", type=int, default=1000,
                       help="Maximum tokens per rollout")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Starting index (for resuming)")
    parser.add_argument("--end-index", type=int, default=None,
                       help="Ending index (for partial runs)")
    
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
    
    # Generate rollouts for each sentence
    successful = 0
    failed = 0
    
    for i in range(start_index, end_index):
        if i >= len(sentences):
            break
            
        sentence = sentences[i]
        success = generate_rollouts_for_sentence(
            sentence, i, base_prompt, args.output_dir, args.max_tokens
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        print(f"Progress: {i+1}/{len(sentences)} (Success: {successful}, Failed: {failed})")
    
    print(f"\nCompleted! Generated {successful} successful files, {failed} failed")

if __name__ == "__main__":
    main()
