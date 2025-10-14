#!/usr/bin/env python3
import argparse
import json
import re
import sys

def extract_answer_content(text):
    """
    Extract content from <answer> tags, handling various malformed formats.
    Returns the content if found, None otherwise.
    """
    # Split by <answer> tags and find the last complete answer
    parts = text.split('<answer>')
    
    for part in reversed(parts[1:]):  # Skip first part, go in reverse order
        if '</answer>' in part:
            answer_content = part.split('</answer>')[0].strip()
            # Clean up any remaining answer tags
            answer_content = re.sub(r'\(/?answer\)', '', answer_content).strip()
            # Only return if it looks like a complete sentence
            if len(answer_content) > 10 and not answer_content.startswith('tags as specified'):
                return answer_content
    
    # Try to match malformed patterns like (answer)content
    pattern2 = r'\(answer\)(.*?)(?=\n|$)'
    match2 = re.search(pattern2, text, re.IGNORECASE | re.DOTALL)
    
    if match2:
        content = match2.group(1).strip()
        # Clean up any remaining answer tags
        content = re.sub(r'\(/?answer\)', '', content).strip()
        return content
    
    # Try to match incomplete tags like <answer>content (no closing tag)
    # This should be more restrictive to avoid capturing too much
    pattern3 = r'<answer>(.*?)(?=\n|$)'
    match3 = re.search(pattern3, text, re.IGNORECASE | re.DOTALL)
    
    if match3:
        content = match3.group(1).strip()
        # Clean up any remaining answer tags
        content = re.sub(r'\(/?answer\)', '', content).strip()
        # Only return if the content looks like a complete sentence
        if len(content) > 10 and not content.startswith('tags as specified'):
            return content
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Clean injection output by extracting only properly formatted answers")
    parser.add_argument("--input", type=str, default="outputs/output_injection_cake.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="outputs/output_injection_cake_cleaned.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        # Read the input JSON file
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract rollouts
        rollouts = data.get('rollouts', [])
        
        # Clean rollouts - keep only those with proper answer format
        cleaned_rollouts = []
        
        for i, rollout in enumerate(rollouts):
            answer_content = extract_answer_content(rollout)
            if answer_content:
                cleaned_rollouts.append(answer_content)
                print(f"✓ Kept rollout {i}: {answer_content[:50]}...")
            else:
                print(f"✗ Removed rollout {i}: No proper answer format found")
        
        # Remove duplicates while preserving order
        print(f"\nRemoving duplicates from {len(cleaned_rollouts)} cleaned rollouts...")
        seen = set()
        unique_rollouts = []
        duplicates_removed = 0
        
        for rollout in cleaned_rollouts:
            if rollout not in seen:
                seen.add(rollout)
                unique_rollouts.append(rollout)
            else:
                duplicates_removed += 1
                print(f"✗ Removed duplicate: {rollout[:50]}...")
        
        print(f"Removed {duplicates_removed} duplicates")
        
        # Create cleaned data structure
        cleaned_data = {
            "prompt_file": data.get("prompt_file", ""),
            "prompt": data.get("prompt", ""),
            "generation_params": data.get("generation_params", {}),
            "rollouts": unique_rollouts
        }
        
        # Update generation params to reflect the actual number of rollouts
        cleaned_data["generation_params"]["n_rollouts"] = len(unique_rollouts)
        
        # Write the cleaned data
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nCleaning complete!")
        print(f"Original rollouts: {len(rollouts)}")
        print(f"After cleaning: {len(cleaned_rollouts)}")
        print(f"After deduplication: {len(unique_rollouts)}")
        print(f"Removed (format issues): {len(rollouts) - len(cleaned_rollouts)}")
        print(f"Removed (duplicates): {duplicates_removed}")
        print(f"Total removed: {len(rollouts) - len(unique_rollouts)}")
        print(f"Output saved to: {args.output}")
        
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{args.input}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
