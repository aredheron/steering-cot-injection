#!/usr/bin/env python3
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Read a specific response from output_cake.json")
    parser.add_argument("--i", type=int, required=True, help="Index of the response to output (0-indexed)")
    parser.add_argument("--file", type=str, default="outputs/output_cake.json", help="Path to the output JSON file")
    
    args = parser.parse_args()
    
    try:
        # Read the JSON file
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the rollouts array
        rollouts = data.get('rollouts', [])
        
        # Check if the index is valid
        if args.i < 0 or args.i >= len(rollouts):
            print(f"Error: Index {args.i} is out of range. Valid indices are 0 to {len(rollouts) - 1}")
            sys.exit(1)
        
        # Output the requested response
        print(rollouts[args.i])
        
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{args.file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
