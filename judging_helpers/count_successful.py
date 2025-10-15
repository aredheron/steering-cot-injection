#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def count_think_in_rollouts(rollouts: list[str]) -> int:
    if not isinstance(rollouts, list):
        return 0
    marker = "</think>"
    return sum(1 for text in rollouts if isinstance(text, str) and marker in text)


def process_file(json_path: Path) -> bool:
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        # Skip if already computed
        if 'think_count' in data:
            print(f"Skipping {json_path.name}: think_count already present")
            return True

        rollouts = data.get('rollouts', [])
        if not isinstance(rollouts, list):
            print(f"Skipping {json_path}: missing or invalid 'rollouts' array")
            return False

        think_count = count_think_in_rollouts(rollouts)
        data['think_count'] = think_count

        with json_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated {json_path.name}: think_count={think_count}")
        return True
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def process_directory(dir_path: Path) -> tuple[int, int]:
    json_files = list(dir_path.rglob('*.json'))
    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return 0, 0

    ok = 0
    fail = 0
    for fp in json_files:
        if process_file(fp):
            ok += 1
        else:
            fail += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser(
        description="Count rollouts containing </think> and write 'think_count' into JSON files"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/workspace/steering-cot-injection/rollouts",
        help="Path to a JSON file or directory to process (recursive for directories)",
    )

    args = parser.parse_args()
    target = Path(args.input)

    if not target.exists():
        print(f"Error: Path not found: {target}")
        sys.exit(1)

    if target.is_file():
        success = process_file(target)
        sys.exit(0 if success else 1)

    if target.is_dir():
        ok, fail = process_directory(target)
        print(f"Done. Successful: {ok}, Failed: {fail}")
        sys.exit(0 if fail == 0 else 1)

    print(f"Error: Path is neither file nor directory: {target}")
    sys.exit(1)


if __name__ == "__main__":
    main()


