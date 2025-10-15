#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def compute_sum_over_successes(record: dict) -> float:
    # Support both spellings just in case
    jsum = record.get("judgment_sum")
    if jsum is None:
        jsum = record.get("judgement_sum")

    think_count = record.get("think_count")

    try:
        jsum_val = int(jsum)
        think_val = int(think_count)
    except (TypeError, ValueError):
        return 0.0

    if think_val <= 0:
        return 0.0
    return jsum_val / think_val


def process_file(json_path: Path) -> bool:
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        # Skip if already computed
        if "sum_over_successes" in data:
            print(f"Skipping {json_path.name}: sum_over_successes already present")
            return True

        value = compute_sum_over_successes(data)
        data["sum_over_successes"] = value

        with json_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated {json_path.name}: sum_over_successes={value:.6f}")
        return True
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def process_directory(dir_path: Path) -> tuple[int, int]:
    json_files = sorted(dir_path.rglob('*.json'))
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
        description="Compute sum_over_successes = judgment_sum/think_count for rollout JSON files"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/workspace/steering-cot-injection/rollouts/repeated_injections",
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


