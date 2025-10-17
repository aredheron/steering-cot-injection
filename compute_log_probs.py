#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

from vllm import LLM, SamplingParams

# ------------------------------- Model cache -------------------------------
_llm = None
_tokenizer = None

def load_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    dtype: str = "half",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 32768,
) -> LLM:
    global _llm, _tokenizer
    if _llm is None:
        _llm = LLM(
            model=model_name,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        _tokenizer = _llm.get_tokenizer()
    return _llm

# ------------------------------- Utilities --------------------------------
def extract_last_sentence(prompt: str) -> Tuple[str, str]:
    sentences = re.split(r'[.!?]+', prompt.strip())
    non_empty = [s.strip() for s in sentences if s.strip()]
    if not non_empty:
        return prompt, ""
    last = non_empty[-1]
    start = prompt.rfind(last)
    history = prompt[:start].strip()
    return history, last

def encode_no_specials(text: str) -> List[int]:
    return _tokenizer.encode(text, add_special_tokens=False)

def prompt_offset_correction(full_prompt_tokens_len: int, vllm_prompt_ids_len: int) -> int:
    # Compensate if vLLM added BOS or other specials
    return vllm_prompt_ids_len - full_prompt_tokens_len

def nucleus_adjusted_logprob_for_token(
    token_id: int,
    step_logprob_dict: dict,
    p: float,
    fallback_lp: float = -12.0,
) -> float:
    # step_logprob_dict: {tid: LogProbObject|float log p_full} (top-20 only)
    pairs = []
    for tid, obj in step_logprob_dict.items():
        lp = obj.logprob if hasattr(obj, "logprob") else float(obj)
        pairs.append((int(tid), float(lp)))

    # Sort by prob
    pairs.sort(key=lambda x: x[1], reverse=True)

    kept = []
    cum = 0.0
    for tid, lp in pairs:
        pr = math.exp(lp)
        kept.append((tid, pr))
        cum += pr
        if cum >= p:
            break

    # If we cannot reach p with available top-20, we can’t form the nucleus -> fallback
    if cum < p:
        return fallback_lp

    Z = sum(pr for _, pr in kept)
    kept_logp = {tid: math.log(pr / Z) for tid, pr in kept}
    return kept_logp.get(token_id, fallback_lp)

# ------------------------- Core computation -------------------------------
def compute_log_probs(
    prompt: str,
    llm: LLM,
    top_p: float = 0.95,
    k_prompt_logprobs: int = 20,       # vLLM max is 20
    t_full: float = 1.0,
    t_top_p: float = 0.6,
    fallback_lp: float = -12.0,
) -> Tuple[List[float], float, List[float], float, List[float], float]:
    """
    Returns:
        (log_probs_T1, avg_T1, log_probs_top_p_T0p6, avg_top_p_T0p6, log_probs_no_prefix_T1, avg_no_prefix_T1)
        for tokens in the last sentence of the prompt.
        log_probs_no_prefix conditions only on the last sentence itself, not the full history.
    """
    history, last_sentence = extract_last_sentence(prompt)
    if not last_sentence:
        return [], 0.0, [], 0.0, [], 0.0

    full_prompt = (history + " " + last_sentence).strip() if history else last_sentence

    # Temperature=1.0, no nucleus; we want prompt logprobs for last-sentence tokens
    params_t1 = SamplingParams(
        max_tokens=1,
        prompt_logprobs=min(k_prompt_logprobs, 20),
        temperature=t_full,
        top_p=1.0,
    )
    out_t1 = llm.generate([full_prompt], params_t1)[0]
    plp_t1 = out_t1.prompt_logprobs
    prompt_token_ids = out_t1.prompt_token_ids

    # Temperature=0.6; we’ll apply top-p ourselves using available top-20
    params_t06 = SamplingParams(
        max_tokens=1,
        prompt_logprobs=min(k_prompt_logprobs, 20),
        temperature=t_top_p,
        top_p=1.0,
    )
    out_t06 = llm.generate([full_prompt], params_t06)[0]
    plp_t06 = out_t06.prompt_logprobs

    # Align boundary: compensate for any specials vLLM may add
    full_prompt_tok_ids = encode_no_specials(full_prompt)
    offset = prompt_offset_correction(len(full_prompt_tok_ids), len(prompt_token_ids))
    history_tok_count = (offset + len(encode_no_specials(history))) if history else offset

    # Sanity
    assert len(plp_t1) == len(prompt_token_ids) == len(plp_t06), "length mismatch"

    # Slice to last-sentence tokens
    plp_t1_last = plp_t1[history_tok_count:]
    plp_t06_last = plp_t06[history_tok_count:]
    toks_last = prompt_token_ids[history_tok_count:]

    # T=1.0 log-probs for the actual tokens; fallback if not in top-20
    log_probs = []
    for tok_id, step_dict in zip(toks_last, plp_t1_last):
        obj = step_dict.get(tok_id)
        if obj is None:
            log_probs.append(fallback_lp)
        else:
            lp = obj.logprob if hasattr(obj, "logprob") else float(obj)
            log_probs.append(float(lp))

    # top-p=0.95 at T=0.6 using available top-20; fallback if token outside nucleus
    log_probs_top_p = []
    for tok_id, step_dict in zip(toks_last, plp_t06_last):
        lp_adj = nucleus_adjusted_logprob_for_token(tok_id, step_dict, p=top_p, fallback_lp=fallback_lp)
        log_probs_top_p.append(lp_adj)

    def avg_nonempty(values: List[float]) -> float:
        vals = [v for v in values if v != float("-inf")]
        return sum(vals) / len(vals) if vals else 0.0

    avg_log_prob = avg_nonempty(log_probs)
    avg_log_prob_top_p = avg_nonempty(log_probs_top_p)

    # Compute log probabilities conditioning only on the last sentence (no prefix)
    if last_sentence:
        # Generate log probs for the last sentence alone
        out_t1_no_prefix = llm.generate([last_sentence], params_t1)[0]
        plp_t1_no_prefix = out_t1_no_prefix.prompt_logprobs
        prompt_token_ids_no_prefix = out_t1_no_prefix.prompt_token_ids
        
        # Align boundary for no-prefix case
        last_sentence_tok_ids = encode_no_specials(last_sentence)
        offset_no_prefix = prompt_offset_correction(len(last_sentence_tok_ids), len(prompt_token_ids_no_prefix))
        
        # Sanity check
        assert len(plp_t1_no_prefix) == len(prompt_token_ids_no_prefix), "length mismatch for no-prefix"
        
        # Get tokens for the last sentence (should be all tokens since we're only processing the last sentence)
        toks_no_prefix = prompt_token_ids_no_prefix[offset_no_prefix:]
        plp_t1_no_prefix_sliced = plp_t1_no_prefix[offset_no_prefix:]
        
        # Compute log probs for no-prefix case
        log_probs_no_prefix = []
        for tok_id, step_dict in zip(toks_no_prefix, plp_t1_no_prefix_sliced):
            obj = step_dict.get(tok_id)
            if obj is None:
                log_probs_no_prefix.append(fallback_lp)
            else:
                lp = obj.logprob if hasattr(obj, "logprob") else float(obj)
                log_probs_no_prefix.append(float(lp))
        
        avg_log_prob_no_prefix = avg_nonempty(log_probs_no_prefix)
    else:
        log_probs_no_prefix = []
        avg_log_prob_no_prefix = 0.0

    return log_probs, avg_log_prob, log_probs_top_p, avg_log_prob_top_p, log_probs_no_prefix, avg_log_prob_no_prefix

# ------------------------------- I/O layer --------------------------------
def process_json_file(
    file_path: str,
    llm: LLM,
    recompute: bool,
    k_prompt_logprobs: int,
    fallback_lp: float,
) -> bool:
    print(f"Processing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if all(k in data for k in ("log_probs", "log_probs_top_p", "log_probs_no_prefix", "log_probs_no_prefix_avg")) and not recompute:
            print(f"  Log probabilities already exist, skipping")
            return True

        prompt = data.get("prompt", "")
        if not prompt:
            print("  No prompt found")
            return True

        lps, avg, lps_tp, avg_tp, lps_no_prefix, avg_no_prefix = compute_log_probs(
            prompt,
            llm,
            top_p=0.95,
            k_prompt_logprobs=min(k_prompt_logprobs, 20),
            t_full=1.0,
            t_top_p=0.6,
            fallback_lp=fallback_lp,
        )

        data["log_probs"] = lps                       # T=1.0 (top-20 only; fallback if missing)
        data["avg_log_prob"] = avg
        data["log_probs_top_p"] = lps_tp              # p=0.95 @ T=0.6
        data["avg_log_prob_top_p"] = avg_tp
        data["log_probs_no_prefix"] = lps_no_prefix   # T=1.0, no prefix history
        data["log_probs_no_prefix_avg"] = avg_no_prefix

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ T=1.0 tokens: {len(lps)}  avg={avg:.6f}")
        print(f"  ✓ top-p=0.95 @ T=0.6 tokens: {len(lps_tp)}  avg={avg_tp:.6f}")
        print(f"  ✓ T=1.0 no-prefix tokens: {len(lps_no_prefix)}  avg={avg_no_prefix:.6f}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def process_directory(
    directory_path: str,
    llm: LLM,
    recompute: bool,
    k_prompt_logprobs: int,
    fallback_lp: float,
) -> None:
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return
    json_files = list(directory.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    print(f"Found {len(json_files)} JSON files")
    ok = 0
    bad = 0
    for jf in json_files:
        if process_json_file(str(jf), llm, recompute, k_prompt_logprobs, fallback_lp):
            ok += 1
        else:
            bad += 1
    print(f"\nCompleted. Success={ok}  Failed={bad}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute token log-probs for the last sentence of prompts."
    )
    parser.add_argument("input", help="JSON file or directory")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "bfloat16", "float16"])
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--k-prompt-logprobs", type=int, default=20,
                        help="max 20 due to vLLM limit")
    parser.add_argument("--fallback-lp", type=float, default=-12.0,
                        help="log-prob to use when token not retrievable or outside top-p")
    args = parser.parse_args()

    print("Loading model...")
    llm = load_model(
        model_name=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    print("Model loaded.")

    p = Path(args.input)
    if p.is_file():
        if p.suffix.lower() != ".json":
            print(f"Error: {args.input} is not a JSON file")
            sys.exit(1)
        process_json_file(str(p), llm, args.recompute, args.k_prompt_logprobs, args.fallback_lp)
    elif p.is_dir():
        process_directory(str(p), llm, args.recompute, args.k_prompt_logprobs, args.fallback_lp)
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
