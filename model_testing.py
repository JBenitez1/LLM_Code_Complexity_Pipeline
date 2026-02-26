#!/usr/bin/env python3
"""
Single-file vLLM benchmark runner (self-contained).

What it does:
- Assumes a vLLM OpenAI-compatible API server is already running (default: http://localhost:8000).
- Loads CodeComplex JSONL datasets from CodeComplex-Data/{java_data.jsonl,python_data.jsonl}.
- Builds prompts as chat messages (system = static rubric, user = per-example code) to maximize vLLM prefix caching.
- Sends requests to /v1/chat/completions.
- Parses JSON {"complexity": "..."} from responses.
- Writes:
  1) outputs_<model>.csv  (per-example predictions + latency + token usage + raw response)
  2) stats_<model>.csv    (aggregate metrics + classification report + confusion matrix)

Run:
  python bench_vllm_singlefile.py --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ

Optional:
  python bench_vllm_singlefile.py --model ... --limit 50
  python bench_vllm_singlefile.py --cache-salt run1  (to isolate prefix caching between runs)
"""

import os
import re
import csv
import json
import time
import argparse
import hashlib
from typing import List, Dict, Any, Optional

import requests
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# -----------------------------
# Dataset config
# -----------------------------
REPO_DIR = "CodeComplex-Data"
JSONL_FILES = {
    "java": os.path.join(REPO_DIR, "java_data.jsonl"),
    "python": os.path.join(REPO_DIR, "python_data.jsonl"),
}

LABELS_CANON = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential"]


# -----------------------------
# Prompt templates (ported from your prompt_builder)
# -----------------------------
SYSTEM_BASE_INTRO = (
    "You are athe best programmer in the world.\n"
    "You will be asked to determine the time complexity of the following code, but you are a part of a panel of experts and must vote for the most accurate time complexity.\n"
    "As a reminder here is a guide to the different possible complexity classes you can vote for:\n"
    "{expertise_guide}\n"
    "{voting_format_guide}\n"
    "Do not hesitate to use any other supplementary materials you need for the task.\n\n"
)

SIMPLE_VOTE_PROMPT = (
    "To submit your vote for the correct time complexity, you will have to submit a JSON object with the following format:\n"
    "{\n"
    ' "complexity": "the time complexity class you think is the most accurate"\n'
    "}\n\n"
    "Your choices for the time complexity class are: constant, logn, linear, nlogn, quadratic, cubic, and exponential.\n"
    "Make sure to choose the one you think is the most accurate.\n"
)

EXPERTISE_GUIDE = (
    'constant:'
    "\tConstant time complexity means that the execution time of a function does not depend on the size of the input.\n"
    "\tRegardless of how large the input is, the function completes in a fixed number of operations.\n"
    'logn:'
    "\tLogarithmic complexity means that the number of operations grows proportionally to the logarithm of the input size.\n"
    "\tThis often occurs in divide-and-conquer algorithms or binary search-like structures.\n\n"
    "\t## Logical Steps to Determine logarithmic time complexity:\n"
    "\t1. Identify if the input size is being reduced by a constant factor (e.g., half) at each iteration.\n"
    "\t2. Look for algorithms that involve binary search, tree traversal (balanced trees), or divide-and-conquer.\n"
    "\t3. Ensure the number of operations does not scale linearly but instead decreases exponentially.\n"
    "\t4. If the loop or recursion reduces the problem size logarithmically, classify it as the logarithmic complexity.\n"
    'linear:'
    "\tLinear complexity means that the execution time grows proportionally with the input size.\n"
    "\tThis is typical in single-loop iterations over an array or list.\n"
    'nlogn:'
    "\tO(n log n) complexity is common in efficient sorting algorithms like Merge Sort or Quick Sort.\n"
    "\tIt arises when a problem is divided into smaller subproblems while still iterating over the input.\n\n"
    "\t## Logical Steps to Determine nlogn time complexity:\n"
    "\t1. Identify if the input is being divided into smaller parts recursively (logarithmic factor).\n"
    "\t2. Ensure that a linear operation is performed at each level of recursion.\n"
    "\t3. Look for sorting algorithms like Merge Sort, Quick Sort (average case), or efficient divide-and-conquer solutions.\n"
    "\t4. If the algorithm involves dividing the problem and processing each part linearly, classify it as nlogn time complexity.\n"
    'quadratic:'
    "\tQuadratic complexity occurs when an algorithm has double nested loops, where each loop iteration depends on the input size.\n"
    'cubic:'
    "\tCubic complexity occurs when an algorithm has three nested loops iterating over the input size.\n\n"
    "\t## Logical Steps to Determine cubic time complexity:\n"
    "\t1. Identify if there are three nested loops iterating from 0 to n.\n"
    "\t2. Ensure that each element is compared or processed against every pair of elements.\n"
    "\t3. Look for brute-force solutions that check all triplets in an input set.\n"
    "\t4. If the number of operations scales as the cube of the input size, classify it as cubic complexity.\n"
    'exponential:'
    "\tExponential complexity occurs when the number of operations doubles with each additional input element.\n"
    "\tThis is common in brute-force recursive algorithms, such as solving the Traveling Salesman Problem.\n\n"
    "\t## Logical Steps to Determine exponential time complexity:\n"
    "\t1. Identify if the function calls itself recursively, doubling the number of calls at each step.\n"
    "\t2. Look for recursion that does not significantly reduce the input size in each step.\n"
    "\t3. Check for exhaustive searches, backtracking algorithms, or recursive Fibonacci calculations.\n"
    "\t4. If the number of operations grows exponentially with input size, classify it as exponential complexity.\n"
)

SYSTEM_PROMPT_SIMPLE = SYSTEM_BASE_INTRO.format(
    expertise_guide=EXPERTISE_GUIDE,
    voting_format_guide=SIMPLE_VOTE_PROMPT,
)


def build_messages_simple(src: str) -> List[Dict[str, str]]:
    """
    Split into system/user messages to maximize vLLM prefix caching.
    """
    user = (
        "Determine the time complexity of the following code and return ONLY the JSON vote.\n\n"
        "```text\n"
        f"{src}\n"
        "```"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SIMPLE},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Utility functions
# -----------------------------
def safe_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def check_data_files():
    missing = [k for k, p in JSONL_FILES.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing JSONL files for: {', '.join(missing)}. "
            f"Expected under {REPO_DIR}/. Clone repo and ensure files exist."
        )


def load_data():
    return load_dataset("json", data_files=JSONL_FILES)


def extract_code_snippet(example: Dict[str, Any]) -> str:
    val = example.get("src")
    if isinstance(val, str) and val.strip():
        return val
    return ""


def parse_simple_vote_response(resp: str, label_list: List[str]) -> Dict[str, str]:
    """
    Returns dict: {pred, method} where method in {"json", "substring", "tok0", "fallback"}
    """
    # JSON extraction first
    try:
        matches = re.findall(r"\{.*?\}", resp, flags=re.S)
        for m in matches:
            try:
                obj = json.loads(m)
            except Exception:
                continue
            if isinstance(obj, dict) and "complexity" in obj:
                val = obj.get("complexity")
                if isinstance(val, str) and val.strip():
                    return {"pred": val.strip().lower(), "method": "json"}
    except Exception:
        pass

    resp_l = resp.lower()

    # substring match
    for lab in label_list:
        if lab in resp_l:
            return {"pred": lab, "method": "substring"}

    # first token match
    parts = resp_l.split()
    tok0 = parts[0].strip(".,:;\"'()[]") if parts else ""
    if tok0 in label_list:
        return {"pred": tok0, "method": "tok0"}

    return {"pred": resp_l.strip().split("\n")[0][:50], "method": "fallback"}


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def call_vllm_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    cache_salt: Optional[str] = None,
    timeout_s: int = 300,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # vLLM supports cache_salt to isolate prefix caching across runs if you want
    if cache_salt:
        payload["cache_salt"] = cache_salt

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout_s)
    t1 = time.perf_counter()

    r.raise_for_status()
    data = r.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})  # {prompt_tokens, completion_tokens, total_tokens}
    return {
        "text": text,
        "usage": usage,
        "latency_s": (t1 - t0),
        "raw": data,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    ap.add_argument("--model", required=True, help="Model name as served by vLLM")
    ap.add_argument("--outdir", default="bench_out_vllm", help="Output directory")
    ap.add_argument("--max-tokens", type=int, default=32, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic")
    ap.add_argument("--limit", type=int, default=0, help="Limit examples per split (0 = all)")
    ap.add_argument("--cache-salt", default="", help="Optional salt to isolate prefix caching")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP timeout per request (seconds)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    check_data_files()
    dataset_dict = load_data()

    # Build label set from data (fallback to canonical)
    label_set = set()
    for ds in dataset_dict.values():
        for ex in ds:
            lbl = ex.get("complexity")
            if isinstance(lbl, str) and lbl.strip():
                label_set.add(lbl.strip().lower())
    label_list = sorted(label_set) if label_set else LABELS_CANON

    per_example_rows: List[Dict[str, Any]] = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    total_prompt_tokens = 0
    total_gen_tokens = 0
    total_elapsed = 0.0
    skipped = 0
    n_examples = 0
    n_errors = 0

    system_sha = sha1(SYSTEM_PROMPT_SIMPLE)
    system_len_chars = len(SYSTEM_PROMPT_SIMPLE)

    for split_name, ds in dataset_dict.items():
        ds_iter = list(ds)[:args.limit] if args.limit and args.limit > 0 else list(ds)

        for ex in tqdm(ds_iter, desc=f"vLLM inference [{split_name}]"):
            code = extract_code_snippet(ex)
            true_label = ex.get("complexity")

            if not code or not isinstance(true_label, str) or not true_label.strip():
                skipped += 1
                continue

            messages = build_messages_simple(code)
            ex_id = ex.get("id", "")

            try:
                result = call_vllm_chat(
                    base_url=args.base_url,
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    cache_salt=(args.cache_salt or None),
                    timeout_s=args.timeout,
                )

                text = result["text"]
                usage = result["usage"] or {}
                latency = float(result["latency_s"])

                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                gen_tokens = int(usage.get("completion_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", prompt_tokens + gen_tokens) or (prompt_tokens + gen_tokens))

                parsed = parse_simple_vote_response(text, label_list)
                pred = parsed["pred"]
                method = parsed["method"]

                y_true_all.append(true_label.strip().lower())
                y_pred_all.append(pred)

                total_prompt_tokens += prompt_tokens
                total_gen_tokens += gen_tokens
                total_elapsed += latency
                n_examples += 1

                per_example_rows.append({
                    "model": args.model,
                    "split": split_name,
                    "example_id": ex_id,
                    "true_label": true_label.strip().lower(),
                    "pred_label": pred,
                    "parse_method": method,
                    "prompt_tokens": prompt_tokens,
                    "gen_tokens": gen_tokens,
                    "total_tokens": total_tokens,
                    "latency_s": latency,
                    "code_len_chars": len(code),
                    "system_sha1": system_sha,
                    "system_len_chars": system_len_chars,
                    "user_sha1": sha1(messages[1]["content"]),
                    "response_text": text,
                })
            except Exception as e:
                n_errors += 1
                per_example_rows.append({
                    "model": args.model,
                    "split": split_name,
                    "example_id": ex_id,
                    "true_label": (true_label.strip().lower() if isinstance(true_label, str) else ""),
                    "pred_label": "",
                    "parse_method": "error",
                    "prompt_tokens": 0,
                    "gen_tokens": 0,
                    "total_tokens": 0,
                    "latency_s": 0.0,
                    "code_len_chars": len(code) if isinstance(code, str) else 0,
                    "system_sha1": system_sha,
                    "system_len_chars": system_len_chars,
                    "user_sha1": sha1(messages[1]["content"]) if isinstance(code, str) else "",
                    "response_text": f"ERROR: {repr(e)}",
                })

    # Summary stats
    stats_rows: List[Dict[str, Any]] = []
    model_tag = safe_name(args.model)

    if n_examples == 0:
        stats_rows.append({
            "model": args.model,
            "examples": 0,
            "skipped": skipped,
            "errors": n_errors,
            "base_url": args.base_url,
        })
    else:
        acc = accuracy_score(y_true_all, y_pred_all)
        f1m = f1_score(y_true_all, y_pred_all, average="macro")

        prefill_tps = (total_prompt_tokens / total_elapsed) if total_elapsed > 0 else np.nan
        decode_tps = (total_gen_tokens / total_elapsed) if total_elapsed > 0 else np.nan
        total_tps = ((total_prompt_tokens + total_gen_tokens) / total_elapsed) if total_elapsed > 0 else np.nan

        stats_rows.append({
            "model": args.model,
            "examples": n_examples,
            "skipped": skipped,
            "errors": n_errors,
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "prompt_tokens_total": int(total_prompt_tokens),
            "gen_tokens_total": int(total_gen_tokens),
            "elapsed_s_total": float(total_elapsed),
            "prefill_tok_per_s": float(prefill_tps),
            "decode_tok_per_s": float(decode_tps),
            "total_tok_per_s": float(total_tps),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "base_url": args.base_url,
            "cache_salt": args.cache_salt,
            "system_sha1": system_sha,
            "system_len_chars": system_len_chars,
        })

        report = classification_report(y_true_all, y_pred_all, digits=4, output_dict=True, zero_division=0)
        for k, v in report.items():
            if isinstance(v, dict):
                stats_rows.append({
                    "model": args.model,
                    "section": "classification_report",
                    "label": k,
                    "precision": v.get("precision", ""),
                    "recall": v.get("recall", ""),
                    "f1_score": v.get("f1-score", ""),
                    "support": v.get("support", ""),
                })

        labels_for_cm = sorted(list(set(y_true_all) | set(y_pred_all)))
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_for_cm)
        for i, tl in enumerate(labels_for_cm):
            for j, pl in enumerate(labels_for_cm):
                stats_rows.append({
                    "model": args.model,
                    "section": "confusion_matrix",
                    "true_label": tl,
                    "pred_label": pl,
                    "count": int(cm[i, j]),
                })

    outputs_csv = os.path.join(args.outdir, f"outputs_{model_tag}.csv")
    stats_csv = os.path.join(args.outdir, f"stats_{model_tag}.csv")
    write_csv(outputs_csv, per_example_rows)
    write_csv(stats_csv, stats_rows)

    print(f"\nWrote per-example outputs to: {outputs_csv}")
    print(f"Wrote summary stats to:     {stats_csv}")
    print(f"Examples: {n_examples}, skipped: {skipped}, errors: {n_errors}")


if __name__ == "__main__":
    main()