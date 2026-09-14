#!/usr/bin/env python3
"""ASCIIBench LLaMA Classification

Separate script for running LLaMA (local HuggingFace) classification.
Requires a CUDA GPU and the dependencies in requirements-llama.txt.

Usage:
    python scripts/run_llama_classification.py                                              # both models
    python scripts/run_llama_classification.py --models meta-llama/Meta-Llama-3.1-8B        # one model
    python scripts/run_llama_classification.py --dataset final_dataset.jsonl --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Few-shot chain-of-thought prompt (from the original evaluation)
# ---------------------------------------------------------------------------

COT_PROMPT = (
    "Classify the following ASCII art into one of the given categories:\n\n"
    "____               /))\n"
    "|;;/;:\\____\\|_____/:/ <\n"
    " V| <:::::::::::::::> |\n"
    "   \\ ##:::::::::::## /\n"
    "   /:::/~\\:::::/~\\:::\\\n"
    "  |##:| @ |:::| @ |:##|\n"
    "  |{__}\\_/ %%% \\_/::::|\n"
    "  |###:::::\\_/:::::###|\n"
    "   \\:::::\\__|__/:::::/\n"
    "     \\##::,___,::##/\n"
    "       /@@:::::@@\\\n"
    "     /###@::@@::###\\\n\n"
    "Choices: ostrich, tents, cats, swan\n\n"
    "Classification: cats\n\n\n"
    "Classify the following ASCII art into one of the given categories:\n\n"
    "\\n~CONFUCIOUS~\\n"
    "                                 .\n"
    "                               .:::.\n"
    "                             .:::::::.\n"
    "                            V^V^V^V^V^V\n"
    "                             (| ^ ^ |)\n"
    "                              | (_) |\n"
    "                              `//=\\\\'\n"
    "                              (((())))\n"
    "                               )))(((\n"
    "                               (())))\n"
    "                                ))((\n"
    "                                (()\n"
    "                                 ))\n"
    "                                 (\n\n"
    "Choices: toystory, insect, confucious, small]\n\n"
    "Classification: confucious\n\n\n"
    "Classify the following ASCII art into one of the given categories:\n\n"
    "                 ___      ___        ::::::::::::::::::::::::\n"
    "                [_ _]    [_ _]   _\n"
    "           /|  ___$________S_   | \\\n"
    "          / |-/        ____  [++| |+\n"
    "         <<<<<---<|  |>____O)|\n"
    "          \\ |-\\___ ________ _[++| |+\n"
    "           \\|    _$_      _S_   |_/       :::::::::::::::::::::::\n"
    "                [___]    [___]            b'\n\n"
    "Choices: land, sofas, grinch, rose\n\n"
    "Classification: land\n\n\n"
    "Classify the following ASCII art into one of the given categories:\n\n"
    '"            o (>,\n'
    "           8 oo |\\      \n"
    '            8 "}| \\\n'
    "            , ., ,'\n"
    "           ('`')'            \n"
    "           )\\____,<)   \n"
    "          / (__,_      \n"
    "          |   (-,/   \n"
    "        .'    ) /  \n"
    "         `._,\\ '`- \n"
    "             `\\      \n"
    "             -`'\n"
    "     --  - -     --   ---   VK/\n\n"
    "Choices: dancing, eye, cats, insect\n\n"
    "Classification: dancing\n\n\n"
    "Classify the following ASCII art into one of the given categories:\n\n"
    '"                             \'\n'
    "                              '\n"
    "~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-\n"
    "                            -~\n"
    "                           -~\n"
    "                           ~-~\n"
    "               ~-~-   -~-~  ~-~\n"
    "                ~-~-~-~-      ~\n"
    "                  ~-~-~\n\n"
    "Choices: sea, romance, thermos, dinosaur\n\n"
    "Classification: sea\n\n\n"
)


# ---------------------------------------------------------------------------
# Dataset & choices
# ---------------------------------------------------------------------------

def load_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            dataset.append({"class": item["class"], "ascii_art": item["ascii_art"]})
    return dataset


def pregenerate_choices(dataset, seed):
    random.seed(seed)
    all_classes = sorted({item["class"] for item in dataset})
    choices_list = []
    for item in dataset:
        others = [c for c in all_classes if c != item["class"]]
        wrong = random.sample(others, 3)
        four = wrong + [item["class"]]
        random.shuffle(four)
        choices_list.append(four)
    return choices_list


# ---------------------------------------------------------------------------
# Response parsing (LLaMA variant: split on "Classification" first)
# ---------------------------------------------------------------------------

def find_classification(response, choices):
    tail = response.split("Classification")[-1].lower()
    counts = {choice: tail.count(choice) for choice in choices}
    max_count = max(counts.values(), default=0)
    if max_count == 0:
        return False
    best = [c for c, n in counts.items() if n == max_count]
    return best[0] if len(best) == 1 else False


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def count_existing_results(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def append_result(path, result):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def results_filename(model_name):
    safe = model_name.replace("/", "_").replace(":", "_")
    return f"{safe}_text_results.jsonl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASCIIBench LLaMA Classification")
    parser.add_argument("--models", nargs="+", default=[
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ], help="HuggingFace model name(s)")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "final_dataset.jsonl"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    # Check for GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("Error: CUDA GPU required for LLaMA inference.")
            sys.exit(1)
        torch.set_default_device("cuda")
    except ImportError:
        print("Error: PyTorch not installed. Run: pip install -r requirements-llama.txt")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("Error: transformers not installed. Run: pip install -r requirements-llama.txt")
        sys.exit(1)

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} items from {args.dataset}")

    choices_list = pregenerate_choices(dataset, args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    for model_name in args.models:
        out_file = results_dir / results_filename(model_name)
        existing = count_existing_results(out_file)

        if existing >= len(dataset):
            print(f"\n{model_name}: all {len(dataset)} results exist — skipping.")
            continue

        status = f"resuming from {existing}" if existing else "starting"
        print(f"\n{model_name}: {status} ({len(dataset)} total)")
        print("Loading model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        correct = 0
        total = 0

        for idx, item in enumerate(tqdm(dataset, desc=model_name)):
            if idx < existing:
                continue

            choices = choices_list[idx]
            prompt = (
                f"{COT_PROMPT}"
                f"Classify the following ASCII art into one of the given categories:\n\n"
                f"{item['ascii_art']}\n\n"
                f"Choices: {', '.join(choices)}\n\n"
                f"Classification: "
            )

            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
            output = model.generate(
                pad_token_id=tokenizer.eos_token_id,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
            )
            response_text = tokenizer.decode(output[0])

            pred = find_classification(response_text, choices)
            is_correct = pred == item["class"]

            append_result(out_file, {
                "model": model_name,
                "modality": "text",
                "ascii_art": item["ascii_art"],
                "choices": choices,
                "predicted_class": pred,
                "actual_class": item["class"],
                "correct": is_correct,
                "response": response_text,
            })

            if is_correct:
                correct += 1
            total += 1

        if total:
            print(f"  Accuracy: {correct}/{total} = {correct / total:.4f}")
        print(f"  Results saved to {out_file}")

        # Free GPU memory before loading next model
        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
