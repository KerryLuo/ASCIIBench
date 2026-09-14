#!/usr/bin/env python3
"""ASCIIBench ASCII Art Generation Pipeline

Generates ASCII art using OpenAI models for each class in the dataset.
Each class gets multiple generation attempts, saved as individual text files.

Usage:
    python run_generation.py                                          # default (gpt-4o, 5 per class)
    python run_generation.py --model gpt-4o-mini                      # specific model
    python run_generation.py --num-generations 3                      # 3 per class
    python run_generation.py --resume                                 # skip existing files
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


SYSTEM_PROMPT = (
    "Please do not respond with any text other than ASCII art. "
    "Once again your response should only return the ASCII art. "
    "Don't reply to me. Just produce art."
)


def load_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))
    return dataset


def generate_ascii_art(client, model, class_name):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": class_name},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content


def safe_request(func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "overloaded" in err:
                wait = 2 ** attempt
                print(f"\n  Rate limit hit. Waiting {wait}s (retry {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries ({max_retries}) exceeded.")


def main():
    parser = argparse.ArgumentParser(description="ASCIIBench ASCII Art Generation")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--dataset", default="final_dataset.jsonl")
    parser.add_argument("--output-dir", default="generations", help="Output directory for generated art")
    parser.add_argument("--num-generations", type=int, default=5,
                        help="Number of generations per item")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items that already have all generation files")
    args = parser.parse_args()

    env_path = Path("secrets") / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to secrets/.env")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} items from {args.dataset}")

    model_safe = args.model.replace("/", "_").replace(":", "_")
    out_dir = Path(args.output_dir) / f"{model_safe}_generations"
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    generated = 0

    for item in tqdm(dataset, desc=f"{args.model} generation"):
        unique_id = item["unique_id"]
        class_name = item["class"]

        if args.resume:
            existing = [f for f in out_dir.iterdir()
                        if f.name.startswith(unique_id) and f.suffix == ".txt"]
            if len(existing) >= args.num_generations:
                skipped += 1
                continue

        for i in range(args.num_generations):
            file_name = f"{unique_id}_{i}.txt"
            file_path = out_dir / file_name

            if args.resume and file_path.exists():
                continue

            try:
                ascii_art = safe_request(generate_ascii_art, client, args.model, class_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(ascii_art)
                generated += 1
            except Exception as e:
                print(f"\n  Error generating {file_name}: {e}")

    print(f"\nDone. Generated {generated} files, skipped {skipped} items.")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
