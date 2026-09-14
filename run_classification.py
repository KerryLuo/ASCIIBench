#!/usr/bin/env python3
"""ASCIIBench Classification Pipeline

Unified pipeline for evaluating LLMs on ASCII art classification.
Supports OpenAI and Anthropic models across text, vision, and text+vision modalities.

Usage:
    python run_classification.py                                # run all models in config.yaml
    python run_classification.py --models gpt-4o               # run specific model(s)
    python run_classification.py --modalities text vision      # run specific modalities
    python run_classification.py --ablation                    # inverted-colors ablation
    python run_classification.py --resume                      # resume interrupted run
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Dataset
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


# ---------------------------------------------------------------------------
# Preprocessing — render ASCII art to an image for vision modalities
# ---------------------------------------------------------------------------

_font_cache = {}


def get_font(font_size=18):
    if font_size in _font_cache:
        return _font_cache[font_size]

    font_dir = Path(__file__).resolve().parent / "fonts"
    font_path = font_dir / "DejaVuSansMono.ttf"

    if not font_path.exists():
        font_dir.mkdir(exist_ok=True)
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSansMono.ttf"
        print(f"Downloading DejaVu Sans Mono font to {font_path} ...")
        urllib.request.urlretrieve(url, str(font_path))

    font = ImageFont.truetype(str(font_path), font_size)
    _font_cache[font_size] = font
    return font


def render_ascii_to_image(ascii_art, preproc):
    w, h = preproc.get("image_size", [1600, 1200])
    font_size = preproc.get("font_size", 18)
    bg = preproc.get("background", "white")
    fg = preproc.get("text_color", "black")
    margin = preproc.get("margin", 10)

    font = get_font(font_size)
    image = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), "A", font=font)
    line_height = bbox[3] + 2

    y = 0
    for line in ascii_art.split("\n"):
        draw.text((10, y), line, font=font, fill=fg)
        y += line_height

    blank = (255, 255, 255) if bg == "white" else (0, 0, 0)
    pixels = image.load()
    width, height = image.size

    def row_blank(r):
        return all(pixels[x, r] == blank for x in range(width))

    def col_blank(c):
        return all(pixels[c, y] == blank for y in range(height))

    top = next((y for y in range(height) if not row_blank(y)), None)
    bot = next((y for y in range(height - 1, -1, -1) if not row_blank(y)), None)
    left = next((x for x in range(width) if not col_blank(x)), None)
    right = next((x for x in range(width - 1, -1, -1) if not col_blank(x)), None)

    if None not in (top, bot, left, right):
        image = image.crop((
            max(0, left - margin),
            max(0, top - margin),
            min(width, right + margin),
            min(height, bot + margin),
        ))

    return image


def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def find_classification(response, choices):
    response_lower = str(response).lower()
    counts = {choice: response_lower.count(choice) for choice in choices}
    max_count = max(counts.values(), default=0)
    if max_count == 0:
        return False
    best = [c for c, n in counts.items() if n == max_count]
    return best[0] if len(best) == 1 else False


# ---------------------------------------------------------------------------
# API classification functions
# ---------------------------------------------------------------------------

def classify_openai(client, model, ascii_art, choices, modality, preproc):
    if modality == "text":
        messages = [{
            "role": "user",
            "content": (
                f"Classify the following ASCII art into one of the given categories:\n\n"
                f"{ascii_art}\n\n"
                f"Choices: {', '.join(choices)}\n\nClassification:"
            ),
        }]
    elif modality == "vision":
        b64 = image_to_base64(render_ascii_to_image(ascii_art, preproc))
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Classify the following ASCII art into one of the given categories:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": f"\n\nChoices: {', '.join(choices)}\n\nClassification:"},
        ]}]
    else:  # text_vision
        b64 = image_to_base64(render_ascii_to_image(ascii_art, preproc))
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Classify the following ASCII art into one of the given categories:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": ascii_art},
            {"type": "text", "text": f"\n\nChoices: {', '.join(choices)}\n\nClassification:"},
        ]}]

    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip().lower()


ANTHROPIC_SYSTEM = (
    "You are an ascii image classifier. "
    "Classify the ascii art into one of the four given categories."
)
ANTHROPIC_FORMAT = (
    'Make sure your answer follows this format: "one of the four given categories." '
    'and then "your reasoning why". For example, if the options were omega, bird, '
    'ocean, and land, your answer would follow this format: "Omega. This ASCII image '
    "clearly resembles an Omega because xyz. If you're unsure, still state one of the "
    "four options in the exact way the option presents it, and then elaborate on why "
    "you're unsure."
)


def classify_anthropic(client, model, ascii_art, choices, modality, preproc):
    choices_str = ", ".join(choices)

    if modality == "text":
        content = [
            {"type": "text", "text": (
                f"Classify the following ASCII art into one of the given categories:"
                f"\n\n{ascii_art}\n\nChoices: {choices_str}\n\nClassification:"
            )},
            {"type": "text", "text": ANTHROPIC_FORMAT},
        ]
    elif modality == "vision":
        b64 = image_to_base64(render_ascii_to_image(ascii_art, preproc))
        content = [
            {"type": "text", "text": "Classify the following ASCII art into one of the given categories:\n\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": f"\n\nChoices: {choices_str}\n\nClassification:"},
            {"type": "text", "text": ANTHROPIC_FORMAT},
        ]
    else:  # text_vision
        b64 = image_to_base64(render_ascii_to_image(ascii_art, preproc))
        content = [
            {"type": "text", "text": "Classify the following ASCII art into one of the given categories:\n\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": (
                f"Classify the following ASCII art into one of the given categories:"
                f"\n\n{ascii_art}\n\nChoices: {choices_str}\n\nClassification:"
            )},
            {"type": "text", "text": ANTHROPIC_FORMAT},
        ]

    msg = client.messages.create(
        model=model,
        max_tokens=50,
        temperature=0,
        system=ANTHROPIC_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(getattr(b, "text", str(b)) for b in msg.content)


# ---------------------------------------------------------------------------
# Rate-limit retry
# ---------------------------------------------------------------------------

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


def results_filename(model, modality, ablation=False):
    safe = model.replace("/", "_").replace(":", "_")
    suffix = "_ablation" if ablation else ""
    return f"{safe}_{modality}{suffix}_results.jsonl"


# ---------------------------------------------------------------------------
# Choice pre-generation (deterministic regardless of resume point)
# ---------------------------------------------------------------------------

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
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_PREPROCESSING = {
    "font_size": 18,
    "image_size": [1600, 1200],
    "background": "white",
    "text_color": "black",
    "margin": 10,
}

DEFAULT_MODELS = [
    {"name": "gpt-4o", "provider": "openai", "modalities": ["text", "vision", "text_vision"]},
    {"name": "gpt-4o-mini", "provider": "openai", "modalities": ["text", "vision", "text_vision"]},
    {"name": "gpt-5-mini", "provider": "openai", "modalities": ["text", "vision", "text_vision"]},
    {"name": "gpt-3.5-turbo", "provider": "openai", "modalities": ["text"]},
    {"name": "claude-3-5-sonnet-20240620", "provider": "anthropic", "modalities": ["text", "vision", "text_vision"]},
]


def load_config(path):
    if yaml is None:
        print("Warning: pyyaml not installed. Using default configuration.")
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config, args):
    dataset_path = config.get("dataset_path", "final_dataset.jsonl")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} items from {dataset_path}")

    seed = config.get("seed", 42)
    choices_list = pregenerate_choices(dataset, seed)

    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)

    preproc = {**DEFAULT_PREPROCESSING, **config.get("preprocessing", {})}
    if args.ablation:
        preproc["background"] = "black"
        preproc["text_color"] = "white"
        print("Ablation mode: inverted colors (black background, white text)")

    models = config.get("models", DEFAULT_MODELS)
    if args.models:
        requested = set(args.models)
        models = [m for m in models if m["name"] in requested]
        if not models:
            print(f"Error: none of {args.models} found in config. Available: "
                  f"{[m['name'] for m in config.get('models', DEFAULT_MODELS)]}")
            sys.exit(1)

    # Initialize API clients lazily
    clients = {}
    needed_providers = {m["provider"] for m in models}

    for provider in needed_providers:
        if provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                print("Warning: OPENAI_API_KEY not set — skipping OpenAI models.")
                continue
            from openai import OpenAI
            clients["openai"] = OpenAI(api_key=key)
        elif provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                print("Warning: ANTHROPIC_API_KEY not set — skipping Anthropic models.")
                continue
            import anthropic
            clients["anthropic"] = anthropic.Anthropic(api_key=key)

    classify_fn = {"openai": classify_openai, "anthropic": classify_anthropic}

    for model_cfg in models:
        name = model_cfg["name"]
        provider = model_cfg["provider"]
        modalities = model_cfg.get("modalities", ["text"])

        if args.modalities:
            modalities = [m for m in modalities if m in args.modalities]
        if not modalities:
            continue

        if provider not in clients:
            print(f"Skipping {name}: no API key for {provider}")
            continue

        client = clients[provider]
        fn = classify_fn[provider]

        for modality in modalities:
            out_file = results_dir / results_filename(name, modality, args.ablation)
            existing = count_existing_results(out_file)

            if existing >= len(dataset):
                print(f"\n{name} ({modality}): all {len(dataset)} results exist — skipping.")
                continue

            status = f"resuming from {existing}" if existing else "starting"
            print(f"\n{name} ({modality}): {status} ({len(dataset)} total)")

            correct = 0
            total = 0

            for idx, item in enumerate(tqdm(dataset, desc=f"{name} ({modality})")):
                if idx < existing:
                    continue

                choices = choices_list[idx]
                try:
                    response = safe_request(
                        fn, client, name, item["ascii_art"], choices, modality, preproc,
                    )
                except Exception as e:
                    print(f"\n  Error on item {idx}: {e}")
                    response = f"ERROR: {e}"

                pred = find_classification(response, choices)
                is_correct = pred == item["class"]

                append_result(out_file, {
                    "model": name,
                    "modality": modality,
                    "ascii_art": item["ascii_art"],
                    "choices": choices,
                    "predicted_class": pred,
                    "actual_class": item["class"],
                    "correct": is_correct,
                    "response": response,
                })

                if is_correct:
                    correct += 1
                total += 1

            if total:
                print(f"  Accuracy so far: {correct}/{total} = {correct / total:.4f}")
            print(f"  Results saved to {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASCIIBench Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--models", nargs="+", help="Model name(s) to run (overrides config)")
    parser.add_argument("--modalities", nargs="+",
                        choices=["text", "vision", "text_vision"],
                        help="Modality/ies to run (overrides config)")
    parser.add_argument("--ablation", action="store_true",
                        help="Inverted-colors ablation (black bg, white text)")
    parser.add_argument("--dataset", help="Path to dataset JSONL (overrides config)")
    parser.add_argument("--results-dir", help="Path to results directory (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    args = parser.parse_args()

    # Load secrets
    env_path = Path("secrets") / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    config = load_config(args.config)
    if args.dataset:
        config["dataset_path"] = args.dataset
    if args.results_dir:
        config["results_dir"] = args.results_dir
    if args.seed is not None:
        config["seed"] = args.seed

    run(config, args)


if __name__ == "__main__":
    main()
