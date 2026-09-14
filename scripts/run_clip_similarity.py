#!/usr/bin/env python3
"""ASCIIBench CLIP Similarity Testing

Computes cosine similarity between original and generated ASCII art using a
fine-tuned CLIP model. Produces a CSV of pairwise similarities and optional
analysis plots (ROC-AUC, distributions, intra-class variance).

Usage:
    python scripts/run_clip_similarity.py --weights checkpoints/clip_weights_epoch_5_lr_1e-06_batch_16.pth
    python scripts/run_clip_similarity.py --weights model.pth --generated-dir generations/gpt-4o_generations
    python scripts/run_clip_similarity.py --weights model.pth --generated-jsonl generated_ascii_GPT4.jsonl
    python scripts/run_clip_similarity.py --weights model.pth --analyze   # also produce plots
"""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Preprocessing — matches the original similarity testing approach
# ---------------------------------------------------------------------------

def preprocess_ascii_for_clip(ascii_art, font_size=12):
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    image = Image.new("RGB", (800, 600), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), ascii_art, fill=(255, 255, 255), font=font)

    bbox = image.getbbox()
    cropped = image.crop(bbox) if bbox else image
    grayscale = cropped.convert("L")
    blurred = grayscale.filter(ImageFilter.GaussianBlur(0))
    normalized = np.array(blurred) / 255.0

    return Image.fromarray((normalized * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ERROR_WORDS = ["Apologies", "apologies", "sorry", "Sorry", "can't assist"]


def has_error_words(text, words=ERROR_WORDS):
    return any(w in text for w in words)


def load_originals(jsonl_path):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append({
                "id": data["unique_id"],
                "class": data["unique_id"].rsplit("_", 1)[0],
                "ascii": data["ascii_art"],
            })
    return items


def load_generated_from_dir(folder):
    items = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if has_error_words(content):
            continue
        gen_class = fname.split("_")[0]
        items.append({"filename": fname, "class": gen_class, "ascii": content})
    return items


def load_generated_from_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            content = entry["content"]
            if has_error_words(content):
                continue
            fname = entry["filename"]
            gen_class = fname.split("_")[0]
            items.append({"filename": fname, "class": gen_class, "ascii": content})
    return items


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def batch_embed(model, processor, ascii_list, device, batch_size=64, label=""):
    embeddings = []
    for i in tqdm(range(0, len(ascii_list), batch_size), desc=f"Embedding {label}"):
        batch = ascii_list[i:i + batch_size]
        images = [preprocess_ascii_for_clip(a["ascii"]) for a in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs).cpu()
        embeddings.append(features)
    return torch.cat(embeddings, dim=0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(df, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError as e:
        print(f"Warning: analysis requires matplotlib, seaborn, scikit-learn ({e})")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Summary stats
    print("\nSame-class similarity:")
    print(df[df["same_class"]]["similarity"].describe())
    print("\nDifferent-class similarity:")
    print(df[~df["same_class"]]["similarity"].describe())

    # ROC-AUC
    if df["same_class"].nunique() >= 2:
        auc = roc_auc_score(df["same_class"], df["similarity"])
        fpr, tpr, _ = roc_curve(df["same_class"], df["similarity"])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve: Similarity vs Class Match")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nROC-AUC: {auc:.4f}")

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="same_class", y="similarity")
    plt.title("Cosine Similarity: Same Class vs Different Class")
    plt.xticks([0, 1], ["Different Class", "Same Class"])
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df["same_class"]]["similarity"], bins=50, color="green",
                 label="Same Class", kde=True, stat="density")
    sns.histplot(df[~df["same_class"]]["similarity"], bins=50, color="red",
                 label="Different Class", kde=True, stat="density")
    plt.title("Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Intra-class variance
    same_df = df[df["same_class"]]
    if len(same_df) > 0:
        stats = same_df.groupby("original_id")["similarity"].agg(
            mean_similarity="mean", std_similarity="std",
            min_similarity="min", max_similarity="max", count="count",
        ).reset_index().sort_values("std_similarity", ascending=False)

        stats_path = os.path.join(output_dir, "intra_class_stats.csv")
        stats.to_csv(stats_path, index=False)
        print(f"Intra-class stats saved to {stats_path}")

        clean = stats[(stats["mean_similarity"] > 0.3) & (stats["std_similarity"] < 0.15)]
        print(f"Clean classes (mean>0.3, std<0.15): {len(clean)}/{len(stats)}")

        if len(clean) > 0:
            clean_ids = set(clean["original_id"])
            df_clean = df[df["original_id"].isin(clean_ids)]
            if df_clean["same_class"].nunique() >= 2:
                clean_auc = roc_auc_score(df_clean["same_class"], df_clean["similarity"])
                print(f"ROC-AUC on clean classes: {clean_auc:.4f}")

    print(f"\nPlots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASCIIBench CLIP Similarity Testing")
    parser.add_argument("--weights", required=True,
                        help="Path to fine-tuned CLIP model weights (.pth)")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "final_dataset.jsonl"),
                        help="Path to original ASCII art JSONL dataset")
    parser.add_argument("--generated-dir", default=None,
                        help="Directory of generated .txt files")
    parser.add_argument("--generated-jsonl", default=None,
                        help="JSONL cache of generated ASCII art")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results/clip_similarity_<model>.csv)")
    parser.add_argument("--model-name", default="GPT4",
                        help="Label for the generation model (used in output naming)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for embedding")
    parser.add_argument("--analyze", action="store_true",
                        help="Produce analysis plots and stats")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for analysis plots (default: results/similarity_plots/)")
    parser.add_argument("--save-generated-jsonl", default=None,
                        help="Cache generated art to this JSONL path for future runs")
    args = parser.parse_args()

    if not args.generated_dir and not args.generated_jsonl:
        default_dir = str(PROJECT_ROOT / "generations" / f"{args.model_name}_generations")
        if os.path.isdir(default_dir):
            args.generated_dir = default_dir
            print(f"Auto-detected generated dir: {default_dir}")
        else:
            parser.error("Provide --generated-dir or --generated-jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load data
    print("Loading original ASCII art...")
    originals = load_originals(args.dataset)
    print(f"  {len(originals)} originals")

    if args.generated_jsonl:
        print(f"Loading generated art from {args.generated_jsonl}...")
        generated = load_generated_from_jsonl(args.generated_jsonl)
    else:
        print(f"Loading generated art from {args.generated_dir}...")
        generated = load_generated_from_dir(args.generated_dir)

    print(f"  {len(generated)} generated (after filtering errors)")

    # Cache generated art if requested
    if args.save_generated_jsonl:
        with open(args.save_generated_jsonl, "w", encoding="utf-8") as f:
            for g in generated:
                f.write(json.dumps({"filename": g["filename"], "content": g["ascii"]},
                                   ensure_ascii=False) + "\n")
        print(f"Cached generated art to {args.save_generated_jsonl}")

    # Embed
    print("\nComputing embeddings...")
    orig_emb = batch_embed(model, processor, originals, device, args.batch_size, "originals")
    gen_emb = batch_embed(model, processor, generated, device, args.batch_size, "generated")

    orig_emb = F.normalize(orig_emb, p=2, dim=1)
    gen_emb = F.normalize(gen_emb, p=2, dim=1)

    # Similarity matrix
    print("Computing similarity matrix...")
    sim_matrix = torch.mm(orig_emb, gen_emb.T)

    # Build results
    print("Formatting results...")
    results = []
    for i, orig in enumerate(tqdm(originals, desc="Building CSV")):
        for j, gen in enumerate(generated):
            results.append({
                "original_id": orig["id"],
                "generated_file": gen["filename"],
                "compared_to_id": gen["filename"].split("_")[0],
                "similarity": sim_matrix[i][j].item(),
                "same_class": orig["class"] == gen["class"],
            })

    df = pd.DataFrame(results)

    # Save
    if args.output:
        csv_path = args.output
    else:
        results_dir = str(PROJECT_ROOT / "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f"clip_similarity_{args.model_name}.csv")

    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} similarity pairs to {csv_path}")

    # Analysis
    if args.analyze:
        plot_dir = args.plot_dir or str(PROJECT_ROOT / "results" / "similarity_plots")
        run_analysis(df, plot_dir)


if __name__ == "__main__":
    main()
