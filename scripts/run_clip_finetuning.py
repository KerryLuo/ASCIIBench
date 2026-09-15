#!/usr/bin/env python3
"""ASCIIBench CLIP Fine-tuning

Fine-tunes a CLIP model on ASCII art using triplet loss with normalized embeddings.
Logs training metrics to Weights & Biases.

Usage:
    python scripts/run_clip_finetuning.py                                    # default settings
    python scripts/run_clip_finetuning.py --lr 1e-6 --batch-size 16          # custom hyperparams
    python scripts/run_clip_finetuning.py --generate-triplets                # create triplets.json first
    python scripts/run_clip_finetuning.py --resume checkpoints/full_ep3.pth  # resume from checkpoint
    python scripts/run_clip_finetuning.py --no-wandb                         # disable wandb logging
"""

import argparse
import json
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Triplet creation / loading
# ---------------------------------------------------------------------------

def create_triplets(data):
    triplets = []
    for anchor in data:
        same_class = [d for d in data if d["class"] == anchor["class"]
                      and d["unique_id"] != anchor["unique_id"]]
        diff_class = [d for d in data if d["class"] != anchor["class"]]
        if not same_class or not diff_class:
            continue
        positive = random.choice(same_class)
        negative = random.choice(diff_class)
        triplets.append((anchor, positive, negative))
    return triplets


def save_triplets(triplets, path):
    data = [{"anchor": a["unique_id"], "positive": p["unique_id"], "negative": n["unique_id"]}
            for a, p, n in triplets]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} triplets to {path}")


def load_triplets(path, data_dict):
    with open(path, "r", encoding="utf-8") as f:
        triplet_ids = json.load(f)
    triplets = []
    for t in triplet_ids:
        a = data_dict.get(t["anchor"])
        p = data_dict.get(t["positive"])
        n = data_dict.get(t["negative"])
        if a and p and n:
            triplets.append((a, p, n))
    return triplets


# ---------------------------------------------------------------------------
# Preprocessing — render ASCII art to image for CLIP
# ---------------------------------------------------------------------------

_font_cache = {}


def get_font(font_size=18):
    if font_size in _font_cache:
        return _font_cache[font_size]
    font_dir = PROJECT_ROOT / "fonts"
    font_path = font_dir / "DejaVuSansMono.ttf"
    if not font_path.exists():
        font_dir.mkdir(exist_ok=True)
        zip_url = ("https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
                   "version_2_37/dejavu-fonts-ttf-2.37.zip")
        zip_path = font_dir / "dejavu-fonts-ttf-2.37.zip"
        print(f"Downloading DejaVu Sans Mono font to {font_path} ...")
        urllib.request.urlretrieve(zip_url, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            member = "dejavu-fonts-ttf-2.37/ttf/DejaVuSansMono.ttf"
            with zf.open(member) as src, open(font_path, "wb") as dst:
                dst.write(src.read())
        zip_path.unlink()
    font = ImageFont.truetype(str(font_path), font_size)
    _font_cache[font_size] = font
    return font


def preprocess_ascii_for_clip(ascii_art):
    image_size = (1600, 1200)
    font_size = 18
    margin = 10

    font = get_font(font_size)
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), "A", font=font)
    line_height = bbox[3] + 2

    y = 0
    for line in ascii_art.split("\n"):
        draw.text((10, y), line, font=font, fill="black")
        y += line_height

    pixels = image.load()
    w, h = image.size

    def row_blank(r):
        return all(pixels[x, r] == (255, 255, 255) for x in range(w))

    def col_blank(c):
        return all(pixels[c, y] == (255, 255, 255) for y in range(h))

    top = next((y for y in range(h) if not row_blank(y)), None)
    bot = next((y for y in range(h - 1, -1, -1) if not row_blank(y)), None)
    left = next((x for x in range(w) if not col_blank(x)), None)
    right = next((x for x in range(w - 1, -1, -1) if not col_blank(x)), None)

    if None not in (top, bot, left, right):
        image = image.crop((
            max(0, left - margin), max(0, top - margin),
            min(w, right + margin), min(h, bot + margin),
        ))

    return image


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def triplet_loss(anchor, positive, negative, margin=1.0):
    dp = (anchor - positive).pow(2).sum(1)
    dn = (anchor - negative).pow(2).sum(1)
    return torch.relu(dp - dn + margin).mean()


def distance_metrics(anchor, positive, negative):
    dp = (anchor - positive).pow(2).sum(1).mean().item()
    dn = (anchor - negative).pow(2).sum(1).mean().item()
    return dp, dn


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if "rng" in ckpt:
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASCIIBench CLIP Fine-tuning")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "final_dataset.jsonl"))
    parser.add_argument("--triplets", default=str(PROJECT_ROOT / "triplets.json"),
                        help="Path to triplets JSON (loaded if exists, otherwise generated)")
    parser.add_argument("--generate-triplets", action="store_true",
                        help="Force regenerate triplets.json from dataset")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--margin", type=float, default=1.0, help="Triplet loss margin")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--resume-epoch", type=int, default=None,
                        help="Resume from epoch N weights (model-only, no optimizer state)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save mid-epoch checkpoint every N batches")
    parser.add_argument("--wandb-project", default="Finetuning_CLIP",
                        help="Weights & Biases project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} items from {args.dataset}")
    data_dict = {d["unique_id"]: d for d in data}

    # Load or generate triplets
    triplets_path = Path(args.triplets)
    if args.generate_triplets or not triplets_path.exists():
        print("Generating triplets...")
        triplets = create_triplets(data)
        save_triplets(triplets, triplets_path)
    else:
        print(f"Loading triplets from {triplets_path}")
        triplets = load_triplets(triplets_path, data_dict)
        print(f"Loaded {len(triplets)} triplets")

    # Import CLIP
    from transformers import CLIPModel, CLIPProcessor

    # wandb setup
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run_id = f"clip_lr{args.lr}_bs{args.batch_size}_m{args.margin}".replace(".", "_")
            wandb.init(
                project=args.wandb_project,
                name=f"LR: {args.lr}, BS: {args.batch_size}, margin: {args.margin}",
                id=run_id,
                resume="allow",
                config=vars(args),
            )
        except ImportError:
            print("Warning: wandb not installed. Continuing without logging.")
            use_wandb = False

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # Data split
    random.shuffle(triplets)
    split = int(0.9 * len(triplets))
    train_triplets = triplets[:split]
    val_triplets = triplets[split:]
    print(f"Train: {len(train_triplets)}, Validation: {len(val_triplets)}")

    # Model setup
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume logic
    start_epoch = 0
    start_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from full checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
    elif args.resume_epoch is not None:
        weights_path = ckpt_dir / f"clip_weights_epoch_{args.resume_epoch}_lr_{args.lr}_batch_{args.batch_size}.pth"
        if weights_path.exists():
            print(f"Resuming model weights from epoch {args.resume_epoch}")
            model.load_state_dict(torch.load(weights_path, map_location=device))
            start_epoch = args.resume_epoch
        else:
            print(f"Warning: {weights_path} not found, starting fresh.")

    print(f"Starting from epoch {start_epoch + 1}, step {start_step}")

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = total_pos = total_neg = 0.0
            batch_count = 0

            num_batches = (len(train_triplets) + args.batch_size - 1) // args.batch_size

            for batch_idx, i in enumerate(
                tqdm(range(0, len(train_triplets), args.batch_size),
                     desc=f"Epoch {epoch + 1}/{args.epochs}")
            ):
                if epoch == start_epoch and batch_idx < start_step:
                    continue

                batch = train_triplets[i:i + args.batch_size]

                anchor_imgs = [preprocess_ascii_for_clip(a["ascii_art"]) for a, _, _ in batch]
                pos_imgs = [preprocess_ascii_for_clip(p["ascii_art"]) for _, p, _ in batch]
                neg_imgs = [preprocess_ascii_for_clip(n["ascii_art"]) for _, _, n in batch]

                inp_a = processor(images=[img.convert("RGB") for img in anchor_imgs],
                                  return_tensors="pt", padding=True).to(device)
                inp_p = processor(images=[img.convert("RGB") for img in pos_imgs],
                                  return_tensors="pt", padding=True).to(device)
                inp_n = processor(images=[img.convert("RGB") for img in neg_imgs],
                                  return_tensors="pt", padding=True).to(device)

                out_a = F.normalize(model.get_image_features(**inp_a), p=2, dim=1)
                out_p = F.normalize(model.get_image_features(**inp_p), p=2, dim=1)
                out_n = F.normalize(model.get_image_features(**inp_n), p=2, dim=1)

                loss = triplet_loss(out_a, out_p, out_n, margin=args.margin)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pd, nd = distance_metrics(out_a, out_p, out_n)
                total_pos += pd
                total_neg += nd
                batch_count += 1

                if use_wandb:
                    ap_dist = (out_a - out_p).norm(dim=1).mean().item()
                    an_dist = (out_a - out_n).norm(dim=1).mean().item()
                    wandb.log({
                        "batch_loss": loss.item(),
                        "anchor_positive_distance": ap_dist,
                        "anchor_negative_distance": an_dist,
                        "dist_gap": an_dist - ap_dist,
                        "cumulative_avg_loss": total_loss / batch_count,
                        "lr": optimizer.param_groups[0]["lr"],
                    })

                if args.save_every and (batch_idx + 1) % args.save_every == 0:
                    mid_path = str(ckpt_dir / f"ckpt_ep{epoch}_step{batch_idx + 1}.pth")
                    save_checkpoint(model, optimizer, scheduler, epoch, batch_idx + 1, mid_path)

            scheduler.step()

            # Validation
            model.eval()
            val_loss = val_pos = val_neg = 0.0
            val_count = 0

            with torch.no_grad():
                for j in tqdm(range(0, len(val_triplets), args.batch_size),
                              desc=f"Validation {epoch + 1}"):
                    batch = val_triplets[j:j + args.batch_size]

                    anchor_imgs = [preprocess_ascii_for_clip(a["ascii_art"]) for a, _, _ in batch]
                    pos_imgs = [preprocess_ascii_for_clip(p["ascii_art"]) for _, p, _ in batch]
                    neg_imgs = [preprocess_ascii_for_clip(n["ascii_art"]) for _, _, n in batch]

                    inp_a = processor(images=[img.convert("RGB") for img in anchor_imgs],
                                      return_tensors="pt", padding=True).to(device)
                    inp_p = processor(images=[img.convert("RGB") for img in pos_imgs],
                                      return_tensors="pt", padding=True).to(device)
                    inp_n = processor(images=[img.convert("RGB") for img in neg_imgs],
                                      return_tensors="pt", padding=True).to(device)

                    out_a = F.normalize(model.get_image_features(**inp_a), p=2, dim=1)
                    out_p = F.normalize(model.get_image_features(**inp_p), p=2, dim=1)
                    out_n = F.normalize(model.get_image_features(**inp_n), p=2, dim=1)

                    loss = triplet_loss(out_a, out_p, out_n, margin=args.margin)
                    val_loss += loss.item()
                    pd, nd = distance_metrics(out_a, out_p, out_n)
                    val_pos += pd
                    val_neg += nd
                    val_count += 1

                    if use_wandb:
                        ap_dist = (out_a - out_p).norm(dim=1).mean().item()
                        an_dist = (out_a - out_n).norm(dim=1).mean().item()
                        wandb.log({
                            "val_anchor_positive_distance": ap_dist,
                            "val_anchor_negative_distance": an_dist,
                            "val_dist_gap": an_dist - ap_dist,
                        })

            avg_loss = total_loss / max(1, batch_count)
            avg_val_loss = val_loss / max(1, val_count)

            print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f}")

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": avg_loss,
                    "epoch_val_loss": avg_val_loss,
                    "epoch_positive_distance": total_pos / max(1, batch_count),
                    "epoch_negative_distance": total_neg / max(1, batch_count),
                    "epoch_val_positive_distance": val_pos / max(1, val_count),
                    "epoch_val_negative_distance": val_neg / max(1, val_count),
                })

            # Save per-epoch weights
            weights_path = ckpt_dir / f"clip_weights_epoch_{epoch + 1}_lr_{args.lr}_batch_{args.batch_size}.pth"
            torch.save(model.state_dict(), weights_path)
            print(f"  Weights saved: {weights_path}")

            # Save full checkpoint
            full_path = str(ckpt_dir / f"full_ep{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, 0, full_path)

    except KeyboardInterrupt:
        emergency = str(ckpt_dir / f"emergency_ep{epoch}_step{batch_idx}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, emergency)
        print(f"\nInterrupted. Emergency checkpoint saved: {emergency}")

    if use_wandb:
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
