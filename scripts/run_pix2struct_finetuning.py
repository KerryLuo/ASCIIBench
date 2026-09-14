#!/usr/bin/env python3
"""ASCIIBench Pix2Struct Fine-tuning

Fine-tunes a Pix2Struct model on ASCII art rendered as images, paired with
their class label as the target text (image captioning / classification).
Logs training metrics to Weights & Biases.

Usage:
    python scripts/run_pix2struct_finetuning.py                                    # default settings
    python scripts/run_pix2struct_finetuning.py --lr 1e-5 --batch-size 4           # custom hyperparams
    python scripts/run_pix2struct_finetuning.py --resume checkpoints/full_ep3.pth  # resume from checkpoint
    python scripts/run_pix2struct_finetuning.py --no-wandb                         # disable wandb logging
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
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
# Preprocessing — render ASCII art to image for Pix2Struct
# ---------------------------------------------------------------------------

_font_cache = {}


def get_font(font_size=18):
    if font_size in _font_cache:
        return _font_cache[font_size]
    font_dir = PROJECT_ROOT / "fonts"
    font_path = font_dir / "DejaVuSansMono.ttf"
    if not font_path.exists():
        font_dir.mkdir(exist_ok=True)
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSansMono.ttf"
        print(f"Downloading DejaVu Sans Mono font to {font_path} ...")
        urllib.request.urlretrieve(url, str(font_path))
    font = ImageFont.truetype(str(font_path), font_size)
    _font_cache[font_size] = font
    return font


def preprocess_ascii_for_pix2struct(ascii_art):
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

    return image.convert("RGB")


# ---------------------------------------------------------------------------
# PyTorch Dataset — ASCII art image paired with its class label
# ---------------------------------------------------------------------------

class AsciiArtCaptioningDataset(Dataset):
    def __init__(self, items, processor, max_patches=1024, max_length=32):
        self.items = items
        self.processor = processor
        self.max_patches = max_patches
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = preprocess_ascii_for_pix2struct(item["ascii_art"])

        encoding = self.processor(images=image, max_patches=self.max_patches, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        labels = self.processor.tokenizer(
            item["class"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


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
    parser = argparse.ArgumentParser(description="ASCIIBench Pix2Struct Fine-tuning")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "final_dataset.jsonl"))
    parser.add_argument("--model-name", default="google/pix2struct-base")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=32, help="Max target token length")
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save mid-epoch checkpoint every N batches")
    parser.add_argument("--wandb-project", default="Finetuning_Pix2Struct",
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

    # Data split
    random.shuffle(data)
    split = int(0.9 * len(data))
    train_items = data[:split]
    val_items = data[split:]
    print(f"Train: {len(train_items)}, Validation: {len(val_items)}")

    # Import Pix2Struct
    from transformers import AutoProcessor, Pix2StructForConditionalGeneration

    # wandb setup
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run_id = f"pix2struct_lr{args.lr}_bs{args.batch_size}".replace(".", "_")
            wandb.init(
                project=args.wandb_project,
                name=f"LR: {args.lr}, BS: {args.batch_size}",
                id=run_id,
                resume="allow",
                config=vars(args),
            )
        except ImportError:
            print("Warning: wandb not installed. Continuing without logging.")
            use_wandb = False

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # Model setup
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        args.model_name, is_encoder_decoder=True
    ).to(device)

    train_dataset = AsciiArtCaptioningDataset(train_items, processor, args.max_patches, args.max_length)
    val_dataset = AsciiArtCaptioningDataset(val_items, processor, args.max_patches, args.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    num_training_steps = len(train_loader) * args.epochs
    optimizer = Adafactor(
        model.parameters(), scale_parameter=False, relative_step=False,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    # Resume logic
    start_epoch = 0
    start_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)

    print(f"Starting from epoch {start_epoch + 1}, step {start_step}")

    # Training loop
    epoch, batch_idx = start_epoch, 0
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            ):
                if epoch == start_epoch and batch_idx < start_step:
                    continue

                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                batch_count += 1

                if use_wandb:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "cumulative_avg_loss": total_loss / batch_count,
                        "lr": optimizer.param_groups[0]["lr"],
                    })

                if args.save_every and (batch_idx + 1) % args.save_every == 0:
                    mid_path = str(ckpt_dir / f"ckpt_ep{epoch}_step{batch_idx + 1}.pth")
                    save_checkpoint(model, optimizer, scheduler, epoch, batch_idx + 1, mid_path)

            # Validation
            model.eval()
            val_loss = 0.0
            val_count = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}"):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    val_count += 1

                    generated = model.generate(
                        flattened_patches=batch["flattened_patches"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=args.max_length,
                    )
                    preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    labels = batch["labels"].clone()
                    labels[labels == -100] = processor.tokenizer.pad_token_id
                    targets = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                    for pred, target in zip(preds, targets):
                        if pred.strip().lower() == target.strip().lower():
                            correct += 1
                        total += 1

            avg_loss = total_loss / max(1, batch_count)
            avg_val_loss = val_loss / max(1, val_count)
            val_accuracy = correct / max(1, total)

            print(f"  Epoch {epoch + 1}: loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f}, "
                  f"val_accuracy={val_accuracy:.4f}")

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": avg_loss,
                    "epoch_val_loss": avg_val_loss,
                    "epoch_val_accuracy": val_accuracy,
                })

            # Save per-epoch weights
            weights_path = ckpt_dir / f"pix2struct_weights_epoch_{epoch + 1}_lr_{args.lr}_batch_{args.batch_size}.pth"
            torch.save(model.state_dict(), weights_path)
            print(f"  Weights saved: {weights_path}")

            # Save full checkpoint
            full_path = str(ckpt_dir / f"pix2struct_full_ep{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, 0, full_path)

    except KeyboardInterrupt:
        emergency = str(ckpt_dir / f"pix2struct_emergency_ep{epoch}_step{batch_idx}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, emergency)
        print(f"\nInterrupted. Emergency checkpoint saved: {emergency}")

    if use_wandb:
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
