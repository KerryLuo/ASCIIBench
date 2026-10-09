#!/usr/bin/env python3
"""ASCIIBench Pix2Struct Fine-tuning on ASCIITune

Fine-tunes Pix2Struct on ASCIITune (https://huggingface.co/datasets/ASCIIEval/ASCIITune) to recognize
the concept depicted by rendered ASCII art, and evaluates on the ASCIIEval test set
(https://huggingface.co/datasets/ASCIIEval/ASCIIEval), a 4-way multiple-choice benchmark (chance = 25%).

Why the training is set up this way
-----------------------------------
A first version trained Pix2Struct to *generate* the label. It learned label frequencies instead of
looking at the art: a text-only baseline that picks the most common training label among the choices
matched it on validation and beat it on the test set. Two properties of the data make that shortcut easy:
ASCIITune's own distractors are mostly made-up or never-correct labels, and the label distribution is
very long-tailed (~2.3K concepts, many seen once). So this version:

  1. Trains with a multiple-choice objective. Each image is scored against its answer plus negatives
     sampled from the training labels, and a cross-entropy over those choice scores is added to the
     usual generation loss. Negatives are drawn from the same label distribution the positives are
     sampled from, so knowing "which labels are common" carries no information — only the image does.
  2. Samples training images with weight count(label)^-balance_power so frequent labels don't dominate.
  3. Validates on choices rebuilt from real training labels (like ASCIIEval's distractors), not on
     ASCIITune's easy distractors, so validation accuracy tracks the test set.
  4. Reports image-free reference points at every eval:
       prior_mc_acc   pick the choice most common as a training answer (never looks at the image)
       blank_mc_acc   the model's own pick when shown a blank image instead of the art
       mc_acc_calib   the model's pick after subtracting each choice's blank-image score
     A model that really uses the image must beat prior_mc_acc and blank_mc_acc by a clear margin.

--train-source curated trains on ASCIITune's train_rational split instead: 6.3K items that GPT-5
recognized correctly from the image (a cleaner, smaller subset). The validation items are held out of
the full training set either way, so both sources are scored on the same validation set.

Outputs in --output-dir (read by scripts/analyze_pix2struct_convergence.py):
  metrics.jsonl  config.json  preds/val_*.jsonl  best/  last.pt  test_results.json

Usage:
    python scripts/run_pix2struct_asciitune.py --output-dir runs/asciitune_mc
    python scripts/run_pix2struct_asciitune.py --output-dir runs/asciitune_mc_curated --train-source curated
    python scripts/run_pix2struct_asciitune.py --output-dir runs/smoke --max-train-samples 64 --max-val-samples 32 --epochs 1
    python scripts/run_pix2struct_asciitune.py --output-dir runs/asciitune_mc --resume runs/asciitune_mc/last.pt
    python scripts/run_pix2struct_asciitune.py --output-dir runs/old_eval --eval-only runs/old_run/best
"""

import argparse
import ast
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# RunPod images set HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer, which makes every
# Hub download fail. Fall back to the normal downloader in that case (must run before HF imports).
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    import importlib.util

    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

from run_pix2struct_finetuning import get_font

TRAIN_REPO = "ASCIIEval/ASCIITune"
TRAIN_FILES = {"full": "train.jsonl", "curated": "train_synthesize.jsonl"}
TEST_REPO = "ASCIIEval/ASCIIEval"
TEST_FILE = "test.jsonl"


# ---------------------------------------------------------------------------
# Dataset loading — ASCIITune / ASCIIEval from the Hugging Face Hub
# ---------------------------------------------------------------------------

def download_jsonl(repo_id, filename, cache_dir=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir)


def clean_label(text):
    """ASCIITune labels come from source URLs (e.g. 'winnie%20the%20pooh', 'musical-instruments')."""
    text = unquote(str(text))
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split()).lower()


def parse_list(value):
    # train_synthesize.jsonl stores choices/labels as Python-literal strings.
    return ast.literal_eval(value) if isinstance(value, str) else value


def load_asciieval_jsonl(path):
    items = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            choices = [clean_label(c) for c in parse_list(row["choices"])]
            labels = parse_list(row["labels"])
            if sum(labels) != 1 or not row.get("ascii_art", "").strip():
                skipped += 1
                continue
            items.append({
                "ascii_art": row["ascii_art"],
                "choices": choices,
                "answer_idx": labels.index(1),
                "answer": choices[labels.index(1)],
                "url": row.get("url", ""),
            })
    if skipped:
        print(f"  Skipped {skipped} malformed rows in {path}")
    return items


class LabelSampler:
    """Draws distractor labels with probability proportional to count(label)^power."""

    def __init__(self, counts, power=1.0):
        self.labels = sorted(counts)
        self.cum_weights = list(itertools.accumulate(counts[l] ** power for l in self.labels))

    def sample(self, rng, exclude, k):
        out = []
        while len(out) < k:
            label = rng.choices(self.labels, cum_weights=self.cum_weights)[0]
            if label != exclude and label not in out:
                out.append(label)
        return out


def with_resampled_choices(items, sampler, seed, num_choices=4):
    """Replace each item's distractors with real labels drawn from `sampler` (fixed seed)."""
    rng = random.Random(seed)
    out = []
    for item in items:
        choices = sampler.sample(rng, item["answer"], num_choices - 1) + [item["answer"]]
        rng.shuffle(choices)
        out.append({**item, "choices": choices, "answer_idx": choices.index(item["answer"])})
    return out


# ---------------------------------------------------------------------------
# Rendering — ASCII art to a tightly cropped image sized to its content
# ---------------------------------------------------------------------------

def render_ascii(ascii_art, font_size=18, margin=10):
    """Render on a canvas sized to the art (wide art is not clipped), then crop to the ink bbox."""
    font = get_font(font_size)
    lines = ascii_art.expandtabs(4).split("\n")
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    char_w = max(1, int(math.ceil(probe.textlength("M", font=font))))
    line_h = probe.textbbox((0, 0), "A", font=font)[3] + 2

    width = max(1, max(len(l) for l in lines)) * char_w + 2 * margin
    height = max(1, len(lines)) * line_h + 2 * margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * line_h), line, font=font, fill="black")

    bbox = ImageChops.invert(image.convert("L")).getbbox()
    if bbox:
        left, top, right, bot = bbox
        image = image.crop((max(0, left - margin), max(0, top - margin),
                            min(width, right + margin), min(height, bot + margin)))
    return image


# ---------------------------------------------------------------------------
# PyTorch Dataset — image plus a set of tokenized choices and the answer's index
# ---------------------------------------------------------------------------

class AsciiTuneDataset(Dataset):
    """With `neg_sampler`, fresh negatives are drawn on every access (training); otherwise the item's
    fixed choices are used (evaluation)."""

    def __init__(self, items, processor, max_patches=1024, max_length=24, font_size=18,
                 neg_sampler=None, num_choices=4):
        self.items = items
        self.processor = processor
        self.max_patches = max_patches
        self.max_length = max_length
        self.font_size = font_size
        self.neg_sampler = neg_sampler
        self.num_choices = num_choices

    def __len__(self):
        return len(self.items)

    def tokenize(self, texts):
        ids = self.processor.tokenizer(
            texts, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids
        ids[ids == self.processor.tokenizer.pad_token_id] = -100
        return ids

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.neg_sampler is not None:
            # DataLoader seeds `random` per worker from the loader's generator, so this is reproducible.
            choices = self.neg_sampler.sample(random, item["answer"], self.num_choices - 1) + [item["answer"]]
            random.shuffle(choices)
            answer_idx = choices.index(item["answer"])
        else:
            choices, answer_idx = item["choices"], item["answer_idx"]

        image = render_ascii(item["ascii_art"], self.font_size)
        encoding = self.processor(images=image, max_patches=self.max_patches, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["choice_labels"] = self.tokenize(choices)
        encoding["answer_idx"] = torch.tensor(answer_idx)
        encoding["idx"] = torch.tensor(idx)
        return encoding


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Choice scoring — shared by the training loss and evaluation
# ---------------------------------------------------------------------------

def score_choices(model, hidden, attention_mask, choice_labels):
    """Score C candidate labels per image against a shared encoder output.

    Returns (mean_logp, sum_logp, n_tokens), each (B, C). mean_logp is the per-token log-likelihood
    used as the choice score; sum_logp / n_tokens give the token-level generation loss.
    """
    from transformers.modeling_outputs import BaseModelOutput

    B, C, L = choice_labels.shape
    flat = choice_labels.view(B * C, L)
    out = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=hidden.repeat_interleave(C, 0)),
        attention_mask=attention_mask.repeat_interleave(C, 0), labels=flat, use_cache=False,
    )
    nll = F.cross_entropy(out.logits.float().reshape(-1, out.logits.size(-1)), flat.reshape(-1),
                          ignore_index=-100, reduction="none").view(B * C, L)
    n_tokens = (flat != -100).sum(-1).clamp(min=1)
    sum_logp = -nll.sum(-1)
    return (sum_logp / n_tokens).view(B, C), sum_logp.view(B, C), n_tokens.view(B, C)


def mc_losses(model, batch, amp_dtype, device, temperature):
    fp, am = batch["flattened_patches"], batch["attention_mask"]
    answer_idx = batch["answer_idx"]
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        hidden = model.encoder(flattened_patches=fp, attention_mask=am).last_hidden_state
        mean_logp, sum_logp, n_tokens = score_choices(model, hidden, am, batch["choice_labels"])
    mc_loss = F.cross_entropy(mean_logp / temperature, answer_idx)
    pick = answer_idx.unsqueeze(1)
    gen_loss = -sum_logp.gather(1, pick).sum() / n_tokens.gather(1, pick).sum()
    mc_correct = (mean_logp.argmax(-1) == answer_idx).float().mean()
    return mc_loss, gen_loss, mc_correct


# ---------------------------------------------------------------------------
# Evaluation — accuracy against image-free reference points
# ---------------------------------------------------------------------------

@torch.no_grad()
def blank_choice_scores(model, dataset, device, amp_dtype, chunk=64):
    """Score every distinct choice string given a blank image: the decoder's image-free preference."""
    blank = Image.new("RGB", (64, 64), "white")
    enc = dataset.processor(images=blank, max_patches=dataset.max_patches, return_tensors="pt")
    fp, am = enc["flattened_patches"].to(device), enc["attention_mask"].to(device)
    strings = sorted({c for item in dataset.items for c in item["choices"]})
    scores = {}
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        hidden = model.encoder(flattened_patches=fp, attention_mask=am).last_hidden_state
        for i in range(0, len(strings), chunk):
            part = strings[i:i + chunk]
            mean_logp, _, _ = score_choices(model, hidden, am, dataset.tokenize(part).to(device).unsqueeze(0))
            scores.update(zip(part, mean_logp[0].float().tolist()))
    return scores


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


@torch.no_grad()
def evaluate(model, loader, processor, device, amp_dtype, max_new_tokens, label_counts,
             generate=True, desc="Eval"):
    model.eval()
    dataset = loader.dataset
    blank = blank_choice_scores(model, dataset, device, amp_dtype)
    nll_sum, nll_tokens, mc_loss_sum = 0.0, 0, 0.0
    records = []

    for batch in tqdm(loader, desc=desc, leave=False):
        fp = batch["flattened_patches"].to(device)
        am = batch["attention_mask"].to(device)
        choice_labels = batch["choice_labels"].to(device)
        answer_idx = batch["answer_idx"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            hidden = model.encoder(flattened_patches=fp, attention_mask=am).last_hidden_state
            mean_logp, sum_logp, n_tokens = score_choices(model, hidden, am, choice_labels)
        pick = answer_idx.unsqueeze(1)
        nll_sum += -sum_logp.gather(1, pick).sum().item()
        nll_tokens += n_tokens.gather(1, pick).sum().item()
        mc_loss_sum += F.cross_entropy(mean_logp, answer_idx, reduction="sum").item()

        if generate:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                gen = model.generate(flattened_patches=fp, attention_mask=am, max_new_tokens=max_new_tokens)
            gen_text = [t.strip().lower() for t in processor.tokenizer.batch_decode(gen, skip_special_tokens=True)]
        else:
            gen_text = [None] * fp.size(0)

        for b in range(fp.size(0)):
            item = dataset.items[batch["idx"][b].item()]
            scores = mean_logp[b].float().tolist()
            blank_scores = [blank[c] for c in item["choices"]]
            prior = [label_counts.get(c, 0) for c in item["choices"]]
            records.append({
                "idx": batch["idx"][b].item(),
                "answer": item["answer"],
                "choices": item["choices"],
                "answer_idx": item["answer_idx"],
                "mc_pred_idx": argmax(scores),
                "calib_pred_idx": argmax([s - z for s, z in zip(scores, blank_scores)]),
                "blank_pred_idx": argmax(blank_scores),
                "prior_pred_idx": argmax(prior),
                "answer_seen": item["answer"] in label_counts,
                "mc_scores": [round(s, 4) for s in scores],
                "blank_scores": [round(s, 4) for s in blank_scores],
                "gen": gen_text[b],
            })

    def acc(key, rows=None):
        rows = records if rows is None else rows
        # Compare by string: a cleaned distractor can coincide with the answer.
        return sum(r["choices"][r[key]] == r["answer"] for r in rows) / max(1, len(rows))

    n = max(1, len(records))
    seen = [r for r in records if r["answer_seen"]]
    unseen = [r for r in records if not r["answer_seen"]]
    metrics = {
        "loss": nll_sum / max(1, nll_tokens),
        "mc_loss": mc_loss_sum / n,
        "mc_acc": acc("mc_pred_idx"),
        "mc_acc_calib": acc("calib_pred_idx"),
        "blank_mc_acc": acc("blank_pred_idx"),
        "prior_mc_acc": acc("prior_pred_idx"),
        "mc_acc_seen": acc("mc_pred_idx", seen),
        "mc_acc_unseen": acc("mc_pred_idx", unseen) if unseen else None,
        "n_unseen": len(unseen),
        "n": len(records),
    }
    if generate:
        metrics["gen_acc"] = sum(r["gen"] == r["answer"] for r in records) / n
        gen_counts = Counter(r["gen"] for r in records)
        metrics["gen_unique"] = len(gen_counts)
        metrics["gen_top_share"] = gen_counts.most_common(1)[0][1] / n if gen_counts else 0.0
    model.train()
    return metrics, records


def format_metrics(m):
    return (f"mc_acc={m['mc_acc']:.4f} calib={m['mc_acc_calib']:.4f} | blank={m['blank_mc_acc']:.4f} "
            f"prior={m['prior_mc_acc']:.4f} | loss={m['loss']:.4f} gen_acc={m.get('gen_acc', 0):.4f} "
            f"unique_gen={m.get('gen_unique')}")


# ---------------------------------------------------------------------------
# Checkpointing and logging
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, step_in_epoch, global_step, best_score):
    tmp = str(path) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_score": best_score,
    }, tmp)
    os.replace(tmp, path)


class MetricsLogger:
    def __init__(self, path, use_wandb):
        self.f = open(path, "a", encoding="utf-8")
        self.use_wandb = use_wandb

    def log(self, record):
        record = {"time": time.time(), **record}
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        if self.use_wandb:
            import wandb

            prefix = record["type"] if record["type"] != "eval" else record.get("split", "eval")
            wandb.log({f"{prefix}/{k}": v for k, v in record.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)
                       and k not in ("time", "global_step")},
                      step=record["global_step"])

    def close(self):
        self.f.close()


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pix2Struct fine-tuning on ASCIITune")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "asciitune"))
    parser.add_argument("--model-name", default="google/pix2struct-base")
    parser.add_argument("--train-source", choices=list(TRAIN_FILES), default="full",
                        help="full = ASCIITune train (11.8K); curated = GPT-5-verified train_rational split (6.3K)")
    parser.add_argument("--hf-cache-dir", default=None, help="Cache dir for dataset downloads")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=4, help="Answer + negatives per training image")
    parser.add_argument("--mc-weight", type=float, default=1.0, help="Weight of the multiple-choice loss")
    parser.add_argument("--gen-weight", type=float, default=0.25, help="Weight of the answer generation loss")
    parser.add_argument("--mc-temperature", type=float, default=1.0,
                        help="Divides choice scores (mean token log-prob) inside the MC softmax")
    parser.add_argument("--balance-power", type=float, default=0.5,
                        help="Sample training images with weight count(label)^-p (0 = natural frequencies). "
                             "Negatives are drawn from the matching distribution, count^(1-p).")
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=24, help="Max target token length")
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="bf16 autocast (falls back to fp32 if the GPU lacks bf16)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--select-metric", default="mc_acc",
                        choices=["mc_acc", "mc_acc_calib"], help="Validation metric used to pick best/")
    parser.add_argument("--log-every", type=int, default=10, help="Log train loss every N optimizer steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save last.pt every N optimizer steps")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after N evals without improvement in --select-metric (0 = off)")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-train-samples", type=int, default=None, help="Subsample for smoke tests")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Skip final ASCIIEval test evaluation")
    parser.add_argument("--eval-only", default=None,
                        help="Path to a saved model dir: evaluate it on val and test, no training")
    parser.add_argument("--resume", default=None, help="Path to last.pt to resume from")
    parser.add_argument("--wandb-project", default="Finetuning_Pix2Struct_ASCIITune")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if args.precision == "bf16" and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            print("Warning: GPU has no bf16 support; using fp32.")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), "
          f"precision: {'bf16' if amp_dtype else 'fp32'}")

    # Data. The validation split always comes from the full training file, so runs with different
    # --train-source values are scored on the same items with the same choices.
    print(f"Downloading {TRAIN_REPO}/{TRAIN_FILES['full']} ...")
    full = load_asciieval_jsonl(download_jsonl(TRAIN_REPO, TRAIN_FILES["full"], args.hf_cache_dir))
    random.Random(args.seed).shuffle(full)
    n_val = max(1, int(args.val_fraction * len(full)))
    val_items, train_items = full[:n_val], full[n_val:]
    full_counts = Counter(d["answer"] for d in train_items)

    if args.train_source == "curated":
        print(f"Downloading {TRAIN_REPO}/{TRAIN_FILES['curated']} ...")
        val_arts = {d["ascii_art"] for d in val_items}
        curated = load_asciieval_jsonl(download_jsonl(TRAIN_REPO, TRAIN_FILES["curated"], args.hf_cache_dir))
        train_items = [d for d in curated if d["ascii_art"] not in val_arts]
        random.Random(args.seed).shuffle(train_items)
        print(f"  Curated split: {len(curated)} items, {len(curated) - len(train_items)} overlap validation and were dropped")

    # Validation distractors: real training labels at their natural frequency, like ASCIIEval's.
    val_items = with_resampled_choices(val_items, LabelSampler(full_counts, 1.0), args.seed + 1)
    if args.max_train_samples:
        train_items = train_items[:args.max_train_samples]
    if args.max_val_samples:
        val_items = val_items[:args.max_val_samples]
    label_counts = Counter(d["answer"] for d in train_items)
    print(f"Train ({args.train_source}): {len(train_items)} items, {len(label_counts)} labels | val: {len(val_items)}")

    from transformers import AutoProcessor, Pix2StructForConditionalGeneration

    get_font(args.font_size)  # Download once here; DataLoader workers would otherwise race for it.
    model_src = args.eval_only or args.model_name
    processor = AutoProcessor.from_pretrained(model_src)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_src).to(device)

    ds_kwargs = dict(max_patches=args.max_patches, max_length=args.max_length, font_size=args.font_size)
    loader_kwargs = dict(collate_fn=collate_fn, num_workers=args.num_workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(AsciiTuneDataset(val_items, processor, **ds_kwargs),
                            batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    def load_test_loader():
        test_items = load_asciieval_jsonl(download_jsonl(TEST_REPO, TEST_FILE, args.hf_cache_dir))
        if args.max_test_samples:
            test_items = test_items[:args.max_test_samples]
        return DataLoader(AsciiTuneDataset(test_items, processor, **ds_kwargs),
                          batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    if args.eval_only:
        results = {}
        for split, loader in (("val", val_loader), ("test", load_test_loader())):
            metrics, records = evaluate(model, loader, processor, device, amp_dtype, args.max_length,
                                        label_counts, desc=split)
            write_jsonl(out_dir / "preds" / f"{split}_eval_only.jsonl", records)
            results[split] = metrics
            print(f"  [{split}] {format_metrics(metrics)}")
        with open(out_dir / "eval_only_results.json", "w") as f:
            json.dump({"model": args.eval_only, **results}, f, indent=2)
        return

    # Balanced image sampling; negatives drawn from the same label distribution as the positives.
    neg_sampler = LabelSampler(label_counts, 1.0 - args.balance_power)
    train_ds = AsciiTuneDataset(train_items, processor, neg_sampler=neg_sampler,
                                num_choices=args.num_choices, **ds_kwargs)
    sample_weights = [label_counts[d["answer"]] ** -args.balance_power for d in train_items]

    batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False,
                          warmup_init=False, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)
    eval_steps = sorted({max(1, round(steps_per_epoch * (k + 1) / args.evals_per_epoch))
                         for k in range(args.evals_per_epoch)})
    print(f"Optimizer steps: {steps_per_epoch}/epoch, {total_steps} total "
          f"(effective batch {args.batch_size * args.grad_accum}, {args.num_choices} choices/image); "
          f"eval at steps {eval_steps} of each epoch")

    start_epoch, start_step, global_step, best_score = 0, 0, 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, start_step = ckpt["epoch"], ckpt["step_in_epoch"]
        global_step, best_score = ckpt["global_step"], ckpt["best_score"]
        print(f"Resumed from {args.resume}: epoch {start_epoch + 1}, step {start_step}, "
              f"global step {global_step}, best {args.select_metric} {best_score:.4f}")
    else:
        with open(out_dir / "config.json", "w") as f:
            json.dump({**vars(args), "n_train": len(train_items), "n_val": len(val_items),
                       "n_labels": len(label_counts), "steps_per_epoch": steps_per_epoch,
                       "total_steps": total_steps}, f, indent=2)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=out_dir.name, id=out_dir.name.replace(".", "_"),
                       resume="allow", config=vars(args))
        except Exception as e:
            print(f"Warning: wandb unavailable ({e}). Continuing without it.")
            use_wandb = False
    logger = MetricsLogger(out_dir / "metrics.jsonl", use_wandb)

    def run_eval(epoch, step_in_epoch):
        nonlocal best_score
        metrics, records = evaluate(model, val_loader, processor, device, amp_dtype, args.max_length,
                                    label_counts, desc=f"Val ep{epoch + 1} step{step_in_epoch}")
        epoch_float = epoch + step_in_epoch / steps_per_epoch
        write_jsonl(out_dir / "preds" / f"val_step{global_step:06d}.jsonl", records)
        improved = metrics[args.select_metric] > best_score
        logger.log({"type": "eval", "split": "val", "global_step": global_step,
                    "epoch": round(epoch_float, 3), "improved": improved, **metrics})
        print(f"  [val @ epoch {epoch_float:.2f}] {format_metrics(metrics)}{'  *best*' if improved else ''}")
        if improved:
            best_score = metrics[args.select_metric]
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            with open(out_dir / "best" / "best_info.json", "w") as f:
                json.dump({"global_step": global_step, "epoch": epoch_float, **metrics}, f, indent=2)
        return improved

    # Training loop
    model.train()
    evals_without_improvement = 0
    stop = False
    epoch, step_in_epoch = start_epoch, start_step
    try:
        for epoch in range(start_epoch, args.epochs):
            # Seeded per-epoch sampling so a mid-epoch resume skips exactly the batches already seen.
            gen = torch.Generator().manual_seed(args.seed + epoch)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds),
                                            replacement=True, generator=gen)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                      generator=gen, **loader_kwargs)
            skip_batches = start_step * args.grad_accum if epoch == start_epoch else 0
            step_in_epoch = start_step if epoch == start_epoch else 0
            window = Counter()
            accum = 0
            t0 = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, batch in enumerate(pbar):
                if batch_idx < skip_batches:
                    continue
                batch = {k: batch[k].to(device, non_blocking=True)
                         for k in ("flattened_patches", "attention_mask", "choice_labels", "answer_idx")}
                mc_loss, gen_loss, mc_correct = mc_losses(model, batch, amp_dtype, device, args.mc_temperature)
                loss = args.mc_weight * mc_loss + args.gen_weight * gen_loss
                if not torch.isfinite(loss):
                    logger.log({"type": "nan", "global_step": global_step, "epoch": epoch})
                    print(f"\n[!] Non-finite loss at global step {global_step}; skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                (loss / args.grad_accum).backward()
                window.update(loss=loss.item(), mc_loss=mc_loss.item(), gen_loss=gen_loss.item(),
                              train_mc_acc=mc_correct.item(), n=1)
                accum += 1

                last_batch = batch_idx == len(train_loader) - 1
                if accum < args.grad_accum and not last_batch:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1
                step_in_epoch += 1

                if global_step % args.log_every == 0 and window["n"]:
                    avg = {k: window[k] / window["n"] for k in ("loss", "mc_loss", "gen_loss", "train_mc_acc")}
                    logger.log({"type": "train", "global_step": global_step,
                                "epoch": round(epoch + step_in_epoch / steps_per_epoch, 3), **avg,
                                "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
                                "sec_per_step": (time.time() - t0) / args.log_every})
                    pbar.set_postfix(loss=f"{avg['loss']:.3f}", mc_acc=f"{avg['train_mc_acc']:.2f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.1e}")
                    window, t0 = Counter(), time.time()

                if step_in_epoch in eval_steps:
                    if run_eval(epoch, step_in_epoch):
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1
                    save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                                    epoch, step_in_epoch, global_step, best_score)
                    if args.early_stop_patience and evals_without_improvement >= args.early_stop_patience:
                        print(f"Early stopping: no {args.select_metric} improvement in {evals_without_improvement} evals.")
                        stop = True
                        break
                elif args.save_every and global_step % args.save_every == 0:
                    save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                                    epoch, step_in_epoch, global_step, best_score)

            if stop:
                break
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                            epoch + 1, 0, global_step, best_score)
    except KeyboardInterrupt:
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                        epoch, step_in_epoch, global_step, best_score)
        print(f"\nInterrupted. Resume with: --resume {out_dir / 'last.pt'}")
        logger.close()
        return

    # Final evaluation of the best model on the ASCIIEval test set
    if not args.skip_test and (out_dir / "best").exists():
        print(f"\nEvaluating best checkpoint on {TEST_REPO}/{TEST_FILE} ...")
        model = Pix2StructForConditionalGeneration.from_pretrained(out_dir / "best").to(device)
        metrics, records = evaluate(model, load_test_loader(), processor, device, amp_dtype,
                                    args.max_length, label_counts, desc="Test")
        write_jsonl(out_dir / "preds" / "test_best.jsonl", records)
        with open(out_dir / "test_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.log({"type": "eval", "split": "test", "global_step": global_step, **metrics})
        print(f"  ASCIIEval test: {format_metrics(metrics)}")

    logger.close()
    if use_wandb:
        import wandb

        wandb.finish()
    print(f"Done. Best val {args.select_metric}: {best_score:.4f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
