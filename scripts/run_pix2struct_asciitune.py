#!/usr/bin/env python3
"""ASCIIBench Pix2Struct Fine-tuning on ASCIITune

Fine-tunes Pix2Struct on the ASCIITune training set (https://huggingface.co/datasets/ASCIIEval/ASCIITune)
instead of the local final_dataset.jsonl. Each ASCII art piece is rendered to an image and the model
learns to generate the name of the depicted concept.

ASCIITune is a 4-way multiple-choice recognition set (11,836 items, ~2.3K concepts), so evaluation
reports two numbers:
  - mc_acc:    multiple-choice accuracy — the model scores all 4 choices by mean token log-likelihood
               and picks the best one. This is the ASCIIEval protocol; chance is 25%.
  - gen_acc:   free-generation exact match against the correct answer (much harder, no choices given).

After training, the best checkpoint (by validation mc_acc) is evaluated on the ASCIIEval test set
(https://huggingface.co/datasets/ASCIIEval/ASCIIEval).

Everything needed by scripts/analyze_pix2struct_convergence.py is written to --output-dir:
  metrics.jsonl         one JSON record per train log step / eval
  preds/val_*.jsonl     per-item validation predictions at every eval
  config.json           run arguments
  best/                 HF save_pretrained() of the best model (+ processor)
  last.pt               full resumable checkpoint (model, optimizer, scheduler, step)
  test_results.json     ASCIIEval test-set results for the best model

Usage:
    python scripts/run_pix2struct_asciitune.py --output-dir runs/asciitune_lr1e-4
    python scripts/run_pix2struct_asciitune.py --output-dir runs/smoke --max-train-samples 64 --max-val-samples 32 --epochs 1
    python scripts/run_pix2struct_asciitune.py --output-dir runs/asciitune_lr1e-4 --resume runs/asciitune_lr1e-4/last.pt
"""

import argparse
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

import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

from run_pix2struct_finetuning import get_font

TRAIN_REPO = "ASCIIEval/ASCIITune"
TRAIN_FILE = "train.jsonl"
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


def load_asciieval_jsonl(path):
    items = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            choices = [clean_label(c) for c in row["choices"]]
            labels = row["labels"]
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
# PyTorch Dataset — image, answer target, and the 4 choices for MC scoring
# ---------------------------------------------------------------------------

class AsciiTuneDataset(Dataset):
    def __init__(self, items, processor, max_patches=1024, max_length=24, font_size=18):
        self.items = items
        self.processor = processor
        self.max_patches = max_patches
        self.max_length = max_length
        self.font_size = font_size

    def __len__(self):
        return len(self.items)

    def _tokenize(self, texts):
        ids = self.processor.tokenizer(
            texts, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids
        ids[ids == self.processor.tokenizer.pad_token_id] = -100
        return ids

    def __getitem__(self, idx):
        item = self.items[idx]
        image = render_ascii(item["ascii_art"], self.font_size)
        encoding = self.processor(images=image, max_patches=self.max_patches, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = self._tokenize([item["answer"]]).squeeze(0)
        encoding["choice_labels"] = self._tokenize(item["choices"])
        encoding["answer_idx"] = torch.tensor(item["answer_idx"])
        encoding["idx"] = torch.tensor(idx)
        return encoding


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Evaluation — loss, multiple-choice accuracy, free-generation accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, processor, device, amp_dtype, max_new_tokens, generate=True, desc="Eval"):
    from transformers.modeling_outputs import BaseModelOutput

    model.eval()
    items = loader.dataset.items
    loss_sum, loss_batches = 0.0, 0
    records = []

    for batch in tqdm(loader, desc=desc, leave=False):
        fp = batch["flattened_patches"].to(device)
        am = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        choice_labels = batch["choice_labels"].to(device)
        B, C, L = choice_labels.shape

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            hidden = model.encoder(flattened_patches=fp, attention_mask=am).last_hidden_state
            out = model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                        attention_mask=am, labels=labels, use_cache=False)
            loss_sum += out.loss.float().item()
            loss_batches += 1

            # Score every choice with the same encoder output.
            cl = choice_labels.view(B * C, L)
            mc_out = model(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden.repeat_interleave(C, 0)),
                attention_mask=am.repeat_interleave(C, 0), labels=cl, use_cache=False,
            )
        logp = F.log_softmax(mc_out.logits.float(), dim=-1)
        mask = cl != -100
        tok_lp = logp.gather(-1, cl.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        scores = (tok_lp.sum(-1) / mask.sum(-1).clamp(min=1)).view(B, C)
        mc_pred = scores.argmax(-1).cpu()

        if generate:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                gen = model.generate(flattened_patches=fp, attention_mask=am, max_new_tokens=max_new_tokens)
            gen_text = [t.strip().lower() for t in processor.tokenizer.batch_decode(gen, skip_special_tokens=True)]
        else:
            gen_text = [None] * B

        for b in range(B):
            item = items[batch["idx"][b].item()]
            records.append({
                "idx": batch["idx"][b].item(),
                "answer": item["answer"],
                "choices": item["choices"],
                "answer_idx": item["answer_idx"],
                "mc_pred_idx": mc_pred[b].item(),
                "mc_scores": [round(s, 4) for s in scores[b].tolist()],
                "gen": gen_text[b],
            })

    n = max(1, len(records))
    metrics = {
        "loss": loss_sum / max(1, loss_batches),
        "mc_acc": sum(r["mc_pred_idx"] == r["answer_idx"] for r in records) / n,
        "n": len(records),
    }
    if generate:
        metrics["gen_acc"] = sum(r["gen"] == r["answer"] for r in records) / n
        gen_counts = Counter(r["gen"] for r in records)
        metrics["gen_unique"] = len(gen_counts)
        metrics["gen_top_share"] = gen_counts.most_common(1)[0][1] / n if gen_counts else 0.0
    model.train()
    return metrics, records


# ---------------------------------------------------------------------------
# Checkpointing and logging
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, step_in_epoch, global_step, best_mc_acc):
    tmp = str(path) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_mc_acc": best_mc_acc,
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

            prefix = record["type"]
            wandb.log({f"{prefix}/{k}": v for k, v in record.items()
                       if isinstance(v, (int, float)) and k not in ("time", "global_step")},
                      step=record["global_step"])

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pix2Struct fine-tuning on ASCIITune")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "asciitune"))
    parser.add_argument("--model-name", default="google/pix2struct-base")
    parser.add_argument("--hf-cache-dir", default=None, help="Cache dir for dataset downloads")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--log-every", type=int, default=10, help="Log train loss every N optimizer steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save last.pt every N optimizer steps")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after N evals without val mc_acc improvement (0 = off)")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-train-samples", type=int, default=None, help="Subsample for smoke tests")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Skip final ASCIIEval test evaluation")
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

    # Data
    print(f"Downloading {TRAIN_REPO}/{TRAIN_FILE} ...")
    data = load_asciieval_jsonl(download_jsonl(TRAIN_REPO, TRAIN_FILE, args.hf_cache_dir))
    rng = random.Random(args.seed)
    rng.shuffle(data)
    n_val = max(1, int(args.val_fraction * len(data)))
    val_items, train_items = data[:n_val], data[n_val:]
    if args.max_train_samples:
        train_items = train_items[:args.max_train_samples]
    if args.max_val_samples:
        val_items = val_items[:args.max_val_samples]
    print(f"Loaded {len(data)} ASCIITune items -> train {len(train_items)}, val {len(val_items)}")
    print(f"Unique answers in train: {len({d['answer'] for d in train_items})}")

    from transformers import AutoProcessor, Pix2StructForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Pix2StructForConditionalGeneration.from_pretrained(args.model_name).to(device)

    ds_kwargs = dict(max_patches=args.max_patches, max_length=args.max_length, font_size=args.font_size)
    train_ds = AsciiTuneDataset(train_items, processor, **ds_kwargs)
    val_ds = AsciiTuneDataset(val_items, processor, **ds_kwargs)
    loader_kwargs = dict(collate_fn=collate_fn, num_workers=args.num_workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

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
          f"(effective batch {args.batch_size * args.grad_accum}); eval at steps {eval_steps} of each epoch")

    start_epoch, start_step, global_step, best_mc_acc = 0, 0, 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, start_step = ckpt["epoch"], ckpt["step_in_epoch"]
        global_step, best_mc_acc = ckpt["global_step"], ckpt["best_mc_acc"]
        print(f"Resumed from {args.resume}: epoch {start_epoch + 1}, step {start_step}, "
              f"global step {global_step}, best mc_acc {best_mc_acc:.4f}")
    else:
        with open(out_dir / "config.json", "w") as f:
            json.dump({**vars(args), "n_train": len(train_items), "n_val": len(val_items),
                       "steps_per_epoch": steps_per_epoch, "total_steps": total_steps}, f, indent=2)

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
        nonlocal best_mc_acc
        metrics, records = evaluate(model, val_loader, processor, device, amp_dtype,
                                    args.max_length, desc=f"Val ep{epoch + 1} step{step_in_epoch}")
        epoch_float = epoch + step_in_epoch / steps_per_epoch
        tag = f"step{global_step:06d}"
        with open(out_dir / "preds" / f"val_{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        improved = metrics["mc_acc"] > best_mc_acc
        logger.log({"type": "eval", "split": "val", "global_step": global_step,
                    "epoch": round(epoch_float, 3), "improved": improved, **metrics})
        print(f"  [val @ epoch {epoch_float:.2f}] loss={metrics['loss']:.4f} mc_acc={metrics['mc_acc']:.4f} "
              f"gen_acc={metrics.get('gen_acc', 0):.4f} unique_gen={metrics.get('gen_unique')}"
              f"{'  *best*' if improved else ''}")
        if improved:
            best_mc_acc = metrics["mc_acc"]
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
            # Seeded per-epoch shuffle so a mid-epoch resume skips exactly the batches already seen.
            gen = torch.Generator().manual_seed(args.seed + epoch)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      generator=gen, **loader_kwargs)
            skip_batches = start_step * args.grad_accum if epoch == start_epoch else 0
            step_in_epoch = start_step if epoch == start_epoch else 0
            window_loss, window_n, accum = 0.0, 0, 0
            t0 = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, batch in enumerate(pbar):
                if batch_idx < skip_batches:
                    continue
                batch = {k: batch[k].to(device, non_blocking=True)
                         for k in ("flattened_patches", "attention_mask", "labels")}
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    logger.log({"type": "nan", "global_step": global_step, "epoch": epoch})
                    print(f"\n[!] Non-finite loss at global step {global_step}; skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                (loss / args.grad_accum).backward()
                window_loss += loss.item()
                window_n += 1
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

                if global_step % args.log_every == 0:
                    avg = window_loss / max(1, window_n)
                    logger.log({"type": "train", "global_step": global_step,
                                "epoch": round(epoch + step_in_epoch / steps_per_epoch, 3),
                                "loss": avg, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
                                "sec_per_step": (time.time() - t0) / args.log_every})
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    window_loss, window_n, t0 = 0.0, 0, time.time()

                if step_in_epoch in eval_steps:
                    if run_eval(epoch, step_in_epoch):
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1
                    save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                                    epoch, step_in_epoch, global_step, best_mc_acc)
                    if args.early_stop_patience and evals_without_improvement >= args.early_stop_patience:
                        print(f"Early stopping: no val mc_acc improvement in {evals_without_improvement} evals.")
                        stop = True
                        break
                elif args.save_every and global_step % args.save_every == 0:
                    save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                                    epoch, step_in_epoch, global_step, best_mc_acc)

            if stop:
                break
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                            epoch + 1, 0, global_step, best_mc_acc)
    except KeyboardInterrupt:
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                        epoch, step_in_epoch, global_step, best_mc_acc)
        print(f"\nInterrupted. Resume with: --resume {out_dir / 'last.pt'}")
        logger.close()
        return

    # Final evaluation of the best model on the ASCIIEval test set
    if not args.skip_test and (out_dir / "best").exists():
        print(f"\nEvaluating best checkpoint on {TEST_REPO}/{TEST_FILE} ...")
        model = Pix2StructForConditionalGeneration.from_pretrained(out_dir / "best").to(device)
        test_items = load_asciieval_jsonl(download_jsonl(TEST_REPO, TEST_FILE, args.hf_cache_dir))
        if args.max_test_samples:
            test_items = test_items[:args.max_test_samples]
        test_loader = DataLoader(AsciiTuneDataset(test_items, processor, **ds_kwargs),
                                 batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        metrics, records = evaluate(model, test_loader, processor, device, amp_dtype,
                                    args.max_length, desc="Test")
        with open(out_dir / "preds" / "test_best.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(out_dir / "test_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.log({"type": "eval", "split": "test", "global_step": global_step, **metrics})
        print(f"  ASCIIEval test: mc_acc={metrics['mc_acc']:.4f} (chance 0.25), gen_acc={metrics['gen_acc']:.4f}")

    logger.close()
    if use_wandb:
        import wandb

        wandb.finish()
    print(f"Done. Best val mc_acc: {best_mc_acc:.4f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
