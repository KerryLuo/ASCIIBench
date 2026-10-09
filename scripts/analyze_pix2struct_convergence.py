#!/usr/bin/env python3
"""Convergence analysis for a Pix2Struct ASCIITune run

Reads the outputs of scripts/run_pix2struct_asciitune.py and answers "did training converge well?":
  - Train loss: total reduction, slope over the final 25% of steps (still falling vs. plateaued), spikes/NaNs
  - Val loss and accuracy: best vs. final eval, rise after the minimum (overfitting), train/val gap
  - Accuracy vs. chance (25%), with a standard error so small wiggles aren't mistaken for progress
  - Prediction health: multiple-choice position bias and free-generation collapse (one answer for everything)

Writes <run_dir>/analysis/report.md and <run_dir>/analysis/convergence.png, and prints the report.
Pass several run dirs to also get a side-by-side comparison table (e.g. an LR sweep).

Usage:
    python scripts/analyze_pix2struct_convergence.py runs/asciitune_lr1e-4
    python scripts/analyze_pix2struct_convergence.py runs/asciitune_lr*
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CHANCE = 0.25

# Reference categorical palette (slots 1-3) plus recessive ink for axes/grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        sys.exit(f"No metrics.jsonl in {run_dir}")

    # A resumed run re-logs some steps; keep the latest record per (type, split, step).
    records = {}
    nan_steps = []
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["type"] == "nan":
                nan_steps.append(r["global_step"])
                continue
            records[(r["type"], r.get("split"), r["global_step"])] = r

    rows = sorted(records.values(), key=lambda r: r["global_step"])
    train = [r for r in rows if r["type"] == "train"]
    val = [r for r in rows if r["type"] == "eval" and r.get("split") == "val"]
    test = next((r for r in rows if r["type"] == "eval" and r.get("split") == "test"), None)

    config = {}
    if (run_dir / "config.json").exists():
        config = json.loads((run_dir / "config.json").read_text())

    preds = sorted((run_dir / "preds").glob("val_step*.jsonl")) if (run_dir / "preds").exists() else []
    return {"dir": run_dir, "name": run_dir.name, "train": train, "val": val, "test": test,
            "config": config, "nan_steps": nan_steps, "pred_files": preds}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ema(values, alpha=0.1):
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return np.array(out)


def acc_stderr(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def analyze_train(run):
    t = run["train"]
    if len(t) < 5:
        return {"ok": False, "reason": f"only {len(t)} train log points"}
    steps = np.array([r["global_step"] for r in t], dtype=float)
    loss = np.array([r["loss"] for r in t], dtype=float)
    n = len(loss)
    head = loss[: max(1, n // 20)].mean()
    tail_n = max(3, n // 10)
    final = loss[-tail_n:].mean()

    # Slope over the last 25% of logged steps, expressed as relative change per 10% of the run.
    k = max(4, n // 4)
    xs, ys = steps[-k:], loss[-k:]
    slope = np.polyfit(xs, ys, 1)[0]
    span = steps[-1] - steps[0] if steps[-1] > steps[0] else 1.0
    tail_rel_change = slope * 0.1 * span / max(abs(ys.mean()), 1e-9)

    # Spikes: points far above a rolling median, measured in robust (MAD) units.
    spikes = 0
    w = 15
    for i in range(w, n):
        win = loss[i - w:i]
        med = np.median(win)
        mad = np.median(np.abs(win - med)) + 1e-9
        if loss[i] > med + 6 * 1.4826 * mad and loss[i] > 1.5 * med:
            spikes += 1

    gn = np.array([r.get("grad_norm", np.nan) for r in t], dtype=float)
    max_gn = run["config"].get("max_grad_norm", 1.0)
    clipped_frac_tail = float(np.mean(gn[-k:] > max_gn)) if np.isfinite(gn).any() else float("nan")

    return {
        "ok": True, "initial": float(head), "final": float(final),
        "reduction": float(1 - final / head) if head > 0 else 0.0,
        "tail_rel_change_per_10pct": float(tail_rel_change),
        "tail_cv": float(ys.std() / max(ys.mean(), 1e-9)),
        "spikes": spikes, "nan_steps": len(run["nan_steps"]),
        "grad_norm_early": float(np.nanmedian(gn[: max(1, n // 10)])),
        "grad_norm_late": float(np.nanmedian(gn[-k:])),
        "clipped_frac_tail": clipped_frac_tail,
        "last_step": int(steps[-1]),
        "total_steps": run["config"].get("total_steps"),
    }


def analyze_val(run, train_stats):
    v = run["val"]
    if not v:
        return {"ok": False, "reason": "no validation evals"}
    loss = np.array([r["loss"] for r in v])
    mc = np.array([r["mc_acc"] for r in v])
    gen = np.array([r.get("gen_acc", np.nan) for r in v])
    n = v[-1]["n"]
    se = acc_stderr(mc[-1], n)

    i_min_loss = int(loss.argmin())
    i_best_mc = int(mc.argmax())
    rise_after_min = float(loss[-1] / loss[i_min_loss] - 1)

    # Improvement over the last third of evals, in standard errors.
    k = max(2, len(mc) // 3)
    recent_gain = float(mc[-1] - mc[-k - 1]) if len(mc) > k else float(mc[-1] - mc[0])

    # Train/val gap at the final eval (train loss averaged around that step).
    gap = None
    if train_stats.get("ok"):
        last_step = v[-1]["global_step"]
        near = [r["loss"] for r in run["train"] if abs(r["global_step"] - last_step) <= 50]
        if near:
            gap = float(loss[-1] - np.mean(near))

    return {
        "ok": True, "n_evals": len(v), "n_val": n,
        "loss_first": float(loss[0]), "loss_min": float(loss[i_min_loss]), "loss_final": float(loss[-1]),
        "loss_min_epoch": v[i_min_loss]["epoch"], "loss_rise_after_min": rise_after_min,
        "mc_first": float(mc[0]), "mc_best": float(mc[i_best_mc]), "mc_final": float(mc[-1]),
        "mc_best_epoch": v[i_best_mc]["epoch"], "mc_stderr": se,
        "mc_recent_gain": recent_gain, "mc_recent_gain_se": recent_gain / max(se, 1e-9),
        "gen_best": float(np.nanmax(gen)) if np.isfinite(gen).any() else float("nan"),
        "gen_final": float(gen[-1]),
        "train_val_gap": gap,
        # Image-free reference points at the best eval (absent in runs from before they were logged).
        "best_prior": v[i_best_mc].get("prior_mc_acc"),
        "best_blank": v[i_best_mc].get("blank_mc_acc"),
        "best_calib": v[i_best_mc].get("mc_acc_calib"),
    }


def analyze_preds(run):
    if not run["pred_files"]:
        return {"ok": False}
    recs = read_jsonl(run["pred_files"][-1])
    n = len(recs)
    pos = Counter(r["mc_pred_idx"] for r in recs)
    true_pos = Counter(r["answer_idx"] for r in recs)
    top_pos, top_pos_n = pos.most_common(1)[0]

    gens = [r["gen"] for r in recs if r.get("gen") is not None]
    vocab = {c for r in recs for c in r["choices"]}
    gen_counts = Counter(gens)
    probs = np.array(list(gen_counts.values()), dtype=float) / max(1, len(gens))
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    max_entropy = math.log2(max(2, len({r["answer"] for r in recs})))

    return {
        "ok": True, "file": run["pred_files"][-1].name, "n": n,
        "mc_pos_dist": [pos.get(i, 0) / n for i in range(4)],
        "true_pos_dist": [true_pos.get(i, 0) / n for i in range(4)],
        "mc_top_pos": top_pos, "mc_top_pos_share": top_pos_n / n,
        "gen_unique": len(gen_counts), "gen_top": gen_counts.most_common(10),
        "gen_top_share": gen_counts.most_common(1)[0][1] / max(1, len(gens)) if gens else 0.0,
        "gen_in_vocab": sum(g in vocab for g in gens) / max(1, len(gens)),
        "gen_entropy_ratio": entropy / max_entropy,
    }


def shortcut_check(split, acc, prior, blank, n):
    """Is accuracy clearly above what's reachable without looking at the image?"""
    refs = {k: v for k, v in (("label-frequency baseline", prior), ("blank-image", blank)) if v is not None}
    if not refs:
        return None
    name, ref = max(refs.items(), key=lambda kv: kv[1])
    se = acc_stderr(acc, n)
    detail = ", ".join(f"{k} {v:.1%}" for k, v in refs.items())
    if acc - ref < 2 * se:
        return ("BAD", f"{split} MC accuracy {acc:.1%} is not clearly above the image-free references ({detail}; ±{se:.1%} SE) "
                       "— the model is answering from label priors, not from the art.")
    return ("OK", f"{split} MC accuracy {acc:.1%} beats the image-free references ({detail}) by {(acc - ref) / se:.1f} SE.")


def verdict(tr, va, pr, test=None):
    """Return (overall verdict, list of (level, message)). Levels: OK / WARN / BAD."""
    findings = []
    flags = set()

    if tr.get("ok"):
        if tr["nan_steps"]:
            findings.append(("BAD", f"{tr['nan_steps']} non-finite loss steps were skipped — training is numerically unstable."))
            flags.add("unstable")
        if tr["final"] >= tr["initial"]:
            findings.append(("BAD", f"Train loss did not decrease ({tr['initial']:.3f} -> {tr['final']:.3f})."))
            flags.add("diverged")
        else:
            findings.append(("OK", f"Train loss fell {tr['reduction']:.0%} ({tr['initial']:.3f} -> {tr['final']:.3f})."))
        rc = tr["tail_rel_change_per_10pct"]
        if rc < -0.03:
            findings.append(("WARN", f"Train loss is still falling at the end ({rc:+.1%} per 10% of the run) — not converged; more epochs or a higher LR would likely help."))
            flags.add("still_improving_train")
        elif rc > 0.03:
            findings.append(("BAD", f"Train loss is rising over the final 25% of steps ({rc:+.1%} per 10% of the run)."))
            flags.add("diverged")
        else:
            findings.append(("OK", f"Train loss has plateaued (final-quarter slope {rc:+.1%} per 10% of the run)."))
        if tr["spikes"] > 3:
            findings.append(("WARN", f"{tr['spikes']} loss spikes detected — consider a lower LR or more warmup."))
            flags.add("spiky")
        if tr["clipped_frac_tail"] == tr["clipped_frac_tail"] and tr["clipped_frac_tail"] > 0.5:
            findings.append(("WARN", f"Gradients were clipped on {tr['clipped_frac_tail']:.0%} of late steps — the LR may be too high."))
        if tr.get("total_steps") and tr["last_step"] < tr["total_steps"]:
            findings.append(("WARN", f"Run stopped at step {tr['last_step']} of {tr['total_steps']} (early stop or interrupted)."))

    if va.get("ok"):
        se = va["mc_stderr"]
        margin = va["mc_best"] - CHANCE
        if margin < 2 * se:
            findings.append(("BAD", f"Val MC accuracy {va['mc_best']:.1%} is within noise of chance (25%, ±{se:.1%} SE) — the model is not learning to recognize the art."))
            flags.add("chance")
        else:
            findings.append(("OK", f"Best val MC accuracy {va['mc_best']:.1%} at epoch {va['mc_best_epoch']:.2f} "
                                   f"({margin / se:.1f} SE above 25% chance)."))
        if va["loss_rise_after_min"] > 0.05 and va["mc_final"] < va["mc_best"] - se:
            findings.append(("WARN", f"Overfitting: val loss rose {va['loss_rise_after_min']:.0%} after its minimum at epoch "
                                     f"{va['loss_min_epoch']:.2f} and val accuracy dropped {va['mc_best'] - va['mc_final']:.1%} from best. "
                                     "Use the saved best/ checkpoint; fewer epochs or more regularization next time."))
            flags.add("overfit")
        elif va["loss_rise_after_min"] > 0.05:
            findings.append(("WARN", f"Val loss rose {va['loss_rise_after_min']:.0%} after its minimum (epoch {va['loss_min_epoch']:.2f}) "
                                     "but accuracy held — mild overconfidence, early overfitting signal."))
        if va["mc_recent_gain_se"] > 2:
            findings.append(("WARN", f"Val MC accuracy was still rising over the last third of evals "
                                     f"(+{va['mc_recent_gain']:.1%}, {va['mc_recent_gain_se']:.1f} SE) — more training likely helps."))
            flags.add("still_improving_val")
        elif abs(va["mc_recent_gain_se"]) <= 2:
            findings.append(("OK", f"Val MC accuracy has plateaued (last-third change {va['mc_recent_gain']:+.1%}, within 2 SE)."))
        if va["train_val_gap"] is not None and va["train_val_gap"] > 1.0:
            findings.append(("WARN", f"Large train/val loss gap at the end ({va['train_val_gap']:.2f} nats) — memorizing train labels."))
        sc = shortcut_check("Val", va["mc_best"], va["best_prior"], va["best_blank"], va["n_val"])
        if sc:
            findings.append(sc)
            if sc[0] == "BAD":
                flags.add("shortcut")
        if va["best_prior"] is None and va["best_blank"] is None:
            findings.append(("WARN", "No image-free baselines were logged (run predates them); "
                                     "accuracy can't be separated from label-prior guessing."))

    if test:
        sc = shortcut_check("Test", test["mc_acc"], test.get("prior_mc_acc"), test.get("blank_mc_acc"), test["n"])
        if sc:
            findings.append(sc)
            if sc[0] == "BAD":
                flags.add("shortcut")

    if pr.get("ok"):
        if pr["mc_top_pos_share"] > 0.5:
            findings.append(("BAD", f"MC position bias: picks choice #{pr['mc_top_pos'] + 1} for {pr['mc_top_pos_share']:.0%} of items "
                                    f"(true answers are spread {', '.join(f'{p:.0%}' for p in pr['true_pos_dist'])})."))
            flags.add("collapsed")
        if pr["gen_top_share"] > 0.3:
            findings.append(("BAD", f"Generation collapse: '{pr['gen_top'][0][0]}' is generated for {pr['gen_top_share']:.0%} of items."))
            flags.add("collapsed")
        elif pr["gen_entropy_ratio"] < 0.5:
            findings.append(("WARN", f"Generations are low-diversity ({pr['gen_unique']} unique strings, entropy ratio {pr['gen_entropy_ratio']:.2f})."))
        elif pr["gen_in_vocab"] < 0.5:
            findings.append(("WARN", f"Only {pr['gen_in_vocab']:.0%} of free generations are valid concept names — "
                                     "the decoder is still emitting pretraining-style text (e.g. '<img_src=...>')."))
        else:
            findings.append(("OK", f"Generations are diverse ({pr['gen_unique']} unique over {pr['n']} items); "
                                   f"{pr['gen_in_vocab']:.0%} are valid concept names."))

    if "diverged" in flags or "unstable" in flags:
        overall = "DIVERGED / UNSTABLE"
    elif "collapsed" in flags:
        overall = "COLLAPSED (degenerate predictions)"
    elif "chance" in flags:
        overall = "NOT LEARNING (at chance)"
    elif "shortcut" in flags:
        overall = "SHORTCUT (no better than image-free baselines)"
    elif "overfit" in flags:
        overall = "CONVERGED, THEN OVERFIT (use best/ checkpoint)"
    elif "still_improving_val" in flags or "still_improving_train" in flags:
        overall = "NOT YET CONVERGED (still improving — train longer)"
    else:
        overall = "CONVERGED"
    return overall, findings


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def style_axis(ax, title, xlabel="epoch", ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_run(run, va, pr, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (pip install matplotlib).")
        return False

    t, v = run["train"], run["val"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#fcfcfb")
    fig.suptitle(f"Pix2Struct on ASCIITune — {run['name']}", x=0.01, ha="left", fontsize=14)

    # 1. Loss
    ax = axes[0, 0]
    if t:
        te = [r["epoch"] for r in t]
        tl = [r["loss"] for r in t]
        ax.plot(te, tl, color=SERIES[0], alpha=0.25, linewidth=1)
        ax.plot(te, ema(tl), color=SERIES[0], linewidth=2, label="train (EMA)")
    if v:
        ax.plot([r["epoch"] for r in v], [r["loss"] for r in v], color=SERIES[1], linewidth=2,
                marker="o", markersize=5, label="val")
    style_axis(ax, "Loss", ylabel="cross-entropy")
    ax.legend(frameon=False, fontsize=8)

    # 2. Accuracy
    ax = axes[0, 1]
    if v:
        ve = [r["epoch"] for r in v]
        mc = np.array([r["mc_acc"] for r in v])
        se = np.array([acc_stderr(p, r["n"]) for p, r in zip(mc, v)])
        ax.fill_between(ve, mc - 2 * se, mc + 2 * se, color=SERIES[0], alpha=0.12, linewidth=0)
        ax.plot(ve, mc, color=SERIES[0], linewidth=2, marker="o", markersize=5, label="multiple-choice (±2 SE)")
        if "blank_mc_acc" in v[0]:
            ax.plot(ve, [r["blank_mc_acc"] for r in v], color=SERIES[1], linewidth=2,
                    marker="o", markersize=5, label="same model, blank image")
        ax.plot(ve, [r.get("gen_acc", np.nan) for r in v], color=SERIES[2], linewidth=2,
                marker="o", markersize=5, label="free generation")
        ax.axhline(CHANCE, color=INK, linestyle="--", linewidth=1)
        ax.text(ve[0], CHANCE, " chance (MC)", color=INK, fontsize=8, va="bottom")
        if v[-1].get("prior_mc_acc") is not None:
            prior = v[-1]["prior_mc_acc"]
            ax.axhline(prior, color=INK, linestyle=":", linewidth=1.2)
            ax.text(ve[-1], prior, f"label-frequency baseline {prior:.0%} ", color=INK, fontsize=8,
                    va="bottom", ha="right")
        if va.get("ok"):
            ax.annotate(f"best {va['mc_best']:.1%}", (va["mc_best_epoch"], va["mc_best"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#0b0b0b")
    if run["test"]:
        ax.text(0.99, 0.02, f"ASCIIEval test MC: {run['test']['mc_acc']:.1%}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#0b0b0b")
    ax.set_ylim(0, 1)
    style_axis(ax, "Validation accuracy", ylabel="accuracy")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # 3. Learning rate
    ax = axes[0, 2]
    if t:
        ax.plot([r["epoch"] for r in t], [r["lr"] for r in t], color=SERIES[0], linewidth=2)
    style_axis(ax, "Learning rate", ylabel="lr")

    # 4. Gradient norm
    ax = axes[1, 0]
    if t:
        gn = [r.get("grad_norm", np.nan) for r in t]
        ax.plot([r["epoch"] for r in t], gn, color=SERIES[0], alpha=0.25, linewidth=1)
        ax.plot([r["epoch"] for r in t], ema(gn), color=SERIES[0], linewidth=2)
        ax.axhline(run["config"].get("max_grad_norm", 1.0), color=INK, linestyle="--", linewidth=1)
        ax.set_yscale("log")
    style_axis(ax, "Gradient norm (pre-clip; dashed = clip)", ylabel="L2 norm (log)")

    # 5. MC choice-position distribution vs. truth
    ax = axes[1, 1]
    if pr.get("ok"):
        x = np.arange(4)
        ax.bar(x - 0.2, pr["true_pos_dist"], width=0.38, color=SERIES[1], label="true answer")
        ax.bar(x + 0.2, pr["mc_pos_dist"], width=0.38, color=SERIES[0], label="model pick")
        ax.set_xticks(x, [f"choice {i + 1}" for i in x])
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
    style_axis(ax, "MC position bias (latest eval)", xlabel="", ylabel="share of items")

    # 6. Generation collapse over time
    ax = axes[1, 2]
    if v and "gen_top_share" in v[0]:
        ve = [r["epoch"] for r in v]
        ax.plot(ve, [r["gen_top_share"] for r in v], color=SERIES[1], linewidth=2, marker="o",
                markersize=5, label="most-common generation share")
        ax.plot(ve, [r["gen_unique"] / r["n"] for r in v], color=SERIES[0], linewidth=2, marker="o",
                markersize=5, label="unique generations / items")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
    style_axis(ax, "Generation diversity", ylabel="fraction")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.4f}"


def build_report(run, tr, va, pr, overall, findings, plotted):
    L = [f"# Convergence report: {run['name']}", "", f"**Verdict: {overall}**", ""]
    icon = {"OK": "[OK]  ", "WARN": "[WARN]", "BAD": "[BAD] "}
    L += [f"- {icon[lvl]} {msg}" for lvl, msg in findings]
    L.append("")

    cfg = run["config"]
    if cfg:
        L += ["## Config", "",
              f"lr={cfg.get('lr')}  batch={cfg.get('batch_size')}x{cfg.get('grad_accum')}  epochs={cfg.get('epochs')}  "
              f"max_patches={cfg.get('max_patches')}  train={cfg.get('n_train')}  val={cfg.get('n_val')}", ""]

    if tr.get("ok"):
        L += ["## Training loss", "", "| metric | value |", "|---|---|",
              f"| initial (first 5%) | {fmt(tr['initial'])} |",
              f"| final (last 10%) | {fmt(tr['final'])} |",
              f"| reduction | {fmt(tr['reduction'], True)} |",
              f"| final-quarter slope (rel. per 10% of run) | {tr['tail_rel_change_per_10pct']:+.2%} |",
              f"| final-quarter noise (CV) | {fmt(tr['tail_cv'])} |",
              f"| spikes / NaN steps | {tr['spikes']} / {tr['nan_steps']} |",
              f"| grad norm early -> late (median) | {fmt(tr['grad_norm_early'])} -> {fmt(tr['grad_norm_late'])} |",
              ""]

    if va.get("ok"):
        L += ["## Validation", "", "| metric | first | best | final |", "|---|---|---|---|",
              f"| loss | {fmt(va['loss_first'])} | {fmt(va['loss_min'])} (ep {va['loss_min_epoch']:.2f}) | {fmt(va['loss_final'])} |",
              f"| MC accuracy | {fmt(va['mc_first'], True)} | {fmt(va['mc_best'], True)} (ep {va['mc_best_epoch']:.2f}) | {fmt(va['mc_final'], True)} |",
              f"| generation accuracy | | {fmt(va['gen_best'], True)} | {fmt(va['gen_final'], True)} |",
              "", f"MC standard error at n={va['n_val']}: ±{va['mc_stderr']:.1%}. Chance = 25%.", ""]
        L += ["| epoch | val loss | MC acc | MC calibrated | blank image | label-freq baseline | gen acc |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {r['epoch']:.2f} | {r['loss']:.4f} | {r['mc_acc']:.1%} | {fmt(r.get('mc_acc_calib'), True)} | "
              f"{fmt(r.get('blank_mc_acc'), True)} | {fmt(r.get('prior_mc_acc'), True)} | {fmt(r.get('gen_acc'), True)} |"
              for r in run["val"]]
        L.append("")

    if run["test"]:
        t = run["test"]
        L += ["## ASCIIEval test set (best checkpoint)", "",
              f"MC accuracy **{t['mc_acc']:.1%}** (n={t['n']}, ±{acc_stderr(t['mc_acc'], t['n']):.1%} SE), "
              f"generation accuracy {fmt(t.get('gen_acc'), True)}.", ""]
        if "prior_mc_acc" in t:
            L += ["| MC acc | MC calibrated | blank image | label-freq baseline | answer seen in train | answer unseen |",
                  "|---|---|---|---|---|---|",
                  f"| {t['mc_acc']:.1%} | {fmt(t.get('mc_acc_calib'), True)} | {fmt(t.get('blank_mc_acc'), True)} | "
                  f"{fmt(t.get('prior_mc_acc'), True)} | {fmt(t.get('mc_acc_seen'), True)} | "
                  f"{fmt(t.get('mc_acc_unseen'), True)} (n={t.get('n_unseen')}) |", ""]

    if pr.get("ok"):
        L += [f"## Predictions ({pr['file']})", "",
              "MC pick distribution by choice position: " + ", ".join(f"{p:.0%}" for p in pr["mc_pos_dist"])
              + "  (truth: " + ", ".join(f"{p:.0%}" for p in pr["true_pos_dist"]) + ")", "",
              "Most common free generations:", ""]
        L += [f"- `{g}` x{c}" for g, c in pr["gen_top"]]
        L.append("")

    if plotted:
        L += ["## Plots", "", "![convergence](convergence.png)", ""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="Analyze convergence of Pix2Struct ASCIITune runs")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    summary = []
    for d in args.run_dirs:
        run = load_run(d)
        tr = analyze_train(run)
        va = analyze_val(run, tr)
        pr = analyze_preds(run)
        overall, findings = verdict(tr, va, pr, run["test"])

        out_dir = run["dir"] / "analysis"
        out_dir.mkdir(exist_ok=True)
        plotted = plot_run(run, va, pr, out_dir / "convergence.png")
        report = build_report(run, tr, va, pr, overall, findings, plotted)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"verdict": overall, "train": tr, "val": va,
                       "preds": {k: v for k, v in pr.items() if k != "gen_top"},
                       "test": run["test"]}, f, indent=2, default=str)

        print(report)
        print(f"\nSaved: {out_dir / 'report.md'}" + (f", {out_dir / 'convergence.png'}" if plotted else ""))
        print("=" * 80)
        summary.append((run, va, overall))

    if len(summary) > 1:
        print("\n## Run comparison\n")
        print("| run | lr | best val MC | final val MC | min val loss | test MC | verdict |")
        print("|---|---|---|---|---|---|---|")
        for run, va, overall in sorted(summary, key=lambda s: -s[1].get("mc_best", 0)):
            print(f"| {run['name']} | {run['config'].get('lr')} | {fmt(va.get('mc_best'), True)} | "
                  f"{fmt(va.get('mc_final'), True)} | {fmt(va.get('loss_min'))} | "
                  f"{fmt(run['test']['mc_acc'], True) if run['test'] else 'n/a'} | {overall} |")


if __name__ == "__main__":
    main()
