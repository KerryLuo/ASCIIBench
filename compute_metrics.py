#!/usr/bin/env python3
"""ASCIIBench Metrics Computation

Computes micro accuracy, macro accuracy, and pass rate from classification results.
Works on individual result files or all files in the results directory.

Usage:
    python compute_metrics.py                                    # all results
    python compute_metrics.py results/gpt-4o_text_results.jsonl  # specific file
    python compute_metrics.py --dedupe                           # remove duplicate entries
    python compute_metrics.py --filter-errors                    # exclude parse errors
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def dedupe_results(results):
    seen = set()
    deduped = []
    for r in results:
        key = r.get("ascii_art", "")
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def remove_parse_errors(results):
    return [r for r in results if r.get("predicted_class") is not False
            and r.get("predicted_class") is not None]


def compute_metrics(results):
    if not results:
        return {"micro_accuracy": 0, "macro_accuracy": 0, "pass_rate": 0,
                "total": 0, "correct": 0, "num_classes": 0, "parse_errors": 0}

    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    parse_errors = sum(1 for r in results
                       if r.get("predicted_class") is False
                       or r.get("predicted_class") is None)

    micro_accuracy = correct / total if total else 0

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cls = r.get("actual_class", "")
        per_class[cls]["total"] += 1
        if r.get("correct", False):
            per_class[cls]["correct"] += 1

    class_accuracies = []
    for cls, counts in per_class.items():
        if counts["total"] > 0:
            class_accuracies.append(counts["correct"] / counts["total"])

    macro_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0

    pass_rate = (total - parse_errors) / total if total else 0

    return {
        "micro_accuracy": micro_accuracy,
        "macro_accuracy": macro_accuracy,
        "pass_rate": pass_rate,
        "total": total,
        "correct": correct,
        "num_classes": len(per_class),
        "parse_errors": parse_errors,
    }


def format_table(rows, headers):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_row, sep]
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute ASCIIBench classification metrics")
    parser.add_argument("files", nargs="*", help="Result JSONL file(s). Defaults to all in results/")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate entries per file")
    parser.add_argument("--filter-errors", action="store_true", help="Exclude parse errors from accuracy")
    parser.add_argument("--per-class", action="store_true", help="Show per-class breakdown")
    parser.add_argument("--csv", action="store_true", help="Output as CSV instead of table")
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: results directory '{results_dir}' not found.")
            sys.exit(1)
        files = sorted(results_dir.glob("*_results.jsonl"))

    if not files:
        print("No result files found.")
        sys.exit(0)

    headers = ["File", "Total", "Correct", "Micro Acc", "Macro Acc", "Pass Rate", "Errors"]
    rows = []

    for fpath in files:
        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping.")
            continue

        results = load_results(fpath)
        if not results:
            print(f"Warning: {fpath} is empty, skipping.")
            continue

        if args.dedupe:
            results = dedupe_results(results)

        if args.filter_errors:
            results = remove_parse_errors(results)

        m = compute_metrics(results)
        label = fpath.stem.replace("_results", "")
        rows.append([
            label,
            m["total"],
            m["correct"],
            f"{m['micro_accuracy']:.4f}",
            f"{m['macro_accuracy']:.4f}",
            f"{m['pass_rate']:.4f}",
            m["parse_errors"],
        ])

        if args.per_class:
            print(f"\n--- Per-class breakdown: {label} ---")
            per_class = defaultdict(lambda: {"correct": 0, "total": 0})
            for r in results:
                cls = r.get("actual_class", "")
                per_class[cls]["total"] += 1
                if r.get("correct", False):
                    per_class[cls]["correct"] += 1

            class_rows = []
            for cls in sorted(per_class.keys()):
                c = per_class[cls]
                acc = c["correct"] / c["total"] if c["total"] else 0
                class_rows.append([cls, c["total"], c["correct"], f"{acc:.4f}"])
            print(format_table(class_rows, ["Class", "Total", "Correct", "Accuracy"]))

    if args.csv:
        print(",".join(headers))
        for row in rows:
            print(",".join(str(c) for c in row))
    else:
        print("\n" + format_table(rows, headers))


if __name__ == "__main__":
    main()
