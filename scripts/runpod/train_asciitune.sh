#!/usr/bin/env bash
# Fine-tune Pix2Struct on ASCIITune on a RunPod GPU pod, then run the convergence analysis.
#
# Runs in the background with nohup, so closing the SSH / web terminal does not kill training.
#
# Usage (from the repo root, e.g. /workspace/ASCIIBench):
#   bash scripts/runpod/train_asciitune.sh smoke              # ~5 min end-to-end check on a tiny subset
#   bash scripts/runpod/train_asciitune.sh train              # full run with defaults (lr 1e-4, 10 epochs)
#   bash scripts/runpod/train_asciitune.sh train --lr 3e-5    # extra args go to run_pix2struct_asciitune.py
#   bash scripts/runpod/train_asciitune.sh sweep              # lr 3e-5, 1e-4, 3e-4 one after another + comparison
#   bash scripts/runpod/train_asciitune.sh analyze runs/asciitune_lr1e-4   # re-run analysis on any run(s)
#
# Watch progress:   tail -f runs/<run_name>/train.log
# Resume a run:     bash scripts/runpod/train_asciitune.sh train --output-dir runs/<run_name> --resume runs/<run_name>/last.pt
#
# Optional env vars:
#   WANDB_API_KEY   enables Weights & Biases logging (otherwise --no-wandb is passed)
#   RUN_NAME        output folder name under runs/ (default: asciitune_lr<lr>_<timestamp>)
#   SKIP_INSTALL=1  skip pip install

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-train}"
shift || true

# Keep model/dataset downloads on the persistent volume so pod restarts don't re-download.
if [ -d /workspace ]; then
    export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

setup() {
    if [ "${SKIP_INSTALL:-0}" != "1" ]; then
        echo ">>> Installing requirements"
        pip install -q -r requirements-pix2struct.txt
    fi
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
}

wandb_flag() {
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "--no-wandb"
    fi
}

# Pull --lr out of the extra args so it can go in the run name.
lr_from_args() {
    local lr="1e-4" prev=""
    for a in "$@"; do
        if [ "$prev" = "--lr" ]; then lr="$a"; fi
        prev="$a"
    done
    echo "$lr"
}

has_arg() {
    local needle="$1"; shift
    for a in "$@"; do [ "$a" = "$needle" ] && return 0; done
    return 1
}

# Train one run in the foreground of the current (already nohup'd) shell, then analyze it.
train_one() {
    local out_dir="$1"; shift
    mkdir -p "$out_dir"
    echo ">>> Training -> $out_dir  (args: $*)"
    python scripts/run_pix2struct_asciitune.py --output-dir "$out_dir" $(wandb_flag) "$@" 2>&1 | tee -a "$out_dir/train.log"         || echo "!!! Training exited with an error (see $out_dir/train.log); analyzing whatever was logged."
    echo ">>> Analyzing $out_dir"
    python scripts/analyze_pix2struct_convergence.py "$out_dir" 2>&1 | tee "$out_dir/analysis.log"
}

launch_bg() {
    # Re-invoke this script under nohup with an internal mode so the whole pipeline survives disconnects.
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    nohup bash "$SCRIPT_PATH" "$@" > "$log" 2>&1 &
    echo ">>> Started in background (PID $!). Follow with:"
    echo "    tail -f $log"
}

STAMP="$(date +%Y%m%d_%H%M%S)"

case "$MODE" in
    smoke)
        setup
        OUT="runs/${RUN_NAME:-smoke_$STAMP}"
        train_one "$OUT" --max-train-samples 64 --max-val-samples 32 --max-test-samples 32 \
            --epochs 2 --log-every 1 --evals-per-epoch 2 --no-wandb "$@"
        echo ">>> Smoke test finished. Check $OUT/analysis/report.md"
        ;;

    train)
        setup
        LR="$(lr_from_args "$@")"
        if has_arg --output-dir "$@"; then
            launch_bg "runs/launcher_$STAMP.log" _train_custom "$@"
        else
            OUT="runs/${RUN_NAME:-asciitune_lr${LR}_$STAMP}"
            launch_bg "$OUT/launcher.log" _train "$OUT" "$@"
        fi
        ;;

    sweep)
        setup
        SWEEP_DIR="runs/sweep_$STAMP"
        launch_bg "$SWEEP_DIR/launcher.log" _sweep "$SWEEP_DIR" "$@"
        ;;

    analyze)
        python scripts/analyze_pix2struct_convergence.py "$@"
        ;;

    # ---- internal modes (run under nohup) ----
    _train)
        OUT="$1"; shift
        train_one "$OUT" "$@"
        ;;

    _train_custom)
        # --output-dir supplied by the user (e.g. when resuming).
        OUT=""; prev=""
        for a in "$@"; do [ "$prev" = "--output-dir" ] && OUT="$a"; prev="$a"; done
        mkdir -p "$OUT"
        echo ">>> Training -> $OUT  (args: $*)"
        python scripts/run_pix2struct_asciitune.py $(wandb_flag) "$@" 2>&1 | tee -a "$OUT/train.log"             || echo "!!! Training exited with an error (see $OUT/train.log)"
        python scripts/analyze_pix2struct_convergence.py "$OUT" 2>&1 | tee "$OUT/analysis.log"
        ;;

    _sweep)
        SWEEP_DIR="$1"; shift
        RUNS=()
        for LR in 3e-5 1e-4 3e-4; do
            OUT="$SWEEP_DIR/lr$LR"
            RUNS+=("$OUT")
            train_one "$OUT" --lr "$LR" "$@" || echo "!!! run lr=$LR failed, continuing"
        done
        echo ">>> Sweep comparison"
        python scripts/analyze_pix2struct_convergence.py "${RUNS[@]}" 2>&1 | tee "$SWEEP_DIR/comparison.log"
        ;;

    *)
        echo "Unknown mode: $MODE (expected smoke | train | sweep | analyze)"
        exit 1
        ;;
esac
