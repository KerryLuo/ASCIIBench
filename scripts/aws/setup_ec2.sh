#!/usr/bin/env bash
# ASCIIBench Pix2Struct — EC2 instance bootstrap
#
# Run this ON the EC2 instance (Deep Learning AMI GPU PyTorch, Ubuntu) after
# SSH'ing in. It clones the repo, installs dependencies, logs into wandb,
# and drops you into a tmux session ready to launch training.
#
# Usage (on the instance):
#   export REPO_URL="https://github.com/<your-org>/ASCIIBench.git"
#   export WANDB_API_KEY="..."          # optional, can also run `wandb login` interactively
#   export S3_BUCKET="my-checkpoint-bucket"   # optional
#   bash setup_ec2.sh

set -euo pipefail

REPO_URL="${REPO_URL:?Set REPO_URL to your git remote, e.g. export REPO_URL=https://github.com/you/ASCIIBench.git}"
REPO_DIR="${REPO_DIR:-$HOME/ASCIIBench}"

echo "==> Cloning repo"
if [ -d "$REPO_DIR/.git" ]; then
  echo "    $REPO_DIR already exists, pulling latest instead"
  git -C "$REPO_DIR" pull
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> Activating PyTorch conda env (Deep Learning AMI)"
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate pytorch

echo "==> Installing Python dependencies"
pip install -q -r requirements-pix2struct.txt

echo "==> GPU check"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU visible')"

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "==> Logging into wandb"
  wandb login "$WANDB_API_KEY"
else
  echo "==> WANDB_API_KEY not set — run 'wandb login' manually before training if you want wandb logging"
fi

if [ -n "${S3_BUCKET:-}" ]; then
  echo "==> Verifying S3 access to bucket: $S3_BUCKET"
  aws s3 ls "s3://$S3_BUCKET" >/dev/null && echo "    OK: bucket reachable" || \
    echo "    Warning: could not list bucket — check the instance's IAM role/permissions"
fi

echo ""
echo "==> Setup complete."
echo "Next steps:"
echo "  tmux new -s pix2struct"
echo "  cd $REPO_DIR"
echo "  python scripts/run_pix2struct_finetuning.py --batch-size 4 \\"
if [ -n "${S3_BUCKET:-}" ]; then
  echo "      --s3-bucket $S3_BUCKET \\"
fi
echo "      # add other flags as needed, e.g. --lr, --epochs"
echo "  # detach with Ctrl+B then D; reattach later with: tmux attach -t pix2struct"
