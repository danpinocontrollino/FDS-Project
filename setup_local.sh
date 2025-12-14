#!/usr/bin/env bash
set -euo pipefail

echo "Creating virtual environment .venv (Python 3)"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "Installing lightweight requirements (no PyTorch)"
python -m pip install -r requirements-no-torch.txt

echo "Done. Activate with: source .venv/bin/activate"
