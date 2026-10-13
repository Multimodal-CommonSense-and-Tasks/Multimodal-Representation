#!/bin/bash
# Creates a Python virtual environment and installs all dependencies.
# Verified on a single NVIDIA RTX PRO 6000 (CUDA 12.8) with Python 3.10.12.
#
#   bash install.sh            # creates ./.venv
#   source .venv/bin/activate

set -e

PYTHON=${PYTHON:-python3.10}
VENV_DIR=${VENV_DIR:-.venv}

${PYTHON} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"
