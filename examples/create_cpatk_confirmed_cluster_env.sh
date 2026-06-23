#!/usr/bin/env bash
# Create the confirmed CPATK + CLIPn Python 3.10 environment used for
# full CPATK testing on the Dundee cluster.
#
# Usage:
#   bash examples/create_cpatk_confirmed_cluster_env.sh
#
# Optional overrides:
#   ENV_NAME=cpatk_test bash examples/create_cpatk_confirmed_cluster_env.sh
#   CUDA_VERSION=12.1 bash examples/create_cpatk_confirmed_cluster_env.sh

set -Eeuo pipefail
IFS=$'\n\t'

ENV_NAME="${ENV_NAME:-cpatk}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"

if command -v mamba >/dev/null 2>&1; then
    CONDA_SOLVER="mamba"
else
    CONDA_SOLVER="conda"
fi

section() {
    printf '\n==== %s ====\n\n' "$*"
}

section "Creating conda environment: ${ENV_NAME}"
"${CONDA_SOLVER}" create -n "${ENV_NAME}" -y -c conda-forge \
  python=3.10 \
  numpy=1.26.4 \
  pandas=2.0 \
  scipy=1.13.1 \
  scikit-learn=1.6.1 \
  umap-learn=0.5.8 \
  matplotlib \
  openpyxl \
  xlsxwriter \
  jinja2 \
  plotly \
  hdbscan \
  shap \
  psutil \
  pyarrow \
  natsort \
  colorcet \
  networkx \
  python-louvain \
  pyvis \
  optuna \
  phate \
  libstdcxx-ng \
  libgcc-ng \
  ruff \
  tqdm \
  duckdb \
  statsmodels \
  protobuf \
  coloredlogs \
  flatbuffers \
  fsspec

section "Activating environment"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
conda config --env --set channel_priority flexible

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

section "Installing PyTorch CUDA ${CUDA_VERSION}"
if [[ "${CUDA_VERSION}" == "11.8" ]]; then
    "${CONDA_SOLVER}" install -y -c pytorch -c nvidia \
      pytorch==2.1.2 \
      torchvision==0.16.2 \
      torchaudio==2.1.2 \
      pytorch-cuda=11.8
elif [[ "${CUDA_VERSION}" == "12.1" ]]; then
    "${CONDA_SOLVER}" install -y -c pytorch -c nvidia \
      pytorch==2.1.2 \
      torchvision==0.16.2 \
      torchaudio==2.1.2 \
      pytorch-cuda=12.1
else
    echo "Unsupported CUDA_VERSION=${CUDA_VERSION}; use 11.8 or 12.1" >&2
    exit 2
fi

section "Installing pip-only packages without dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps kmapper
python -m pip install --no-deps clipn
python -m pip install --no-deps copairs
python -m pip install --no-deps onnx==1.14.1 onnxruntime==1.16.3

section "Running import checks"
python -m pip check
python - <<'PY'
import sys
import numpy
import pandas
import scipy
import scipy.stats
import scipy.interpolate
import scipy.special
import sklearn
import umap
import torch
import copairs
import duckdb
import statsmodels
import onnx
import onnxruntime
import kmapper
import tqdm

print("python", sys.version)
print("numpy", numpy.__version__, numpy.__file__)
print("pandas", pandas.__version__, pandas.__file__)
print("scipy", scipy.__version__, scipy.__file__)
print("sklearn", sklearn.__version__, sklearn.__file__)
print("umap", umap.__file__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("core imports OK")
PY

section "Done"
echo "Activate with: conda activate ${ENV_NAME}"
echo "Install CPATK from the repository with: python -m pip install -e ."
