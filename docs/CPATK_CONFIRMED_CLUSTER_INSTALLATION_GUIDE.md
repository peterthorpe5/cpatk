# CPATK confirmed cluster installation guide

This document records the tested CPATK + CLIPn environment recipe that worked on the Dundee cluster after resolving several SciPy, scikit-learn, UMAP and PyTorch dependency conflicts.

The confirmed environment name used during testing was:

```text
cpatk
```

The confirmed CPATK unit-test result after the fixes was:

```text
Ran 179 tests in 19.218s
OK
```

## Why this environment is pinned

CPATK can run many workflows with standard scientific Python packages, but the full workflow also uses optional packages for CLIPn, UMAP, SHAP, HDBSCAN, KeplerMapper, ONNX and PyTorch. On HPC systems these packages can easily pull in incompatible binary builds if conda and pip are mixed too freely.

The safest rule is:

- use conda-forge or mamba for compiled scientific packages;
- use pip only for packages not available cleanly through conda;
- use `pip install --no-deps` for CLIPn-related pip packages so pip does not replace the conda-built NumPy/SciPy/scikit-learn stack;
- use Python 3.10, not the newest Python;
- keep SciPy pinned to `1.13.1` for this environment;
- keep scikit-learn at `1.6.1` to satisfy the installed `umap-learn` requirement;
- use flexible channel priority for the PyTorch/CUDA solve if libmamba reports strict-repository-priority warnings.

## Start clean

Do not try to rescue an old mixed environment if SciPy or scikit-learn imports are failing. Create a fresh environment.

```bash
conda deactivate 2>/dev/null || true

mamba create -n cpatk -y -c conda-forge \
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
```

If `mamba` is unavailable, use:

```bash
conda install --solver=classic ...
```

or create the environment with `conda create`, but mamba is strongly preferred for this dependency set.

## Activate and set flexible priority

```bash
conda activate cpatk
conda config --env --set channel_priority flexible
```

Flexible priority was needed on the cluster when solving the PyTorch/CUDA install, because libmamba repeatedly printed:

```text
warning  libmamba Problem type not implemented SOLVER_RULE_STRICT_REPO_PRIORITY
```

This warning is annoying but not necessarily fatal. Setting flexible priority for this environment allowed the solve to proceed.

## Install PyTorch with CUDA 11.8

```bash
mamba install -y -c pytorch -c nvidia \
  pytorch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  pytorch-cuda=11.8
```

If `mamba` struggles, use:

```bash
conda install --solver=classic -y -c pytorch -c nvidia \
  pytorch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  pytorch-cuda=11.8
```

## Install pip-only packages without dependencies

Use `--no-deps` deliberately. The required shared dependencies have already been installed from conda-forge above.

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-deps kmapper
python -m pip install --no-deps clipn
python -m pip install --no-deps copairs
python -m pip install --no-deps onnx==1.14.1 onnxruntime==1.16.3
```

Do not run broad commands such as:

```bash
python -m pip install --upgrade numpy scipy scikit-learn pandas matplotlib umap-learn hdbscan shap
```

Those can overwrite the conda-built numerical stack and recreate the import failures.

## Install CPATK

From the CPATK repository or unpacked release directory:

```bash
cd /home/pthorpe001/data/2025_jason_cell_painting/cpatk
python -m pip install -e .
```

## Add library-path protection in interactive sessions and SGE jobs

After activating the environment, especially inside SGE jobs, use:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
```

This reduces the risk that Python uses the system `/lib64/libstdc++.so.6` instead of the conda copy.

## Environment validation

Run this before launching a full CPATK job:

```bash
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
```

`pip check` should not report missing dependencies. During debugging, the following missing packages were fixed by installing them with conda-forge:

```text
duckdb
statsmodels
protobuf
coloredlogs
flatbuffers
fsspec
tqdm
```

## Run CPATK unit tests

Limit BLAS/threaded libraries during unit tests:

```bash
cd /home/pthorpe001/data/2025_jason_cell_painting/cpatk

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python -m unittest discover -s tests -q
```

Expected successful result from the confirmed environment:

```text
Ran 179 tests in 19.218s
OK
```

## Packages to avoid in this environment unless explicitly needed

Avoid these in the default CPATK environment:

```text
umap
community
fastcluster
ace_tools_open
```

Use `umap-learn`, not `umap`. Use `python-louvain`, not the ambiguous `community` package. Leave `fastcluster` out unless a specific old workflow absolutely requires it.

## Troubleshooting notes

### GLIBCXX error

If you see:

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

then Python is probably loading the system `libstdc++` before the conda environment copy. Add the `LD_LIBRARY_PATH` and `LD_PRELOAD` lines above.

### SciPy `sph_legendre_p` or `_fitpack_impl` import errors

These indicate a broken SciPy/NumPy binary combination. Rebuild cleanly with the pinned versions in this guide rather than trying repeated partial upgrades.

### UMAP scikit-learn mismatch

If `pip check` reports:

```text
umap-learn ... has requirement scikit-learn>=1.6
```

install `scikit-learn=1.6.1` from conda-forge while keeping `scipy=1.13.1`.

### Sidecar files in raw data folders

CPATK v0.2.15 and later skip macOS AppleDouble sidecar files beginning `._`. If using an older release, remove them before inspecting raw CellProfiler folders:

```bash
find /path/to/CellProfiler_folder -name '._*' -type f -print -delete
```
