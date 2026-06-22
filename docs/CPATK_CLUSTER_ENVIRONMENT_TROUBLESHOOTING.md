# CPATK cluster environment troubleshooting

## GLIBCXX_3.4.30 import errors

If importing SciPy or scikit-learn fails with an error similar to:

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

then Python is loading the system `libstdc++` before the conda environment copy.
This is an environment/library-order problem, not a CPATK algorithm failure.

Use this after activating the conda environment:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
```

Then test:

```bash
python -c "import scipy, sklearn; print(scipy.__version__, sklearn.__version__)"
```

If the conda `libstdc++` does not contain `GLIBCXX_3.4.30`, reinstall the runtime
and scientific stack from conda-forge:

```bash
conda install -c conda-forge libstdcxx-ng libgcc-ng scipy scikit-learn
```

A lighter diagnostic script is provided at:

```text
examples/check_cpatk_cluster_environment.sh
```
