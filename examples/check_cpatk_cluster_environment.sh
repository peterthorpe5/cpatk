#!/usr/bin/env bash
#$ -jc rhel9
#$ -j y
#$ -N cpatk_env_check
#$ -cwd

set -Eeuo pipefail
IFS=$'\n\t'

section() { echo -e "\n==== $* ====\n"; }

section "Conda environment"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "PATH=$PATH"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
        export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
fi

section "libstdc++ GLIBCXX symbols"
if [[ -f /lib64/libstdc++.so.6 ]]; then
    echo "System /lib64/libstdc++.so.6:"
    strings /lib64/libstdc++.so.6 | grep 'GLIBCXX_3.4.30' || true
fi
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    echo "Conda ${CONDA_PREFIX}/lib/libstdc++.so.6:"
    strings "${CONDA_PREFIX}/lib/libstdc++.so.6" | grep 'GLIBCXX_3.4.30' || true
fi

section "Python import checks"
python - <<'PY'
import os
import sys
print('python', sys.version)
print('CONDA_PREFIX', os.environ.get('CONDA_PREFIX'))
for module_name in ['numpy', 'scipy', 'sklearn', 'pandas', 'matplotlib']:
    try:
        module = __import__(module_name)
        print(module_name, 'OK', getattr(module, '__version__', 'unknown'), getattr(module, '__file__', ''))
    except Exception as exc:
        print(module_name, 'FAILED', repr(exc))
        raise
PY

section "CPATK import check"
python - <<'PY'
import cpatk
print('cpatk', cpatk.__version__)
PY
