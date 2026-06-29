# CPATK v0.2.27 sanity pass and safe threading controls

## Purpose

CPATK v0.2.27 is a final sanity and hardening pass after the v0.2.26 native contrastive backend release.

The main goals were:

1. sanity-check the v0.2.26 native contrastive changes;
2. keep CPATK-native contrastive as the default latent backend;
3. keep the published external CLIPn package as an explicitly requested compatibility backend only;
4. add safe multi-threading controls where they are useful and unlikely to cause nested oversubscription;
5. update the large multi-dataset SGE shell so it passes the cluster thread allocation through to supported CPATK steps.

## Default latent backend policy

The default remains:

```bash
--backend_module cpatk_contrastive
```

The published external CLIPn backend is still available only when explicitly requested:

```bash
--backend_module clipn
```

## Threading additions

A new helper module was added:

```text
cpatk/threading_utils.py
```

It provides:

```text
normalise_thread_count
configure_threading
configure_torch_threads
```

These functions set common BLAS/OpenMP environment variables and, when available, use `threadpoolctl` to apply a live process-level native-library thread limit.

## Commands now accepting `--threads`

The following command-line tools now accept an explicit thread count:

```text
cpatk-classical
cpatk-batch
cpatk-ml
cpatk-explain
cpatk-latent
cpatk-clipn
```

The legacy `cpatk-clipn` command name is retained, but the default backend remains CPATK-native contrastive unless `--backend_module clipn` is supplied.

## Where threading is used

Threading is applied only where it is reasonably safe:

- pairwise distance calculation in `cpatk-classical`;
- scikit-learn cross-validation scheduling in `cpatk-ml`;
- scikit-learn cross-validation scheduling in `cpatk-batch`;
- permutation importance and local explainability models in `cpatk-explain`;
- PyTorch CPU thread pools and latent nearest-neighbour diagnostics in `cpatk-latent`;
- common native-library thread pools through environment variables and `threadpoolctl`.

For cross-validation workflows, CPATK avoids obvious nested oversubscription by letting the cross-validation scheduler own the thread count while keeping the fitted tree estimator inside each fold single-threaded. For direct non-CV model fitting, supported tree estimators can still use their own `n_jobs` setting.

## Large multi-dataset shell update

The example full SGE shell is now:

```text
examples/run_cpatk_v0_2_27_multidataset_stb_selleck_mitotox_full_sge.sh
```

It requests 16 SGE slots by default and passes:

```bash
--threads "${THREADS}"
```

to supported downstream tools. It also retains the corrected image-level combined-profile key:

```bash
Metadata_Profile_Source,Metadata_Plate,ImageNumber,Metadata_Well
```

## Validation performed in the sandbox

The following checks passed:

```bash
python -m compileall -q cpatk tests
python -m unittest tests.test_cpatk_v0_2_26 tests.test_cpatk_v0_2_27 -v
python -m unittest tests.test_cpatk_v0_2 tests.test_cpatk_v0_2_7 tests.test_cpatk_v0_2_21 tests.test_cpatk_v0_2_23 tests.test_cpatk_v0_2_25 tests.test_cpatk_v0_2_26 tests.test_cpatk_v0_2_27 -v
bash -n run_cpatk_v0_2_27_multidataset_stb_selleck_mitotox_full_sge.sh
```

The combined targeted unittest command ran 60 tests and passed.

A full package-wide `unittest discover` was not claimed here, because previous full-suite discovery attempts have timed out in the sandbox due to older long-running workflows.

## Practical recommendation

For the large cluster run, keep:

```bash
#$ -pe smp 16
THREADS="${THREADS:-${NSLOTS:-16}}"
```

and let the v0.2.27 shell pass those threads to the supported CPATK steps. This is safer than trying to parallelise every loop manually.
