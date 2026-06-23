# CPATK v0.2.15 mitotox fast TMPDIR/Parquet stress-test shell

This shell is a broad CPATK stress test for the mitotox Cell Painting dataset at:

```bash
/home/pthorpe001/data/2025_jason_cell_painting/data/mitotox
```

It expects these inputs by default:

```bash
raw/
metadata/KVP_MitotoxPlate_IXM_07042025_cleaned.csv
metadata/KVP_MitotoxPlate_IXM_07042025.csv
mitotox_all_plates_image_level.tsv
```

The cleaned metadata is used first. The raw metadata is only used if the cleaned file is absent.

## Recommended run

```bash
qsub -v CONDA_ENV_NAME=cpatk run_mitotox_cpatk_v0_2_15_fast_tmp_parquet.sh
```

or interactively:

```bash
conda activate cpatk
bash run_mitotox_cpatk_v0_2_15_fast_tmp_parquet.sh
```

## What it does

The script:

1. uses `$TMPDIR` when available;
2. copies raw CellProfiler exports to local scratch;
3. excludes macOS sidecar files such as `._*` and `.DS_Store`;
4. validates metadata with explicit plate/well columns;
5. treats `BC` and `comp_s` as source/transfer columns, not assay keys;
6. optionally derives a pseudo-anchor label table from `Seahorse_alert`;
7. inspects the raw CellProfiler folder;
8. runs drift QC;
9. builds merged profiles from Image/object tables;
10. falls back to `mitotox_all_plates_image_level.tsv` if raw profile building fails;
11. runs baseline preprocessing as the required primary strategy;
12. runs DMSO/reference normalisation and batch-corrected strategies as optional strategies;
13. runs classical analysis, visualisation, stability, batch diagnostics, neighbours, MOA, ML, explanation and CLIPn where possible;
14. syncs results back to the project filesystem at the end or on normal failure.

## Why baseline preprocessing is primary by default

This dataset may or may not have a clean `DMSO` control label in `Metadata_Compound`. The script therefore makes baseline preprocessing the required primary strategy, then attempts DMSO robust-z normalisation as an optional strategy. If DMSO robust-z succeeds, it is used as the primary strategy for downstream interpretation.

## Useful overrides

Fast smoke test:

```bash
RUN_DRIFT_QC=0 \
RUN_ML=0 \
RUN_EXPLAIN=0 \
RUN_CLIPN=0 \
RUN_MOA=0 \
STABILITY_BOOTSTRAPS=5 \
STABILITY_PERMUTATIONS=5 \
qsub -v CONDA_ENV_NAME=cpatk run_mitotox_cpatk_v0_2_15_fast_tmp_parquet.sh
```

Full stress test with scratch retained:

```bash
KEEP_SCRATCH=1 qsub -v CONDA_ENV_NAME=cpatk run_mitotox_cpatk_v0_2_15_fast_tmp_parquet.sh
```

Use a different pseudo-anchor label source column:

```bash
MITOTOX_LABEL_SOURCE_COLUMN=Library qsub -v CONDA_ENV_NAME=cpatk run_mitotox_cpatk_v0_2_15_fast_tmp_parquet.sh
```

Disable premerged fallback:

```bash
ALLOW_PREMERGED_PROFILE_FALLBACK=0 qsub -v CONDA_ENV_NAME=cpatk run_mitotox_cpatk_v0_2_15_fast_tmp_parquet.sh
```

## Failure behaviour

Core metadata/profile/preprocessing failures still stop the run if no usable fallback exists. Optional downstream modules are run in soft-fail mode where possible. Results are synced back from scratch on normal command failure through an exit trap.

Hard scheduler kills, node crashes or scratch deletion may still prevent sync-back.
