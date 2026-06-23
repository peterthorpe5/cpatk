# CPATK v0.2.15 malaria fast TMPDIR/Parquet shell, fixed argument passing

This is the corrected version of the fast malaria shell.

The previous version failed at preprocessing with an error like:

```text
cpatk-preprocess: error: unrecognized arguments: --reference_normalisation_method none --batch_correction_method none
```

The cause was shell argument packing. Because the script set:

```bash
IFS=$'\n\t'
```

preprocessing strategy arguments supplied as one string were not split on spaces. The fixed script now passes each strategy argument as a real array element.

Recommended run:

```bash
qsub -v CONDA_ENV_NAME=cpatk run_malaria_cpatk_v0_2_15_fast_tmp_parquet_fixed.sh
```

To rerun into a new output folder, simply submit again. To force rerun in an existing output folder:

```bash
FORCE_RERUN=1 qsub -v CONDA_ENV_NAME=cpatk run_malaria_cpatk_v0_2_15_fast_tmp_parquet_fixed.sh
```

The script still stages inputs to `$TMPDIR`, excludes sidecar files, prefers Parquet, and syncs partial results back on normal failure.
