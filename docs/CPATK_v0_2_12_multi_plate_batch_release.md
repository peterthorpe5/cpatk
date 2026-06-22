# CPATK v0.2.12 multi-plate, control-QC and batch-correction release notes

## Purpose

This release hardens CPATK for real multi-plate Cell Painting studies. The focus is not to add a new downstream plot, but to make the core data-building and preprocessing path safer when several CellProfiler exports, plates, DMSO controls and replicate groups are analysed together.

## New command-line entry point

```bash
cpatk-combine-profiles \
  --profile_tables plate_01/merged_profiles.tsv.gz,plate_02/merged_profiles.tsv.gz \
  --output_dir results/01_profile_build/all_plates \
  --source_labels plate_01,plate_02 \
  --key_columns Metadata_Plate,Metadata_Well \
  --feature_join union \
  --duplicate_policy error \
  --log_level INFO
```

This command combines already-built profile tables after each plate or export has been reviewed. It writes:

- `combined_profiles.tsv.gz`
- `combine_profile_summary.tsv`
- `input_profile_report.tsv`
- `combined_duplicate_key_report.tsv`
- `feature_presence_matrix.tsv`
- `retained_combined_features.tsv`
- `combine_profiles_summary.xlsx`
- `combine_profiles_report.html`
- `combine_profiles.log`

Use `feature_join union` when different plates have partially different features and you want preprocessing QC to decide what survives. Use `feature_join intersection` only when you deliberately want to keep features measured in every table.

## Native composite-key profile building

`cpatk-build-profiles` and `cpatk-preprocess --input_dir` now support explicit composite image/object merge keys:

```bash
cpatk-build-profiles \
  --input_dir raw_cellprofiler_exports \
  --output_dir results/01_profile_build \
  --image_merge_keys Metadata_Plate,ImageNumber \
  --metadata_table results/00_metadata_check/formatted_metadata.tsv \
  --aggregate_statistic median
```

This matters because independent CellProfiler exports often restart `ImageNumber` at 1. Using `ImageNumber` alone can silently mis-merge object summaries from different plates. CPATK now prefers composite keys when they are available and fails if a profile backbone has duplicated `ImageNumber` values without a plate/export key.

Recommended behaviour:

- If Image, Cell, Nuclei and Cytoplasm tables all contain assay plate metadata, pooled multi-plate profile building can use `--image_merge_keys Metadata_Plate,ImageNumber`.
- If object tables do not contain plate/export metadata, build each plate/export separately, then use `cpatk-combine-profiles`.
- Do not use source/robot plate or well columns as assay profile keys.

## Control QC before reference normalisation

`cpatk-preprocess` now writes `reference_control_qc_before_normalisation.tsv` whenever reference normalisation is requested. This report is calculated before DMSO/reference normalisation and before imputation, so weak control sets are visible rather than hidden.

Example per-plate DMSO robust-z normalisation:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/all_plates/combined_profiles.tsv.gz \
  --output_dir results/02_preprocess \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate
```

Inspect the control QC report for:

- no DMSO/reference rows in a plate;
- fewer than two reference rows;
- high reference missingness;
- many features with zero or near-zero DMSO MAD.

If control QC is poor, reference normalisation can create unstable values. In that case, compare with no reference normalisation and consider whether the controls or plate design are usable.

## Optional ComBat-style correction

`cpatk-preprocess` now includes optional ComBat-style location/scale correction:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/all_plates/combined_profiles.tsv.gz \
  --output_dir results/02_preprocess_combat_style \
  --batch_correction_method combat_location_scale \
  --batch_column Metadata_Plate \
  --batch_protect_columns Metadata_Compound,Metadata_MOA,Metadata_Dose \
  --batch_correction_min_batch_size 3
```

This is deliberately named `combat_location_scale`. It is a transparent feature-wise location/scale harmonisation inspired by the practical goal of ComBat, but it is not claimed to be a full empirical-Bayes ComBat implementation.

It writes:

- `batch_correction_report.tsv`
- `batch_confounding_report.tsv`

Use it only after checking confounding. If compound, MOA or dose is strongly confounded with plate, batch correction may remove biology or fail to separate technical and biological structure.

## Before/after replicate and batch reports

`cpatk-preprocess` can now generate before/after reports around the optional batch-correction step:

```bash
cpatk-preprocess \
  --input_table combined_profiles.tsv.gz \
  --output_dir results/02_preprocess \
  --batch_correction_method combat_location_scale \
  --batch_column Metadata_Plate \
  --batch_protect_columns Metadata_Compound,Metadata_MOA,Metadata_Dose \
  --replicate_group_columns Metadata_Compound,Metadata_Dose \
  --batch_report_columns Metadata_Plate,Metadata_Batch
```

It writes:

- `before_after_replicate_correlations.tsv`
- `before_after_replicate_summary.tsv`
- `before_after_batch_pc_association.tsv`

Use these reports to check whether correction improves replicate consistency and reduces batch association, rather than simply making plots look cleaner.

## Recommended multi-plate analysis strategy

For production work, run at least two preprocessing strategies and compare them:

1. No reference normalisation, robust scaling.
2. Per-plate DMSO robust-z normalisation.
3. Optional ComBat-style correction, only if the design is not badly confounded.

Then compare:

- DMSO/control QC;
- replicate correlations;
- batch PC association;
- batch diagnostics from `cpatk-batch`;
- PCA/UMAP coloured by plate, compound and MOA;
- nearest-neighbour stability.

Do not treat batch correction as automatically better. A good publication-quality analysis should show that the chosen preprocessing improves technical consistency without erasing expected biological structure.
