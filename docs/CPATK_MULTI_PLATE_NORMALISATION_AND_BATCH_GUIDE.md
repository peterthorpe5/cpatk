# CPATK multi-plate normalisation and batch guide

Version: 0.2.11 documentation expansion

## Why this matters

Real Cell Painting datasets often contain several plates. Plate effects can arise from staining, incubation, imaging date, microscope state, reagent batches, plate position, cell density, acquisition order and instrument drift. Multi-plate analysis therefore needs both normalisation and diagnostics.

The older project-specific workflow used per-plate DMSO robust-z normalisation with median and MAD. This remains a sensible option when each plate has suitable controls. However, drift and stronger batch effects mean it should be compared against unnormalised and batch-diagnostic outputs rather than applied blindly.

## What CPATK v0.2.11 currently supports

CPATK currently supports:

- multi-plate metadata once each profile row has reliable `Metadata_Plate` and `Metadata_Well`;
- per-plate DMSO/reference normalisation through `--reference_group_columns Metadata_Plate`;
- simple batch centring by batch columns;
- batch/domain-shift diagnostics with `cpatk-batch`;
- replicate and stability checks with `cpatk-stability`;
- acquisition drift QC on object-level files with `cpatk-drift-qc`.

## Current multi-plate limitation

v0.2.11 should not yet be considered fully native for several independent CellProfiler export folders when `ImageNumber` restarts per plate/export and files are pooled into one folder.

The current profile builder aggregates object tables by `ImageNumber`. This is safe when `ImageNumber` is unique within the analysed export. It is risky when several independent exports reuse the same `ImageNumber` values.

## Recommended safe multi-plate workflow for v0.2.11

### 1. Validate metadata once

```bash
cpatk-metadata \
  --metadata_table metadata/raw_plate_map.tsv \
  --output_dir results/00_metadata_check \
  --plate_column Assay_Plate_Barcode \
  --well_column Destination_Well \
  --source_plate_column Source_Plate_Barcode \
  --source_well_column Source_Well \
  --annotation_tables metadata/compound_annotations.tsv \
  --annotation_source_plate_column Barcode \
  --annotation_source_well_column Well \
  --merge_keys Metadata_Source_Plate,Metadata_Source_Well \
  --duplicate_policy error \
  --log_level INFO
```

### 2. Build profiles per plate/export

```bash
cpatk-build-profiles \
  --input_dir raw_cellprofiler/plate_01 \
  --output_dir results/01_profile_build/plate_01 \
  --recursive \
  --metadata_table results/00_metadata_check/formatted_metadata.tsv \
  --aggregate_statistic median \
  --duplicate_image_policy error \
  --metadata_duplicate_policy error \
  --log_level INFO
```

Repeat for each plate/export.

### 3. Combine the per-plate profile tables

At v0.2.11, review this step manually or with a small audited script. The combined table must have:

```text
Metadata_Plate
Metadata_Well
Metadata_Compound or cpd_id
feature columns with matching names
one row per imaged profile or intended analysis unit
```

A native `cpatk-combine-profiles` command should be added in the next code pass.

### 4. Run one joint preprocessing pass

Do not preprocess each plate independently and then compare them unless that is an explicit sensitivity analysis. For the main cross-plate analysis, combine profiles first and then run one preprocessing workflow.

## Baseline preprocessing without plate normalisation

Run this first as a baseline:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/all_plates_merged_profiles.tsv.gz \
  --output_dir results/02_preprocess_baseline \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO
```

Use this to see the raw scale of plate effects.

## Per-plate DMSO robust-z normalisation

Use this when each plate has suitable DMSO controls:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/all_plates_merged_profiles.tsv.gz \
  --output_dir results/02_preprocess_dmso_robust_z_by_plate \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO
```

Interpretation:

```text
normalised_feature = (feature - median of DMSO controls on that plate) / MAD of DMSO controls on that plate
```

This is robust to outlier controls compared with mean/standard deviation normalisation.

## When DMSO robust-z is appropriate

Use it when:

- every plate contains enough DMSO/vehicle controls;
- controls are distributed across the plate or otherwise representative;
- DMSO controls are measured in the same imaging/acquisition conditions as treated wells;
- you want an interpretable plate-level normalisation.

Avoid or treat cautiously when:

- some plates lack controls;
- DMSO controls are in only one area of the plate;
- DMSO wells are visibly affected by drift or edge effects;
- there are too few controls to estimate MAD reliably;
- all treated compounds on a plate are from one biological class.

## Simple batch centring

CPATK can median- or mean-centre features by batch:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/all_plates_merged_profiles.tsv.gz \
  --output_dir results/02_preprocess_batch_median_centered \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --batch_centering_method median_center \
  --batch_centering_columns Metadata_Plate \
  --imputation_method median \
  --scaling_method robust \
  --log_level INFO
```

This is a sensitivity analysis, not automatically the final answer.

## ComBat-style correction

ComBat is not implemented as a first-class CPATK method in v0.2.11. It should be added only with:

- unit tests;
- clear design-matrix arguments;
- checks for confounding between batch and biology;
- enough samples per batch;
- before/after PCA/UMAP;
- before/after replicate QC;
- before/after control separation;
- an audit table showing which features were corrected.

Use ComBat only when batch and biology are not perfectly confounded. If every compound or class appears on only one plate, ComBat may remove real biology or create misleading separation.

## Recommended diagnostics after each normalisation choice

For each preprocessing strategy, run:

```bash
cpatk-batch \
  --input_table results/02_preprocess_dmso_robust_z_by_plate/preprocessed.tsv.gz \
  --output_dir results/05_batch_dmso_robust_z_by_plate \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --batch_column Metadata_Plate \
  --columns_to_test Metadata_Plate,Metadata_Batch,Metadata_MOA,Metadata_Compound \
  --log_level INFO
```

and:

```bash
cpatk-stability \
  --input_table results/02_preprocess_dmso_robust_z_by_plate/preprocessed.tsv.gz \
  --output_dir results/04_replicate_qc_dmso_robust_z_by_plate \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --replicate_group_columns Metadata_Compound,Metadata_Dose \
  --n_neighbours 10 \
  --n_bootstraps 100 \
  --n_permutations 100 \
  --log_level INFO
```

A normalisation method is more convincing when it reduces plate dominance while preserving or improving replicate consistency and known control behaviour.

## Instrument drift

Run object-level drift QC before or alongside profile building:

```bash
cpatk-drift-qc \
  --input_dir raw_cellprofiler/plate_01 \
  --output_dir results/00_drift_qc/plate_01 \
  --recursive \
  --log_level INFO
```

Use drift output to decide whether plate normalisation is enough or whether acquisition-order effects require additional modelling or experimental caution.

## Recommended next code pass

The next production code pass should add:

1. native multi-plate profile building with composite keys;
2. a `cpatk-combine-profiles` command for safely combining per-plate profile builds;
3. explicit per-plate control QC before reference normalisation;
4. optional ComBat-style correction with design checks;
5. before/after batch and replicate comparison reports.
