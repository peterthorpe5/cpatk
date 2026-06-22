# CPATK next code pass recommendations

Version: 0.2.11 documentation expansion

This document separates documentation-level guidance from code changes that should be made in the next production pass.

## Highest priority: native multi-plate profile building

Problem:

- v0.2.11 profile building aggregates object tables by `ImageNumber`.
- This is safe when `ImageNumber` is unique in the analysed export.
- It is risky when several independent CellProfiler exports are pooled and `ImageNumber` restarts for each plate/export.

Recommended code changes:

1. Add a stable internal profile key such as `Metadata_CPATK_Profile_ID`.
2. When an Image table has plate/well metadata, attach these fields before object/profile merging.
3. Aggregate object tables using a composite key when possible.
4. Add explicit CLI options:

```text
--profile_key_columns Metadata_Plate,ImageNumber
--plate_id STATIC_OR_COLUMN
--image_number_scope per_plate|global
--fail_on_reused_image_number
```

5. Add unit tests for:

```text
single plate, unique ImageNumber
multi-plate, globally unique ImageNumber
multi-plate, ImageNumber restarts and should fail unless composite keys are available
multi-plate, object table lacks plate and must be joined through Image table first
```

## Add `cpatk-combine-profiles`

A dedicated command should combine per-plate profile builds safely.

Recommended behaviour:

- accepts a manifest TSV with `plate`, `path`, optional `batch` and `notes`;
- reads profile tables;
- checks required metadata columns;
- checks duplicate profile keys;
- supports feature intersection or union with missing-value audit;
- writes a combined profile table;
- writes a combine report and log.

Example future command:

```bash
cpatk-combine-profiles \
  --manifest profile_manifest.tsv \
  --output_dir results/01_profile_build_combined \
  --feature_strategy intersection \
  --profile_id_columns Metadata_Plate,ImageNumber \
  --duplicate_policy error \
  --log_level INFO
```

## Add stronger control QC before reference normalisation

Before DMSO robust-z normalisation, CPATK should report:

- number of reference controls per plate;
- missingness of reference controls;
- MAD-zero features per plate;
- spatial distribution of controls;
- reference-control replicate correlations;
- plates lacking adequate controls.

The workflow should be able to fail or warn based on minimum control count.

## Add ComBat-style correction carefully

Recommended CLI design:

```text
--batch_correction_method none|median_center|mean_center|combat
--batch_column Metadata_Plate
--combat_covariates Metadata_MOA,Metadata_Dose
--combat_reference_column Metadata_Compound
--combat_reference_values DMSO
--combat_fail_on_confounding
```

Required reports:

```text
batch_design_report.tsv
batch_confounding_report.tsv
combat_feature_report.tsv
batch_before_after_summary.tsv
replicate_qc_before_after.tsv
```

Required tests:

- balanced two-batch synthetic data where ComBat reduces batch shift;
- confounded design where ComBat must warn or fail;
- batch with too few samples;
- missing covariate columns;
- output remains finite and same shape.

## Expand replicate QC

Recommended additions:

1. replicate correlation distribution plots;
2. per-group warning labels;
3. control-specific replicate QC;
4. replicate-centroid distance summaries;
5. per-plate replicate QC;
6. before/after normalisation comparison;
7. replicate-aware nearest-neighbour summaries;
8. group-aware cross-validation for ML and MOA classification.

## Add a full synthetic multi-plate CellProfiler fixture

This should include:

```text
plate_01_Image.tsv
plate_01_Cells.tsv
plate_01_Nuclei.tsv
plate_02_Image.tsv
plate_02_Cells.tsv
plate_02_Nuclei.tsv
metadata with assay wells and source wells
compound annotations by source well
DMSO controls on each plate
positive controls on each plate
repeated reference compounds across plates
ImageNumber restarting per plate
```

The tests should cover:

- metadata validation;
- profile building;
- duplicate ImageNumber risk detection;
- per-plate build and safe combine;
- DMSO robust-z by plate;
- batch diagnostics;
- replicate QC.

## Documentation updates to keep

The expanded documentation in this pass should be retained and updated after the next code pass:

```text
docs/CPATK_USER_GUIDE.md
docs/CPATK_METHOD_SELECTION_GUIDE.md
docs/CPATK_METADATA_AND_ANNOTATION_GUIDE.md
docs/CPATK_MULTI_PLATE_NORMALISATION_AND_BATCH_GUIDE.md
docs/CPATK_REPLICATE_QC_GUIDE.md
docs/CPATK_CLIPN_GUIDE.md
```
