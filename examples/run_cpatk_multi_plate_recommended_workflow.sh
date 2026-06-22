#!/usr/bin/env bash
set -euo pipefail

# CPATK multi-plate recommended workflow, v0.2.12.
# This shell intentionally keeps the current safest workflow explicit:
#   1. validate metadata once;
#   2. build profiles per plate/export;
#   3. combine per-plate profiles after review;
#   4. run one joint preprocessing pass with optional per-plate DMSO normalisation;
#   5. run replicate QC and batch diagnostics.
#
# Note: v0.2.12 does not yet include native cpatk-combine-profiles. If your
# CellProfiler ImageNumber restarts per plate/export, do not pool raw exports
# into one input folder and rely on ImageNumber alone.

: "${PROJECT_DIR:?Set PROJECT_DIR to the project root}"
: "${METADATA_TABLE:?Set METADATA_TABLE to the raw metadata table}"
: "${ANNOTATION_TABLE:?Set ANNOTATION_TABLE to the compound annotation table}"
: "${PLATE01_DIR:?Set PLATE01_DIR to the CellProfiler export folder for plate 1}"
: "${PLATE02_DIR:?Set PLATE02_DIR to the CellProfiler export folder for plate 2}"

OUT_DIR="${PROJECT_DIR}/results"
mkdir -p "${OUT_DIR}"

cpatk-metadata \
  --metadata_table "${METADATA_TABLE}" \
  --output_dir "${OUT_DIR}/00_metadata_check" \
  --plate_column Assay_Plate_Barcode \
  --well_column Destination_Well \
  --source_plate_column Source_Plate_Barcode \
  --source_well_column Source_Well \
  --annotation_tables "${ANNOTATION_TABLE}" \
  --annotation_source_plate_column Barcode \
  --annotation_source_well_column Well \
  --merge_keys Metadata_Source_Plate,Metadata_Source_Well \
  --duplicate_policy error \
  --log_level INFO

cpatk-build-profiles \
  --input_dir "${PLATE01_DIR}" \
  --output_dir "${OUT_DIR}/01_profile_build/plate_01" \
  --recursive \
  --metadata_table "${OUT_DIR}/00_metadata_check/formatted_metadata.tsv" \
  --aggregate_statistic median \
  --image_merge_keys Metadata_Plate,ImageNumber \
  --duplicate_image_policy error \
  --metadata_duplicate_policy error \
  --log_level INFO

cpatk-build-profiles \
  --input_dir "${PLATE02_DIR}" \
  --output_dir "${OUT_DIR}/01_profile_build/plate_02" \
  --recursive \
  --metadata_table "${OUT_DIR}/00_metadata_check/formatted_metadata.tsv" \
  --aggregate_statistic median \
  --image_merge_keys Metadata_Plate,ImageNumber \
  --duplicate_image_policy error \
  --metadata_duplicate_policy error \
  --log_level INFO

PLATE01_PROFILE="${OUT_DIR}/01_profile_build/plate_01/merged_profiles.tsv.gz"
if [[ ! -f "${PLATE01_PROFILE}" ]]; then
  PLATE01_PROFILE="${OUT_DIR}/01_profile_build/plate_01/merged_profiles.parquet"
fi
PLATE02_PROFILE="${OUT_DIR}/01_profile_build/plate_02/merged_profiles.tsv.gz"
if [[ ! -f "${PLATE02_PROFILE}" ]]; then
  PLATE02_PROFILE="${OUT_DIR}/01_profile_build/plate_02/merged_profiles.parquet"
fi

cpatk-combine-profiles \
  --profile_tables "${PLATE01_PROFILE},${PLATE02_PROFILE}" \
  --output_dir "${OUT_DIR}/01_profile_build/all_plates" \
  --source_labels plate_01,plate_02 \
  --key_columns Metadata_Plate,Metadata_Well \
  --feature_join union \
  --duplicate_policy error \
  --log_level INFO

COMBINED_PROFILE_TABLE="${OUT_DIR}/01_profile_build/all_plates/combined_profiles.tsv.gz"

cpatk-preprocess \
  --input_table "${COMBINED_PROFILE_TABLE}" \
  --output_dir "${OUT_DIR}/02_preprocess_dmso_robust_z_by_plate" \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Dose,Metadata_Batch,cpd_type \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate \
  --batch_correction_method combat_location_scale \
  --batch_column Metadata_Plate \
  --batch_protect_columns Metadata_Compound,Metadata_MOA,Metadata_Dose \
  --replicate_group_columns Metadata_Compound,Metadata_Dose \
  --batch_report_columns Metadata_Plate,Metadata_Batch \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO

PREPROCESSED="${OUT_DIR}/02_preprocess_dmso_robust_z_by_plate/preprocessed.tsv.gz"
if [[ ! -f "${PREPROCESSED}" ]]; then
  PREPROCESSED="${OUT_DIR}/02_preprocess_dmso_robust_z_by_plate/preprocessed.parquet"
fi

cpatk-stability \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/04_replicate_qc_and_stability" \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Dose,Metadata_Batch,cpd_type \
  --replicate_group_columns Metadata_Compound,Metadata_Dose \
  --n_neighbours 10 \
  --n_bootstraps 100 \
  --n_permutations 100 \
  --k_values 2,3,4,5,6,7,8,9,10 \
  --log_level INFO

cpatk-batch \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/05_batch" \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Dose,Metadata_Batch,cpd_type \
  --batch_column Metadata_Plate \
  --columns_to_test Metadata_Plate,Metadata_Batch,Metadata_MOA,Metadata_Compound \
  --log_level INFO
