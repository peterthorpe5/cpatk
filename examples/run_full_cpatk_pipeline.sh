#!/usr/bin/env bash
set -euo pipefail

# CPATK full example workflow
#
# This script assumes you have already installed CPATK, for example:
#   python -m pip install -e .[all]
#
# It runs a defensible start-to-finish workflow from a folder of CellProfiler
# exports and a plate-map/metadata file.

CELLPROFILER_DIR="${CELLPROFILER_DIR:-/path/to/cellprofiler_exports}"
METADATA_TSV="${METADATA_TSV:-examples/example_metadata.tsv}"
OUT_DIR="${OUT_DIR:-cpatk_results}"
ID_COLUMN="${ID_COLUMN:-Metadata_Compound}"
MOA_COLUMN="${MOA_COLUMN:-Metadata_MOA}"
COLOUR_COLUMNS="${COLOUR_COLUMNS:-Metadata_Compound,Metadata_MOA,cpd_type,Metadata_Batch}"
METADATA_COLUMNS="${METADATA_COLUMNS:-Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Dose,Metadata_Batch,cpd_type,Replicate,Donor,Timepoint}"

mkdir -p "${OUT_DIR}"

echo "Step 1: object-level per-compartment acquisition-drift QC"
cpatk-drift-qc \
  --input_dir "${CELLPROFILER_DIR}" \
  --output_dir "${OUT_DIR}/00_drift_qc" \
  --image_col ImageNumber \
  --max_features 200 \
  --plot_top_n 8 \
  --log_level INFO

echo "Step 2: merge CellProfiler folder into profile table"
cpatk-build-profiles \
  --input_dir "${CELLPROFILER_DIR}" \
  --output_dir "${OUT_DIR}/01_profile_build" \
  --recursive \
  --metadata_table "${METADATA_TSV}" \
  --aggregate_statistic median \
  --log_level INFO

PROFILE_TABLE="${OUT_DIR}/01_profile_build/merged_profiles.parquet"
if [ ! -f "${PROFILE_TABLE}" ]; then
  PROFILE_TABLE="${OUT_DIR}/01_profile_build/merged_profiles.tsv.gz"
fi

echo "Step 3: preprocess merged profiles with imputation"
cpatk-preprocess \
  --input_table "${PROFILE_TABLE}" \
  --output_dir "${OUT_DIR}/02_preprocess" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO

PREPROCESSED="${OUT_DIR}/02_preprocess/preprocessed.parquet"
if [ ! -f "${PREPROCESSED}" ]; then
  PREPROCESSED="${OUT_DIR}/02_preprocess/preprocessed.tsv.gz"
fi

echo "Step 4: classical PCA/UMAP/distances/neighbours/clustering"
cpatk-classical \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/03_classical" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --id_column "${ID_COLUMN}" \
  --colour_column "${MOA_COLUMN}" \
  --cluster_group_columns "${MOA_COLUMN},${ID_COLUMN}" \
  --distance_metric cosine \
  --n_neighbours 15 \
  --n_clusters 8 \
  --log_level INFO

echo "Step 5: cluster/neighbour stability"
cpatk-stability \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/04_stability" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --replicate_group_columns "${ID_COLUMN},${MOA_COLUMN}" \
  --n_clusters 8 \
  --k_values 2,3,4,5,6,7,8,9,10 \
  --n_bootstraps 100 \
  --n_permutations 100 \
  --log_level INFO

echo "Step 6: batch-effect checks"
cpatk-batch \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/05_batch" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --batch_column Metadata_Batch \
  --log_level INFO

echo "Step 7: visualisation report: PCA, UMAP, heatmap and topology"
cpatk-visualise \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/06_visualisation" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --id_column "${ID_COLUMN}" \
  --colour_columns "${COLOUR_COLUMNS}" \
  --aggregate_by_id \
  --aggregate_method median \
  --log_level INFO

echo "Step 8: MOA prediction and pseudo-anchor scoring"
cpatk-moa \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/07_moa" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --id_column "${ID_COLUMN}" \
  --class_column "${MOA_COLUMN}" \
  --make_pseudo_anchors \
  --pseudo_anchor_method bootstrap \
  --k_values 8,12,16,24,32 \
  --n_bootstraps 50 \
  --aggregate_method median \
  --centroid_method median \
  --adaptive_shrinkage \
  --score_method cosine \
  --n_permutations 100 \
  --make_projection_plots \
  --projection both \
  --interactive \
  --log_level INFO

echo "Step 9: optional ML classifier benchmark"
cpatk-ml \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/08_ml" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --class_column "${MOA_COLUMN}" \
  --log_level INFO

echo "Step 10: optional feature attribution"
cpatk-explain \
  --input_table "${PREPROCESSED}" \
  --output_dir "${OUT_DIR}/09_explain" \
  --metadata_columns "${METADATA_COLUMNS}" \
  --class_column "${MOA_COLUMN}" \
  --model_name random_forest \
  --n_repeats 20 \
  --include_shap \
  --log_level INFO

echo "Step 11: nearest-neighbour plots"
if [ -f "${OUT_DIR}/03_classical/nearest_neighbours.tsv" ]; then
  cpatk-neighbours \
    --input_neighbours "${OUT_DIR}/03_classical/nearest_neighbours.tsv" \
    --output_dir "${OUT_DIR}/10_neighbours" \
    --compounds "DMSO,Compound_001" \
    --top_n 10 \
    --log_level INFO
fi

echo "Step 12: final HTML report"
cpatk-report \
  --output_html "${OUT_DIR}/CPATK_full_analysis_report.html" \
  --title "CPATK full Cell Painting analysis report" \
  --table "preprocessing=${OUT_DIR}/02_preprocess/preprocessing_summary.tsv" \
  --table "stability=${OUT_DIR}/04_stability/stability_summary.tsv" \
  --table "moa=${OUT_DIR}/07_moa/moa_prediction_summary.tsv" \
  --plot "${OUT_DIR}/02_preprocess/plots/preprocessing_retention_summary.svg" \
  --plot "${OUT_DIR}/06_visualisation/plots/profile_feature_heatmap.svg" \
  --narrative "Start-to-finish CPATK analysis from CellProfiler folder and metadata file." \
  --log_level INFO

echo "CPATK workflow complete: ${OUT_DIR}"
