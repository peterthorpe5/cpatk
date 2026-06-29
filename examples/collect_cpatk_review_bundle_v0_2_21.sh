#!/usr/bin/env bash
# Collect a compact CPATK review bundle for upload.
#
# This script copies the main CPATK logs, reports and summary tables from
# malaria and/or mitotox v0.2.18/v0.2.21 stress-test runs into a small review folder,
# then compresses that folder as a .tar.gz archive.
#
# Usage examples:
#   bash collect_cpatk_review_bundle.sh
#   MALARIA_DIR=/path/to/malaria/results MITOTOX_DIR=/path/to/mitotox/results bash collect_cpatk_review_bundle.sh
#   MAX_NN_ROWS=5000 bash collect_cpatk_review_bundle.sh
#
# The script is deliberately conservative: it avoids large matrices by default
# and truncates large nearest-neighbour/SHAP tables into preview files.

set -Eeuo pipefail
IFS=$'\n\t'

############################################
# User-editable defaults
############################################

MALARIA_BASE="${MALARIA_BASE:-/home/pthorpe001/data/2025_jason_cell_painting/data/malaria}"
MITOTOX_BASE="${MITOTOX_BASE:-/home/pthorpe001/data/2025_jason_cell_painting/data/mitotox}"

# Override these if the latest folder is not the one you want.
MALARIA_DIR="${MALARIA_DIR:-}"
MITOTOX_DIR="${MITOTOX_DIR:-}"

# Override these if the qsub log names differ.
MALARIA_LOG="${MALARIA_LOG:-}"
MITOTOX_LOG="${MITOTOX_LOG:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PWD}/cpatk_review_bundle_${RUN_TAG}}"
ARCHIVE="${ARCHIVE:-${OUT_ROOT}.tar.gz}"

# Keep previews small enough for upload and inspection.
MAX_TABLE_ROWS="${MAX_TABLE_ROWS:-5000}"
MAX_NN_ROWS="${MAX_NN_ROWS:-5000}"
MAX_SHAP_ROWS="${MAX_SHAP_ROWS:-5000}"

# Set to 1 if you really want latent/profile tables included. Default is off
# because these can make the archive huge.
INCLUDE_LARGE_MATRICES="${INCLUDE_LARGE_MATRICES:-0}"

############################################
# Helpers
############################################

section() {
  printf '\n==== %s ====\n\n' "$*"
}

warn() {
  echo "WARN: $*" >&2
}

find_latest_dir() {
  local base_dir="$1"
  local pattern="$2"
  find "${base_dir}" -maxdepth 1 -type d -name "${pattern}" -printf '%T@\t%p\n' 2>/dev/null \
    | sort -nr \
    | awk -F '\t' 'NR == 1 {print $2}'
}

find_latest_log() {
  local base_dir="$1"
  local pattern="$2"
  find "${base_dir}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@\t%p\n' 2>/dev/null \
    | sort -nr \
    | awk -F '\t' 'NR == 1 {print $2}'
}

copy_if_present() {
  local source_file="$1"
  local target_file="$2"
  if [[ -s "${source_file}" ]]; then
    mkdir -p "$(dirname "${target_file}")"
    cp "${source_file}" "${target_file}"
    echo "copied\t${source_file}\t${target_file}" >> "${MANIFEST}"
  else
    echo "missing\t${source_file}\t${target_file}" >> "${MANIFEST}"
  fi
}

copy_glob_if_present() {
  local source_glob="$1"
  local target_dir="$2"
  local found=0
  local file
  shopt -s nullglob
  for file in ${source_glob}; do
    found=1
    mkdir -p "${target_dir}"
    cp "${file}" "${target_dir}/"
    echo "copied\t${file}\t${target_dir}/$(basename "${file}")" >> "${MANIFEST}"
  done
  shopt -u nullglob
  if [[ "${found}" -eq 0 ]]; then
    echo "missing_glob\t${source_glob}\t${target_dir}" >> "${MANIFEST}"
  fi
}

copy_table_preview() {
  local source_file="$1"
  local target_file="$2"
  local max_rows="$3"
  if [[ ! -s "${source_file}" ]]; then
    echo "missing\t${source_file}\t${target_file}" >> "${MANIFEST}"
    return 0
  fi
  mkdir -p "$(dirname "${target_file}")"
  case "${source_file}" in
    *.gz)
      gzip -cd "${source_file}" | awk -v max_rows="${max_rows}" 'NR <= max_rows + 1 {print}' > "${target_file%.gz}.preview.tsv"
      echo "preview\t${source_file}\t${target_file%.gz}.preview.tsv" >> "${MANIFEST}"
      ;;
    *.parquet)
      echo "skipped_large_or_binary\t${source_file}\t${target_file}" >> "${MANIFEST}"
      ;;
    *)
      awk -v max_rows="${max_rows}" 'NR <= max_rows + 1 {print}' "${source_file}" > "${target_file}"
      echo "preview\t${source_file}\t${target_file}" >> "${MANIFEST}"
      ;;
  esac
}

copy_dataset_outputs() {
  local label="$1"
  local result_dir="$2"
  local qsub_log="$3"
  local dest_root="${OUT_ROOT}/${label}"

  section "Collect ${label}"

  if [[ -z "${result_dir}" || ! -d "${result_dir}" ]]; then
    warn "No result directory found for ${label}; skipping."
    echo "missing_result_dir\t${label}\t${result_dir}" >> "${MANIFEST}"
    return 0
  fi

  echo "Using ${label} result dir: ${result_dir}"
  mkdir -p "${dest_root}"

  cat > "${dest_root}/source_paths.tsv" <<PATHS
item\tpath
result_dir\t${result_dir}
qsub_log\t${qsub_log}
PATHS

  if [[ -n "${qsub_log}" && -s "${qsub_log}" ]]; then
    copy_if_present "${qsub_log}" "${dest_root}/qsub_log/$(basename "${qsub_log}")"
  fi

  # Run-level files and reports.
  copy_if_present "${result_dir}/run_configuration.tsv" "${dest_root}/run_configuration.tsv"
  copy_glob_if_present "${result_dir}/CPATK_*_full_report.html" "${dest_root}/html_reports"
  copy_glob_if_present "${result_dir}/CPATK_*_report.html" "${dest_root}/html_reports"

  # Metadata validation.
  copy_if_present "${result_dir}/00_metadata_validation/metadata_validation_summary.tsv" "${dest_root}/00_metadata_validation/metadata_validation_summary.tsv"
  copy_if_present "${result_dir}/00_metadata_validation/metadata_key_validation.tsv" "${dest_root}/00_metadata_validation/metadata_key_validation.tsv"
  copy_if_present "${result_dir}/00_metadata_validation/metadata_merge_report.tsv" "${dest_root}/00_metadata_validation/metadata_merge_report.tsv"
  copy_if_present "${result_dir}/00_metadata_validation/metadata_prepare_summary.tsv" "${dest_root}/00_metadata_validation/metadata_prepare_summary.tsv"
  copy_if_present "${result_dir}/00_metadata_validation/metadata_annotation_merge_report.tsv" "${dest_root}/00_metadata_validation/metadata_annotation_merge_report.tsv"

  # Profile building.
  copy_if_present "${result_dir}/02_profile_build/profile_build_summary.tsv" "${dest_root}/02_profile_build/profile_build_summary.tsv"
  copy_if_present "${result_dir}/02_profile_build/metadata_merge_report.tsv" "${dest_root}/02_profile_build/metadata_merge_report.tsv"
  copy_if_present "${result_dir}/02_profile_build/object_aggregation_report.tsv
object_key_propagation_report.tsv" "${dest_root}/02_profile_build/object_aggregation_report.tsv
object_key_propagation_report.tsv"
  copy_if_present "${result_dir}/02_profile_build/input_table_inventory.tsv" "${dest_root}/02_profile_build/input_table_inventory.tsv"
  copy_if_present "${result_dir}/02_profile_build/retained_profile_features.tsv" "${dest_root}/02_profile_build/retained_profile_features.tsv"

  # Preprocessing strategies: copy summaries from all completed strategies.
  local strategy_dir strategy_name
  shopt -s nullglob
  for strategy_dir in "${result_dir}"/03_preprocess_strategy_comparison/*; do
    [[ -d "${strategy_dir}" ]] || continue
    strategy_name="$(basename "${strategy_dir}")"
    copy_if_present "${strategy_dir}/preprocessing_summary.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/preprocessing_summary.tsv"
    copy_if_present "${strategy_dir}/final_matrix_validation.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/final_matrix_validation.tsv"
    copy_if_present "${strategy_dir}/control_qc_before_normalisation.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/control_qc_before_normalisation.tsv"
    copy_if_present "${strategy_dir}/reference_control_qc_before_normalisation.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/reference_control_qc_before_normalisation.tsv"
    copy_if_present "${strategy_dir}/before_after_replicate_summary.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/before_after_replicate_summary.tsv"
    copy_if_present "${strategy_dir}/before_after_batch_pc_association.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/before_after_batch_pc_association.tsv"
    copy_if_present "${strategy_dir}/excel_export_notes.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/excel_export_notes.tsv"
    copy_if_present "${strategy_dir}/feature_qc_report.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/feature_qc_report.tsv"
    copy_if_present "${strategy_dir}/row_qc_report.tsv" "${dest_root}/03_preprocess_strategy_comparison/${strategy_name}/row_qc_report.tsv"
  done

  # Classical reports and summaries.
  for strategy_dir in "${result_dir}"/04_classical/*; do
    [[ -d "${strategy_dir}" ]] || continue
    strategy_name="$(basename "${strategy_dir}")"
    copy_if_present "${strategy_dir}/classical_analysis_report.html" "${dest_root}/04_classical/${strategy_name}/classical_analysis_report.html"
    copy_if_present "${strategy_dir}/pca_explained_variance.tsv" "${dest_root}/04_classical/${strategy_name}/pca_explained_variance.tsv"
    copy_if_present "${strategy_dir}/cluster_summary.tsv" "${dest_root}/04_classical/${strategy_name}/cluster_summary.tsv"
    copy_if_present "${strategy_dir}/cluster_silhouette_summary.tsv" "${dest_root}/04_classical/${strategy_name}/cluster_silhouette_summary.tsv"
    copy_table_preview "${strategy_dir}/nearest_neighbours.tsv" "${dest_root}/04_classical/${strategy_name}/nearest_neighbours.preview.tsv" "${MAX_NN_ROWS}"
    copy_glob_if_present "${strategy_dir}/*.png" "${dest_root}/04_classical/${strategy_name}/plots"
    copy_glob_if_present "${strategy_dir}/*.svg" "${dest_root}/04_classical/${strategy_name}/plots"
    copy_glob_if_present "${strategy_dir}/*.html" "${dest_root}/04_classical/${strategy_name}/html_outputs"
  done

  # Visualisation, stability and batch summaries/reports.
  copy_glob_if_present "${result_dir}/05_visualise/*/*.html" "${dest_root}/05_visualise/html_outputs"
  copy_glob_if_present "${result_dir}/05_visualise/*/*.png" "${dest_root}/05_visualise/plots"
  copy_glob_if_present "${result_dir}/06_stability/*/*.html" "${dest_root}/06_stability/html_outputs"
  copy_glob_if_present "${result_dir}/06_stability/*/*.tsv" "${dest_root}/06_stability/tables"
  copy_glob_if_present "${result_dir}/07_batch/*/*.html" "${dest_root}/07_batch/html_outputs"
  copy_glob_if_present "${result_dir}/07_batch/*/*.tsv" "${dest_root}/07_batch/tables"

  # MOA summaries.
  copy_if_present "${result_dir}/09_moa/advanced_moa_top_predictions.tsv" "${dest_root}/09_moa/advanced_moa_top_predictions.tsv"
  copy_if_present "${result_dir}/09_moa/pseudo_anchor_summary.tsv" "${dest_root}/09_moa/pseudo_anchor_summary.tsv"
  copy_if_present "${result_dir}/09_moa/pseudo_anchor_phenotype_summary.tsv" "${dest_root}/09_moa/pseudo_anchor_phenotype_summary.tsv"
  copy_if_present "${result_dir}/09_moa/pseudo_anchor_phenotype_label_audit.tsv" "${dest_root}/09_moa/pseudo_anchor_phenotype_label_audit.tsv"
  copy_glob_if_present "${result_dir}/09_moa/*.html" "${dest_root}/09_moa/html_outputs"
  copy_glob_if_present "${result_dir}/09_moa/*.png" "${dest_root}/09_moa/plots"

  # Explainability summaries and per-query reports. Keep previews only for big tables.
  copy_if_present "${result_dir}/11_explain/explainability_summary.tsv" "${dest_root}/11_explain/explainability_summary.tsv"
  copy_if_present "${result_dir}/11_explain/query_background_shap_summary.tsv" "${dest_root}/11_explain/query_background_shap_summary.tsv"
  copy_glob_if_present "${result_dir}/11_explain/query_neighbourhoods/*/query_explanation_report.html" "${dest_root}/11_explain/query_reports"
  copy_glob_if_present "${result_dir}/11_explain/query_neighbourhoods/*/*.png" "${dest_root}/11_explain/query_plots"
  copy_glob_if_present "${result_dir}/11_explain/query_background_shap/*.html" "${dest_root}/11_explain/query_background_shap_html"
  copy_glob_if_present "${result_dir}/11_explain/query_background_shap/*.png" "${dest_root}/11_explain/query_background_shap_plots"
  copy_glob_if_present "${result_dir}/11_explain/query_background_shap/*summary*.tsv" "${dest_root}/11_explain/query_background_shap_tables"
  for table in "${result_dir}"/11_explain/query_background_shap/*.tsv; do
    [[ -s "${table}" ]] || continue
    copy_table_preview "${table}" "${dest_root}/11_explain/query_background_shap_tables/$(basename "${table%.tsv}").preview.tsv" "${MAX_SHAP_ROWS}"
  done

  # CLIPn summaries and latent-space MOA summaries. Avoid latent matrices unless requested.
  copy_if_present "${result_dir}/12_clipn/clipn_status.tsv" "${dest_root}/12_clipn/clipn_status.tsv"
  copy_if_present "${result_dir}/12_clipn/clipn_run_status.tsv" "${dest_root}/12_clipn/clipn_run_status.tsv"
  copy_if_present "${result_dir}/12_clipn/clipn_backend_provenance.tsv" "${dest_root}/12_clipn/clipn_backend_provenance.tsv"
  copy_if_present "${result_dir}/normalisation_strategy_comparison.tsv" "${dest_root}/normalisation_strategy_comparison.tsv"
  copy_if_present "${result_dir}/method_guides/ml_nn_method_guide.tsv" "${dest_root}/method_guides/ml_nn_method_guide.tsv"
  copy_if_present "${result_dir}/method_guides/ML_NN_METHOD_GUIDE.md" "${dest_root}/method_guides/ML_NN_METHOD_GUIDE.md"
  copy_if_present "${result_dir}/12_clipn/clipn_preprocessing_summary.tsv" "${dest_root}/12_clipn/clipn_preprocessing_summary.tsv"
  copy_if_present "${result_dir}/12_clipn/clipn_feature_summary.tsv" "${dest_root}/12_clipn/clipn_feature_summary.tsv"
  copy_if_present "${result_dir}/12_clipn/clipn_feature_report.tsv" "${dest_root}/12_clipn/clipn_feature_report.tsv"
  copy_if_present "${result_dir}/12_clipn/clipn_label_report.tsv" "${dest_root}/12_clipn/clipn_label_report.tsv"
  copy_if_present "${result_dir}/12_clipn/latent_diagnostic_summary.tsv" "${dest_root}/12_clipn/latent_diagnostic_summary.tsv"
  copy_if_present "${result_dir}/12_clipn/pca_fallback_explained_variance.tsv" "${dest_root}/12_clipn/pca_fallback_explained_variance.tsv"
  copy_if_present "${result_dir}/12_clipn/latent_variance.tsv" "${dest_root}/12_clipn/latent_variance.tsv"
  copy_table_preview "${result_dir}/12_clipn/nearest_neighbours.tsv" "${dest_root}/12_clipn/nearest_neighbours.preview.tsv" "${MAX_NN_ROWS}"
  copy_glob_if_present "${result_dir}/12_clipn/*.html" "${dest_root}/12_clipn/html_outputs"
  copy_glob_if_present "${result_dir}/12_clipn/plots/*.svg" "${dest_root}/12_clipn/plots"
  copy_glob_if_present "${result_dir}/12_clipn/plots/*.png" "${dest_root}/12_clipn/plots"
  copy_if_present "${result_dir}/13_clipn_latent_moa/advanced_moa_top_predictions.tsv" "${dest_root}/13_clipn_latent_moa/advanced_moa_top_predictions.tsv"
  copy_if_present "${result_dir}/13_clipn_latent_moa/pseudo_anchor_summary.tsv" "${dest_root}/13_clipn_latent_moa/pseudo_anchor_summary.tsv"
  copy_if_present "${result_dir}/13_clipn_latent_moa/pseudo_anchor_phenotype_summary.tsv" "${dest_root}/13_clipn_latent_moa/pseudo_anchor_phenotype_summary.tsv"
  copy_if_present "${result_dir}/13_clipn_latent_moa/pseudo_anchor_phenotype_label_audit.tsv" "${dest_root}/13_clipn_latent_moa/pseudo_anchor_phenotype_label_audit.tsv"
  copy_if_present "${result_dir}/13_clipn_latent_moa/pseudo_anchor_k_selection.tsv" "${dest_root}/13_clipn_latent_moa/pseudo_anchor_k_selection.tsv"
  copy_glob_if_present "${result_dir}/13_clipn_latent_moa/*.html" "${dest_root}/13_clipn_latent_moa/html_outputs"
  if [[ "${INCLUDE_LARGE_MATRICES}" == "1" ]]; then
    copy_glob_if_present "${result_dir}/12_clipn/clipn_latent.*" "${dest_root}/12_clipn/clipn_latent"
    copy_glob_if_present "${result_dir}/12_clipn/latent_profiles.*" "${dest_root}/12_clipn/latent_profiles"
  fi
  shopt -u nullglob
}

############################################
# Main
############################################

mkdir -p "${OUT_ROOT}"
MANIFEST="${OUT_ROOT}/collection_manifest.tsv"
echo -e "status\tsource\tdestination" > "${MANIFEST}"

if [[ -z "${MALARIA_DIR}" ]]; then
  MALARIA_DIR="$(find_latest_dir "${MALARIA_BASE}" 'cpatk_v0_2_*_malaria_fast_*')"
fi
if [[ -z "${MITOTOX_DIR}" ]]; then
  MITOTOX_DIR="$(find_latest_dir "${MITOTOX_BASE}" 'cpatk_v0_2_*_mitotox_fast_*')"
fi
if [[ -z "${MALARIA_LOG}" ]]; then
  MALARIA_LOG="$(find_latest_log "${MALARIA_BASE}" 'cpatk_malaria_fast.o*')"
fi
if [[ -z "${MITOTOX_LOG}" ]]; then
  MITOTOX_LOG="$(find_latest_log "${MITOTOX_BASE}" 'cpatk_mitotox_fast.o*')"
fi

cat > "${OUT_ROOT}/collection_summary.tsv" <<SUMMARY
item\tvalue
run_tag\t${RUN_TAG}
malaria_base\t${MALARIA_BASE}
mitotox_base\t${MITOTOX_BASE}
malaria_dir\t${MALARIA_DIR}
mitotox_dir\t${MITOTOX_DIR}
malaria_log\t${MALARIA_LOG}
mitotox_log\t${MITOTOX_LOG}
max_table_rows\t${MAX_TABLE_ROWS}
max_nn_rows\t${MAX_NN_ROWS}
max_shap_rows\t${MAX_SHAP_ROWS}
include_large_matrices\t${INCLUDE_LARGE_MATRICES}
SUMMARY

copy_dataset_outputs "malaria" "${MALARIA_DIR}" "${MALARIA_LOG}"
copy_dataset_outputs "mitotox" "${MITOTOX_DIR}" "${MITOTOX_LOG}"

section "Compress review bundle"
tar -czf "${ARCHIVE}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"

section "Done"
ls -lh "${ARCHIVE}"
echo "Archive written to: ${ARCHIVE}"
echo "Manifest: ${MANIFEST}"
echo "If the archive is too large, rerun with smaller MAX_NN_ROWS/MAX_SHAP_ROWS or INCLUDE_LARGE_MATRICES=0."
