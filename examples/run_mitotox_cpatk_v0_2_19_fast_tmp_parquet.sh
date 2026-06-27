#!/usr/bin/env bash
#$ -jc rhel9
#$ -j y
#$ -N cpatk_mitotox_fast
#$ -pe smp 16
#$ -mods l_hard mfree 240G
#$ -adds l_hard h_vmem 240G
#$ -cwd
# For a real GPU-backed CLIPn backend, uncomment the two lines below before qsub.
##$ -adds l_hard gpu 1
##$ -adds l_hard cuda.0.name 'NVIDIA A40'

# CPATK v0.2.19 fast mitotox Cell Painting stress-test workflow.
#
# This is designed as a broad validation/stress test rather than a final
# biological analysis. It stages inputs to job-local scratch, prefers Parquet
# for downstream tables, runs the core CPATK suite, and treats most downstream
# interpretation layers as soft-fail optional stages so a single optional method
# does not prevent useful QC outputs being copied back.

set -Eeuo pipefail
IFS=$'\n\t'

############################################
# User-editable configuration
############################################

BASE_DIR="${BASE_DIR:-/home/pthorpe001/data/2025_jason_cell_painting/data/mitotox}"
RAW_DIR="${RAW_DIR:-${BASE_DIR}/raw}"
CLEANED_METADATA="${CLEANED_METADATA:-${BASE_DIR}/metadata/KVP_MitotoxPlate_IXM_07042025_cleaned.csv}"
RAW_METADATA="${RAW_METADATA:-${BASE_DIR}/metadata/KVP_MitotoxPlate_IXM_07042025.csv}"
PREMERGED_PROFILE_TABLE="${PREMERGED_PROFILE_TABLE:-${BASE_DIR}/mitotox_all_plates_image_level.tsv}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_OUT_DIR="${PROJECT_OUT_DIR:-${OUT_DIR:-${BASE_DIR}/cpatk_v0_2_19_mitotox_fast_${RUN_TAG}}}"

USE_LOCAL_SCRATCH="${USE_LOCAL_SCRATCH:-1}"
COPY_INPUTS_TO_SCRATCH="${COPY_INPUTS_TO_SCRATCH:-1}"
COPY_BACK_ON_EXIT="${COPY_BACK_ON_EXIT:-1}"
KEEP_SCRATCH="${KEEP_SCRATCH:-0}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp/${USER:-cpatk}}}"
WORK_ROOT=""
WORK_OUT_DIR=""
OUT_DIR="${PROJECT_OUT_DIR}"

THREADS="${THREADS:-16}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
FORCE_RERUN="${FORCE_RERUN:-0}"
ALLOW_OPTIONAL_FAILURES="${ALLOW_OPTIONAL_FAILURES:-1}"

# Profile building key choice.
TRY_COMPOSITE_KEYS="${TRY_COMPOSITE_KEYS:-1}"
ALLOW_IMAGENUMBER_FALLBACK="${ALLOW_IMAGENUMBER_FALLBACK:-1}"
ALLOW_PREMERGED_PROFILE_FALLBACK="${ALLOW_PREMERGED_PROFILE_FALLBACK:-1}"
COMPOSITE_IMAGE_MERGE_KEYS="${COMPOSITE_IMAGE_MERGE_KEYS:-Metadata_Plate,ImageNumber}"
FALLBACK_IMAGE_MERGE_KEYS="${FALLBACK_IMAGE_MERGE_KEYS:-ImageNumber}"

# Method toggles. Keep these as 1 for a full stress test. Set selected values to 0
# for a faster smoke test.
RUN_METADATA="${RUN_METADATA:-1}"
RUN_INSPECT="${RUN_INSPECT:-1}"
RUN_DRIFT_QC="${RUN_DRIFT_QC:-1}"
RUN_PROFILE_BUILD="${RUN_PROFILE_BUILD:-1}"
RUN_PREPROCESSING="${RUN_PREPROCESSING:-1}"
RUN_CLASSICAL="${RUN_CLASSICAL:-1}"
RUN_VISUALISE="${RUN_VISUALISE:-1}"
RUN_STABILITY="${RUN_STABILITY:-1}"
RUN_BATCH="${RUN_BATCH:-1}"
RUN_NEIGHBOURS="${RUN_NEIGHBOURS:-1}"
RUN_MOA="${RUN_MOA:-1}"
RUN_ML="${RUN_ML:-1}"
RUN_EXPLAIN="${RUN_EXPLAIN:-1}"
RUN_CLIPN="${RUN_CLIPN:-1}"
RUN_CLIPN_LATENT_MOA="${RUN_CLIPN_LATENT_MOA:-1}"
RUN_PCA_FALLBACK_LATENT_MOA="${RUN_PCA_FALLBACK_LATENT_MOA:-0}"
RUN_FINAL_REPORT="${RUN_FINAL_REPORT:-1}"

# CLIPn settings.
CLIPN_LATENT_DIM="${CLIPN_LATENT_DIM:-10}"
CLIPN_EPOCHS="${CLIPN_EPOCHS:-80}"
CLIPN_STRICT_DROP_ANY_ZERO="${CLIPN_STRICT_DROP_ANY_ZERO:-0}"
CLIPN_ZERO_POLICY="${CLIPN_ZERO_POLICY:-keep}"
CLIPN_ALLOW_PCA_FALLBACK="${CLIPN_ALLOW_PCA_FALLBACK:-1}"

# Lightweight first-pass settings. Increase for final analyses.
N_CLUSTERS="${N_CLUSTERS:-8}"
N_NEIGHBOURS="${N_NEIGHBOURS:-15}"
STABILITY_BOOTSTRAPS="${STABILITY_BOOTSTRAPS:-20}"
STABILITY_PERMUTATIONS="${STABILITY_PERMUTATIONS:-20}"
MOA_BOOTSTRAPS="${MOA_BOOTSTRAPS:-20}"
MOA_PERMUTATIONS="${MOA_PERMUTATIONS:-100}"
ML_CV_SPLITS="${ML_CV_SPLITS:-3}"

# Metadata columns expected after cpatk-metadata formatting.
METADATA_COLUMNS="${METADATA_COLUMNS:-Metadata_Plate,Metadata_Well,Metadata_Source_Plate,Metadata_Source_Well,Metadata_Compound,cpd_id,Plate_Metadata,Well_Metadata,Library,Seahorse_alert,Destination_Concentration,cpd_type,COMPOUND_NUMBER}"
ID_COLUMN="${ID_COLUMN:-Metadata_Compound}"
REFERENCE_COLUMN="${REFERENCE_COLUMN:-Metadata_Compound}"
REFERENCE_VALUES="${REFERENCE_VALUES:-DMSO}"
REPLICATE_GROUP_COLUMNS="${REPLICATE_GROUP_COLUMNS:-Metadata_Compound}"
BATCH_COLUMN="${BATCH_COLUMN:-Metadata_Plate}"
BATCH_REPORT_COLUMNS="${BATCH_REPORT_COLUMNS:-Metadata_Plate,Metadata_Compound,Library,Seahorse_alert}"
BATCH_PROTECT_COLUMNS="${BATCH_PROTECT_COLUMNS:-Metadata_Compound}"

# Optional pseudo-anchor phenotype labels made from metadata.
# For mitotox this defaults to Seahorse_alert; if this is too broad, override
# MITOTOX_LABEL_SOURCE_COLUMN=Library or provide PHENOTYPE_LABEL_TABLE manually.
USE_PHENOTYPE_LABELS_FOR_MOA="${USE_PHENOTYPE_LABELS_FOR_MOA:-1}"
PHENOTYPE_LABEL_TABLE="${PHENOTYPE_LABEL_TABLE:-}"
MITOTOX_LABEL_SOURCE_COLUMN="${MITOTOX_LABEL_SOURCE_COLUMN:-Seahorse_alert}"
PHENOTYPE_LABEL_ID_COLUMN="${PHENOTYPE_LABEL_ID_COLUMN:-cpd_id}"
PHENOTYPE_LABEL_COLUMN="${PHENOTYPE_LABEL_COLUMN:-label}"
PSEUDO_ANCHOR_FINAL_MOA_COLUMN="${PSEUDO_ANCHOR_FINAL_MOA_COLUMN:-moa_final}"

# Compounds to highlight where present.
COMPOUNDS=(
  DMSO
  Antimycine
  Oligomycine
  "Cytochalasin B"
  Rotenone
  CCCP
)

############################################
# Optional environment setup
############################################

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "${CONDA_ENV_NAME}"
fi

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  if [[ -s "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi

if [[ -n "${CPATK_SOURCE_DIR:-}" ]]; then
  python -m pip install -e "${CPATK_SOURCE_DIR}"
fi

############################################
# Helpers
############################################

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"

section() {
  printf '\n==== %s ====\n\n' "$*"
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

run_soft() {
  set +e
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    if [[ "${ALLOW_OPTIONAL_FAILURES}" == "1" ]]; then
      echo "WARN: optional command failed with status ${status}; continuing." >&2
      return 0
    fi
    echo "ERROR: command failed with status ${status}." >&2
    exit "${status}"
  fi
}

run_step() {
  local stamp="$1"
  shift
  if [[ -s "${stamp}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "Skipping completed step: ${stamp}"
    return 0
  fi
  run "$@"
  mkdir -p "$(dirname "${stamp}")"
  date > "${stamp}"
}

run_soft_step() {
  local stamp="$1"
  shift
  if [[ -s "${stamp}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "Skipping completed optional step: ${stamp}"
    return 0
  fi
  run_soft "$@"
  mkdir -p "$(dirname "${stamp}")"
  date > "${stamp}"
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: required file is missing or empty: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: required directory is missing: ${path}" >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: command not found: ${command_name}" >&2
    exit 1
  fi
}

find_existing_table() {
  local base="$1"
  local candidates=(
    "${base}.parquet"
    "${base}.tsv.gz"
    "${base}.tsv"
    "${base}.csv.gz"
    "${base}.csv"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

first_existing_table() {
  local base="$1"
  if ! find_existing_table "${base}"; then
    echo "ERROR: no table found for base path: ${base}" >&2
    exit 1
  fi
}

join_by_comma() {
  local IFS=','
  printf '%s' "$*"
}

copy_dir_filtered() {
  local source_dir="$1"
  local target_dir="$2"
  mkdir -p "${target_dir}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '._*' \
      --exclude '.DS_Store' \
      --exclude '~$*' \
      "${source_dir}/" "${target_dir}/"
  else
    cp -a "${source_dir}/." "${target_dir}/"
    find "${target_dir}" \
      \( -name '._*' -o -name '.DS_Store' -o -name '~$*' \) \
      -type f -print -delete
  fi
}

copy_file_if_present() {
  local source_file="$1"
  local target_file="$2"
  if [[ -s "${source_file}" ]]; then
    mkdir -p "$(dirname "${target_file}")"
    cp "${source_file}" "${target_file}"
  fi
}

sync_results_back() {
  if [[ "${USE_LOCAL_SCRATCH}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${WORK_OUT_DIR:-}" || ! -d "${WORK_OUT_DIR}" ]]; then
    return 0
  fi
  mkdir -p "${PROJECT_OUT_DIR}"
  echo "Syncing CPATK results back to: ${PROJECT_OUT_DIR}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${WORK_OUT_DIR}/" "${PROJECT_OUT_DIR}/"
  else
    cp -a "${WORK_OUT_DIR}/." "${PROJECT_OUT_DIR}/"
  fi
}

cleanup_scratch() {
  if [[ "${USE_LOCAL_SCRATCH}" == "1" && "${KEEP_SCRATCH}" != "1" && -n "${WORK_ROOT:-}" && -d "${WORK_ROOT}" ]]; then
    rm -rf "${WORK_ROOT}"
  fi
}

on_exit() {
  local status=$?
  if [[ "${COPY_BACK_ON_EXIT}" == "1" ]]; then
    sync_results_back || true
  fi
  if [[ "${status}" -eq 0 ]]; then
    cleanup_scratch || true
  else
    echo "Workflow exited with status ${status}. Scratch retained at: ${WORK_ROOT:-not-created}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT TERM INT HUP

add_table_if_present() {
  local label="$1"
  local path="$2"
  if [[ -s "${path}" ]]; then
    REPORT_ARGS+=(--table "${label}=${path}")
  else
    echo "WARN: report table missing, skipping: ${path}" >&2
  fi
}

make_phenotype_label_table() {
  local metadata_table="$1"
  local output_table="$2"
  local source_col="$3"
  python - "${metadata_table}" "${output_table}" "${source_col}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

metadata_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
source_col = sys.argv[3]

df = pd.read_csv(metadata_path, sep="\t")
id_col = "cpd_id" if "cpd_id" in df.columns else "Metadata_Compound"
if id_col not in df.columns:
    raise SystemExit(f"No cpd_id or Metadata_Compound column in {metadata_path}")

fallback_cols = [source_col, "Library", "cpd_type", "Metadata_Compound"]
label_cols = [col for col in fallback_cols if col in df.columns]
if not label_cols:
    raise SystemExit("No suitable label columns found")

records = []
for _, row in df.iterrows():
    cid = row.get(id_col)
    if pd.isna(cid) or str(cid).strip() == "":
        continue
    label = None
    for col in label_cols:
        value = row.get(col)
        if not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}:
            label = str(value).strip()
            break
    if label is not None:
        records.append({"cpd_id": str(cid).strip(), "label": label})

out = pd.DataFrame(records).drop_duplicates()
output_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_path, sep="\t", index=False)
print(f"Wrote phenotype label table: {output_path} ({len(out)} rows)")
PY
}

############################################
# Pre-flight and staging
############################################

section "Pre-flight checks"
mkdir -p "${OUT_DIR}"
require_dir "${RAW_DIR}"

if [[ -s "${CLEANED_METADATA}" ]]; then
  METADATA_INPUT="${CLEANED_METADATA}"
  METADATA_KIND="cleaned_kvp"
elif [[ -s "${RAW_METADATA}" ]]; then
  METADATA_INPUT="${RAW_METADATA}"
  METADATA_KIND="raw_kvp"
else
  echo "ERROR: no mitotox metadata file found." >&2
  echo "Tried: ${CLEANED_METADATA}" >&2
  echo "Tried: ${RAW_METADATA}" >&2
  exit 1
fi

SOURCE_RAW_DIR="${RAW_DIR}"
SOURCE_METADATA_INPUT="${METADATA_INPUT}"
SOURCE_PREMERGED_PROFILE_TABLE="${PREMERGED_PROFILE_TABLE}"
SOURCE_PHENOTYPE_LABEL_TABLE="${PHENOTYPE_LABEL_TABLE:-auto_from_metadata}"

if [[ "${USE_LOCAL_SCRATCH}" == "1" ]]; then
  section "Stage inputs to local scratch"
  WORK_ROOT="${SCRATCH_ROOT%/}/cpatk_mitotox_${RUN_TAG}"
  WORK_OUT_DIR="${WORK_ROOT}/results"
  STAGED_INPUT_DIR="${WORK_ROOT}/inputs"
  STAGED_RAW_DIR="${STAGED_INPUT_DIR}/raw"
  STAGED_METADATA_INPUT="${STAGED_INPUT_DIR}/$(basename "${METADATA_INPUT}")"
  STAGED_PREMERGED_PROFILE_TABLE="${STAGED_INPUT_DIR}/$(basename "${PREMERGED_PROFILE_TABLE}")"
  STAGED_PHENOTYPE_LABEL_TABLE="${STAGED_INPUT_DIR}/$(basename "${PHENOTYPE_LABEL_TABLE:-phenotype_labels.tsv}")"
  mkdir -p "${STAGED_INPUT_DIR}" "${WORK_OUT_DIR}" "${PROJECT_OUT_DIR}"
  echo "Project output directory: ${PROJECT_OUT_DIR}"
  echo "Scratch work directory: ${WORK_ROOT}"
  if [[ "${COPY_INPUTS_TO_SCRATCH}" == "1" ]]; then
    echo "Copying raw mitotox CellProfiler exports to local scratch..."
    copy_dir_filtered "${RAW_DIR}" "${STAGED_RAW_DIR}"
    copy_file_if_present "${METADATA_INPUT}" "${STAGED_METADATA_INPUT}"
    copy_file_if_present "${PREMERGED_PROFILE_TABLE}" "${STAGED_PREMERGED_PROFILE_TABLE}"
    if [[ -n "${PHENOTYPE_LABEL_TABLE:-}" ]]; then
      copy_file_if_present "${PHENOTYPE_LABEL_TABLE}" "${STAGED_PHENOTYPE_LABEL_TABLE}"
      if [[ -s "${STAGED_PHENOTYPE_LABEL_TABLE}" ]]; then
        PHENOTYPE_LABEL_TABLE="${STAGED_PHENOTYPE_LABEL_TABLE}"
      fi
    fi
    RAW_DIR="${STAGED_RAW_DIR}"
    METADATA_INPUT="${STAGED_METADATA_INPUT}"
    if [[ -s "${STAGED_PREMERGED_PROFILE_TABLE}" ]]; then
      PREMERGED_PROFILE_TABLE="${STAGED_PREMERGED_PROFILE_TABLE}"
    fi
  fi
  OUT_DIR="${WORK_OUT_DIR}"
else
  mkdir -p "${PROJECT_OUT_DIR}"
  OUT_DIR="${PROJECT_OUT_DIR}"
fi

require_command cpatk-metadata
require_command cpatk-inspect
require_command cpatk-build-profiles
require_command cpatk-preprocess
require_command cpatk-classical
require_command cpatk-visualise
require_command cpatk-stability
require_command cpatk-batch
require_command cpatk-report
# These are expected in the full cpatk environment, but downstream stages use run_soft.
for optional_command in cpatk-drift-qc cpatk-neighbours cpatk-moa cpatk-ml cpatk-explain cpatk-clipn; do
  if ! command -v "${optional_command}" >/dev/null 2>&1; then
    echo "WARN: optional command not found: ${optional_command}" >&2
  fi
done

cat > "${OUT_DIR}/run_configuration.tsv" <<EOF
item	value
base_dir	${BASE_DIR}
source_raw_dir	${SOURCE_RAW_DIR}
staged_raw_dir	${RAW_DIR}
source_metadata_input	${SOURCE_METADATA_INPUT}
staged_metadata_input	${METADATA_INPUT}
metadata_kind	${METADATA_KIND}
source_premerged_profile_table	${SOURCE_PREMERGED_PROFILE_TABLE}
staged_premerged_profile_table	${PREMERGED_PROFILE_TABLE}
source_phenotype_label_table	${SOURCE_PHENOTYPE_LABEL_TABLE}
phenotype_label_table	${PHENOTYPE_LABEL_TABLE:-auto_from_metadata}
project_out_dir	${PROJECT_OUT_DIR}
work_out_dir	${OUT_DIR}
use_local_scratch	${USE_LOCAL_SCRATCH}
scratch_root	${SCRATCH_ROOT}
work_root	${WORK_ROOT}
run_tag	${RUN_TAG}
threads	${THREADS}
try_composite_keys	${TRY_COMPOSITE_KEYS}
composite_image_merge_keys	${COMPOSITE_IMAGE_MERGE_KEYS}
allow_imagenumber_fallback	${ALLOW_IMAGENUMBER_FALLBACK}
fallback_image_merge_keys	${FALLBACK_IMAGE_MERGE_KEYS}
allow_premerged_profile_fallback	${ALLOW_PREMERGED_PROFILE_FALLBACK}
reference_values	${REFERENCE_VALUES}
clipn_strict_drop_any_zero	${CLIPN_STRICT_DROP_ANY_ZERO}
clipn_zero_policy	${CLIPN_ZERO_POLICY}
run_clipn_latent_moa	${RUN_CLIPN_LATENT_MOA}
run_pca_fallback_latent_moa	${RUN_PCA_FALLBACK_LATENT_MOA}
mitotox_label_source_column	${MITOTOX_LABEL_SOURCE_COLUMN}
EOF

############################################
# 00 Metadata validation
############################################

METADATA_DIR="${OUT_DIR}/00_metadata_validation"
FORMATTED_METADATA="${METADATA_DIR}/formatted_metadata.tsv"

if [[ "${RUN_METADATA}" == "1" ]]; then
  section "Step 00: metadata validation"
  mkdir -p "${METADATA_DIR}"
  run_step "${METADATA_DIR}/.metadata.done" \
    cpatk-metadata \
      --metadata_table "${METADATA_INPUT}" \
      --output_dir "${METADATA_DIR}" \
      --plate_column Plate_Metadata \
      --well_column Well_Metadata \
      --source_plate_column BC \
      --source_well_column comp_s \
      --duplicate_policy error \
      --log_level "${LOG_LEVEL}"
fi
require_file "${FORMATTED_METADATA}"

if [[ -z "${PHENOTYPE_LABEL_TABLE:-}" ]]; then
  PHENOTYPE_LABEL_TABLE="${OUT_DIR}/00_metadata_validation/mitotox_pseudo_anchor_labels.tsv"
  section "Step 00b: derive mitotox pseudo-anchor labels from metadata"
  make_phenotype_label_table "${FORMATTED_METADATA}" "${PHENOTYPE_LABEL_TABLE}" "${MITOTOX_LABEL_SOURCE_COLUMN}"
fi

############################################
# 01 Inspect and drift QC
############################################

if [[ "${RUN_INSPECT}" == "1" ]]; then
  section "Step 01a: inspect staged raw CellProfiler folder"
  run_step "${OUT_DIR}/01_inspect/.inspect.done" \
    cpatk-inspect \
      --input_dir "${RAW_DIR}" \
      --output_dir "${OUT_DIR}/01_inspect" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_DRIFT_QC}" == "1" ]] && command -v cpatk-drift-qc >/dev/null 2>&1; then
  section "Step 01b: raw acquisition/instrument drift QC"
  run_soft_step "${OUT_DIR}/01_drift_qc/.drift_qc.done" \
    cpatk-drift-qc \
      --input_dir "${RAW_DIR}" \
      --output_dir "${OUT_DIR}/01_drift_qc" \
      --image_col ImageNumber \
      --plot_top_n 12 \
      --min_points 20 \
      --log_level "${LOG_LEVEL}"
fi

############################################
# 02 Build merged profiles
############################################

PROFILE_DIR="${OUT_DIR}/02_profile_build"
PROFILE_TABLE=""
PROFILE_BUILD_STATUS=0

if [[ "${RUN_PROFILE_BUILD}" == "1" ]]; then
  section "Step 02: build profiles from CellProfiler exports"
  mkdir -p "${PROFILE_DIR}"
  if [[ "${TRY_COMPOSITE_KEYS}" == "1" ]]; then
    echo "Trying composite image/object merge keys: ${COMPOSITE_IMAGE_MERGE_KEYS}"
    set +e
    cpatk-build-profiles \
      --input_dir "${RAW_DIR}" \
      --output_dir "${PROFILE_DIR}" \
      --metadata_table "${FORMATTED_METADATA}" \
      --aggregate_statistic median \
      --duplicate_image_policy error \
      --metadata_duplicate_policy error \
      --image_merge_keys "${COMPOSITE_IMAGE_MERGE_KEYS}" \
      --log_level "${LOG_LEVEL}"
    PROFILE_BUILD_STATUS=$?
    set -e
    if [[ "${PROFILE_BUILD_STATUS}" -ne 0 && "${ALLOW_IMAGENUMBER_FALLBACK}" == "1" ]]; then
      echo "WARN: composite-key build failed with status ${PROFILE_BUILD_STATUS}; retrying ImageNumber fallback." >&2
      rm -rf "${PROFILE_DIR}"
      mkdir -p "${PROFILE_DIR}"
      set +e
      cpatk-build-profiles \
        --input_dir "${RAW_DIR}" \
        --output_dir "${PROFILE_DIR}" \
        --metadata_table "${FORMATTED_METADATA}" \
        --aggregate_statistic median \
        --duplicate_image_policy error \
        --metadata_duplicate_policy error \
        --image_merge_keys "${FALLBACK_IMAGE_MERGE_KEYS}" \
        --log_level "${LOG_LEVEL}"
      PROFILE_BUILD_STATUS=$?
      set -e
    fi
  else
    set +e
    cpatk-build-profiles \
      --input_dir "${RAW_DIR}" \
      --output_dir "${PROFILE_DIR}" \
      --metadata_table "${FORMATTED_METADATA}" \
      --aggregate_statistic median \
      --duplicate_image_policy error \
      --metadata_duplicate_policy error \
      --image_merge_keys "${FALLBACK_IMAGE_MERGE_KEYS}" \
      --log_level "${LOG_LEVEL}"
    PROFILE_BUILD_STATUS=$?
    set -e
  fi

  if [[ "${PROFILE_BUILD_STATUS}" -eq 0 ]]; then
    date > "${PROFILE_DIR}/.profile_build.done"
  elif [[ "${ALLOW_PREMERGED_PROFILE_FALLBACK}" == "1" && -s "${PREMERGED_PROFILE_TABLE}" ]]; then
    echo "WARN: profile building failed; using premerged profile table fallback: ${PREMERGED_PROFILE_TABLE}" >&2
    FALLBACK_PROFILE_DIR="${OUT_DIR}/02_profile_build_premerged_fallback"
    mkdir -p "${FALLBACK_PROFILE_DIR}"
    cp "${PREMERGED_PROFILE_TABLE}" "${FALLBACK_PROFILE_DIR}/merged_profiles.tsv"
    cat > "${FALLBACK_PROFILE_DIR}/profile_build_summary.tsv" <<EOF
item	value
profile_build_status	failed
fallback_used	premerged_profile_table
premerged_profile_table	${PREMERGED_PROFILE_TABLE}
EOF
    PROFILE_DIR="${FALLBACK_PROFILE_DIR}"
  else
    echo "ERROR: profile building failed and no premerged fallback is available." >&2
    exit "${PROFILE_BUILD_STATUS}"
  fi
fi

PROFILE_TABLE="$(first_existing_table "${PROFILE_DIR}/merged_profiles")"
echo "Using merged profile table: ${PROFILE_TABLE}"

############################################
# 03 Preprocessing strategy comparison
############################################

PREPROCESS_ROOT="${OUT_DIR}/03_preprocess_strategy_comparison"
PRIMARY_STRATEGY="baseline_no_reference_normalisation"
PRIMARY_TABLE=""

run_preprocess_strategy() {
  local strategy_name="$1"
  local required="$2"
  shift 2
  local strategy_dir="${PREPROCESS_ROOT}/${strategy_name}"
  local extra_args=("$@")
  mkdir -p "${strategy_dir}"
  if [[ -s "${strategy_dir}/.preprocess.done" && "${FORCE_RERUN}" != "1" ]]; then
    echo "Skipping completed preprocessing strategy: ${strategy_name}"
    return 0
  fi
  if [[ "${required}" == "required" ]]; then
    run cpatk-preprocess \
      --input_table "${PROFILE_TABLE}" \
      --output_dir "${strategy_dir}" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --additional_metadata_columns "${METADATA_COLUMNS}" \
      --imputation_method median \
      --scaling_method robust \
      --max_feature_missing_fraction 0.20 \
      --max_sample_missing_fraction 0.50 \
      --max_absolute_correlation 0.98 \
      --max_features_for_correlation 7000 \
      --max_zero_fraction 1.0 \
      --replicate_group_columns "${REPLICATE_GROUP_COLUMNS}" \
      --batch_report_columns "${BATCH_REPORT_COLUMNS}" \
      "${extra_args[@]}" \
      --log_level "${LOG_LEVEL}"
    date > "${strategy_dir}/.preprocess.done"
  else
    set +e
    cpatk-preprocess \
      --input_table "${PROFILE_TABLE}" \
      --output_dir "${strategy_dir}" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --additional_metadata_columns "${METADATA_COLUMNS}" \
      --imputation_method median \
      --scaling_method robust \
      --max_feature_missing_fraction 0.20 \
      --max_sample_missing_fraction 0.50 \
      --max_absolute_correlation 0.98 \
      --max_features_for_correlation 7000 \
      --max_zero_fraction 1.0 \
      --replicate_group_columns "${REPLICATE_GROUP_COLUMNS}" \
      --batch_report_columns "${BATCH_REPORT_COLUMNS}" \
      "${extra_args[@]}" \
      --log_level "${LOG_LEVEL}"
    local status=$?
    set -e
    if [[ "${status}" -eq 0 ]]; then
      date > "${strategy_dir}/.preprocess.done"
    else
      echo "WARN: optional preprocessing strategy failed: ${strategy_name} (status ${status}); continuing." >&2
    fi
  fi
}

if [[ "${RUN_PREPROCESSING}" == "1" ]]; then
  section "Step 03: preprocessing strategy comparison"
  mkdir -p "${PREPROCESS_ROOT}"

  run_preprocess_strategy \
    "baseline_no_reference_normalisation" \
    required \
    --reference_normalisation_method none \
    --batch_correction_method none

  run_preprocess_strategy \
    "dmso_robust_z" \
    optional \
    --reference_normalisation_method robust_z \
    --reference_column "${REFERENCE_COLUMN}" \
    --reference_values "${REFERENCE_VALUES}" \
    --reference_group_columns Metadata_Plate \
    --batch_correction_method none

  run_preprocess_strategy \
    "dmso_robust_z_combat_location_scale" \
    optional \
    --reference_normalisation_method robust_z \
    --reference_column "${REFERENCE_COLUMN}" \
    --reference_values "${REFERENCE_VALUES}" \
    --reference_group_columns Metadata_Plate \
    --batch_correction_method combat_location_scale \
    --batch_column "${BATCH_COLUMN}" \
    --batch_protect_columns "${BATCH_PROTECT_COLUMNS}"
fi

if [[ -s "${PREPROCESS_ROOT}/dmso_robust_z/.preprocess.done" ]]; then
  PRIMARY_STRATEGY="dmso_robust_z"
fi
PRIMARY_TABLE="$(first_existing_table "${PREPROCESS_ROOT}/${PRIMARY_STRATEGY}/preprocessed")"
echo "Using primary preprocessed table: ${PRIMARY_TABLE}"
echo "Primary strategy: ${PRIMARY_STRATEGY}"

############################################
# 04 Classical, visualisation, replicate and batch QC for each completed strategy
############################################

for STRATEGY_DIR in "${PREPROCESS_ROOT}"/*; do
  [[ -d "${STRATEGY_DIR}" ]] || continue
  STRATEGY_NAME="$(basename "${STRATEGY_DIR}")"
  if ! STRATEGY_TABLE="$(find_existing_table "${STRATEGY_DIR}/preprocessed")"; then
    echo "Skipping strategy without a completed preprocessed table: ${STRATEGY_NAME}" >&2
    continue
  fi

  if [[ "${RUN_CLASSICAL}" == "1" ]]; then
    section "Step 04a: classical analysis for ${STRATEGY_NAME}"
    run_soft_step "${OUT_DIR}/04_classical/${STRATEGY_NAME}/.classical.done" \
      cpatk-classical \
        --input_table "${STRATEGY_TABLE}" \
        --output_dir "${OUT_DIR}/04_classical/${STRATEGY_NAME}" \
        --metadata_columns "${METADATA_COLUMNS}" \
        --id_column "${ID_COLUMN}" \
        --colour_column Metadata_Compound \
        --cluster_group_columns Metadata_Compound,Metadata_Plate \
        --distance_metric cosine \
        --n_neighbours "${N_NEIGHBOURS}" \
        --n_clusters "${N_CLUSTERS}" \
        --log_level "${LOG_LEVEL}"
  fi

  if [[ "${RUN_VISUALISE}" == "1" ]]; then
    section "Step 04b: visualisation for ${STRATEGY_NAME}"
    run_soft_step "${OUT_DIR}/05_visualise/${STRATEGY_NAME}/.visualise.done" \
      cpatk-visualise \
        --input_table "${STRATEGY_TABLE}" \
        --output_dir "${OUT_DIR}/05_visualise/${STRATEGY_NAME}" \
        --metadata_columns "${METADATA_COLUMNS}" \
        --id_column "${ID_COLUMN}" \
        --colour_columns Metadata_Compound,Metadata_Plate \
        --aggregate_by_id \
        --aggregate_method median \
        --log_level "${LOG_LEVEL}"
  fi

  if [[ "${RUN_STABILITY}" == "1" ]]; then
    section "Step 04c: replicate/neighbour/cluster stability for ${STRATEGY_NAME}"
    run_soft_step "${OUT_DIR}/06_stability/${STRATEGY_NAME}/.stability.done" \
      cpatk-stability \
        --input_table "${STRATEGY_TABLE}" \
        --output_dir "${OUT_DIR}/06_stability/${STRATEGY_NAME}" \
        --metadata_columns "${METADATA_COLUMNS}" \
        --replicate_group_columns "${REPLICATE_GROUP_COLUMNS}" \
        --n_clusters "${N_CLUSTERS}" \
        --k_values 4,6,8,10,12 \
        --n_bootstraps "${STABILITY_BOOTSTRAPS}" \
        --n_permutations "${STABILITY_PERMUTATIONS}" \
        --n_neighbours "${N_NEIGHBOURS}" \
        --log_level "${LOG_LEVEL}"
  fi

  if [[ "${RUN_BATCH}" == "1" ]]; then
    section "Step 04d: batch association diagnostics for ${STRATEGY_NAME}"
    run_soft_step "${OUT_DIR}/07_batch/${STRATEGY_NAME}/.batch.done" \
      cpatk-batch \
        --input_table "${STRATEGY_TABLE}" \
        --output_dir "${OUT_DIR}/07_batch/${STRATEGY_NAME}" \
        --metadata_columns "${METADATA_COLUMNS}" \
        --batch_column "${BATCH_COLUMN}" \
        --columns_to_test "${BATCH_REPORT_COLUMNS}" \
        --log_level "${LOG_LEVEL}"
  fi
done

############################################
# 05 Primary-strategy interpretation layers
############################################

PRIMARY_CLASSICAL_DIR="${OUT_DIR}/04_classical/${PRIMARY_STRATEGY}"
PRIMARY_NN="${PRIMARY_CLASSICAL_DIR}/nearest_neighbours.tsv"
COMPOUND_LIST="$(join_by_comma "${COMPOUNDS[@]}")"

if [[ "${RUN_NEIGHBOURS}" == "1" && -s "${PRIMARY_NN}" ]] && command -v cpatk-neighbours >/dev/null 2>&1; then
  section "Step 05a: nearest-neighbour plotting for primary strategy"
  run_soft_step "${OUT_DIR}/08_neighbours/.neighbours.done" \
    cpatk-neighbours \
      --input_neighbours "${PRIMARY_NN}" \
      --output_dir "${OUT_DIR}/08_neighbours" \
      --compounds "${COMPOUND_LIST}" \
      --top_n 15 \
      --include_ties_at_k \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_MOA}" == "1" ]] && command -v cpatk-moa >/dev/null 2>&1; then
  section "Step 05b: pseudo-anchor MOA-style analysis for primary strategy"
  MOA_LABEL_ARGS=()
  if [[ "${USE_PHENOTYPE_LABELS_FOR_MOA}" == "1" && -s "${PHENOTYPE_LABEL_TABLE}" ]]; then
    MOA_LABEL_ARGS=(
      --pseudo_anchor_label_table "${PHENOTYPE_LABEL_TABLE}"
      --pseudo_anchor_label_id_column "${PHENOTYPE_LABEL_ID_COLUMN}"
      --pseudo_anchor_label_column "${PHENOTYPE_LABEL_COLUMN}"
      --pseudo_anchor_final_moa_column "${PSEUDO_ANCHOR_FINAL_MOA_COLUMN}"
    )
  fi
  run_soft_step "${OUT_DIR}/09_moa/.moa.done" \
    cpatk-moa \
      --input_table "${PRIMARY_TABLE}" \
      --output_dir "${OUT_DIR}/09_moa" \
      --id_column "${ID_COLUMN}" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --make_pseudo_anchors \
      --pseudo_anchor_method bootstrap \
      --auto_k \
      --k_values 4,6,8,10,12 \
      --n_bootstraps "${MOA_BOOTSTRAPS}" \
      --n_permutations "${MOA_PERMUTATIONS}" \
      --aggregate_method median \
      --centroid_method median \
      --adaptive_shrinkage \
      --score_method cosine \
      --make_projection_plots \
      --projection both \
      --interactive \
      "${MOA_LABEL_ARGS[@]}" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_ML}" == "1" ]] && command -v cpatk-ml >/dev/null 2>&1; then
  section "Step 05c: supervised classifier smoke test using compound IDs as labels"
  run_soft_step "${OUT_DIR}/10_ml/.ml.done" \
    cpatk-ml \
      --input_table "${PRIMARY_TABLE}" \
      --output_dir "${OUT_DIR}/10_ml" \
      --class_column Metadata_Compound \
      --metadata_columns "${METADATA_COLUMNS}" \
      --compare_models \
      --n_splits "${ML_CV_SPLITS}" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_EXPLAIN}" == "1" && -s "${PRIMARY_NN}" ]] && command -v cpatk-explain >/dev/null 2>&1; then
  section "Step 05d: feature tests and optional SHAP/neighbourhood explanation"
  run_soft_step "${OUT_DIR}/11_explain/.explain.done" \
    cpatk-explain \
      --input_table "${PRIMARY_TABLE}" \
      --output_dir "${OUT_DIR}/11_explain" \
      --class_column Metadata_Compound \
      --metadata_columns "${METADATA_COLUMNS}" \
      --id_column "${ID_COLUMN}" \
      --query_ids "${COMPOUNDS[@]}" \
      --nn_file "${PRIMARY_NN}" \
      --n_neighbours 4 \
      --run_feature_tests \
      --run_query_background_shap \
      --run_neighbourhood_shap \
      --include_shap \
      --background_column Metadata_Compound \
      --background_values "${REFERENCE_VALUES}" \
      --n_top_features 20 \
      --log_level "${LOG_LEVEL}"
fi

############################################
# 06 CLIPn / PCA-fallback test
############################################

if [[ "${RUN_CLIPN}" == "1" ]] && command -v cpatk-clipn >/dev/null 2>&1; then
  section "Step 06: CLIPn test with single-dataset split by compound"
  CLIPN_ZERO_ARGS=()
  if [[ "${CLIPN_STRICT_DROP_ANY_ZERO}" == "1" ]]; then
    CLIPN_ZERO_ARGS+=(--drop_rows_with_any_zero)
  fi
  CLIPN_FALLBACK_ARGS=()
  if [[ "${CLIPN_ALLOW_PCA_FALLBACK}" == "1" ]]; then
    CLIPN_FALLBACK_ARGS+=(--allow_pca_fallback)
  fi
  run_soft_step "${OUT_DIR}/12_clipn/.clipn.done" \
    cpatk-clipn \
      --dataset "mitotox=${PRIMARY_TABLE}" \
      --output_dir "${OUT_DIR}/12_clipn" \
      --experiment mitotox_cpatk_v0_2_19_fast \
      --mode integrate_all \
      --split_single_dataset_by_column Metadata_Compound \
      --single_dataset_split_names reference_like,query_like \
      --single_dataset_split_seed 42 \
      --id_column "${ID_COLUMN}" \
      --label_column Metadata_Compound \
      --metadata_columns "${METADATA_COLUMNS}" \
      --latent_dim "${CLIPN_LATENT_DIM}" \
      --epochs "${CLIPN_EPOCHS}" \
      --imputation_method median \
      --scaling_method robust \
      --n_neighbours "${N_NEIGHBOURS}" \
      --clipn_zero_policy "${CLIPN_ZERO_POLICY}" \
      "${CLIPN_ZERO_ARGS[@]}" \
      "${CLIPN_FALLBACK_ARGS[@]}" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_CLIPN_LATENT_MOA}" == "1" ]] && command -v cpatk-moa >/dev/null 2>&1; then
  CLIPN_LATENT_TABLE="${OUT_DIR}/12_clipn/clipn_latent.tsv.gz"
  CLIPN_RUN_STATUS="${OUT_DIR}/12_clipn/clipn_run_status.tsv"
  CLIPN_BACKEND_RUN="unknown"
  if [[ -s "${CLIPN_RUN_STATUS}" ]]; then
    CLIPN_BACKEND_RUN="$(awk -F '\t' 'NR == 2 {print $1}' "${CLIPN_RUN_STATUS}")"
  fi
  if [[ -s "${CLIPN_LATENT_TABLE}" && ( "${CLIPN_BACKEND_RUN}" == "success" || "${RUN_PCA_FALLBACK_LATENT_MOA}" == "1" ) ]]; then
    if [[ "${CLIPN_BACKEND_RUN}" == "success" ]]; then
      section "Step 06b: CLIPn latent-space pseudo-anchor MOA analysis"
    else
      section "Step 06b: PCA-fallback latent-space pseudo-anchor MOA analysis"
      echo "WARN: CLIPn backend did not complete successfully (${CLIPN_BACKEND_RUN}); running latent-space MOA on PCA fallback because RUN_PCA_FALLBACK_LATENT_MOA=1." >&2
    fi
  CLIPN_MOA_LABEL_ARGS=()
  if [[ "${USE_PHENOTYPE_LABELS_FOR_MOA:-0}" == "1" && -s "${PHENOTYPE_LABEL_TABLE:-}" ]]; then
    CLIPN_MOA_LABEL_ARGS=(
      --pseudo_anchor_label_table "${PHENOTYPE_LABEL_TABLE}"
      --pseudo_anchor_label_id_column "${PHENOTYPE_LABEL_ID_COLUMN}"
      --pseudo_anchor_label_column "${PHENOTYPE_LABEL_COLUMN}"
      --pseudo_anchor_final_moa_column "${PSEUDO_ANCHOR_FINAL_MOA_COLUMN}"
    )
  fi
    run_soft_step "${OUT_DIR}/13_clipn_latent_moa/.clipn_latent_moa.done" \
      cpatk-moa \
        --input_table "${CLIPN_LATENT_TABLE}" \
        --output_dir "${OUT_DIR}/13_clipn_latent_moa" \
        --id_column "${ID_COLUMN}" \
        --metadata_columns "${METADATA_COLUMNS},Dataset,Sample" \
        --make_pseudo_anchors \
        --pseudo_anchor_method bootstrap \
        --auto_k \
        --k_values 4,6,8,10,12 \
        --n_bootstraps "${MOA_BOOTSTRAPS}" \
        --n_permutations "${MOA_PERMUTATIONS}" \
        --aggregate_method median \
        --centroid_method median \
        --adaptive_shrinkage \
        --score_method cosine \
        --make_projection_plots \
        --projection both \
        --interactive \
        "${CLIPN_MOA_LABEL_ARGS[@]}" \
        --log_level "${LOG_LEVEL}"
  else
    if [[ ! -s "${CLIPN_LATENT_TABLE}" ]]; then
      echo "WARN: RUN_CLIPN_LATENT_MOA=1 but no latent table found: ${CLIPN_LATENT_TABLE}" >&2
    else
      echo "WARN: RUN_CLIPN_LATENT_MOA=1 but CLIPn backend_run=${CLIPN_BACKEND_RUN}; skipping latent-space MOA. Set RUN_PCA_FALLBACK_LATENT_MOA=1 to run this on PCA fallback output." >&2
    fi
  fi
fi

############################################
# 07 Final report index
############################################

if [[ "${RUN_FINAL_REPORT}" == "1" ]]; then
  section "Step 07: final HTML report index"
  REPORT_ARGS=(
    --output_html "${OUT_DIR}/CPATK_mitotox_v0_2_19_fast_full_report.html"
    --title "CPATK v0.2.19 mitotox Cell Painting stress-test report"
    --narrative "End-to-end CPATK v0.2.19 stress test on mitotox Cell Painting data. Review metadata merge rates, profile-building/fallback status, preprocessing strategy comparison, replicate QC, batch QC, classical plots, neighbour tables, pseudo-anchor labels and optional ML/CLIPn/explainability outputs before interpreting biology."
    --warning "This is a validation workflow. Optional modules are allowed to fail so that earlier QC and preprocessing outputs are preserved."
    --warning "The ML section uses compound IDs as labels for a software smoke test unless a genuine MOA label column is supplied."
    --log_level "${LOG_LEVEL}"
  )
  add_table_if_present "Run configuration" "${OUT_DIR}/run_configuration.tsv"
  add_table_if_present "Metadata validation summary" "${METADATA_DIR}/metadata_validation_summary.tsv"
  add_table_if_present "Metadata key validation" "${METADATA_DIR}/metadata_key_validation.tsv"
  add_table_if_present "Profile build summary" "${PROFILE_DIR}/profile_build_summary.tsv"
  add_table_if_present "Primary preprocessing summary" "${PREPROCESS_ROOT}/${PRIMARY_STRATEGY}/preprocessing_summary.tsv"
  add_table_if_present "Primary final matrix validation" "${PREPROCESS_ROOT}/${PRIMARY_STRATEGY}/final_matrix_validation.tsv"
  add_table_if_present "Primary before-after replicate summary" "${PREPROCESS_ROOT}/${PRIMARY_STRATEGY}/before_after_replicate_summary.tsv"
  add_table_if_present "Primary before-after batch PC association" "${PREPROCESS_ROOT}/${PRIMARY_STRATEGY}/before_after_batch_pc_association.tsv"
  add_table_if_present "Primary nearest neighbours" "${PRIMARY_NN}"
  add_table_if_present "Pseudo-anchor phenotype labels" "${OUT_DIR}/09_moa/pseudo_anchor_phenotype_labels.tsv"
  add_table_if_present "CLIPn run status" "${OUT_DIR}/12_clipn/clipn_run_status.tsv"
  add_table_if_present "CLIPn preprocessing summary" "${OUT_DIR}/12_clipn/clipn_preprocessing_summary.tsv"
  add_table_if_present "CLIPn latent diagnostic summary" "${OUT_DIR}/12_clipn/latent_diagnostic_summary.tsv"
  add_table_if_present "Latent-space MOA predictions" "${OUT_DIR}/13_clipn_latent_moa/advanced_moa_top_predictions.tsv"
  run_soft cpatk-report "${REPORT_ARGS[@]}"
fi

if [[ "${USE_LOCAL_SCRATCH}" == "1" ]]; then
  section "Copy final results back to project filesystem"
  sync_results_back
  COPY_BACK_ON_EXIT=0
fi

section "Workflow complete"
echo "Scratch output directory: ${OUT_DIR}"
echo "Project output directory: ${PROJECT_OUT_DIR}"
echo "Primary strategy: ${PRIMARY_STRATEGY}"
echo "Primary preprocessed table: ${PRIMARY_TABLE}"
echo "Primary report on project filesystem: ${PROJECT_OUT_DIR}/CPATK_mitotox_v0_2_19_fast_full_report.html"
