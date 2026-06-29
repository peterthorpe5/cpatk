#!/usr/bin/env bash
#$ -jc rhel9
#$ -j y
#$ -N cpatk_multi_cp
#$ -jc long
#$ -pe smp 16
#$ -mods l_hard mfree 240G
#$ -adds l_hard h_vmem 240G
#$ -cwd
# GPU resources are enabled here because this full rerun includes the optional
# CPATK-native contrastive latent step by default. Comment these two lines if submitting to a CPU queue.
#$ -adds l_hard gpu 1
#$ -adds l_hard cuda.0.name 'NVIDIA A40'

# CPATK v0.2.26 large multi-dataset Cell Painting stress test.
#
# Datasets covered by the manifest:
#   STB1, STB2, SelleckChem batches B1-B6, and mitotox.
#
# The workflow deliberately builds profiles per raw export, then combines the
# finished profile tables. This is safer than pooling raw CellProfiler folders,
# because ImageNumber can restart in independent exports.
#
# The shell stages only common compartment files by default:
#   Image, Acrosome/Arosome, FilteredNuclei and Mitochondria.
# This keeps the first multi-dataset run comparable across STB, SelleckChem and
# mitotox and avoids the SelleckChem-only SpermCells/FilterSpermCells tables
# dominating missingness/feature-union behaviour.

set -Eeuo pipefail
IFS=$'\n\t'

############################################
# User-editable configuration
############################################

INPUT_CONFIG_DIR="${INPUT_CONFIG_DIR:-${PWD}}"
MANIFEST="${MANIFEST:-${INPUT_CONFIG_DIR}/cpatk_multidataset_manifest.tsv}"
MASTER_METADATA="${MASTER_METADATA:-${INPUT_CONFIG_DIR}/cpatk_multidataset_master_metadata.tsv}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_OUT_DIR="${PROJECT_OUT_DIR:-${OUT_DIR:-/home/pthorpe001/data/2025_jason_cell_painting/data/cpatk_v0_2_26_multidataset_${RUN_TAG}}}"
OUT_DIR="${PROJECT_OUT_DIR}"

THREADS="${THREADS:-${NSLOTS:-16}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
FORCE_RERUN="${FORCE_RERUN:-0}"
ALLOW_OPTIONAL_FAILURES="${ALLOW_OPTIONAL_FAILURES:-1}"

# Scratch mode. Raw folders are copied one at a time to scratch, filename-normalised,
# then removed after profile building. This improves filesystem performance without
# needing enough scratch for every raw dataset at once.
USE_LOCAL_SCRATCH="${USE_LOCAL_SCRATCH:-1}"
COPY_EACH_RAW_TO_SCRATCH="${COPY_EACH_RAW_TO_SCRATCH:-1}"
COPY_BACK_ON_EXIT="${COPY_BACK_ON_EXIT:-1}"
KEEP_SCRATCH="${KEEP_SCRATCH:-0}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp/${USER:-cpatk}}}"
WORK_ROOT=""
WORK_OUT_DIR=""

# Keep the first cross-dataset run conservative and comparable.
STAGE_COMMON_COMPARTMENTS_ONLY="${STAGE_COMMON_COMPARTMENTS_ONLY:-1}"
NORMALISE_AROSOME_TO_ACROSOME="${NORMALISE_AROSOME_TO_ACROSOME:-1}"

# Analysis toggles.
RUN_METADATA="${RUN_METADATA:-1}"
RUN_PROFILE_BUILD="${RUN_PROFILE_BUILD:-1}"
RUN_COMBINE="${RUN_COMBINE:-1}"
RUN_PREPROCESSING="${RUN_PREPROCESSING:-1}"
RUN_CLASSICAL="${RUN_CLASSICAL:-1}"
RUN_VISUALISE="${RUN_VISUALISE:-1}"
RUN_STABILITY="${RUN_STABILITY:-1}"
RUN_BATCH="${RUN_BATCH:-1}"
RUN_NEIGHBOURS="${RUN_NEIGHBOURS:-1}"
RUN_MOA="${RUN_MOA:-1}"
RUN_ML="${RUN_ML:-0}"
RUN_EXPLAIN="${RUN_EXPLAIN:-1}"
RUN_CLIPN="${RUN_CLIPN:-1}"
RUN_CLIPN_LATENT_MOA="${RUN_CLIPN_LATENT_MOA:-1}"
RUN_FINAL_REPORT="${RUN_FINAL_REPORT:-1}"

# Latent settings. Default is CPATK-native contrastive; published CLIPn runs only if LATENT_BACKEND_MODULE=clipn.
CLIPN_LATENT_DIM="${CLIPN_LATENT_DIM:-10}"
CLIPN_EPOCHS="${CLIPN_EPOCHS:-120}"
CLIPN_EARLY_STOPPING="${CLIPN_EARLY_STOPPING:-1}"
CLIPN_PATIENCE="${CLIPN_PATIENCE:-20}"
CLIPN_MIN_DELTA="${CLIPN_MIN_DELTA:-0.0001}"
CLIPN_EPOCH_CHUNK_SIZE="${CLIPN_EPOCH_CHUNK_SIZE:-10}"
CLIPN_VALIDATION_FRACTION="${CLIPN_VALIDATION_FRACTION:-0.15}"
CLIPN_ALLOW_PCA_FALLBACK="${CLIPN_ALLOW_PCA_FALLBACK:-0}"
CLIPN_ZERO_POLICY="${CLIPN_ZERO_POLICY:-keep}"
LATENT_BACKEND_MODULE="${LATENT_BACKEND_MODULE:-cpatk_contrastive}"
NATIVE_HIDDEN_DIMS="${NATIVE_HIDDEN_DIMS:-512,256}"
NATIVE_BATCH_SIZE="${NATIVE_BATCH_SIZE:-256}"
NATIVE_STEPS_PER_EPOCH="${NATIVE_STEPS_PER_EPOCH:-0}"
NATIVE_DEVICE="${NATIVE_DEVICE:-auto}"

# Optional user-requested features to keep through ordinary feature filters.
PROTECTED_FEATURES="${PROTECTED_FEATURES:-}"
PROTECTED_FEATURES_FILE="${PROTECTED_FEATURES_FILE:-}"

# Lightweight settings for the first ambitious stress test.
N_CLUSTERS="${N_CLUSTERS:-12}"
N_NEIGHBOURS="${N_NEIGHBOURS:-15}"
STABILITY_BOOTSTRAPS="${STABILITY_BOOTSTRAPS:-20}"
STABILITY_PERMUTATIONS="${STABILITY_PERMUTATIONS:-20}"
MOA_BOOTSTRAPS="${MOA_BOOTSTRAPS:-20}"
MOA_PERMUTATIONS="${MOA_PERMUTATIONS:-100}"
ML_CV_SPLITS="${ML_CV_SPLITS:-3}"

# Metadata/analysis columns after metadata/profile combining.
METADATA_COLUMNS="${METADATA_COLUMNS:-Metadata_Plate,Metadata_Well,ImageNumber,Metadata_Profile_Source,Metadata_Assay_Family,Metadata_Input_Group,Metadata_Source_Dataset,Metadata_Batch,Metadata_Compound,cpd_id,cpd_type,Library,compound_name,Metadata_Concentration}"
ID_COLUMN="${ID_COLUMN:-Metadata_Compound}"
REFERENCE_COLUMN="${REFERENCE_COLUMN:-cpd_type}"
REFERENCE_VALUES="${REFERENCE_VALUES:-DMSO}"
REFERENCE_GROUP_COLUMNS="${REFERENCE_GROUP_COLUMNS:-Metadata_Plate}"
REPLICATE_GROUP_COLUMNS="${REPLICATE_GROUP_COLUMNS:-Metadata_Compound}"
BATCH_COLUMN="${BATCH_COLUMN:-Metadata_Profile_Source}"
BATCH_REPORT_COLUMNS="${BATCH_REPORT_COLUMNS:-Metadata_Profile_Source,Metadata_Assay_Family,Metadata_Plate,cpd_type,Library}"
BATCH_PROTECT_COLUMNS="${BATCH_PROTECT_COLUMNS:-Metadata_Compound,cpd_type,Library}"

# Query compounds for explanation plots. Missing IDs are skipped/audited by CPATK.
QUERY_IDS="${QUERY_IDS:-DMSO,A23187,Tafenoquine,DDD02443214,CCCP,Rotenone,Oligomycine,Antimycine,Disulfiram}"

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

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export POLARS_MAX_THREADS="${THREADS}"

############################################
# Helpers
############################################

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
    return "${status}"
  fi
}

run_step() {
  local sentinel="$1"
  shift
  if [[ -s "${sentinel}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "Skipping completed step: ${sentinel}"
    return 0
  fi
  mkdir -p "$(dirname "${sentinel}")"
  run "$@"
  date > "${sentinel}"
}

run_soft_step() {
  local sentinel="$1"
  shift
  if [[ -s "${sentinel}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "Skipping completed optional step: ${sentinel}"
    return 0
  fi
  mkdir -p "$(dirname "${sentinel}")"
  run_soft "$@"
  date > "${sentinel}"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

safe_label() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '_'
}

csv_join() {
  local IFS=,
  echo "$*"
}

copy_back() {
  local status=$?
  if [[ "${COPY_BACK_ON_EXIT}" == "1" && -n "${WORK_OUT_DIR}" && -d "${WORK_OUT_DIR}" && "${WORK_OUT_DIR}" != "${OUT_DIR}" ]]; then
    section "Copying results back to ${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
    rsync -a "${WORK_OUT_DIR}/" "${OUT_DIR}/"
  fi
  if [[ "${KEEP_SCRATCH}" != "1" && -n "${WORK_ROOT}" && -d "${WORK_ROOT}" ]]; then
    rm -rf "${WORK_ROOT}"
  fi
  exit "${status}"
}
trap copy_back EXIT

find_profile_table() {
  local profile_dir="$1"
  for candidate in \
    "${profile_dir}/merged_profiles.tsv.gz" \
    "${profile_dir}/merged_profiles.tsv" \
    "${profile_dir}/merged_profiles.parquet"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "ERROR: no merged profile table found in ${profile_dir}" >&2
  return 1
}

stage_common_raw_files() {
  local src_dir="$1"
  local dest_dir="$2"
  mkdir -p "${dest_dir}"
  shopt -s nullglob
  local copied=0
  local file base dest_base
  local patterns=(
    "*Image.csv.gz" "*Image.csv"
    "*Acrosome.csv.gz" "*Acrosome.csv"
    "*Arosome.csv.gz" "*Arosome.csv"
    "*FilteredNuclei.csv.gz" "*FilteredNuclei.csv"
    "*Mitochondria.csv.gz" "*Mitochondria.csv"
  )
  for pattern in "${patterns[@]}"; do
    for file in "${src_dir}"/${pattern}; do
      [[ -f "${file}" ]] || continue
      base="$(basename "${file}")"
      [[ "${base}" == ._* ]] && continue
      dest_base="${base}"
      if [[ "${NORMALISE_AROSOME_TO_ACROSOME}" == "1" ]]; then
        dest_base="${dest_base//Arosome/Acrosome}"
        dest_base="${dest_base//arosome/acrosome}"
      fi
      cp -L "${file}" "${dest_dir}/${dest_base}"
      copied=$((copied + 1))
    done
  done
  shopt -u nullglob
  if [[ "${copied}" -eq 0 ]]; then
    echo "ERROR: no common CellProfiler tables were staged from ${src_dir}" >&2
    return 1
  fi
  find "${dest_dir}" -maxdepth 1 -type f -printf '%f\n' | sort > "${dest_dir}/staged_file_list.txt"
}

add_table_if_present() {
  local label="$1"
  local path="$2"
  if [[ -s "${path}" ]]; then
    REPORT_ARGS+=(--table "${label}=${path}")
  fi
}

add_plot_if_present() {
  local label="$1"
  local path="$2"
  if [[ -s "${path}" ]]; then
    REPORT_ARGS+=(--plot "${label}=${path}")
  fi
}

############################################
# Setup
############################################

section "Initial checks"

for command in \
  cpatk-metadata cpatk-build-profiles cpatk-combine-profiles cpatk-preprocess \
  cpatk-strategy-summary cpatk-classical cpatk-visualise cpatk-stability cpatk-batch \
  cpatk-neighbours cpatk-moa cpatk-explain cpatk-latent cpatk-report; do
  require_command "${command}"
done

python - <<'PY'
import cpatk


def parse_version(value):
    return tuple(int(part) for part in str(value).split('.')[:3])


version = getattr(cpatk, '__version__', 'unknown')
print(f'Detected CPATK version: {version}')
if version == 'unknown' or parse_version(version) < parse_version('0.2.26'):
    raise SystemExit(f'CPATK 0.2.26 or newer is required; found {version}')
PY

[[ -s "${MANIFEST}" ]] || { echo "ERROR: manifest not found: ${MANIFEST}" >&2; exit 1; }
[[ -s "${MASTER_METADATA}" ]] || { echo "ERROR: master metadata not found: ${MASTER_METADATA}" >&2; exit 1; }

if [[ "${USE_LOCAL_SCRATCH}" == "1" ]]; then
  WORK_ROOT="$(mktemp -d "${SCRATCH_ROOT%/}/cpatk_multidataset_${RUN_TAG}.XXXXXX")"
  WORK_OUT_DIR="${WORK_ROOT}/results"
else
  WORK_ROOT="${OUT_DIR}"
  WORK_OUT_DIR="${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}" "${WORK_OUT_DIR}" "${WORK_OUT_DIR}/00_inputs"

cp "${MANIFEST}" "${WORK_OUT_DIR}/00_inputs/cpatk_multidataset_manifest.tsv"
cp "${MASTER_METADATA}" "${WORK_OUT_DIR}/00_inputs/cpatk_multidataset_master_metadata.tsv"
MASTER_METADATA_WORK="${WORK_OUT_DIR}/00_inputs/cpatk_multidataset_master_metadata.tsv"
MANIFEST_WORK="${WORK_OUT_DIR}/00_inputs/cpatk_multidataset_manifest.tsv"

{
  echo -e "item\tvalue"
  echo -e "run_tag\t${RUN_TAG}"
  echo -e "out_dir\t${OUT_DIR}"
  echo -e "work_out_dir\t${WORK_OUT_DIR}"
  echo -e "manifest\t${MANIFEST}"
  echo -e "master_metadata\t${MASTER_METADATA}"
  echo -e "cpatk_version\t$(python - <<'PY'
try:
    import cpatk
    print(getattr(cpatk, '__version__', 'unknown'))
except Exception as exc:
    print(f'unknown: {exc}')
PY
)"
  echo -e "protected_features\t${PROTECTED_FEATURES}"
  echo -e "protected_features_file\t${PROTECTED_FEATURES_FILE}"
  echo -e "stage_common_compartments_only\t${STAGE_COMMON_COMPARTMENTS_ONLY}"
  echo -e "normalise_arosome_to_acrosome\t${NORMALISE_AROSOME_TO_ACROSOME}"
} > "${WORK_OUT_DIR}/run_configuration.tsv"

############################################
# 00. Metadata validation
############################################

if [[ "${RUN_METADATA}" == "1" ]]; then
  section "Metadata validation"
  run_step "${WORK_OUT_DIR}/00_metadata_validation/.metadata.done" \
    cpatk-metadata \
      --metadata_table "${MASTER_METADATA_WORK}" \
      --output_dir "${WORK_OUT_DIR}/00_metadata_validation" \
      --plate_column Metadata_Plate \
      --well_column Metadata_Well \
      --duplicate_policy error \
      --log_level "${LOG_LEVEL}"
  FORMATTED_METADATA="${WORK_OUT_DIR}/00_metadata_validation/formatted_metadata.tsv"
else
  FORMATTED_METADATA="${MASTER_METADATA_WORK}"
fi

############################################
# 01. Build profiles per dataset/export
############################################

PROFILE_TABLES=()
SOURCE_LABELS=()

if [[ "${RUN_PROFILE_BUILD}" == "1" ]]; then
  section "Per-dataset profile building"
  tail -n +2 "${MANIFEST_WORK}" | while IFS=$'\t' read -r dataset raw_path metadata_table assay_family batch_hint notes; do
    [[ -n "${dataset}" ]] || continue
    label="$(safe_label "${dataset}")"
    build_dir="${WORK_OUT_DIR}/01_profile_build/${label}"
    mkdir -p "${build_dir}"
    echo -e "dataset\t${dataset}\nraw_path\t${raw_path}\nassay_family\t${assay_family}\nbatch_hint\t${batch_hint}" > "${build_dir}/dataset_build_configuration.tsv"

    input_dir="${raw_path}"
    staged_dir=""
    if [[ "${COPY_EACH_RAW_TO_SCRATCH}" == "1" ]]; then
      staged_dir="${WORK_ROOT}/raw_inputs/${label}"
      rm -rf "${staged_dir}"
      mkdir -p "${staged_dir}"
      if [[ "${STAGE_COMMON_COMPARTMENTS_ONLY}" == "1" ]]; then
        stage_common_raw_files "${raw_path}" "${staged_dir}"
      else
        rsync -a --exclude='._*' --exclude='.DS_Store' --exclude='~$*' "${raw_path%/}/" "${staged_dir}/"
        if [[ "${NORMALISE_AROSOME_TO_ACROSOME}" == "1" ]]; then
          find "${staged_dir}" -maxdepth 1 -type f -name '*Arosome*' | while read -r typo_file; do
            fixed_file="${typo_file//Arosome/Acrosome}"
            mv "${typo_file}" "${fixed_file}"
          done
        fi
      fi
      input_dir="${staged_dir}"
    fi

    run_step "${build_dir}/.build.done" \
      cpatk-build-profiles \
        --input_dir "${input_dir}" \
        --output_dir "${build_dir}" \
        --metadata_table "${FORMATTED_METADATA}" \
        --aggregate_statistic median \
        --image_merge_keys Metadata_Plate,ImageNumber \
        --duplicate_image_policy error \
        --metadata_duplicate_policy error \
        --log_level "${LOG_LEVEL}"

    profile_table="$(find_profile_table "${build_dir}")"
    echo -e "dataset\tprofile_table" > "${build_dir}/profile_table_location.tsv"
    echo -e "${dataset}\t${profile_table}" >> "${build_dir}/profile_table_location.tsv"

    if [[ -n "${staged_dir}" && "${KEEP_SCRATCH}" != "1" ]]; then
      rm -rf "${staged_dir}"
    fi
  done
fi

# Build arrays after the subshell above has completed.
while IFS=$'\t' read -r dataset raw_path metadata_table assay_family batch_hint notes; do
  [[ "${dataset}" == "dataset" ]] && continue
  [[ -n "${dataset}" ]] || continue
  label="$(safe_label "${dataset}")"
  profile_table="$(find_profile_table "${WORK_OUT_DIR}/01_profile_build/${label}")"
  PROFILE_TABLES+=("${profile_table}")
  SOURCE_LABELS+=("${dataset}")
done < "${MANIFEST_WORK}"

PROFILE_TABLES_CSV="$(csv_join "${PROFILE_TABLES[@]}")"
SOURCE_LABELS_CSV="$(csv_join "${SOURCE_LABELS[@]}")"

############################################
# 02. Combine profile tables
############################################

if [[ "${RUN_COMBINE}" == "1" ]]; then
  section "Combining profiles"
  run_step "${WORK_OUT_DIR}/02_combined_profiles/.combine.done" \
    cpatk-combine-profiles \
      --profile_tables "${PROFILE_TABLES_CSV}" \
      --output_dir "${WORK_OUT_DIR}/02_combined_profiles" \
      --source_labels "${SOURCE_LABELS_CSV}" \
      --key_columns Metadata_Profile_Source,Metadata_Plate,ImageNumber,Metadata_Well \
      --feature_join union \
      --duplicate_policy error \
      --log_level "${LOG_LEVEL}"
fi

COMBINED_PROFILE_TABLE="${WORK_OUT_DIR}/02_combined_profiles/combined_profiles.tsv.gz"
[[ -s "${COMBINED_PROFILE_TABLE}" ]] || { echo "ERROR: combined profile table not found: ${COMBINED_PROFILE_TABLE}" >&2; exit 1; }

############################################
# 03. Preprocessing strategy comparison
############################################

run_preprocess_strategy() {
  local strategy_name="$1"
  shift
  local strategy_dir="${WORK_OUT_DIR}/03_preprocess_strategy_comparison/${strategy_name}"
  local args=(
    cpatk-preprocess
    --input_table "${COMBINED_PROFILE_TABLE}"
    --output_dir "${strategy_dir}"
    --metadata_columns "${METADATA_COLUMNS}"
    --imputation_method median
    --scaling_method robust
    --max_feature_missing_fraction 0.30
    --max_sample_missing_fraction 0.60
    --min_feature_variance 1e-12
    --max_absolute_correlation 0.95
    --correlation_method spearman
    --correlation_filter_strategy variance
    --max_features_for_correlation 5000
    --max_zero_fraction 1.0
    --replicate_group_columns "${REPLICATE_GROUP_COLUMNS}"
    --batch_report_columns "${BATCH_REPORT_COLUMNS}"
    --log_level "${LOG_LEVEL}"
  )
  if [[ -n "${PROTECTED_FEATURES}" ]]; then
    args+=(--protected_features "${PROTECTED_FEATURES}")
  fi
  if [[ -n "${PROTECTED_FEATURES_FILE}" ]]; then
    args+=(--protected_features_file "${PROTECTED_FEATURES_FILE}")
  fi
  args+=("$@")
  run_step "${strategy_dir}/.preprocess.done" "${args[@]}"
}

if [[ "${RUN_PREPROCESSING}" == "1" ]]; then
  section "Preprocessing strategy comparison"
  run_preprocess_strategy "01_none_robust_scale" \
    --reference_normalisation_method none \
    --batch_correction_method none

  run_preprocess_strategy "02_dmso_robust_z_by_plate" \
    --reference_normalisation_method robust_z \
    --reference_column "${REFERENCE_COLUMN}" \
    --reference_values "${REFERENCE_VALUES}" \
    --reference_group_columns "${REFERENCE_GROUP_COLUMNS}" \
    --batch_correction_method none

  COMBAT_ARGS=(
    cpatk-preprocess
    --input_table "${COMBINED_PROFILE_TABLE}"
    --output_dir "${WORK_OUT_DIR}/03_preprocess_strategy_comparison/03_dmso_robust_z_plus_combat_source"
    --metadata_columns "${METADATA_COLUMNS}"
    --imputation_method median
    --scaling_method robust
    --max_feature_missing_fraction 0.30
    --max_sample_missing_fraction 0.60
    --min_feature_variance 1e-12
    --max_absolute_correlation 0.95
    --correlation_method spearman
    --correlation_filter_strategy variance
    --max_features_for_correlation 5000
    --max_zero_fraction 1.0
    --reference_normalisation_method robust_z
    --reference_column "${REFERENCE_COLUMN}"
    --reference_values "${REFERENCE_VALUES}"
    --reference_group_columns "${REFERENCE_GROUP_COLUMNS}"
    --batch_correction_method combat_location_scale
    --batch_column "${BATCH_COLUMN}"
    --batch_protect_columns "${BATCH_PROTECT_COLUMNS}"
    --replicate_group_columns "${REPLICATE_GROUP_COLUMNS}"
    --batch_report_columns "${BATCH_REPORT_COLUMNS}"
    --log_level "${LOG_LEVEL}"
  )
  if [[ -n "${PROTECTED_FEATURES}" ]]; then
    COMBAT_ARGS+=(--protected_features "${PROTECTED_FEATURES}")
  fi
  if [[ -n "${PROTECTED_FEATURES_FILE}" ]]; then
    COMBAT_ARGS+=(--protected_features_file "${PROTECTED_FEATURES_FILE}")
  fi
  run_soft_step "${WORK_OUT_DIR}/03_preprocess_strategy_comparison/03_dmso_robust_z_plus_combat_source/.preprocess.done" "${COMBAT_ARGS[@]}"

  run_step "${WORK_OUT_DIR}/03_preprocess_strategy_comparison/.strategy_summary.done" \
    cpatk-strategy-summary \
      --strategy_root "${WORK_OUT_DIR}/03_preprocess_strategy_comparison" \
      --output_table "${WORK_OUT_DIR}/03_preprocess_strategy_comparison/normalisation_strategy_comparison.tsv" \
      --batch_column "${BATCH_COLUMN}" \
      --compound_column "${ID_COLUMN}" \
      --log_level "${LOG_LEVEL}"
fi

PRIMARY_STRATEGY_DIR="${PRIMARY_STRATEGY_DIR:-${WORK_OUT_DIR}/03_preprocess_strategy_comparison/02_dmso_robust_z_by_plate}"
PRIMARY_MATRIX="${PRIMARY_STRATEGY_DIR}/preprocessed.tsv.gz"
if [[ ! -s "${PRIMARY_MATRIX}" ]]; then
  PRIMARY_MATRIX="${PRIMARY_STRATEGY_DIR}/preprocessed.parquet"
fi
[[ -s "${PRIMARY_MATRIX}" ]] || { echo "ERROR: primary preprocessed matrix not found in ${PRIMARY_STRATEGY_DIR}" >&2; exit 1; }

############################################
# 04+. Downstream analysis
############################################

if [[ "${RUN_CLASSICAL}" == "1" ]]; then
  section "Classical analysis"
  run_step "${WORK_OUT_DIR}/04_classical/.classical.done" \
    cpatk-classical \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/04_classical" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --id_column "${ID_COLUMN}" \
      --colour_column "${BATCH_COLUMN}" \
      --cluster_group_columns "${REPLICATE_GROUP_COLUMNS}" \
      --n_neighbours "${N_NEIGHBOURS}" \
      --n_clusters "${N_CLUSTERS}" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_VISUALISE}" == "1" ]]; then
  section "Visualisation"
  run_soft_step "${WORK_OUT_DIR}/05_visualisation/.visualise.done" \
    cpatk-visualise \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/05_visualisation" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --id_column "${ID_COLUMN}" \
      --colour_columns "${BATCH_REPORT_COLUMNS}" \
      --aggregate_by_id \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_STABILITY}" == "1" ]]; then
  section "Replicate/stability QC"
  run_soft_step "${WORK_OUT_DIR}/06_stability/.stability.done" \
    cpatk-stability \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/06_stability" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --replicate_group_columns "${REPLICATE_GROUP_COLUMNS}" \
      --n_clusters "${N_CLUSTERS}" \
      --n_bootstraps "${STABILITY_BOOTSTRAPS}" \
      --n_permutations "${STABILITY_PERMUTATIONS}" \
      --n_neighbours "${N_NEIGHBOURS}" \
      --k_values 2,3,4,5,6,7,8,9,10 \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_BATCH}" == "1" ]]; then
  section "Batch/domain diagnostics"
  run_soft_step "${WORK_OUT_DIR}/07_batch/.batch.done" \
    cpatk-batch \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/07_batch" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --batch_column "${BATCH_COLUMN}" \
      --columns_to_test "${BATCH_REPORT_COLUMNS},${ID_COLUMN}" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_NEIGHBOURS}" == "1" ]]; then
  section "Neighbour analysis"
  NN_FILE="${WORK_OUT_DIR}/04_classical/nearest_neighbours.tsv"
  if [[ -s "${NN_FILE}" ]]; then
    run_soft_step "${WORK_OUT_DIR}/08_neighbours/.neighbours.done" \
      cpatk-neighbours \
        --input_neighbours "${NN_FILE}" \
        --output_dir "${WORK_OUT_DIR}/08_neighbours" \
        --compounds "${QUERY_IDS}" \
        --top_n 20 \
        --log_level "${LOG_LEVEL}"
  else
    echo "WARN: nearest-neighbour file not found; skipping cpatk-neighbours" >&2
  fi
fi

if [[ "${RUN_MOA}" == "1" ]]; then
  section "MOA / pseudo-anchor analysis"
  run_soft_step "${WORK_OUT_DIR}/09_moa/.moa.done" \
    cpatk-moa \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/09_moa" \
      --id_column "${ID_COLUMN}" \
      --metadata_columns "${METADATA_COLUMNS}" \
      --metric cosine \
      --top_n 10 \
      --run_knn \
      --n_neighbors "${N_NEIGHBOURS}" \
      --make_pseudo_anchors \
      --auto_k \
      --n_clusters 30 \
      --n_bootstraps "${MOA_BOOTSTRAPS}" \
      --n_permutations "${MOA_PERMUTATIONS}" \
      --distance_metrics cosine,spearman \
      --make_projection_plots \
      --interactive \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_ML}" == "1" ]]; then
  section "Optional ML"
  run_soft_step "${WORK_OUT_DIR}/10_ml/.ml.done" \
    cpatk-ml \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/10_ml" \
      --class_column cpd_type \
      --metadata_columns "${METADATA_COLUMNS}" \
      --compare_models \
      --n_splits "${ML_CV_SPLITS}" \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_EXPLAIN}" == "1" ]]; then
  section "Explainability"
  run_soft_step "${WORK_OUT_DIR}/11_explain/.explain.done" \
    cpatk-explain \
      --input_table "${PRIMARY_MATRIX}" \
      --output_dir "${WORK_OUT_DIR}/11_explain" \
      --class_column cpd_type \
      --metadata_columns "${METADATA_COLUMNS}" \
      --id_column "${ID_COLUMN}" \
      --query_ids "${QUERY_IDS}" \
      --run_neighbourhood_shap \
      --run_feature_tests \
      --background_column cpd_type \
      --background_values DMSO,control,negative_control \
      --include_shap \
      --max_shap_background 200 \
      --max_shap_explain 200 \
      --log_level "${LOG_LEVEL}"
fi

if [[ "${RUN_CLIPN}" == "1" ]]; then
  section "CLIPn multi-dataset integration"
  CLIPN_DATASET_DIR="${WORK_OUT_DIR}/12_clipn_datasets"
  mkdir -p "${CLIPN_DATASET_DIR}"
  run_step "${CLIPN_DATASET_DIR}/.split.done" \
    python - "${PRIMARY_MATRIX}" "${CLIPN_DATASET_DIR}" <<'PY'
from pathlib import Path
import re
import sys
import pandas as pd

matrix = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
if matrix.suffix == ".parquet":
    df = pd.read_parquet(matrix)
else:
    df = pd.read_csv(matrix, sep="\t", low_memory=False)
source_col = "Metadata_Profile_Source"
if source_col not in df.columns:
    raise SystemExit(f"{source_col} is missing from {matrix}")
records = []
for source, group in df.groupby(source_col, dropna=False, sort=False):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source)).strip("_") or "unknown_source"
    path = out_dir / f"{label}.tsv.gz"
    group.to_csv(path, sep="\t", index=False, compression="gzip")
    records.append({"dataset": label, "path": str(path), "n_rows": int(group.shape[0])})
manifest = pd.DataFrame.from_records(records)
manifest[["dataset", "path"]].to_csv(out_dir / "clipn_datasets.tsv", sep="\t", index=False)
manifest.to_csv(out_dir / "clipn_dataset_split_summary.tsv", sep="\t", index=False)
print(manifest.to_string(index=False))
PY

  CLIPN_ARGS=(
    cpatk-latent
    --backend_module "${LATENT_BACKEND_MODULE}"
    --datasets_csv "${CLIPN_DATASET_DIR}/clipn_datasets.tsv"
    --output_dir "${WORK_OUT_DIR}/12_clipn"
    --experiment cpatk_multidataset_clipn
    --mode integrate_all
    --latent_dim "${CLIPN_LATENT_DIM}"
    --epochs "${CLIPN_EPOCHS}"
    --clipn_patience "${CLIPN_PATIENCE}"
    --clipn_min_delta "${CLIPN_MIN_DELTA}"
    --clipn_epoch_chunk_size "${CLIPN_EPOCH_CHUNK_SIZE}"
    --clipn_validation_fraction "${CLIPN_VALIDATION_FRACTION}"
    --metadata_columns "${METADATA_COLUMNS}"
    --id_column "${ID_COLUMN}"
    --label_column cpd_type
    --n_neighbours "${N_NEIGHBOURS}"
    --clipn_zero_policy "${CLIPN_ZERO_POLICY}"
    --native_hidden_dims "${NATIVE_HIDDEN_DIMS}"
    --native_batch_size "${NATIVE_BATCH_SIZE}"
    --native_steps_per_epoch "${NATIVE_STEPS_PER_EPOCH}"
    --native_device "${NATIVE_DEVICE}"
    --native_positive_column "${ID_COLUMN}"
    --log_level "${LOG_LEVEL}"
  )
  if [[ "${CLIPN_EARLY_STOPPING}" == "1" ]]; then
    CLIPN_ARGS+=(--clipn_early_stopping)
  fi
  if [[ "${CLIPN_ALLOW_PCA_FALLBACK}" == "1" ]]; then
    CLIPN_ARGS+=(--allow_pca_fallback)
  fi
  run_soft_step "${WORK_OUT_DIR}/12_clipn/.clipn.done" "${CLIPN_ARGS[@]}"
fi

if [[ "${RUN_CLIPN_LATENT_MOA}" == "1" ]]; then
  section "CLIPn latent-space MOA"
  CLIPN_LATENT_TABLE="${WORK_OUT_DIR}/12_clipn/clipn_latent.tsv.gz"
  CLIPN_STATUS="${WORK_OUT_DIR}/12_clipn/clipn_run_status.tsv"
  if [[ -s "${CLIPN_LATENT_TABLE}" && -s "${CLIPN_STATUS}" ]] && grep -q $'backend_run\tsuccess' "${CLIPN_STATUS}"; then
    run_soft_step "${WORK_OUT_DIR}/13_clipn_latent_moa/.clipn_latent_moa.done" \
      cpatk-moa \
        --input_table "${CLIPN_LATENT_TABLE}" \
        --output_dir "${WORK_OUT_DIR}/13_clipn_latent_moa" \
        --id_column "${ID_COLUMN}" \
        --metadata_columns "Dataset,Sample,${METADATA_COLUMNS}" \
        --feature_columns latent_1,latent_2,latent_3,latent_4,latent_5,latent_6,latent_7,latent_8,latent_9,latent_10 \
        --metric cosine \
        --top_n 10 \
        --run_knn \
        --n_neighbors "${N_NEIGHBOURS}" \
        --make_pseudo_anchors \
        --auto_k \
        --n_clusters 30 \
        --n_bootstraps "${MOA_BOOTSTRAPS}" \
        --n_permutations "${MOA_PERMUTATIONS}" \
        --distance_metrics cosine,spearman \
        --log_level "${LOG_LEVEL}"
  else
    echo "Skipping CLIPn latent MOA: true CLIPn backend success was not recorded." >&2
  fi
fi

############################################
# Final report
############################################

if [[ "${RUN_FINAL_REPORT}" == "1" ]]; then
  section "Final report"
  REPORT_ARGS=(
    --output_html "${WORK_OUT_DIR}/CPATK_multidataset_v0_2_26_full_report.html"
    --title "CPATK v0.2.26 multi-dataset Cell Painting stress test"
    --narrative "Large CPATK multi-dataset stress test across STB1/STB2, six SelleckChem batches and mitotox. Profiles were built per export, common raw compartments were staged, the SelleckChem Arosome filename typo was normalised to Acrosome in scratch, then profile tables were combined and analysed jointly."
    --warning "This is an ambitious stress test. Interpret biological clustering only after checking profile build, metadata merge, missingness, batch and replicate diagnostics."
    --auto_plot_root "${WORK_OUT_DIR}"
    --strategy_root "${WORK_OUT_DIR}/03_preprocess_strategy_comparison"
    --strategy_batch_column "${BATCH_COLUMN}"
    --strategy_compound_column "${ID_COLUMN}"
    --export_method_guide
    --max_auto_plots 250
    --max_table_rows 40
    --log_level "${LOG_LEVEL}"
  )

  add_table_if_present "Run configuration" "${WORK_OUT_DIR}/run_configuration.tsv"
  add_table_if_present "Input manifest" "${WORK_OUT_DIR}/00_inputs/cpatk_multidataset_manifest.tsv"
  add_table_if_present "Metadata validation summary" "${WORK_OUT_DIR}/00_metadata_validation/metadata_validation_summary.tsv"
  add_table_if_present "Combined profile summary" "${WORK_OUT_DIR}/02_combined_profiles/combine_profile_summary.tsv"
  add_table_if_present "Feature presence matrix" "${WORK_OUT_DIR}/02_combined_profiles/feature_presence_matrix.tsv"
  add_table_if_present "Normalisation strategy comparison" "${WORK_OUT_DIR}/03_preprocess_strategy_comparison/normalisation_strategy_comparison.tsv"
  add_table_if_present "Primary preprocessing summary" "${PRIMARY_STRATEGY_DIR}/preprocessing_summary.tsv"
  add_table_if_present "Feature selection summary" "${PRIMARY_STRATEGY_DIR}/feature_selection_summary.tsv"
  add_table_if_present "Protected feature audit" "${PRIMARY_STRATEGY_DIR}/protected_feature_audit.tsv"
  add_table_if_present "Correlation filter report" "${PRIMARY_STRATEGY_DIR}/correlation_filter_report.tsv"
  add_table_if_present "Final matrix validation" "${PRIMARY_STRATEGY_DIR}/final_matrix_validation.tsv"
  add_table_if_present "Classical PCA variance" "${WORK_OUT_DIR}/04_classical/pca_explained_variance.tsv"
  add_table_if_present "Classical cluster summary" "${WORK_OUT_DIR}/04_classical/cluster_summary.tsv"
  add_table_if_present "Batch association" "${WORK_OUT_DIR}/07_batch/metadata_pc_association.tsv"
  add_table_if_present "MOA predictions" "${WORK_OUT_DIR}/09_moa/advanced_moa_top_predictions.tsv"
  add_table_if_present "Explainability summary" "${WORK_OUT_DIR}/11_explain/explainability_summary.tsv"
  add_table_if_present "CLIPn backend provenance" "${WORK_OUT_DIR}/12_clipn/clipn_backend_provenance.tsv"
  add_table_if_present "CLIPn training summary" "${WORK_OUT_DIR}/12_clipn/clipn_training_summary.tsv"
  add_table_if_present "CLIPn latent diagnostics" "${WORK_OUT_DIR}/12_clipn/latent_diagnostic_summary.tsv"
  add_table_if_present "CLIPn latent MOA predictions" "${WORK_OUT_DIR}/13_clipn_latent_moa/advanced_moa_top_predictions.tsv"

  run_soft_step "${WORK_OUT_DIR}/.final_report.done" cpatk-report "${REPORT_ARGS[@]}"
fi

section "Workflow complete"
echo "Project output directory: ${OUT_DIR}"
echo "Working output directory: ${WORK_OUT_DIR}"
