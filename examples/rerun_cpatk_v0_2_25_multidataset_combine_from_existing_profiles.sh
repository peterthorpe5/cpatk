#!/usr/bin/env bash
#$ -cwd
#$ -V
#$ -pe smp 4
#$ -l h_rt=04:00:00
#$ -l h_vmem=8G
#$ -N cpatk_multi_combine_v025
#$ -o cpatk_multi_combine_v025.o$JOB_ID
#$ -e cpatk_multi_combine_v025.e$JOB_ID

set -euo pipefail

log_section() {
  printf '\n==== %s ====\n\n' "$1"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Required file not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: Required directory not found: ${path}" >&2
    exit 1
  fi
}

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

if [[ -z "${OUT_DIR:-}" ]]; then
  echo "ERROR: Set OUT_DIR to the existing failed CPATK multi-dataset output directory." >&2
  echo "Example:" >&2
  echo "  OUT_DIR=/path/to/cpatk_multidataset_output bash $0" >&2
  exit 1
fi

OUT_DIR="$(readlink -f "${OUT_DIR}")"
RESULTS_DIR="${RESULTS_DIR:-${OUT_DIR}}"
if [[ -d "${OUT_DIR}/results/01_profile_build" && ! -d "${RESULTS_DIR}/01_profile_build" ]]; then
  RESULTS_DIR="${OUT_DIR}/results"
fi
RESULTS_DIR="$(readlink -f "${RESULTS_DIR}")"
PROFILE_DIR="${RESULTS_DIR}/01_profile_build"
COMBINE_DIR="${RESULTS_DIR}/02_combined_profiles"
MANIFEST="${RESULTS_DIR}/00_inputs/cpatk_multidataset_manifest.tsv"
KEY_COLUMNS="${KEY_COLUMNS:-Metadata_Profile_Source,Metadata_Plate,ImageNumber,Metadata_Well}"
FEATURE_JOIN="${FEATURE_JOIN:-union}"
DUPLICATE_POLICY="${DUPLICATE_POLICY:-error}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

require_dir "${PROFILE_DIR}"
require_file "${MANIFEST}"

log_section "Environment"
python --version
cpatk-combine-profiles --help >/dev/null
python - <<'PY'
import cpatk
print(f"CPATK version: {cpatk.__version__}")
PY

log_section "Discover profile tables from manifest"
PROFILE_TABLES=""
SOURCE_LABELS=""
while IFS=$'\t' read -r dataset _rest; do
  if [[ "${dataset}" == "dataset" || -z "${dataset}" ]]; then
    continue
  fi
  profile_table="${PROFILE_DIR}/${dataset}/merged_profiles.parquet"
  if [[ ! -f "${profile_table}" ]]; then
    profile_table="${PROFILE_DIR}/${dataset}/merged_profiles.tsv.gz"
  fi
  require_file "${profile_table}"
  if [[ -z "${PROFILE_TABLES}" ]]; then
    PROFILE_TABLES="${profile_table}"
    SOURCE_LABELS="${dataset}"
  else
    PROFILE_TABLES="${PROFILE_TABLES},${profile_table}"
    SOURCE_LABELS="${SOURCE_LABELS},${dataset}"
  fi
  echo "${dataset}: ${profile_table}"
done < "${MANIFEST}"

if [[ -z "${PROFILE_TABLES}" ]]; then
  echo "ERROR: No profile tables were discovered from ${MANIFEST}" >&2
  exit 1
fi

log_section "Prepare output directory"
if [[ -d "${COMBINE_DIR}" ]]; then
  BACKUP_DIR="${COMBINE_DIR}.before_v0_2_25_$(date +%Y%m%d_%H%M%S)"
  echo "Existing combine directory found; moving to ${BACKUP_DIR}"
  mv "${COMBINE_DIR}" "${BACKUP_DIR}"
fi
mkdir -p "${COMBINE_DIR}"

log_section "Run cpatk-combine-profiles"
set -x
cpatk-combine-profiles \
  --profile_tables "${PROFILE_TABLES}" \
  --output_dir "${COMBINE_DIR}" \
  --source_labels "${SOURCE_LABELS}" \
  --key_columns "${KEY_COLUMNS}" \
  --feature_join "${FEATURE_JOIN}" \
  --duplicate_policy "${DUPLICATE_POLICY}" \
  --log_level "${LOG_LEVEL}"
set +x

log_section "Key reports"
python - <<PY
from pathlib import Path
import pandas as pd
combine_dir = Path("${COMBINE_DIR}")
for name in ["combine_profile_summary.tsv", "combined_duplicate_key_report.tsv", "combined_key_candidate_report.tsv"]:
    path = combine_dir / name
    print(f"\n### {name}")
    if not path.exists():
        print("missing")
        continue
    data_frame = pd.read_csv(path, sep="\t")
    print(data_frame.head(20).to_string(index=False))
PY

log_section "Done"
echo "Combined profile table: ${COMBINE_DIR}/combined_profiles.tsv.gz"
echo "Next step: rerun downstream preprocessing/report steps using this combined table."
