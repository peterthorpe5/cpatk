#!/usr/bin/env bash
#$ -jc rhel9
#$ -j y
#$ -N cpatk_synth_v032
#$ -jc long
#$ -pe smp 16
#$ -mods l_hard mfree 96G
#$ -adds l_hard h_vmem 96G
#$ -cwd
# Optional GPU-backed native contrastive benchmark. Uncomment if desired and
# if these resource names match the local SGE setup.
##$ -adds l_hard gpu 1
##$ -adds l_hard cuda.0.name 'NVIDIA A40'

# CPATK v0.2.32 comprehensive synthetic latent validation.
# This is deliberately more demanding than the v0.2.30 smoke benchmark. It runs
# repeated random seeds across clean, weak, confounded, noisy-label, missingness,
# missing-compartment, outlier and negative-control scenarios. The primary metric
# is held-out validation-to-train compound retrieval, not all-row retrieval.

set -Eeuo pipefail
IFS=$'\n\t'

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-/home/${USER}/data/2025_jason_cell_painting/data/cpatk_synthetic_benchmarks}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/cpatk_v0_2_32_comprehensive_synthetic_latent_${RUN_TAG}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpatk}"
THREADS="${THREADS:-${NSLOTS:-16}}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

BENCHMARK_MODE="${BENCHMARK_MODE:-comprehensive}"
# Leave SCENARIOS empty to use the v0.2.32 comprehensive preset.
SCENARIOS="${SCENARIOS:-}"
SEED_VALUES="${SEED_VALUES:-42,101,202,303,404}"
N_COMPOUNDS="${N_COMPOUNDS:-48}"
N_MOA_CLASSES="${N_MOA_CLASSES:-8}"
N_BATCHES="${N_BATCHES:-4}"
N_DATASETS="${N_DATASETS:-2}"
REPLICATES_PER_COMPOUND_DATASET="${REPLICATES_PER_COMPOUND_DATASET:-4}"
N_FEATURES="${N_FEATURES:-240}"
N_INFORMATIVE_FEATURES="${N_INFORMATIVE_FEATURES:-80}"
LATENT_DIM="${LATENT_DIM:-16}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.15}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TEMPERATURE="${TEMPERATURE:-0.10}"
HIDDEN_DIMS="${HIDDEN_DIMS:-512,256}"
DROPOUT="${DROPOUT:-0.10}"
N_NEIGHBOURS="${N_NEIGHBOURS:-5}"
RANDOM_STATE="${RANDOM_STATE:-42}"

section() {
  printf '\n==== %s ====\n\n' "$*"
}

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

section "Comprehensive synthetic latent validation configuration"
mkdir -p "${OUT_DIR}"
{
  printf 'item\tvalue\n'
  printf 'run_tag\t%s\n' "${RUN_TAG}"
  printf 'out_dir\t%s\n' "${OUT_DIR}"
  printf 'threads\t%s\n' "${THREADS}"
  printf 'benchmark_mode\t%s\n' "${BENCHMARK_MODE}"
  printf 'scenarios\t%s\n' "${SCENARIOS:-v0.2.32_comprehensive_preset}"
  printf 'seed_values\t%s\n' "${SEED_VALUES}"
  printf 'n_compounds\t%s\n' "${N_COMPOUNDS}"
  printf 'n_moa_classes\t%s\n' "${N_MOA_CLASSES}"
  printf 'n_batches\t%s\n' "${N_BATCHES}"
  printf 'n_datasets\t%s\n' "${N_DATASETS}"
  printf 'replicates_per_compound_dataset\t%s\n' "${REPLICATES_PER_COMPOUND_DATASET}"
  printf 'n_features\t%s\n' "${N_FEATURES}"
  printf 'n_informative_features\t%s\n' "${N_INFORMATIVE_FEATURES}"
  printf 'latent_dim\t%s\n' "${LATENT_DIM}"
  printf 'epochs\t%s\n' "${EPOCHS}"
  printf 'batch_size\t%s\n' "${BATCH_SIZE}"
  printf 'steps_per_epoch\t%s\n' "${STEPS_PER_EPOCH}"
  printf 'validation_fraction\t%s\n' "${VALIDATION_FRACTION}"
  printf 'learning_rate\t%s\n' "${LEARNING_RATE}"
  printf 'temperature\t%s\n' "${TEMPERATURE}"
  printf 'hidden_dims\t%s\n' "${HIDDEN_DIMS}"
  printf 'dropout\t%s\n' "${DROPOUT}"
  printf 'n_neighbours\t%s\n' "${N_NEIGHBOURS}"
  printf 'random_state\t%s\n' "${RANDOM_STATE}"
} > "${OUT_DIR}/run_configuration.tsv"

section "Run CPATK comprehensive synthetic latent validation"
CMD=(
  cpatk-synthetic-latent-benchmark
  --output_dir "${OUT_DIR}"
  --benchmark_mode "${BENCHMARK_MODE}"
  --seed_values "${SEED_VALUES}"
  --n_compounds "${N_COMPOUNDS}"
  --n_moa_classes "${N_MOA_CLASSES}"
  --n_batches "${N_BATCHES}"
  --n_datasets "${N_DATASETS}"
  --replicates_per_compound_dataset "${REPLICATES_PER_COMPOUND_DATASET}"
  --n_features "${N_FEATURES}"
  --n_informative_features "${N_INFORMATIVE_FEATURES}"
  --latent_dim "${LATENT_DIM}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --steps_per_epoch "${STEPS_PER_EPOCH}"
  --validation_fraction "${VALIDATION_FRACTION}"
  --learning_rate "${LEARNING_RATE}"
  --temperature "${TEMPERATURE}"
  --hidden_dims "${HIDDEN_DIMS}"
  --random_state "${RANDOM_STATE}"
  --n_neighbours "${N_NEIGHBOURS}"
  --threads "${THREADS}"
  --log_level "${LOG_LEVEL}"
)
if [[ -n "${SCENARIOS}" ]]; then
  CMD+=(--scenarios "${SCENARIOS}")
fi
set -x
"${CMD[@]}"
set +x

section "Comprehensive synthetic latent validation complete"
echo "Output directory: ${OUT_DIR}"
echo "Metric summary: ${OUT_DIR}/synthetic_metric_summary.tsv"
echo "Pass/fail summary: ${OUT_DIR}/synthetic_pass_fail_summary.tsv"
echo "Decision summary: ${OUT_DIR}/synthetic_validation_decision_summary.tsv"
echo "Excel summary: ${OUT_DIR}/synthetic_latent_benchmark_summary.xlsx"
