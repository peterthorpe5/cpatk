# CPATK v0.2.29 synthetic latent benchmark release notes

Date: 2026-06-30

## Purpose

CPATK v0.2.29 adds a controlled synthetic benchmark for developing and testing
CPATK-native contrastive latent embeddings. The aim is to avoid judging the
latent method only from messy real datasets where true biological structure,
batch effects and label quality are unknown.

The benchmark generates Cell Painting-like profile tables with known compound,
MOA, dataset and batch structure, then compares:

- raw feature-space nearest-neighbour retrieval;
- PCA latent nearest-neighbour retrieval;
- CPATK-native supervised contrastive latent retrieval.

## New command

```bash
cpatk-synthetic-latent-benchmark \
  --output_dir cpatk_synthetic_latent_benchmark \
  --threads 16
```

The command runs four default scenarios:

1. `clean_biology` — compound biology is strong and batch effects are modest.
2. `batch_confounded_biology` — real compound biology exists but dataset/batch
   effects dominate raw/PCA feature space.
3. `weak_biology` — biology is present but deliberately subtle.
4. `no_biology_negative_control` — no compound biology is simulated, so the
   contrastive layer should not invent a strong compound-specific signal.

## Key outputs

At the top level:

- `synthetic_metric_summary.tsv`
- `synthetic_pass_fail_summary.tsv`
- `synthetic_scenario_configs.tsv`
- `native_contrastive_training_summary.tsv`
- `native_contrastive_quality_summary.tsv`
- `synthetic_latent_benchmark_summary.xlsx`

Within each scenario folder:

- `synthetic_profiles.tsv.gz`
- `synthetic_ground_truth.tsv`
- `raw_feature_neighbours.tsv.gz`
- `pca_latent.tsv.gz`
- `pca_neighbours.tsv.gz`
- `cpatk_contrastive_latent.tsv.gz`
- `cpatk_contrastive_training_loss.tsv`
- `cpatk_contrastive_training_summary.tsv`
- `cpatk_contrastive_neighbours.tsv.gz`
- `scenario_metric_summary.tsv`
- `scenario_pass_fail.tsv`

## Smoke-test behaviour observed in the sandbox

A small benchmark was run with 16 compounds, 2 datasets, 3 replicates per
compound per dataset, 60 features, 15 epochs and 2 sampled mini-batches per
epoch.

Selected top-1 same-compound retrieval rates:

| scenario | raw features | PCA | CPATK-native contrastive |
|---|---:|---:|---:|
| clean_biology | 0.833 | 0.385 | 0.917 |
| batch_confounded_biology | 0.031 | 0.000 | 0.729 |
| weak_biology | 0.000 | 0.000 | 0.521 |
| no_biology_negative_control | 0.000 | 0.000 | 0.115 |

This is the expected broad pattern: the native contrastive model recovers
compound structure when biology is simulated, helps under batch-confounded
conditions, and stays low in the no-biology negative control. This is not a
claim that the method is fully validated on real data; it is a controlled
software/method-development benchmark.

## New SGE shell

A cluster shell is included:

```bash
qsub run_cpatk_v0_2_29_synthetic_latent_benchmark.sh
```

It writes real tab-separated configuration output and avoids the literal `\t`
quoting bug that was seen in earlier collector scripts.

## Validation run

Passed in the sandbox:

```text
python -m compileall -q cpatk tests
python -m unittest tests.test_cpatk_v0_2_29 -v
python -m unittest tests.test_cpatk_v0_2_26 tests.test_cpatk_v0_2_27 tests.test_cpatk_v0_2_28 tests.test_cpatk_v0_2_29 -v
bash -n run_cpatk_v0_2_29_synthetic_latent_benchmark.sh
python -m cpatk.cli.synthetic_latent_benchmark --output_dir /mnt/data/cpatk_029_synth_smoke ...
```

The combined focused test run covered 17 tests and passed.

## Interpretation

This benchmark should become the development harness for the CPATK-native latent
layer. Useful future extensions include:

- a small hyperparameter grid over latent dimension, temperature, dropout and
  positive-pair sampling;
- repeated random seeds for stability intervals;
- more explicit simulation of plate-wise DMSO normalisation;
- a real-data benchmark wrapper comparing malaria and mitotox latent results
  against the same metrics used here.
