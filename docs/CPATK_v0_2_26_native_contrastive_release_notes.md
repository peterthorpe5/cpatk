# CPATK v0.2.26 native contrastive latent backend

Date: 2026-06-29

## Main purpose

This release adds a CPATK-native supervised contrastive latent embedding backend and makes it the default latent method. The published external CLIPn package is retained only as an explicitly requested compatibility backend.

This follows the project decision that CPATK should not depend on the published CLIPn implementation as the default, because the inspected code is too thin for a robust, reviewer-facing analysis backend.

## User-facing policy change

Default backend:

```bash
--backend_module cpatk_contrastive
```

External published CLIPn backend, only when explicitly requested:

```bash
--backend_module clipn
```

The older command name remains available:

```bash
cpatk-clipn
```

A clearer alias has been added:

```bash
cpatk-latent
```

## Native backend features

The new native backend includes:

- a real PyTorch `nn.Module` MLP encoder with configurable hidden layers;
- activation functions, dropout and optional layer/batch normalisation;
- row-wise L2-normalised latent embeddings;
- supervised contrastive loss with positive-pair checking;
- positive-pair batch sampling;
- multiple sampled mini-batches per epoch rather than one batch per epoch;
- configurable automatic or fixed `steps_per_epoch`;
- train/validation splitting that preserves positive pairs where possible;
- early stopping using validation loss when possible, otherwise sampled training loss;
- best-model state restoration before final encoding;
- reproducible NumPy and PyTorch seeding;
- CPU/GPU device resolution;
- chunked latent encoding for larger matrices;
- backend provenance and PyTorch/CUDA status reporting;
- positive-label replication audit;
- split audit;
- training-loss table;
- latent nearest-neighbour diagnostics;
- latent quality warnings when retrieval is poor or dataset/source structure dominates.

## New/updated outputs

When the native backend runs, the latent output folder includes:

```text
latent_backend_policy.tsv
clipn_run_status.tsv
clipn_backend_provenance.tsv
clipn_training_summary.tsv
clipn_training_loss.tsv
clipn_latent.tsv.gz
cpatk_contrastive_positive_label_report.tsv
cpatk_contrastive_split_report.tsv
cpatk_contrastive_backend_status.tsv
latent_diagnostic_summary.tsv
latent_quality_warnings.tsv
nearest_neighbours.tsv
latent_variance.tsv
clipn_report.html
clipn_summary.xlsx
```

The `clipn_` prefixes are retained for backward compatibility with older CPATK collectors and reports, but the backend provenance and `latent_backend_policy.tsv` state whether the run used CPATK-native contrastive learning or the external published CLIPn package.

## Multi-dataset shell update

A new full SGE shell is included:

```text
run_cpatk_v0_2_26_multidataset_stb_selleck_mitotox_full_sge.sh
```

It uses `cpatk-latent` and defaults to:

```bash
LATENT_BACKEND_MODULE=cpatk_contrastive
```

Set this only if the external published backend is deliberately wanted:

```bash
LATENT_BACKEND_MODULE=clipn
```

## Validation performed in the sandbox

Passed:

```text
python -m compileall -q cpatk tests
python -m unittest tests.test_cpatk_v0_2_26 -v
python -m unittest tests.test_cpatk_v0_2 tests.test_cpatk_v0_2_7 tests.test_cpatk_v0_2_21 tests.test_cpatk_v0_2_23 tests.test_cpatk_v0_2_25 -v
```

A command-line smoke test also passed using the default backend through `python -m cpatk.cli.clipn`, confirming that the selected backend was `cpatk_contrastive` and that `latent_backend_policy.tsv`, `clipn_run_status.tsv`, `clipn_latent.tsv.gz`, native audit tables and the HTML report were written.

A long combined unittest command including all the above plus the v0.2.26 torch test reached the final native test and timed out in the sandbox after many tests had already passed. The v0.2.26 test module passed when run directly, and the older relevant modules passed separately. Therefore, do not claim a clean full-suite discovery run from the sandbox.

## Important interpretation note

The native contrastive backend is better engineered than the inspected published CLIPn package, but it is still an optional latent-learning layer. Biological interpretation should be based on retrieval metrics, replicate/control behaviour, batch/source diagnostics and the classical CPATK outputs, not on the existence of a latent embedding alone.
