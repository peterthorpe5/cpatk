# CPATK v0.2.7 CLIPn adapter audit

## Why this pass was needed

The old project-specific CLIPn script contained a lot of hard-earned practical
behaviour: dataset manifests, feature intersection, label encoding, metadata
restoration, chunked prediction, loss curves, latent-space nearest-neighbour
checks, precision@k, dataset-mixing diagnostics and UMAP-style plots.  The
previous CPATK adapter was intentionally safe but too thin.  It checked whether a
backend could be imported and called generic `fit`/`predict` methods, but it did
not yet provide enough auditable preparation and diagnostics for real Cell
Painting integration work.

## Main design decisions

1. CLIPn remains optional.  CPATK must still work without a CLIPn installation.
2. Feature harmonisation happens before fitting.  Only the shared numeric feature
   intersection is used unless the user provides an explicit feature list.
3. Metadata and technical columns are excluded from CLIPn feature matrices, even
   if they are numeric or have been label encoded.
4. Missingness handling is explicit and logged before fitting.
5. `reference_only` and `integrate_all` are treated as separate modes.
6. If the backend fails, CPATK writes status/audit files instead of pretending the
   analysis succeeded.
7. PCA fallback is available only as a debugging aid and is explicitly labelled.

## Outputs added or strengthened

- `clipn_adapter_config.json`
- `clipn_metadata_alias_report.tsv`
- `clipn_status.tsv`
- `clipn_run_status.tsv`
- `clipn_feature_summary.tsv`
- `clipn_feature_report.tsv`
- `clipn_preprocessing_summary.tsv`
- `clipn_label_report.tsv`
- `clipn_latent.tsv.gz` when latent output is available
- `clipn_training_loss.tsv` when the backend returns loss values
- `latent_variance.tsv`
- `nearest_neighbours.tsv`
- `latent_diagnostic_summary.tsv`
- `clipn_summary.xlsx`
- `clipn_report.html`
- `plots/clipn_latent_pca.*`
- `plots/clipn_latent_umap_or_pca.*`
- `plots/clipn_latent_interactive.html`

## Remaining caveats

The adapter supports the common `clipn.model.CLIPn` API but CLIPn installations
can vary.  If the local CLIPn class has a different constructor or projection
method, pass `--backend_module`, `--model_class`, `--fit_method` and
`--predict_method`, or add a small project-specific backend shim.  The adapter
will still preserve all feature and preprocessing audit outputs if backend
execution fails.

The latent-space diagnostics are descriptive.  They should be interpreted with
classical CPATK PCA/UMAP, distance, replicate, batch-effect, MOA and SHAP outputs
rather than used as the sole evidence for mechanism-of-action assignment.
