# CPATK v0.2.23 protected features and CLIPn training controls

This update adds three related hardening changes.

## Protected feature input

`cpatk-preprocess` now accepts user-requested features that should be protected from ordinary feature filters when they are usable:

```bash
--protected_features FeatureA,FeatureB
--protected_features_file protected_features.txt
```

The file format is one feature name per line. Blank lines and lines beginning with `#` are ignored.

Protected features are protected from ordinary feature selection steps such as missingness/variance/unique-value/zero-fraction filtering and correlation redundancy filtering. They are not rescued if they are absent, non-numeric or entirely missing.

New outputs:

- `protected_feature_audit.tsv`
- `feature_selection_report.tsv`
- `feature_selection_summary.tsv`

These tables report requested protected features, whether they were present and usable, whether they were retained, and why every feature was retained or removed.

## Feature-selection reporting

The preprocessing report now includes explicit feature-selection sections. The report distinguishes features removed by feature QC from features removed by the Spearman/variance-prioritised correlation filter, and links the retained feature table.

## CLIPn training controls

`cpatk-clipn` still supports fixed epoch training, but now also exposes conservative chunked early stopping:

```bash
--epochs 200 \
--clipn_early_stopping \
--clipn_patience 20 \
--clipn_min_delta 0.0001 \
--clipn_epoch_chunk_size 10
```

Because common CLIPn backends expose a simple `fit(X, y, lr, epochs)` API rather than a validation callback, this early stopping monitors reported training loss plateauing. It should not be treated as validation-loss optimisation. CPATK writes this provenance clearly.

New/expanded CLIPn outputs:

- `clipn_training_summary.tsv`
- `clipn_training_loss.tsv`
- expanded `clipn_backend_provenance.tsv`

The safest default remains to interpret CLIPn alongside PCA, UMAP, nearest-neighbour, replicate QC, batch QC, feature-space MOA and explainability results rather than treating the CLIPn latent space as intrinsically superior.
