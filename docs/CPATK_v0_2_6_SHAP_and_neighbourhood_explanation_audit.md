# CPATK v0.2.6 SHAP and neighbourhood explanation audit

This release reviews the feature-attribution layer against the older project-specific scripts `shap_explain_nn_similarity.py` and `explain_feature_driven_results.py`.  The old scripts were useful because they recognised two different explanation tasks:

1. explaining a query compound against its nearest neighbours using SHAP; and
2. comparing query wells against a background or neighbour set feature-by-feature using non-parametric statistics and effect sizes.

The earlier CPATK v0.2.5 code already supported global supervised permutation importance and optional global SHAP, but it did not yet fully reproduce those query-level workflows.  v0.2.6 adds a dedicated `cpatk.neighbourhood_explain` module and extends `cpatk-explain` so that global MOA attribution and local query-neighbourhood explanation can be run from the same command.

## Main design decisions

- Neighbourhood SHAP is treated as a local explanation: it answers which features help a small classifier distinguish a query compound from its selected neighbours.  It should not be interpreted as causal biology.
- Feature-level statistical testing is kept separate from SHAP.  SHAP explains a fitted classifier; Mann-Whitney/Kolmogorov-Smirnov tests, median differences and Wasserstein distances describe direct distributional feature differences.
- Numeric CellProfiler provenance and QC columns are excluded again at the local explanation stage.  This prevents columns such as `ImageNumber`, `ObjectNumber`, `Count_*`, `ExecutionTime_*`, paths, checksums and grouped image bookkeeping from driving explanations accidentally.
- Query ID parsing now handles plain one-ID-per-line files without dropping the first line as a header.
- SHAP model handling is defensive across SHAP versions.  Small local problems use logistic regression with a linear SHAP explainer; larger local problems use a random forest with tree SHAP where possible.
- SHAP plotting is optional and failure-tolerant.  A failed beeswarm, heatmap or dependence plot is logged without stopping the whole CPATK run.

## New outputs from query-neighbourhood explanation

For each query, CPATK can now write:

- `selected_neighbours.tsv`
- `query_neighbourhood_metadata.tsv`
- `query_neighbourhood_column_audit.tsv`
- `query_vs_background_feature_statistics.tsv`
- `top_query_increased_features.tsv`
- `top_query_decreased_features.tsv`
- `top_shap_features_driving_query_difference.tsv`
- `low_contribution_shap_features.tsv`
- `sample_feature_shap_values.tsv.gz`
- `neighbourhood_shap_status.tsv`
- `top_shap_feature_family_summary.tsv`
- `query_neighbourhood_explanation_summary.xlsx`
- signed feature-shift plots
- SHAP summary bar, beeswarm, bar, heatmap and dependence plots where supported by the installed SHAP backend

## Recommended interpretation

Use feature statistics to describe what differs, and SHAP to describe which features a local model used to separate the query from its neighbours.  Stronger confidence comes when the same feature families appear across direct statistics, SHAP/permutation attribution, replicate consistency and biological plausibility.
