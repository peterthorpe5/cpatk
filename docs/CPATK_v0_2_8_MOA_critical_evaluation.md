# CPATK v0.2.8 MOA critical evaluation and upgrade notes

## Why this pass was needed

The earlier CPATK MOA workflow supported centroid and KNN classification, but it did not yet capture several useful behaviours from the older MOA analysis scripts. The old scripts were project-specific, but they contained important ideas that are worth preserving in a generic package:

- pseudo-anchor generation from CLIPn latent embeddings or CellProfiler features;
- bootstrap K-means stability for selecting a pseudo-anchor cluster number;
- centroid and sub-centroid MOA scoring;
- optional adaptive shrinkage for small anchor groups;
- cosine and CSLS-style scoring;
- anchor/permutation-based confidence diagnostics;
- PCA/UMAP maps with MOA centroids overlaid;
- pairwise distance matrices, nearest-neighbour outputs and heatmaps.

v0.2.8 implements these ideas in a package-style, testable, logged and generic form.

## Important methodological changes

### 1. Known MOA and pseudo-MOA are both supported

`cpatk-moa` can now run three related workflows:

1. supervised known-MOA centroid/KNN classification when `--class_column` is supplied;
2. anchor-based centroid scoring when `--anchor_table` is supplied;
3. pseudo-anchor generation followed by centroid scoring when `--make_pseudo_anchors` is supplied.

This makes the tool useful both for annotated reference datasets and for exploratory query datasets where only latent/feature structure is available.

### 2. Pseudo-anchor generation is now generic

The package no longer assumes a particular project folder or metadata schema. It uses the same feature/metadata separation as the rest of CPATK, aggregates replicates by identifier, L2-normalises embeddings and then clusters at compound level.

### 3. Bootstrap stability is more defensible than a single silhouette score

The old bootstrap script used silhouette summaries over bootstrap solutions. v0.2.8 keeps silhouette information but also evaluates the agreement between the full-data clustering and bootstrap-extended clusterings using adjusted Rand index. This better reflects whether a selected cluster number gives stable compound groupings under resampling.

### 4. The permutation test was redesigned

A common pitfall is to shuffle labels on an already-built score matrix. If there is one centroid per class, the maximum score does not change, so the resulting p-values are not meaningful. v0.2.8 instead permutes the anchor MOA labels, rebuilds centroids and rescoring compounds to create an empirical null distribution of top scores.

### 5. Centroid scoring now supports sub-centroids and shrinkage

For heterogeneous MOA groups, a single centroid can be too crude. v0.2.8 supports multiple sub-centroids per MOA and optional shrinkage towards the global mean, including size-aware adaptive shrinkage. These options are useful but should be reported carefully because they can change classification behaviour.

### 6. Plots and reports are integrated

The MOA workflow now writes TSV outputs, an Excel summary, static PDF/SVG plots, optional interactive projection HTML and a CPATK HTML report. This keeps the workflow closer to a reproducible analysis package than a set of one-off scripts.

## Key outputs

Typical advanced outputs include:

- `pseudo_anchors.tsv`
- `pseudo_anchor_summary.tsv`
- `pseudo_anchor_clusters.tsv`
- `pseudo_anchor_k_selection.tsv`
- `moa_anchor_table.tsv`
- `advanced_moa_centroids.tsv`
- `advanced_moa_centroid_summary.tsv`
- `advanced_moa_scores_long.tsv`
- `advanced_moa_top_predictions.tsv`
- `advanced_moa_score_matrix.tsv`
- `advanced_moa_permutation_summary.tsv`
- `advanced_moa_permutation_null.tsv`
- `pairwise_distance_cosine.tsv`
- `nearest_neighbours_cosine.tsv`
- `moa_summary.xlsx`
- `moa_report.html`
- plots under `plots/`

## Interpretation cautions

- Pseudo-MOA clusters are exploratory phenotypic groups, not biological mechanism labels unless externally validated.
- CLIPn latent space, PCA/UMAP space and preprocessed CellProfiler feature space can produce different neighbourhoods. Use CPATK outputs to compare them rather than assuming they are equivalent.
- Sub-centroids can improve heterogeneous classes but may overfit small groups if used without enough anchors.
- Permutation p-values provide empirical support relative to randomised anchor labels; they do not prove biological mechanism.
- Strong MOA calls should be supported by replicate reproducibility, known-control recovery, nearest-neighbour consistency, classifier validation and feature attribution.
