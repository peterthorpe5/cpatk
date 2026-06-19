# CPATK v0.2.0 methods notes

CPATK v0.2.0 treats Cell Painting analysis as a sequence of defensible steps: inspection, preprocessing, QC, exploratory embedding, distance analysis, clustering, stability testing, MOA prediction, feature attribution and reporting.

The uploaded example files show the kinds of generic input CPATK is intended to handle: image-level CellProfiler/Cell Painting tables with many image, channel, count, granularity, intensity, texture and mean object feature columns; object-level tables with `ImageNumber`, `ObjectNumber`, `AreaShape`, intensity, texture, radial distribution, neighbour and parent columns; and project-specific metadata containing plate, well, concentration, compound and `cpd_id` fields. CPATK therefore avoids hard-coding a specific metadata schema.

## Why stability testing was added

Cell Painting analyses often use PCA, UMAP and clustering because morphology profiles are high-dimensional and difficult to interpret directly. However, a visually plausible UMAP cluster is not enough. CPATK therefore adds:

- replicate profile correlations;
- bootstrap nearest-neighbour stability;
- bootstrap cluster stability using adjusted Rand index;
- consensus co-clustering matrices;
- permutation tests of cluster structure using silhouette scores after feature-wise permutation.

These outputs help distinguish robust structure from unstable visual clustering.

## Why several MOA classifiers are included

Different MOA datasets have different class sizes, imbalance, batch effects and replicate structures. CPATK includes simple centroid scoring, KNN, random forest, extra trees, gradient boosting, logistic regression and calibrated linear SVM so that performance can be compared rather than assumed.

## Why SHAP and permutation importance are both included

Permutation importance is model-agnostic and asks whether a feature matters for held-out classification. SHAP is more detailed and can provide global and local explanations, especially for tree-based models. Both are useful, but both are still model explanations rather than causal biological evidence.

## Why the CLIPn adapter is optional

AI/CLIPn workflows are valuable, but they should not be the only analysis path. CPATK uses a defensive adapter so CLIPn can be used when available, while keeping classical analysis, QC, and MOA workflows operational without any AI dependency.
