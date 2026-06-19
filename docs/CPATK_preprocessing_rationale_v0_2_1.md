# CPATK v0.2.1 preprocessing rationale

This note records the rationale for the upgraded CPATK preprocessing workflow.
The main aim is to make Cell Painting profile preparation reproducible,
defensible and auditable across projects with variable metadata conventions.

## Why v0.2.0 needed upgrading

The v0.2.0 preprocessing workflow already separated metadata and features,
filtered features, imputed missing values, scaled features and optionally removed
correlated features. That was useful, but too thin for routine Cell Painting
work because real project tables often contain messy metadata, accidental CSV
index columns, mixed CellProfiler feature families, missing values, plate/batch
structure and unclear provenance.

The upgraded v0.2.1 workflow therefore adds a more explicit preprocessing audit:
column-name cleaning, metadata alias standardisation, missing-value reporting,
optional group-wise imputation, optional missingness indicators, feature-family
summaries, preprocessing plots and an HTML report.

## Recommended default strategy

For most profile-level Cell Painting data, the recommended starting strategy is:

1. Keep metadata separate from numeric features.
2. Standardise obvious metadata aliases, but do not force all projects into one
   schema.
3. Drop accidental exported CSV index columns such as sequential `Unnamed: 0`.
4. Remove features with excessive missingness.
5. Remove profiles with excessive missingness.
6. Remove constant or near-constant features.
7. Impute the remaining missing values using median imputation by default.
8. Robust-scale features by default.
9. Optionally remove highly correlated features.
10. Write all QC, filtering, imputation and scaling decisions to TSV, Excel,
    plots, logs and HTML.

This is deliberately conservative. It avoids treatment-specific filtering or
outcome-dependent filtering, both of which can bias downstream biological
interpretation.

## Imputation choices

CPATK supports median, mean, zero, KNN, group-median and group-mean imputation.
Median imputation is the safest default because Cell Painting features are often
skewed and can contain outliers. KNN imputation may be useful for richer profile
tables, but it should be used cautiously because it can blur treatment effects
when missingness is structured by plate, batch or treatment. Group-wise median
imputation can be useful when missingness is strongly plate- or batch-specific,
but the grouping columns must be chosen carefully. The imputation report should
always be reviewed.

## Missingness indicators

Missingness indicators are optional. They can be useful for machine-learning
models because the fact that a feature was missing may itself be informative.
However, they can also encode plate, batch, segmentation or acquisition artefacts.
They are therefore disabled by default and should be enabled deliberately.

## Scaling choices

Robust scaling is the default because Cell Painting feature distributions are
frequently skewed and outlier-prone. Standard scaling is available when classical
z-scoring is preferred. Min-max scaling is available but should be used with care
because it is sensitive to extreme values. No scaling is also supported when the
input has already been normalised.

## Correlation filtering

Highly correlated feature filtering can reduce redundancy and speed downstream
analysis. However, it can remove biologically interpretable features, and the
retained feature from a correlated group is not necessarily the most biologically
meaningful one. CPATK records all removed features and the correlated partner so
that this step remains auditable.

## Metadata standardisation

Metadata names vary widely between projects. CPATK recognises common aliases such
as `Plate_Metadata`, `Well_Metadata`, `Compound`, `cpd_id`, `cpd_type`, dose,
concentration and batch/library fields. Canonical aliases such as
`Metadata_Plate`, `Metadata_Well`, `Metadata_Compound`, `Metadata_MOA`,
`Metadata_Dose` and `Metadata_Batch` are added where possible without deleting
or replacing the original columns.

## What CPATK does not assume

CPATK does not assume one biological assay, one stain, one cell type, one plate
layout or one specific metadata file format. It also does not assume that all
numeric columns are meaningful phenotypic features. Users should still inspect
feature inventories and may need to explicitly provide metadata and feature
columns for complex projects.

## Downstream implications

The preprocessing output is suitable for the non-AI workflows in CPATK, including
PCA, UMAP, distance analysis, clustering, nearest-neighbour analysis, MOA
classification and feature attribution. It is also suitable for optional CLIPn
adapter workflows, provided feature columns are aligned across datasets.

For publication-facing analysis, preprocessing should always be reported along
with: number of input profiles, number of retained profiles, number of detected
features, number of retained features, imputation method, scaling method,
correlation-filter threshold and any project-specific normalisation decisions.
