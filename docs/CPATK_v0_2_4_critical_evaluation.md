# CPATK v0.2.4 critical evaluation notes

## Why this pass was needed

The v0.2.3 package had the correct high-level architecture: build profiles from a folder, preprocess them defensibly, then run classical analysis, stability analysis, MOA prediction, ML and reporting. The most important remaining risks were in the practical details:

1. real metadata files are messy and may split row and column rather than providing a clean well identifier;
2. more than one metadata or annotation table may need to be merged;
3. CellProfiler tables may contain non-finite values, BOM-prefixed column names, accidental index columns and zero-heavy technical features;
4. cluster stability needs more than one summary statistic;
5. MOA prediction needs separability diagnostics, not just classifier output.

## Preprocessing decisions

The default preprocessing remains deliberately conservative. CPATK separates metadata and features, excludes obvious numeric QC/provenance columns, removes high-missingness/low-variance/low-uniqueness features, applies sample missingness QC, imputes remaining missing values, scales features and optionally removes highly correlated features.

v0.2.4 adds non-finite value handling before feature QC. Positive and negative infinity are converted to missing values and reported in `nonfinite_value_report.tsv`. This is more defensible than treating infinite values as extreme biology.

v0.2.4 also adds an optional exact-zero fraction filter. This is disabled by default because exact zeros can sometimes be biologically meaningful, but it can be enabled with `--max_zero_fraction` when zero-heavy features are clearly technical or uninformative.

Median imputation remains the recommended default. KNN imputation is available but should be used cautiously because it can borrow information across treatments, plates or batches.

## Folder merging/profile building

CPATK continues to avoid unsafe cross-compartment object joins. Cell, Nuclei, Cytoplasm and similar object-level tables are aggregated to `ImageNumber` before merging. This is safer than assuming `ObjectNumber` is equivalent across object compartments.

v0.2.4 supports multiple metadata/platemap tables, supplied as a comma-separated list. Metadata aliases are standardised before merging. If row and column metadata are available, CPATK can derive `Metadata_Well`, for example `A01` or `B03`.

## Cluster stability and permutation testing

A single UMAP or K-means result is not enough evidence for a stable cluster structure. CPATK therefore provides:

- feature-subsampling nearest-neighbour stability;
- sample-subsampling cluster stability using adjusted Rand index;
- consensus co-clustering matrices;
- feature-wise permutation tests for cluster structure;
- K-range evaluation combining silhouette, bootstrap ARI and permutation p-values.

The permutation test shuffles each feature independently across profiles, preserving each feature's marginal distribution but breaking coordinated multivariate morphology. A low p-value supports structure relative to this null model, but does not prove the biologically correct number of clusters.

## MOA prediction and SHAP/feature attribution

MOA prediction should not rely on one model. CPATK retains centroid scoring, KNN, random forest, extra trees, gradient boosting, logistic regression and linear SVM options. v0.2.4 adds MOA separability diagnostics: known profiles from the same MOA should tend to be closer than profiles from different MOAs.

Permutation feature importance remains the robust default explanation method because it does not require SHAP. SHAP remains optional and is attempted when installed. SHAP results should be treated as model explanations, not causal biological proof.

## Remaining caveats

The package is now much stronger, but preprocessing is still not a substitute for good experimental design. Randomisation, control placement, plate balancing, donor/batch structure and known positive/negative controls still determine how strong the final biological interpretation can be.
