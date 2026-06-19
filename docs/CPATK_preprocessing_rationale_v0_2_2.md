# CPATK preprocessing rationale v0.2.2

The main principle is to avoid silently manufacturing biological signal from technical columns, missing data, batch structure or metadata leakage.

## Column role assignment

Many Cell Painting exports contain numeric columns that are not biological features. CPATK therefore excludes common QC/provenance columns from default feature inference, including execution times, object counts, image identifiers, object identifiers, file hashes, file paths and image dimensions. These can still be included explicitly if needed.

## Missing data

Missing values can arise from segmentation failures, object detection limits, channel-specific measurement failures or table-merging problems. CPATK reports missingness before imputation and writes a per-feature imputation report.

Recommended default: median imputation.

Alternative options:

- `group_median`: useful for plate/batch-aware imputation.
- `mean`: less robust than median.
- `zero`: only defensible for features where zero has a true measurement meaning.
- `knn`: exploratory; can smooth biological and batch structure.

## Scaling

Robust scaling is the default because Cell Painting features can be skewed and outlier-prone. Standard scaling is available when distributions are already well-behaved. Min-max scaling is available mainly for algorithms that require bounded values, but is sensitive to outliers.

## Control/reference normalisation

Reference normalisation can be powerful when each plate has suitable controls. It should be done within plate/batch where possible. CPATK supports median-centering, robust-z and z-score normalisation against reference profiles.

## Correlation filtering

Cell Painting features are highly redundant. CPATK can remove one feature from highly correlated pairs. v0.2.2 preferentially removes the feature with more missingness and/or lower variance, rather than relying only on column order.
