# CPATK method selection guide

Version: 0.2.11 documentation expansion

This guide explains when to use each major CPATK workflow and when not to use it.

## Summary table

| Workflow | Use when | Do not use as |
|---|---|---|
| `cpatk-metadata` | Any new metadata or annotation file arrives | A replacement for understanding assay/source plate design |
| `cpatk-inspect` | You need to inventory CellProfiler exports | Proof that files are biologically valid |
| `cpatk-build-profiles` | You have Image/object CellProfiler outputs to merge | A safe native merger for several independent exports where `ImageNumber` restarts |
| `cpatk-preprocess` | You need QC, imputation, normalisation, scaling and feature filtering | A way to rescue poor experimental design |
| `cpatk-layout` | You need plate-layout/row/column diagnostics | A replacement for actual replicate and control checks |
| `cpatk-drift-qc` | You need acquisition/order drift checks on object-level files | A final biological profile analysis |
| `cpatk-batch` | You need to quantify plate or batch dominance | A batch correction method |
| `cpatk-stability` | You need replicate, neighbour and cluster stability evidence | Proof that one cluster number is biologically true |
| `cpatk-classical` | You need PCA/UMAP/distances/neighbours/clusters | A final mechanistic interpretation by itself |
| `cpatk-visualise` | You need static and interactive plots | A substitute for QC tables |
| `cpatk-neighbours` | You need focused nearest-neighbour comparison/plots | Proof of shared mechanism without controls |
| `cpatk-moa` | You have known MOA labels or anchors | Discovery of true MOA from unlabelled data alone |
| `cpatk-ml` | You have enough labelled classes and proper validation | A good idea for tiny or confounded datasets |
| `cpatk-explain` | You need feature attribution or query-vs-neighbour explanation | Causal proof |
| `cpatk-clipn` | You have two datasets or a justified split | A replacement for preprocessing, replicate QC or batch checks |
| `cpatk-report` | You need a final report index | A statistical analysis method |

## Metadata validation

Use `cpatk-metadata` first for any collaborator metadata. This is especially important when:

- the metadata contains both destination/assay wells and source/library wells;
- wells are inconsistent, for example `A1`, `A01`, `a1`;
- compound annotations are in separate files;
- column names have changed across experiments;
- there are concentration, plate handler, robot or library transfer fields.

The user should explicitly identify the true assay plate/well columns when there is any ambiguity. Source plate/well columns should remain annotations unless deliberately used for annotation merging.

## Profile building

Use `cpatk-build-profiles` to merge CellProfiler outputs. The core design is:

1. use the Image/profile table as the backbone;
2. aggregate each object table to ImageNumber;
3. prefix object features by table/compartment;
4. merge all image-level summaries;
5. merge metadata by plate/well.

This avoids unsafe Cell/Cytoplasm/Nuclei object-number joins.

Do not use a single multi-plate input folder if several independent CellProfiler exports restart `ImageNumber`. Build each export separately first until native composite-key multi-plate merging is added.

## Imputation methods

### Median

Recommended default. Use when:

- missingness is moderate;
- dataset size is small or medium;
- you need a stable first-pass analysis.

Avoid interpreting imputed values biologically.

### Mean

Rarely preferred. More sensitive to outliers than median.

### Zero

Use only when zero is a meaningful measured value, not a missing-value placeholder.

### KNN

Use cautiously. It can smooth local structure and can introduce treatment leakage if missingness is treatment- or plate-associated. CPATK caps neighbours for small datasets, but KNN is still not the safest default.

### Group median / group mean

Use when batch-aware or plate-aware imputation is justified. For example, group by `Metadata_Plate` if each plate has enough data and missingness is plate-specific.

## Scaling methods

### Robust scaling

Recommended default for Cell Painting. It is less sensitive to outliers than standard scaling.

### Standard scaling

Use when distributions are approximately symmetric and outliers are already controlled.

### Min-max scaling

Use mainly for visualisation or compatibility with methods that require bounded inputs. It is sensitive to outliers.

### No scaling

Use only when the input table is already scaled appropriately.

## Reference normalisation

### Robust z to DMSO / vehicle controls

Use when each plate has sufficient reference controls. This is often appropriate for multi-plate Cell Painting.

Command pattern:

```bash
cpatk-preprocess \
  --input_table all_plates_merged_profiles.tsv.gz \
  --output_dir results/02_preprocess_dmso_by_plate \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate \
  --imputation_method median \
  --scaling_method robust
```

Use robust-z rather than simple z-score when DMSO controls may include outliers.

### Median centring

Use when controls are present but scale estimates are unstable. It removes plate/control location shifts but does not scale by control spread.

### Z-score

Use only when control distributions are reasonably well behaved and there are enough controls per group.

## Batch centring

CPATK currently supports simple mean or median centring by batch columns. Use as sensitivity analysis, not as the only result.

Use when:

- batch effects are visible but not fully confounded with treatment;
- you have enough shared controls or replicated compounds across batches;
- you compare before/after plots and replicate QC.

Do not use when:

- every treatment appears on only one plate or one batch;
- the batch correction would remove the biological contrast of interest;
- you cannot show what changed.

## ComBat and stronger batch correction

ComBat-style empirical Bayes correction is not yet implemented as a first-class CPATK method in v0.2.11. It is a good candidate for the next code pass, but it should be added carefully with tests, clear warnings and before/after diagnostics.

Use ComBat only when:

- batch effects are substantial;
- each batch has overlapping biology or common controls;
- treatment is not perfectly confounded with batch;
- the design matrix is specified correctly;
- before/after replicate QC improves without destroying known control separation.

Do not use ComBat when:

- one treatment or MOA occurs in only one batch;
- the batch variable is the biological variable;
- there are too few samples per batch;
- you cannot explain the design matrix.

## Correlation filtering

Use correlation filtering to remove redundant features after merging, QC and imputation. Do not apply it independently to each CellProfiler compartment before merging. CPATK skips full correlation filtering above the configured feature-count guard to avoid memory blow-up.

## All-zero row filtering

Keep the default all-zero row filter for most CellProfiler workflows. It removes profiles whose retained merged features contain no non-zero evidence. This should happen after merging but before imputation.

Disable only if a true biological profile could genuinely be all zero.

## Replicate QC

Replicate QC should be run for every project. It is more important than a beautiful UMAP. Use the most specific sensible replicate grouping, usually compound plus dose plus timepoint, not MOA alone.

## Classical analysis

Use PCA/UMAP/distance/neighbour analysis as the main first-pass interpretation layer. These methods are transparent and help detect obvious batch effects, replicate failures or annotation errors.

## CLIPn

Use only after preprocessing and QC. CLIPn needs at least two datasets. If only one dataset exists, CPATK can split it by compound, but that should be treated as an exploratory internal contrast, not a true independent validation.

## MOA, ML and SHAP

Use these only after basic QC. They need enough labelled examples and careful validation. Feature attribution explains a model or local contrast; it does not prove causality.
