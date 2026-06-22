# CPATK CLIPn guide

Version: 0.2.11 documentation expansion

## Purpose

CLIPn is an optional integration layer. It should be used after standard CPATK metadata validation, preprocessing, replicate QC and batch diagnostics. It is not a substitute for these steps.

## Key requirements

CLIPn needs at least two non-empty datasets. CPATK v0.2.11 enforces this.

Input datasets should:

- have compatible biological features;
- have clean metadata;
- have no missing or infinite feature values after preprocessing;
- have all-zero rows/features removed;
- have clear dataset labels;
- ideally contain some comparable references, controls or shared compounds.

## Two real datasets

Preferred command pattern:

```bash
cpatk-clipn \
  --dataset reference1=results/reference1/preprocessed.tsv.gz \
  --dataset reference2=results/reference2/preprocessed.tsv.gz \
  --dataset query=results/query/preprocessed.tsv.gz \
  --output_dir results/11_clipn \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --mode integrate_all \
  --imputation_method median \
  --scaling_method robust \
  --n_neighbours 15 \
  --log_level INFO
```

## One dataset split into two parts

If only one dataset exists, CPATK can split it by compound:

```bash
cpatk-clipn \
  --dataset full=results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/11_clipn_single_dataset_split \
  --split_single_dataset_by_column Metadata_Compound \
  --single_dataset_split_names reference_like,query_like \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --allow_pca_fallback \
  --log_level INFO
```

This is exploratory. It is not an independent validation dataset. Use it to check whether the workflow runs and whether internal contrast structure is sensible.

## Zeros

CLIPn should not receive all-zero profiles or all-zero features. CPATK removes all-zero rows and features before CLIPn fitting.

Do not automatically remove every literal zero value unless you know zeros are impossible in the scaled feature space. After centring/scaling, zero can be meaningful because it can represent the reference centre or feature median. CPATK therefore provides strict zero-row/feature protection by default and an optional stricter setting for unusual cases.

## Metadata and features

Only biological morphology features should enter CLIPn. Exclude:

```text
plate identifiers
well identifiers
compound labels
MOA labels
batch labels
ImageNumber
ObjectNumber
row/column IDs
file paths
missingness indicators unless explicitly intended
```

## Interpretation

CLIPn latent spaces should be interpreted with:

- classical PCA/UMAP results;
- nearest-neighbour tables;
- replicate QC;
- batch diagnostics;
- known positive/negative controls;
- same-compound or same-MOA neighbour enrichment where available.

Do not treat a CLIPn UMAP alone as evidence of mechanism.

## Outputs to inspect

```text
clipn_status.tsv
clipn_feature_report.tsv
clipn_preprocessing_summary.tsv
clipn_latent.tsv.gz
nearest_neighbours.tsv
latent_diagnostic_summary.tsv
clipn_summary.xlsx
clipn_report.html
plots/
```

## Common failure modes

### Only one dataset

Fix: provide at least two datasets or split a single dataset by compound for exploratory work.

### No shared features

Fix: check preprocessing, feature selection and column naming. Avoid running feature selection separately in ways that produce incompatible feature sets.

### Batch dominates latent space

Fix: return to preprocessing and batch diagnostics. Consider per-plate control normalisation or batch correction only if design supports it.

### Replicates are poor

Fix: do not overinterpret CLIPn. Check metadata, compound annotation, image QC, plate effects and acquisition drift.
