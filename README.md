# CPATK: Cell Painting Analysis Toolkit

CPATK is a generic, extensible toolkit for Cell Painting / high-content profiling analysis. It supports defensive preprocessing, QC, classical analysis, optional CLIPn/AI integration, replicate and cluster stability, batch/domain-shift diagnostics, MOA classification, feature attribution and HTML/Excel reporting.

Version: **0.2.5**

## Design principles

- Generic across Cell Painting projects rather than tied to one assay.
- Metadata and feature handling must be explicit and auditable.
- Classical non-AI analysis is a first-class workflow, not just a fallback.
- Optional AI/CLIPn integration must be defensive and not required for basic use.
- Every workflow should write TSV outputs, formatted Excel summaries, logs and HTML reports where appropriate.
- Unit tests use `unittest`.

## Installation

```bash
cd cpatk_v0_2_5_full
python -m pip install -e .
python -m unittest discover -s tests -v
```

Optional dependencies:

```bash
python -m pip install pyarrow plotly umap-learn shap
```

`pyarrow` enables Parquet output. Without it, CPATK falls back to `.tsv.gz` and logs the reason.

## Command-line tools

```text
cpatk-inspect
cpatk-preprocess
cpatk-classical
cpatk-layout
cpatk-stability
cpatk-batch
cpatk-ml
cpatk-explain
cpatk-clipn
cpatk-moa
cpatk-report
```

## Preprocessing

The v0.2.5 preprocessing workflow is intentionally conservative. It writes an auditable column role table, removes obvious technical/provenance columns from default features, imputes missing values, optionally normalises to reference controls, scales features and optionally removes highly correlated features.

Basic preprocessing:

```bash
cpatk-preprocess \
  --input_table profiles.tsv \
  --output_dir results/01_preprocess \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO
```

Control/reference normalisation, for example DMSO within plate:

```bash
cpatk-preprocess \
  --input_table profiles.tsv \
  --output_dir results/01_preprocess_dmso_reference \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate \
  --imputation_method median \
  --scaling_method robust
```

Optional batch centering sensitivity analysis:

```bash
cpatk-preprocess \
  --input_table profiles.tsv \
  --output_dir results/01_preprocess_batch_centered \
  --batch_centering_method median_center \
  --batch_centering_columns Metadata_Batch \
  --imputation_method median \
  --scaling_method robust
```

Main preprocessing outputs:

```text
preprocessed.parquet or preprocessed.tsv.gz
imputed_unscaled_features_with_metadata.tsv
feature_qc.tsv
sample_qc.tsv
all_zero_row_report.tsv
column_role_report.tsv
imputation_report.tsv
reference_normalisation_report.tsv
batch_centering_report.tsv
correlation_filter_report.tsv
retained_features.tsv
feature_family_summary.tsv
preprocessing_decision_log.tsv
preprocessing_config.tsv
preprocessing_summary.xlsx
preprocessing_report.html
preprocess.log
plots/
```


## v0.2.5 merge-first zero-row preprocessing

The v0.2.5 release adds an explicit all-zero profile filter. This is important for CellProfiler exports where failed images, empty wells or failed object aggregation can produce rows whose retained feature values are all zero. The filter is deliberately applied only after any folder of CellProfiler outputs has been merged into one profile matrix, and before imputation, scaling and correlation filtering. This means a profile is not removed just because one compartment table is zero; it is removed only when the merged retained feature evidence for that profile contains no non-zero observed values.

```bash
cpatk-preprocess \
  --input_dir /path/to/cellpainting_exports \
  --output_dir results/01_preprocess \
  --recursive \
  --metadata_table plate_map.tsv \
  --imputation_method median \
  --scaling_method robust
```

The all-zero row filter is enabled by default. It can be disabled for unusual assays where a true biological profile could genuinely be all zero:

```bash
cpatk-preprocess \
  --input_table merged_profiles.tsv.gz \
  --output_dir results/01_preprocess_keep_zero_rows \
  --disable_all_zero_row_filter
```

The relevant audit files are:

```text
all_zero_row_report.tsv
preprocessing_decision_log.tsv
preprocessing_summary.tsv
preprocessing_report.html
plots/all_zero_feature_row_qc.pdf
```

Additional v0.2.5 checks include replacing finite values with absolute value greater than 1e10 with missing values before imputation. This follows the defensive behaviour in the older project-specific preprocessing script, where extreme ratio artefacts were not treated as reliable biological morphology.

## Classical non-AI analysis

```bash
cpatk-classical \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/02_classical \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA \
  --id_column Metadata_Compound \
  --colour_column Metadata_MOA \
  --cluster_group_columns Metadata_MOA,Metadata_Compound \
  --distance_metric cosine \
  --n_neighbours 10 \
  --n_clusters 8 \
  --run_tsne
```

This generates PCA, UMAP/PCA fallback, optional t-SNE, pairwise distances, nearest neighbours, clustering outputs, static plots and interactive Plotly outputs when Plotly is installed.

## Stability and clustering confidence

```bash
cpatk-stability \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/03_stability \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA \
  --cluster_group_columns Metadata_MOA,Metadata_Compound \
  --n_clusters 8 \
  --n_bootstrap 100 \
  --n_permutations 100
```

Cluster permutation testing compares the observed clustering score with feature-wise permutation null scores. This does not prove the true number of clusters, but it helps determine whether clustering is stronger than expected from scrambled feature structure.

## Batch and domain-shift diagnostics

```bash
cpatk-batch \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/04_batch \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch \
  --batch_column Metadata_Batch
```

## MOA analysis

Centroid MOA scoring with confidence margins:

```bash
cpatk-moa \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/05_moa \
  --class_column Metadata_MOA \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA \
  --min_class_size 2 \
  --metric cosine \
  --top_n 5 \
  --run_knn
```

Outputs include centroid scores, top predictions, leave-one-out centroid validation, optional KNN predictions, confidence plots, Excel summary and `moa_report.html`.

## Supervised ML classifiers

```bash
cpatk-ml \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/06_ml \
  --class_column Metadata_MOA \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA \
  --compare_models \
  --n_splits 5
```

Supported models include KNN, random forest, extra trees, gradient boosting, logistic regression and calibrated linear SVM.

## Feature attribution: permutation importance and SHAP

```bash
cpatk-explain \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/07_feature_attribution \
  --class_column Metadata_MOA \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA \
  --model_name random_forest \
  --n_repeats 20 \
  --include_shap
```

Permutation importance is the safest default. SHAP is optional and will write a clear status table if the dependency or model-specific explainer is unavailable.

## CLIPn adapter

The CLIPn adapter is defensive. It aligns datasets, checks backend availability and writes status/provenance tables. It does not make CLIPn a hard dependency of CPATK.

```bash
cpatk-clipn \
  --dataset reference1=results/ref1_preprocessed.parquet \
  --dataset reference2=results/ref2_preprocessed.parquet \
  --dataset query=results/query_preprocessed.parquet \
  --output_dir results/08_clipn \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA
```

## Critical evaluation notes

See:

```text
docs/CPATK_v0_2_2_critical_evaluation.md
docs/CPATK_preprocessing_rationale_v0_2_2.md
docs/CPATK_v0_2_5_merge_first_preprocessing_audit.md
```

## Test status

The v0.2.5 package passed module-level tests:

```text
Ran 91 tests
OK
```

## v0.2.3 folder input and profile building

CPATK can now start from a folder of Cell Painting files instead of requiring a
single pre-merged profile table.  The new command is:

```bash
cpatk-build-profiles \
  --input_dir /path/to/cellpainting_folder \
  --output_dir results/00_profile_build \
  --recursive \
  --aggregate_statistic median
```

The profile builder accepts `.csv`, `.csv.gz`, `.tsv`, `.tsv.gz`, `.parquet` and
Excel files.  It infers Image, object and metadata tables from the file headers,
uses the Image/profile table as the backbone, aggregates object tables to
`ImageNumber`, and merges external plate/well metadata where available.  It does
not blindly join separate Cell/Cytoplasm/Nuclei object tables by `ObjectNumber`.

You can also build and preprocess in one command:

```bash
cpatk-preprocess \
  --input_dir /path/to/cellpainting_folder \
  --output_dir results/01_preprocess \
  --recursive \
  --metadata_table /path/to/metadata.tsv \
  --imputation_method median \
  --scaling_method robust
```

This writes `results/01_preprocess/00_profile_build/` plus all normal
preprocessing outputs, including imputation reports, plots, Excel summaries and
HTML reports.


## v0.2.4 critical refinements

This release strengthens the parts of CPATK that are most important for defensible Cell Painting analysis:

- folder-level profile building now supports multiple metadata/platemap tables supplied as a comma-separated list;
- row and column metadata can be standardised to `Metadata_Row` and `Metadata_Column`, and `Metadata_Well` can be derived as `A01`, `B03`, etc. when needed;
- CSV/CSV.GZ reading uses UTF-8-SIG handling to avoid BOM problems in messy metadata files;
- preprocessing now converts positive/negative infinity to missing values before QC and writes `nonfinite_value_report.tsv`;
- feature QC now reports exact-zero fractions and supports an optional `--max_zero_fraction` filter;
- cluster permutation testing now has a detailed mode that writes the null distribution as well as the summary p-value;
- stability workflows can evaluate a range of K values using silhouette, permutation testing and bootstrap ARI;
- MOA analysis now includes a separability diagnostic comparing within-MOA and between-MOA distances against shuffled labels;
- stability CLI now writes a richer HTML report describing the caveats around cluster-number selection.

Recommended first-pass workflow from a raw Cell Painting folder:

```bash
cpatk-preprocess   --input_dir /path/to/cellpainting_exports   --output_dir results/01_preprocess   --recursive   --metadata_table plate_map.tsv,compound_annotations.csv   --imputation_method median   --scaling_method robust   --max_feature_missing_fraction 0.2   --max_sample_missing_fraction 0.5   --max_absolute_correlation 0.95   --log_level INFO
```

Optional zero-heavy feature filtering can be enabled when exact-zero-heavy features are clearly artefactual:

```bash
cpatk-preprocess   --input_table profiles.tsv   --output_dir results/01_preprocess_zero_filter   --max_zero_fraction 0.95
```

Cluster-number confidence should be assessed with several diagnostics rather than a single UMAP:

```bash
cpatk-stability   --input_table results/01_preprocess/preprocessed.parquet   --output_dir results/04_stability   --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA   --replicate_group_columns Metadata_Compound,Metadata_MOA   --n_clusters 8   --k_values 2,3,4,5,6,7,8,9,10   --n_bootstraps 100   --n_permutations 100
```

## v0.2.6 update: query-neighbourhood SHAP and feature statistics

CPATK v0.2.6 substantially expands `cpatk-explain`.  In addition to global supervised feature attribution from known classes or MOA labels, it can now explain why a query compound separates from, or resembles, its nearest neighbours.

This was added after reviewing older project-specific SHAP scripts.  The old scripts produced useful query-vs-neighbour SHAP plots and query-vs-background feature statistics, but were tightly coupled to specific column names.  CPATK v0.2.6 generalises those ideas into a reusable module while retaining the outputs that were most useful biologically.

Example query-neighbourhood explanation:

```bash
cpatk-explain \
  --input_table results/01_preprocess/preprocessed.parquet \
  --output_dir results/07_explain \
  --metadata_columns Metadata_Plate,Metadata_Well,cpd_id,cpd_type,Library,Metadata_MOA \
  --id_column cpd_id \
  --query_ids MCP09 \
  --nn_file results/02_classical/nearest_neighbours.tsv \
  --n_neighbours 5 \
  --run_feature_tests \
  --run_neighbourhood_shap \
  --background_column cpd_type \
  --background_values DMSO,control,negative_control \
  --include_shap \
  --n_top_features 20 \
  --log_level INFO
```

Important outputs include:

```text
query_neighbourhoods/<query>/selected_neighbours.tsv
query_neighbourhoods/<query>/query_vs_background_feature_statistics.tsv
query_neighbourhoods/<query>/top_query_increased_features.tsv
query_neighbourhoods/<query>/top_query_decreased_features.tsv
query_neighbourhoods/<query>/top_shap_features_driving_query_difference.tsv
query_neighbourhoods/<query>/low_contribution_shap_features.tsv
query_neighbourhoods/<query>/sample_feature_shap_values.tsv.gz
query_neighbourhoods/<query>/neighbourhood_shap_status.tsv
query_neighbourhoods/<query>/top_shap_feature_family_summary.tsv
query_neighbourhoods/<query>/query_neighbourhood_explanation_summary.xlsx
feature_explanation_report.html
feature_explanation_summary.xlsx
```

Interpretation: the feature-statistics tables describe direct query-vs-background differences using median differences, Wasserstein distances, Mann-Whitney or Kolmogorov-Smirnov tests and BH-FDR correction.  The SHAP tables and plots explain the fitted local query-vs-neighbour classifier.  They are useful for biological hypothesis generation, but they do not prove causality.

## v0.2.7 CLIPn adapter upgrade

CPATK v0.2.7 substantially upgrades the optional CLIPn layer.  The previous
adapter only checked for a backend and made a best-effort call to a generic
model.  The new workflow is closer to the mature project scripts while keeping
CPATK generic and dependency-safe.

New CLIPn features include:

- dataset manifests with `dataset` and `path` columns;
- repeated `--dataset name=path` inputs for quick runs;
- CSV, TSV, gzipped CSV/TSV, Parquet and Excel input through the standard CPATK reader;
- metadata alias standardisation for compound, MOA/class, plate, well and library columns;
- shared-feature intersection across datasets with full audit tables;
- strict exclusion of metadata, IDs and technical columns from CLIPn features;
- non-finite and extreme-value cleanup before imputation;
- median, mean, KNN or no imputation;
- robust, standard or no scaling;
- `integrate_all` and `reference_only` modes;
- optional PCA fallback for debugging when CLIPn is unavailable;
- backend run status tables rather than silent failure;
- latent-space diagnostics including nearest-neighbour summaries, latent variance and silhouette checks;
- static PCA/UMAP-or-PCA plots and interactive HTML embeddings;
- formatted Excel and HTML summary reports.

Example manifest:

```text

dataset	path
reference1	results/reference1/preprocessed.tsv.gz
reference2	results/reference2/preprocessed.tsv.gz
query	results/query/preprocessed.tsv.gz
```

Example command:

```bash
cpatk-clipn \
  --datasets_csv datasets.tsv \
  --output_dir results/08_clipn \
  --experiment cellpainting_clipn \
  --mode reference_only \
  --reference_names reference1,reference2 \
  --latent_dim 20 \
  --epochs 500 \
  --lr 1e-5 \
  --imputation_method median \
  --scaling_method robust \
  --n_neighbours 15 \
  --log_level INFO
```

If the CLIPn backend is not installed, the workflow still writes the feature,
metadata, preprocessing and backend-status audit files.  Add
`--allow_pca_fallback` only for debugging the downstream reporting structure;
that output is labelled as PCA fallback and should not be reported as CLIPn.

## v0.2.9 additions: QC, visualisation and full workflow

CPATK v0.2.9 adds three new command-line workflows:

```bash
cpatk-drift-qc
cpatk-visualise
cpatk-neighbours
```

`cpatk-drift-qc` inspects object-level CellProfiler files before profile aggregation. It reports per-compartment acquisition drift using `ImageNumber`, Spearman correlation on per-image medians, early-vs-late median shifts, Cliff's delta and scalable drift plots.

`cpatk-visualise` creates PCA, optional UMAP, optional PHATE, latent/profile norm checks, clustered heatmaps and k-nearest-neighbour topology plots from either CLIPn latent outputs or processed Cell Painting profiles.

`cpatk-neighbours` creates top-neighbour plots, shared-neighbour scatter plots and overlap/RBO summaries for comparing nearest-neighbour outputs across runs.

A full start-to-finish example shell script is provided at:

```text
examples/run_full_cpatk_pipeline.sh
```

An example metadata file and column guidance are provided at:

```text
examples/example_metadata.tsv
examples/METADATA_REQUIREMENTS.md
```

Minimal recommended metadata columns are:

```text
Metadata_Plate
Metadata_Well
Metadata_Compound
cpd_type
```

Strongly recommended columns include:

```text
Metadata_MOA
Metadata_Dose
Metadata_Batch
Replicate
Donor / CellLine / Timepoint where relevant
```
