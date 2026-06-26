# CPATK: Cell Painting Analysis Toolkit

CPATK is a generic, extensible toolkit for Cell Painting / high-content profiling analysis. It supports defensive preprocessing, QC, classical analysis, optional CLIPn/AI integration, replicate and cluster stability, batch/domain-shift diagnostics, MOA classification, feature attribution and HTML/Excel reporting.

Version: **0.2.16**

## Design principles

- Generic across Cell Painting projects rather than tied to one assay.
- Metadata and feature handling must be explicit and auditable.
- Classical non-AI analysis is a first-class workflow, not just a fallback.
- Optional AI/CLIPn integration must be defensive and not required for basic use.
- Every workflow should write TSV outputs, formatted Excel summaries, logs and HTML reports where appropriate.
- Unit tests use `unittest`.

## Installation

```bash
cd cpatk_v0_2_11_full
python -m pip install -e .
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

Optional dependencies:

```bash
python -m pip install pyarrow plotly umap-learn shap
```

`pyarrow` enables Parquet output. Without it, CPATK falls back to `.tsv.gz` and logs the reason.

## v0.2.15 release-hardening update

This release is focused on making CPATK safer for production and publication workflows rather than adding a new analysis method. Key changes are:

- Added `cpatk-metadata` as a step-one metadata and annotation validation workflow.
- Canonicalises messy assay and source well metadata, including `A1`/`A01` formats, while preserving raw values in `__raw` audit columns.
- Requires explicit assay plate/well columns when metadata are ambiguous; source/robot plate-well columns are never promoted to the CellProfiler assay keys.
- Supports strict annotation merging with duplicate-key reports.
- Fails by default on dangerous duplicate image rows or metadata merge keys.
- Fixes copied interactive HTML report links so they point into `report_assets/`.
- Returns a real visualisation output manifest from `cpatk-visualise`.
- Calculates sample/profile QC both before and after feature-level QC.
- Applies reference/control normalisation before imputation.
- Caps KNN imputation neighbours for small datasets.
- Separates missingness-indicator features from biological correlation filtering by default.
- Adds final feature-matrix validation for empty, NaN or infinite outputs.
- Adds a memory guard for full correlation filtering on very wide matrices.
- Hardens CLIPn inputs by requiring at least two non-empty datasets, adding a compound-preserving single-table split helper, and removing all-zero rows/features before CLIPn fitting.

Recommended first metadata check:

```bash
cpatk-metadata \
  --metadata_table raw_metadata.csv \
  --output_dir results/00_metadata_check \
  --plate_column Assay_Plate_Barcode \
  --well_column Destination_Well \
  --source_plate_column Source_Plate_Barcode \
  --source_well_column Source_Well \
  --annotation_tables annotation_file.csv,compound_library.tsv \
  --annotation_source_plate_column Barcode \
  --annotation_source_well_column Well \
  --merge_keys Metadata_Source_Plate,Metadata_Source_Well \
  --duplicate_policy error \
  --log_level INFO
```

The main output from this step is `formatted_metadata.tsv`, which should be used as the safer metadata input for later CPATK steps.



## Recommended workflow at a glance

For a real project, CPATK should normally be run as a staged, auditable workflow rather than as one opaque command:

```text
raw CellProfiler exports
  -> cpatk-metadata          # validate and standardise plate-map / annotations
  -> cpatk-inspect           # inspect CellProfiler files and inferred roles
  -> cpatk-build-profiles    # merge Image + object compartments into one profile table
  -> cpatk-preprocess        # QC, all-zero filtering, reference normalisation, imputation, scaling, feature filtering
  -> cpatk-classical         # PCA/UMAP/distances/neighbours/clustering
  -> cpatk-stability         # replicate QC, neighbour stability, cluster stability
  -> cpatk-batch             # plate/batch/domain-shift diagnostics
  -> cpatk-visualise         # static and interactive plots
  -> optional cpatk-moa / cpatk-ml / cpatk-explain / cpatk-clipn
  -> cpatk-report            # final report index
```

The safest first analysis is deliberately classical: metadata validation, profile building, preprocessing, PCA/UMAP, heatmaps, nearest neighbours, replicate QC and batch diagnostics. CLIPn, MOA classification, ML and SHAP should be added only after these checks look credible.

## Expanded documentation

The following documents give more detailed practical guidance than the README:

```text
docs/CPATK_USER_GUIDE.md
docs/CPATK_METHOD_SELECTION_GUIDE.md
docs/CPATK_METADATA_AND_ANNOTATION_GUIDE.md
docs/CPATK_MULTI_PLATE_NORMALISATION_AND_BATCH_GUIDE.md
docs/CPATK_REPLICATE_QC_GUIDE.md
docs/CPATK_CLIPN_GUIDE.md
docs/CPATK_PHENOTYPE_LABELLED_MOA_GUIDE.md
docs/CPATK_NEXT_CODE_PASS_RECOMMENDATIONS.md
```

The most important production caveat at v0.2.15 is multi-plate CellProfiler export handling. CPATK can analyse multi-plate profile tables once each row has reliable `Metadata_Plate` and `Metadata_Well` values, and `cpatk-preprocess` already supports per-plate DMSO/reference normalisation with `--reference_group_columns Metadata_Plate`. However, if several independent CellProfiler exports are placed in one folder and `ImageNumber` restarts at 1 for each plate/export, v0.2.15 should not be treated as fully native multi-plate folder merging. In that case, build profiles per plate/export first, preserve plate provenance, then combine the resulting profile tables before one joint preprocessing pass. Native composite-key multi-plate folder merging should be the next code pass.

## Command-line tools

```text
cpatk-metadata
cpatk-inspect
cpatk-build-profiles
cpatk-preprocess
cpatk-classical
cpatk-layout
cpatk-stability
cpatk-batch
cpatk-ml
cpatk-explain
cpatk-clipn
cpatk-moa
cpatk-visualise
cpatk-drift-qc
cpatk-neighbours
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

Pseudo-anchor clusters can also be annotated using a curated compound-to-phenotype table. CPATK writes phenotype-label audits and a conservative `moa_final` column. Weakly labelled or mixed clusters retain their pseudo-anchor IDs rather than being over-interpreted.

```bash
cpatk-moa \
  --input_table results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/09_moa \
  --id_column cpd_id \
  --make_pseudo_anchors \
  --pseudo_anchor_label_table cpd_id_to_phenotype.tsv \
  --pseudo_anchor_label_id_column cpd_id \
  --pseudo_anchor_label_column label \
  --pseudo_anchor_final_moa_column moa_final
```

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

The v0.2.15 package passed full unittest discovery in this sandbox with native numerical thread limits set:

```text
Ran 163 tests
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


## v0.2.15 multi-plate and batch-correction additions

CPATK now includes safer native support for multi-plate CellProfiler workflows:

- `cpatk-build-profiles --image_merge_keys Metadata_Plate,ImageNumber` for pooled exports where image and object tables share assay plate metadata.
- `cpatk-combine-profiles` for combining already-reviewed per-plate or per-export profile tables.
- pre-normalisation DMSO/reference control QC in `cpatk-preprocess`.
- optional `--batch_correction_method combat_location_scale` with confounding reports.
- before/after replicate and batch PC-association reports using `--replicate_group_columns` and `--batch_report_columns`.

See `docs/CPATK_v0_2_12_multi_plate_batch_release.md` and `examples/run_cpatk_multi_plate_recommended_workflow.sh`.


## v0.2.16 reporting and large-workbook hardening

CPATK v0.2.16 fixes failures seen during real malaria and mitotox stress tests.
Excel workbooks are now treated as readable summaries, not lossless exports: very
large sheets are previewed and an `Excel_export_notes` sheet records the original
shape and whether a sheet was truncated. Full data remain available in the TSV or
Parquet files written beside the workbook.

The HTML report heading now uses the plainer `Summary` wording. The
report generator also auto-discovers nearby plots by default, adds links back to
full source tables, includes a clearer results map, and uses plainer text aimed
at making the output easier to interpret.

For CLIPn smoke tests, the example shells no longer drop every row containing
any zero by default, because this was too strict for real preprocessed Cell
Painting matrices and could leave zero samples. The strict mode is still
available by setting `CLIPN_STRICT_DROP_ANY_ZERO=1`.

## v0.2.15 note: macOS sidecar files and cluster SciPy imports

CPATK v0.2.15 skips macOS AppleDouble sidecar files such as `._table.tsv`
during table discovery. These files are not real data tables and can contain
binary metadata bytes that break UTF-8 decoding. The inspection workflow also
writes `inspection_failure_report.tsv` for any recognised table that cannot be
read cleanly.

The acquisition-drift module now imports SciPy lazily for Spearman p-values and
falls back to a NumPy/pandas Spearman rho if `scipy.stats` cannot be imported in
a problematic HPC environment. Full drift QC, ML and PCA/UMAP workflows still
need a working SciPy/scikit-learn stack for production analysis.
