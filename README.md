# CPATK: Cell Painting Analysis Toolkit

CPATK is a generic toolkit for Cell Painting and high-content profiling analyses. It is designed to replace project-specific scripts with a modular, testable and reproducible workflow.

This first development version is deliberately generic. It does not assume sperm data, acrosome staining, one particular plate layout, or one particular biological question. The earlier `clipn` scripts were used as inspiration for the workflow design, but CPATK is organised as a reusable package.

## Design goals

- Work with generic CellProfiler/Cell Painting profile tables.
- Support object-level to profile-level aggregation.
- Perform robust preprocessing and QC.
- Write Parquet for large intermediate tables.
- Write TSV tables, formatted Excel summaries and HTML reports.
- Produce static PDF/SVG plots and optional interactive HTML plots.
- Support non-AI classical analysis as a first-class workflow.
- Keep AI/CLIPn-style analysis optional, so the rest of the toolkit remains usable without AI dependencies.
- Provide a route towards mode-of-action classification.
- Include extensive logging and unit tests.

## Current modules

```text
cpatk/
    ai.py                Optional AI/CLIPn backend status and future adapter hooks
    clustering.py        K-means, agglomerative clustering, DBSCAN and summaries
    distances.py         Pairwise distances and nearest-neighbour analysis
    embedding.py         PCA, UMAP-if-available, and t-SNE helpers
    features.py          Metadata/feature inference and feature summaries
    inspection.py        Generic table inspection workflow
    io.py                TSV, Parquet, Excel, HTML table helpers
    logging_utils.py     Logging setup
    moa.py               Centroid and KNN mode-of-action classification
    plotting.py          Static and optional interactive plotting
    preprocessing.py     QC, imputation, scaling and correlation filtering
    qc.py                Feature-level, sample-level and outlier QC helpers
    reporting.py         Simple HTML report generation
    cli/                 Command-line entry points
```

## Installation

From the package folder:

```bash
python -m pip install -e .
```

Recommended extras if available:

```bash
python -m pip install -e '.[all]'
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The current development version passes 42 unit tests in the build environment.

## Example workflow

### 1. Inspect input tables

```bash
cpatk-inspect \
  --input_dir /path/to/input_tables \
  --output_dir /path/to/results/00_inspection \
  --log_level INFO
```

Outputs include:

```text
file_inventory.tsv
file_summary.tsv
column_inventory.tsv
inspection_summary.xlsx
inspect.log
```

### 2. Preprocess profiles

For a profile-level table:

```bash
cpatk-preprocess \
  --input_table /path/to/profiles.tsv \
  --output_dir /path/to/results/01_preprocess \
  --metadata_columns Metadata_Plate,Metadata_Well,compound,moa \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO
```

For an object-level table that needs aggregation:

```bash
cpatk-preprocess \
  --input_table /path/to/object_level.tsv.gz \
  --output_dir /path/to/results/01_preprocess \
  --metadata_columns Metadata_Plate,Metadata_Well,compound,moa \
  --aggregate_by Metadata_Plate,Metadata_Well,compound,moa \
  --aggregate_statistic median \
  --log_level INFO
```

Outputs include:

```text
preprocessed.parquet
feature_qc.tsv
sample_qc.tsv
correlation_filter_report.tsv
retained_features.tsv
preprocessing_summary.tsv
preprocessing_summary.xlsx
preprocess.log
```

### 3. Run non-AI classical analysis

```bash
cpatk-classical \
  --input_table /path/to/results/01_preprocess/preprocessed.parquet \
  --output_dir /path/to/results/02_classical \
  --metadata_columns Metadata_Plate,Metadata_Well,compound,moa \
  --id_column compound \
  --colour_column moa \
  --cluster_group_columns moa,compound \
  --distance_metric cosine \
  --n_neighbours 10 \
  --n_clusters 8 \
  --log_level INFO
```

Outputs include:

```text
pca_scores.tsv
pca_explained_variance.tsv
embedding.tsv
pairwise_distances.tsv.gz
nearest_neighbours.tsv
clusters.tsv
cluster_summary.tsv
cluster_silhouette_summary.tsv
embedding_static.pdf
embedding_static.svg
embedding_interactive.html
classical_analysis_summary.xlsx
classical.log
```

### 4. Run centroid-based MOA scoring

```bash
cpatk-moa \
  --input_table /path/to/results/01_preprocess/preprocessed.parquet \
  --output_dir /path/to/results/03_moa \
  --metadata_columns Metadata_Plate,Metadata_Well,compound,moa \
  --class_column moa \
  --min_class_size 2 \
  --metric cosine \
  --top_n 5 \
  --log_level INFO
```

Outputs include:

```text
moa_centroids.tsv
moa_class_summary.tsv
moa_centroid_scores.tsv
moa_top_predictions.tsv
moa_prediction_summary.tsv
moa_summary.xlsx
moa.log
```

### 5. Create an HTML report

```bash
cpatk-report \
  --output_html /path/to/results/summary_report.html \
  --title "Cell Painting analysis report" \
  --narrative "Distribution-first Cell Painting analysis using CPATK." \
  --table /path/to/results/01_preprocess/preprocessing_summary.tsv \
  --table /path/to/results/02_classical/cluster_summary.tsv \
  --table /path/to/results/03_moa/moa_prediction_summary.tsv \
  --plot /path/to/results/02_classical/embedding_static.svg \
  --log_level INFO
```

## QC philosophy

CPATK separates technical quality control from biological interpretation. The default QC steps include:

- feature missingness summaries
- sample/profile missingness summaries
- near-zero variance filtering
- minimum unique value checks
- optional robust profile outlier detection
- imputation after QC
- robust or standard scaling
- optional highly correlated feature filtering

For assay-specific workflows, CPATK should avoid filtering directly on the biological outcome feature unless there is an independent technical justification.

## Classical analysis versus AI analysis

The classical route uses distances, embeddings, clustering, nearest neighbours and centroid/KNN MOA scoring. This is always available.

The AI route is optional. Version 0.1.0 includes explicit backend availability checks and a placeholder adapter for future CLIPn integration. This is intentional: users should be able to run robust non-AI analysis even when CLIPn or other AI models are not installed.

## Development roadmap

Planned future additions:

- full CLIPn adapter with save/load projection workflow
- richer interactive dashboards
- plate-layout diagnostics and batch-effect modelling
- replicate-correlation analysis
- consensus nearest-neighbour stability across runs
- richer MOA classification with calibration and uncertainty
- SHAP/permutation feature attribution for classification decisions
- CellProfiler-specific object/table merge helpers
- project templates for common Cell Painting experiments
