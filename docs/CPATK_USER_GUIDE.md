# CPATK user guide

Version: 0.2.11 documentation expansion

## Purpose

CPATK is intended to be a reusable Cell Painting / high-content morphology analysis toolkit. It is not just a plotting package. It is designed to make the analysis auditable from the first metadata check through to preprocessing, replicate QC, batch diagnostics, classical analysis, optional AI/CLIPn integration, MOA scoring, feature attribution and final HTML reporting.

The central principle is:

```text
merge first, then preprocess, then analyse
```

For CellProfiler folders, this means Image, Cell, Nuclei, Cytoplasm and other object/compartment outputs should be assembled into one coherent profile matrix before all-zero profile removal, imputation, scaling, correlation filtering, clustering, CLIPn or supervised modelling.

## Recommended folder layout

A practical project layout is:

```text
project_root/
  raw_cellprofiler/
    plate_01/
    plate_02/
    plate_03/
  metadata/
    raw_plate_map.tsv
    compound_annotations.tsv
  results/
    00_metadata_check/
    01_profile_build/
    02_preprocess/
    03_classical/
    04_replicate_qc_and_stability/
    05_batch/
    06_visualisation/
    07_neighbours/
    08_moa/
    09_ml/
    10_explain/
    11_clipn/
    12_report/
```

The exact folder names are not required, but keeping a staged output structure makes review and debugging much easier.

## Installation

From the unpacked package folder:

```bash
python -m pip install -e .
```

Optional extras can be installed when needed:

```bash
python -m pip install -e .[all]
```

For HPC or older systems, it may be safer to install optional packages separately through conda-forge. The core workflow does not require Plotly, UMAP, SHAP, PHATE or Parquet support. CPATK should fall back gracefully where optional dependencies are absent.

## Recommended test command

Full test discovery is expected to pass with native numerical thread limits set:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m unittest discover -s tests -q
```

## Stage 0: inspect input files

Use this before deciding how to run the rest of the pipeline:

```bash
cpatk-inspect \
  --input_dir raw_cellprofiler \
  --output_dir results/00_inspect \
  --recursive \
  --log_level INFO
```

Use this when:

- you have a folder of unknown CellProfiler outputs;
- you are not sure which file is Image-level and which files are object-level;
- there are several plates or several exports;
- you want an inventory before preprocessing.

Do not rely on filename assumptions alone. The inspection table should be reviewed before a publication analysis.

## Stage 1: validate and format metadata

Use `cpatk-metadata` to make the plate map safer before using it for CellProfiler merging.

A typical messy plate-map command is:

```bash
cpatk-metadata \
  --metadata_table metadata/raw_plate_map.tsv \
  --output_dir results/00_metadata_check \
  --plate_column Assay_Plate_Barcode \
  --well_column Destination_Well \
  --source_plate_column Source_Plate_Barcode \
  --source_well_column Source_Well \
  --annotation_tables metadata/compound_annotations.tsv \
  --annotation_source_plate_column Barcode \
  --annotation_source_well_column Well \
  --merge_keys Metadata_Source_Plate,Metadata_Source_Well \
  --duplicate_policy error \
  --log_level INFO
```

The important output is:

```text
results/00_metadata_check/formatted_metadata.tsv
```

Use this output as the metadata input for profile building and preprocessing.

When to use this:

- always for a new dataset;
- always when plate-map files come from robots, plate handlers or collaborators;
- when wells are mixed between `A1` and `A01`;
- when metadata includes both assay/destination wells and source/library wells;
- when annotations need to be merged from compound library files.

When not to rely on automatic inference:

- when both source wells and assay wells are present;
- when columns are called generic names such as `Well`, `Barcode`, `Plate`, `SW` or `DW`;
- when source-plate annotations are being merged onto destination wells.

In these cases, specify `--plate_column`, `--well_column`, `--source_plate_column` and `--source_well_column` explicitly.

## Stage 2: build profiles from CellProfiler outputs

For a single CellProfiler export where `ImageNumber` is unique within the export:

```bash
cpatk-build-profiles \
  --input_dir raw_cellprofiler/plate_01 \
  --output_dir results/01_profile_build/plate_01 \
  --recursive \
  --metadata_table results/00_metadata_check/formatted_metadata.tsv \
  --aggregate_statistic median \
  --duplicate_image_policy error \
  --metadata_duplicate_policy error \
  --log_level INFO
```

This step:

- identifies Image/profile and object-level files;
- aggregates object-level CellProfiler outputs to image-level profiles using median or mean;
- prefixes features by compartment/table label;
- merges object summaries onto the Image/profile backbone;
- merges external metadata using canonical plate/well keys where available;
- writes inventory and merge reports.

Use median aggregation by default. Mean aggregation is more sensitive to object-level outliers.

## Multi-plate profile building

There are two different multi-plate cases.

### Case A: one CellProfiler export containing all plates with globally unique ImageNumber

This is the easier case. If the Image table contains reliable `Metadata_Plate` and `Metadata_Well`, and `ImageNumber` is globally unique across all plates in that export, `cpatk-build-profiles` can be run on the full folder.

### Case B: several independent CellProfiler exports where ImageNumber restarts

This is the risky case. Many CellProfiler workflows restart `ImageNumber` at 1 for each plate or export. If several such exports are placed in one folder, object-level tables can collide if the software merges by `ImageNumber` alone.

For v0.2.11, the safest production approach is:

1. run `cpatk-build-profiles` separately for each plate/export;
2. ensure each resulting table has a reliable `Metadata_Plate` value;
3. combine the per-plate profile tables only after profile building;
4. run one joint `cpatk-preprocess` command on the combined profile table.

A future code pass should add native composite-key multi-plate profile building, using a key such as `Metadata_Plate + ImageNumber` or an internally generated `Metadata_CPATK_Profile_ID` after plate/well have been attached from the Image table.

## Stage 3: preprocess profiles

A conservative first pass is:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/merged_profiles.tsv.gz \
  --output_dir results/02_preprocess \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5 \
  --max_absolute_correlation 0.95 \
  --log_level INFO
```

This is suitable when:

- you want a defensible baseline;
- you do not yet know whether plate effects are large;
- you want all QC reports before trying stronger correction.

Do not start with complex batch correction. First run the conservative baseline and inspect controls, replicate QC, PCA/UMAP and batch diagnostics.

## Stage 4: plate-level DMSO/reference normalisation

When each plate contains enough DMSO or vehicle controls:

```bash
cpatk-preprocess \
  --input_table results/01_profile_build/all_plates_merged_profiles.tsv.gz \
  --output_dir results/02_preprocess_dmso_by_plate \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate \
  --imputation_method median \
  --scaling_method robust \
  --log_level INFO
```

Use this when:

- DMSO/vehicle controls are present on each plate;
- plate-level intensity or morphology shifts are expected;
- you need a simple, interpretable normalisation method;
- the biological design includes comparable controls across plates.

Be careful when:

- some plates have too few DMSO wells;
- DMSO wells are spatially clustered and affected by edge or drift effects;
- the controls themselves are poor quality;
- plate identity is confounded with treatment or compound library.

## Stage 5: batch diagnostics

After preprocessing, always test whether plate or batch dominates the feature space:

```bash
cpatk-batch \
  --input_table results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/05_batch \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --batch_column Metadata_Plate \
  --columns_to_test Metadata_Plate,Metadata_Batch,Metadata_MOA,Metadata_Compound \
  --log_level INFO
```

Use this before interpreting MOA, CLIPn, ML or SHAP.

## Stage 6: replicate QC and stability

Replicate QC is central for Cell Painting. Run it before trusting clusters or nearest neighbours:

```bash
cpatk-stability \
  --input_table results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/04_replicate_qc_and_stability \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --replicate_group_columns Metadata_Compound,Metadata_Dose \
  --n_neighbours 10 \
  --n_bootstraps 100 \
  --n_permutations 100 \
  --k_values 2,3,4,5,6,7,8,9,10 \
  --log_level INFO
```

Use replicate groups that represent true biological or technical replicates. Do not group unrelated wells just because they share an MOA label.

## Stage 7: classical analysis

```bash
cpatk-classical \
  --input_table results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/03_classical \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --id_column Metadata_Compound \
  --colour_column Metadata_MOA \
  --cluster_group_columns Metadata_MOA,Metadata_Compound \
  --distance_metric cosine \
  --n_neighbours 15 \
  --n_clusters 8 \
  --log_level INFO
```

Use this as the main visual and neighbour baseline. PCA, UMAP, nearest neighbours and distance tables should be reviewed before more complex models.

## Stage 8: visualisation

```bash
cpatk-visualise \
  --input_table results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/06_visualisation \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Batch,cpd_type \
  --colour_column Metadata_MOA \
  --label_column Metadata_Compound \
  --interactive \
  --log_level INFO
```

Use both static and interactive outputs. Static plots are safer for manuscripts; interactive plots are better for exploration and reviewing individual compounds.

## Optional downstream analyses

Use optional downstream workflows only after metadata, preprocessing, replicate QC and batch diagnostics look credible.

### MOA scoring

Use `cpatk-moa` when you have known classes or reference compounds.

### ML classification

Use `cpatk-ml` only when there are enough labelled examples and a leakage-aware validation design is possible.

### Feature attribution / SHAP

Use `cpatk-explain` to explain model behaviour or query-neighbour differences. Interpret SHAP alongside direct feature statistics, not as causal proof.

### CLIPn

Use `cpatk-clipn` when you have at least two datasets or a justified split of one dataset. CLIPn should not be used to compensate for poor metadata, poor replicate consistency or unresolved batch effects.

## What to check before publication

Before a result is considered publication-ready, inspect:

```text
metadata validation report
metadata merge report
input table inventory
object aggregation report
sample_qc.tsv
sample_qc_post_feature_filter.tsv
feature_qc.tsv
all_zero_row_report.tsv
reference_normalisation_report.tsv
batch_centering_report.tsv
correlation_filter_report.tsv
replicate_correlations.tsv
replicate_summary.tsv
batch_prediction.tsv
pc_metadata_association.tsv
classical nearest-neighbour tables
PCA/UMAP plots coloured by plate, batch, compound and MOA
final HTML report
```

A convincing biological signal should survive basic changes in preprocessing choices, should show reasonable replicate consistency, and should not be explainable purely by plate, batch, acquisition order or metadata merge failure.
