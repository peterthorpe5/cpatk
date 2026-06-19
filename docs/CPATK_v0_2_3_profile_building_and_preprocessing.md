# CPATK v0.2.3 profile building and preprocessing rationale

CPATK v0.2.3 adds a dedicated profile-building layer for projects where the
input is a folder of Cell Painting outputs rather than one clean profile table.
This is common for CellProfiler exports, where Image, Cell, Cytoplasm, Nuclei
and external plate-map/metadata files may be supplied separately.

## Critical point

CPATK does not blindly merge different object compartments by `ObjectNumber`.
Object identifiers are not always guaranteed to have the same biological meaning
across CellProfiler object tables.  Therefore the defensible default is to:

1. use the Image/profile table as the row-level backbone;
2. aggregate each object-level table within `ImageNumber`;
3. prefix aggregated object features with the source table label;
4. merge aggregated object summaries onto the Image/profile backbone by
   `ImageNumber`; and
5. merge external metadata by canonical plate/well aliases where possible.

This design is intentionally conservative.  It favours traceability and avoids
creating apparent profiles from unsafe row-order or object-number joins.

## Supported input formats

The folder profile builder supports:

- `.csv`
- `.csv.gz`
- `.tsv`
- `.tsv.gz`
- `.parquet`
- `.xlsx` / `.xls`

## New commands

Build profiles only:

```bash
cpatk-build-profiles \
  --input_dir /path/to/cellpainting_folder \
  --output_dir results/00_profile_build \
  --recursive \
  --aggregate_statistic median
```

Build profiles from a folder and then preprocess them in one step:

```bash
cpatk-preprocess \
  --input_dir /path/to/cellpainting_folder \
  --output_dir results/01_preprocess \
  --recursive \
  --metadata_table /path/to/plate_map.tsv \
  --imputation_method median \
  --scaling_method robust \
  --max_feature_missing_fraction 0.2 \
  --max_sample_missing_fraction 0.5
```

The original single-table workflow remains valid:

```bash
cpatk-preprocess \
  --input_table profiles.tsv \
  --output_dir results/01_preprocess \
  --imputation_method median \
  --scaling_method robust
```

## Output reports

When folder input is used, CPATK writes a `00_profile_build` folder inside the
preprocessing output directory.  This includes:

- `merged_profiles.parquet`, or `merged_profiles.tsv.gz` if Parquet support is unavailable;
- `input_table_inventory.tsv`;
- `object_aggregation_report.tsv`;
- `metadata_merge_report.tsv`;
- `profile_build_summary.xlsx`;
- `profile_build_report.html`; and
- the usual preprocessing outputs, plots, workbook and `preprocessing_report.html`.

## Missing data

Missing data are handled after profile building.  The default remains median
imputation because it is robust and does not borrow morphology from nearby
profiles in feature space.  Group-wise imputation is available for deliberately
plate- or batch-aware workflows, and KNN imputation is available for exploratory
use, but KNN should be interpreted cautiously because it can smooth treatment or
batch structure.
