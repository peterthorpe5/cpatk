# CPATK v0.2.10 release-hardening notes

Date: 2026-06-22

CPATK v0.2.10 is a release-hardening pass focused on production and publication safety. It addresses release blockers identified during the v0.2.9 audit and adds a new step-one metadata validation workflow for messy real-world Cell Painting metadata and annotation files.

## Main changes

### Release hygiene

- Updated the package version to `0.2.10` in `pyproject.toml` and `cpatk/__init__.py`.
- Updated the README installation path and command list.
- Added `cpatk-metadata` as a first-class command-line tool.
- The final release archive should be built after removing `__pycache__`, `.pyc`, and other generated cache files.

### Metadata and annotation step one

A new module, `cpatk.metadata_validation`, and CLI, `cpatk-metadata`, provide a pre-flight metadata check before profile building or preprocessing.

The workflow:

- reads CSV, TSV, gzipped CSV/TSV, Parquet and Excel tables using CPATK IO helpers;
- removes accidental CSV index columns;
- standardises metadata aliases;
- canonicalises well names such as `A1` and `A01` to a consistent format;
- adds legacy compatibility columns such as `cpd_id`, `cpd_type`, `Library`, `Plate_Metadata` and `Well_Metadata` where possible;
- validates missing plate/well keys;
- reports duplicate merge keys;
- optionally collapses duplicate keys using a strict, first, last or consensus policy;
- merges one or more annotation tables using strict key validation;
- writes TSV, Excel, log and HTML outputs.

Recommended command:

```bash
cpatk-metadata \
  --metadata_table raw_metadata.csv \
  --output_dir results/00_metadata_check \
  --annotation_tables annotation_file.csv,compound_library.tsv \
  --merge_keys Metadata_Source_Plate,Metadata_Source_Well \
  --duplicate_policy error \
  --log_level INFO
```

The main downstream input is:

```text
results/00_metadata_check/formatted_metadata.tsv
```

### Report-link fix

Copied interactive HTML assets are now linked using their true relative path under `report_assets/`. This fixes the v0.2.9 bug where the generated report could link only to the basename of the copied file.

### Visualisation manifest fix

`run_visualisation_workflow()` now returns a populated output manifest containing the paths of the feature table, norm summary, PCA outputs, UMAP/PHATE outputs where available, heatmaps, topology outputs and generated reports. This makes downstream report assembly and reproducibility safer.

### Merge safety

Dangerous duplicate rows are no longer silently collapsed by default.

- Duplicate image/profile rows now fail by default before image/profile merging.
- Duplicate external metadata keys now fail by default before annotation or metadata merging.
- Permissive collapse policies are still available for deliberately messy exploratory work, but they are explicit choices rather than silent defaults.

### Preprocessing hardening

The preprocessing workflow now includes additional safety checks:

- sample/profile QC is reported before and after feature-level QC;
- reference/control normalisation is applied before imputation so reference statistics are not biased by imputed values;
- KNN imputation caps `n_neighbors` to the available number of rows;
- small KNN groups fall back to median-style behaviour rather than failing;
- missingness indicators are labelled as `missingness_indicator` and are excluded from biological correlation filtering by default;
- final feature matrices are explicitly validated for non-empty rows, non-empty features, no NaN values and no infinite values;
- full correlation filtering has a maximum-feature guard for very wide matrices;
- report fields avoid comma-joined values where possible and prefer semicolon-style summaries in TSV-compatible cells.

### Test and CI notes

The new regression tests cover:

- copied interactive HTML report links;
- visualisation output manifests;
- strict duplicate image row handling;
- strict duplicate metadata key handling;
- KNN neighbour capping for small tables;
- reference normalisation before imputation;
- metadata well canonicalisation and legacy alias creation;
- source-plate/source-well annotation merging;
- step-one metadata validation workflow outputs.

Full test discovery passed in this sandbox when native numerical-library thread counts were limited:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m unittest discover -s tests -q
```

`ruff` was not available in this sandbox, so formal style linting could not be run here. The package was syntax-checked with `compileall`.

## Remaining risks

- The new metadata workflow has been smoke-tested on representative messy metadata/annotation examples, but more real datasets will reveal further aliases and edge cases.
- Full correlation filtering is still inherently expensive on very wide Cell Painting matrices; the new guard prevents accidental memory blow-ups but blockwise correlation filtering would be a useful future improvement.
- SHAP, UMAP, Plotly and CLIPn remain optional environment-sensitive components and should continue to fail gracefully.
- A real full CellProfiler folder end-to-end run should still be performed before manuscript-facing analysis.
