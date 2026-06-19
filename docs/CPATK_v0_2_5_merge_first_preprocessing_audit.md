# CPATK v0.2.5 merge-first preprocessing audit

This note records the critical preprocessing decision made after reviewing the legacy project-specific Cell Painting preprocessing script and the CPATK v0.2.4 implementation.

## Issue identified

The v0.2.4 workflow could build profiles from a folder of CellProfiler outputs, then preprocess the merged table. However, it did not explicitly remove profile rows whose retained merged feature values were all zero. It did remove constant or all-zero features through feature-level QC, but that is different from removing failed profiles/rows.

For Cell Painting, an all-zero row after merging Cell, Nuclei, Cytoplasm/Image-derived features is usually not a valid morphology profile. It can indicate an empty field, failed segmentation, failed object aggregation, failed export, or a merge artefact. Keeping such rows means the imputation and scaling steps can turn failed measurements into apparently valid profiles.

## Required order

The correct order is:

1. read all CellProfiler output tables from the folder;
2. choose an Image/profile backbone;
3. aggregate each object-level table to ImageNumber;
4. merge all object/profile summaries and external metadata;
5. infer and QC feature columns on the fully merged profile matrix;
6. remove all-zero profile rows using the retained merged feature set;
7. perform sample missingness filtering;
8. impute remaining missing values;
9. optionally normalise to reference controls or centre batches;
10. scale and correlation-filter features.

The all-zero row check must not be applied to individual CellProfiler object files before merging, because a compartment can have zero-like features while another compartment contains valid signal.

## v0.2.5 implementation

CPATK now includes `calculate_all_zero_row_report()` and applies it within `preprocess_profiles()` after feature-level QC and before sample QC, imputation, normalisation, scaling and correlation filtering.

A row is flagged when it has at least one observed retained feature value and no observed retained feature value is non-zero. Missing values do not count as non-zero evidence. Fully missing rows are handled separately by the sample/profile missingness filter.

The filter is enabled by default and can be disabled with `--disable_all_zero_row_filter`.

## Outputs

The preprocessing workflow now writes:

- `all_zero_row_report.tsv`
- `preprocessing_summary.tsv` with all-zero row counts
- `preprocessing_decision_log.tsv`
- `preprocessing_report.html` warning when rows are removed
- `plots/all_zero_feature_row_qc.pdf` and `.svg`

## Legacy script comparison

The old project-specific script already encoded several useful defensive ideas: merge metadata before imputation, replace infinite and very large finite numeric values with missing values, drop all-NaN columns before imputation, then impute and feature-select. CPATK keeps those principles but makes the workflow more generic, auditable and test-covered.

## Test coverage

v0.2.5 adds tests that confirm:

- all-zero rows are detected;
- all-zero rows are removed before imputation;
- the filter can be disabled;
- when a folder contains separate Cell and Nuclei tables, a row is removed only if the merged profile has no non-zero observed feature evidence across the retained merged features.
