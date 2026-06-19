# CPATK v0.2.2 critical evaluation and upgrade notes

## Why another preprocessing pass was needed

The earlier v0.2.0/v0.2.1 preprocessing workflow was useful, but it was still too permissive for heterogeneous Cell Painting projects. Real CellProfiler exports often contain thousands of columns, many of which are numeric but not biological profile features. Examples include `ImageNumber`, `ObjectNumber`, `ExecutionTime_*`, `Count_*`, image dimensions, file names, file hashes and grouping indices. Treating these as morphology features can create spurious clustering, artificial MOA classification, and misleading SHAP/permutation importance.

v0.2.2 therefore makes column role assignment explicit. Each column is classified as metadata, selected feature, excluded numeric QC/provenance, numeric-not-selected or non-numeric-not-selected. This role assignment is written to `column_role_report.tsv` and included in the HTML report.

## Defensible default preprocessing

The current default workflow is:

1. Normalise column names, including BOM and trailing spaces.
2. Drop likely accidental CSV index columns such as `Unnamed: 0`.
3. Standardise common metadata aliases while preserving original columns.
4. Infer metadata and feature columns conservatively.
5. Exclude obvious numeric QC/provenance columns from default features.
6. Remove features with excessive missingness, near-zero variance or too few unique values.
7. Remove profiles with excessive feature missingness.
8. Optionally winsorise extreme values if explicitly requested.
9. Impute remaining missing values, with median imputation as the default.
10. Optionally normalise to reference/control wells, such as DMSO within plate.
11. Optionally centre features within batch groups.
12. Scale features, using robust scaling by default.
13. Optionally remove highly correlated redundant features.
14. Write all decisions to TSV, Excel, plots, HTML and logs.

Median imputation remains the recommended default because it is robust and does not borrow information across perturbations. KNN imputation remains available but should be treated as exploratory because it can smooth true perturbation or batch structure.

## Reference/control normalisation

v0.2.2 adds optional control/reference normalisation. For example, users can normalise each plate to its DMSO wells using robust-z or median-centering. This is often appropriate for Cell Painting, but it is not enabled by default because it depends on correct metadata and a suitable experimental design.

Recommended use when controls are available:

```bash
cpatk-preprocess \
  --input_table profiles.tsv \
  --output_dir results/01_preprocess_dmso_normalised \
  --reference_normalisation_method robust_z \
  --reference_column Metadata_Compound \
  --reference_values DMSO \
  --reference_group_columns Metadata_Plate
```

## Batch centering

v0.2.2 adds optional batch centering. This is useful for sensitivity analysis, but should not be used to hide a poor design. A batch-corrected analysis should usually be presented alongside the uncorrected analysis.

## MOA analysis

MOA analysis now includes:

- centroid-based scoring;
- top-N centroid rankings;
- distance, similarity, softmax-style confidence and top1-vs-top2 margin;
- leave-one-out centroid validation;
- optional KNN predictions;
- optional KNN neighbour tables;
- HTML reports and confidence plots.

The confidence scores are useful for triage, but they do not prove mechanism of action. They should be interpreted alongside replicate reproducibility, class size, cross-validation performance, nearest-neighbour stability and batch diagnostics.

## SHAP and feature attribution

v0.2.2 improves SHAP support by adding:

- class-aware row subsampling;
- tree-explainer preference for tree models;
- fallback to generic SHAP explainer;
- robust handling of multiclass SHAP arrays;
- global SHAP importance;
- class-level SHAP importance where available;
- explicit SHAP status reporting;
- HTML feature-attribution reports.

Permutation importance remains the default recommended feature-attribution method because it directly measures held-out performance loss when each feature is shuffled. SHAP is valuable, especially for explaining MOA classifiers, but it is still a model explanation rather than proof of biological causality.

## Remaining caveats

CPATK is becoming a strong analysis toolkit, but preprocessing still cannot rescue a poor experimental design. Plate-layout diagnostics, replicate checks and batch/domain-shift diagnostics remain essential. In particular, if treatment, plate row, donor, batch or imaging run are confounded, the package can expose that problem but cannot make the resulting inference fully causal.
