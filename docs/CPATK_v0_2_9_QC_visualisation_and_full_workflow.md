# CPATK v0.2.9 QC, visualisation and full-workflow audit

## Why this pass was needed

The legacy plotting and QC scripts contained useful analyses that were not fully represented in CPATK v0.2.8. These included latent-space norm checks, latent heatmaps, UMAP/PHATE/topology-style views, per-compartment acquisition-drift QC, nearest-neighbour overlap testing and shared-neighbour plots. v0.2.9 ports these ideas into generic CPATK modules with defensive input handling, logging, TSV/Excel/HTML outputs and unit tests.

## New command-line tools

- `cpatk-drift-qc`: object-level per-compartment acquisition-drift QC before profile aggregation.
- `cpatk-visualise`: PCA, UMAP-if-available, PHATE-if-available, heatmaps, latent norm checks and kNN topology plots.
- `cpatk-neighbours`: nearest-neighbour top-N plots, shared-neighbour scatter plots and overlap/RBO comparisons across runs.

## Important design choices

1. Per-compartment drift QC runs before profile aggregation. This is deliberate: acquisition drift can affect Cell, Cytoplasm, Nuclei or other object compartments differently, and collapsing too early can hide a problem.
2. The visualisation workflow is generic. It can use digit-named CLIPn latent dimensions, prefixed latent columns, or ordinary processed Cell Painting features.
3. Topological graphs are implemented as a robust k-nearest-neighbour graph by default. This avoids requiring KeplerMapper/PyVis but preserves the useful idea of inspecting local connectivity in latent/profile space.
4. UMAP and PHATE are optional. If dependencies are missing, CPATK logs the omission and still produces PCA, heatmap and topology outputs.
5. Nearest-neighbour overlap supports both long and wide NN schemas, automatic k detection, optional tie expansion and rank-biased overlap.
6. The full shell script uses only CPATK command-line tools and shell logic; there is no inline Python.

## Metadata expectations

The recommended minimal metadata file contains:

- `Metadata_Plate`
- `Metadata_Well`
- `Metadata_Compound`
- `cpd_type`

Strongly recommended:

- `Metadata_MOA`
- `Metadata_Dose`
- `Metadata_Batch`
- `Replicate`
- relevant biological covariates such as `Donor`, `CellLine`, `Timepoint` or `TreatmentDuration`

See `examples/example_metadata.tsv` and `examples/METADATA_REQUIREMENTS.md`.

## Outputs added in v0.2.9

`cpatk-drift-qc` writes:

- per-compartment `drift_statistics.tsv`
- per-compartment `per_image_summary.tsv`
- drift plots for top drifting features
- `drift_input_inventory.tsv`
- `drift_qc_summary.xlsx`
- `drift_qc_report.html`
- `drift_qc.log`

`cpatk-visualise` writes:

- `visualisation_feature_columns.tsv`
- `latent_norm_summary.tsv`
- `pca_coordinates.tsv`
- `pca_variance.tsv`
- optional `umap_coordinates.tsv`
- optional `phate_coordinates.tsv`
- `heatmap_matrix.tsv`
- `heatmap_row_order.tsv`
- `heatmap_column_order.tsv`
- `topology_nodes.tsv`
- `topology_edges.tsv`
- `topology_coordinates.tsv`
- static plots and optional interactive HTML files
- `visualisation_report.html`
- `visualise.log`

`cpatk-neighbours` writes:

- top-neighbour TSV and plots for requested compounds
- shared-neighbour TSV and static/interactive scatter for compound pairs
- neighbour-overlap per-query and summary TSVs
- `neighbour_analysis_summary.xlsx`
- `neighbour_analysis_report.html`
- `neighbour_analysis.log`

## Caveats

- Drift QC should be interpreted as QC evidence, not as automatic correction. Strong drift suggests a need to inspect imaging order, staining batches, plate order and controls.
- UMAP and topology plots are exploratory. They should not replace distance matrices, stability testing, replicate reproducibility or MOA validation.
- Neighbour-overlap scores depend on the chosen neighbour depth and distance metric. Use them to compare analysis runs, not as standalone biological truth.
