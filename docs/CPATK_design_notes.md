# CPATK design notes

## Scope of this first version

This package is an initial refactor of the mature ideas from the earlier `clipn` folder into a generic package named `cpatk`.

The package intentionally does not use the anomaly-detection folder or make assumptions about sperm-specific staining. It is designed for generic Cell Painting/high-content screening experiments.

## Main design decision

CPATK separates the analysis into reusable layers:

1. Inspection and schema discovery.
2. QC and preprocessing.
3. Classical non-AI analysis.
4. Optional AI/CLIPn-style analysis hooks.
5. Mode-of-action scoring.
6. Plotting, Excel summaries and HTML reporting.

This avoids having one very large analysis script that is hard to test, debug or reuse.

## Why classical analysis is first-class

The non-AI workflow is not just a fallback. It provides important interpretable outputs:

- PCA/UMAP-style embeddings.
- Pairwise distances.
- Nearest-neighbour tables.
- K-means/agglomerative/DBSCAN clustering.
- Cluster composition summaries.
- Silhouette summaries.
- Centroid-based MOA scoring.

These analyses are useful even when CLIPn or other AI methods are also used, because they provide a transparent baseline.

## Why AI/CLIPn is optional

AI integration is kept optional because:

- CLIPn may not be installed in every environment.
- Different projects may need different CLIPn wrappers or saved-model logic.
- The classical workflow should remain reproducible without AI dependencies.
- The package can later add a fully tested CLIPn adapter without breaking the base toolkit.

## QC philosophy

QC is performed on technical features and missingness/variance structure rather than on biological outcome labels. For assay-specific analyses, outcome-based filtering should be avoided unless an independent technical reason is documented.

## Output policy

CPATK writes TSV, TSV.GZ, Parquet, Excel, PDF, SVG and HTML outputs. It does not use comma-separated outputs for generated result tables.

Parquet writing requires `pyarrow` or `fastparquet`. If unavailable in the command-line preprocessing workflow, CPATK writes a TSV.GZ fallback and logs the reason.

## Future work

Planned additions include:

- Full CLIPn save/load/project adapter.
- Replicate reproducibility metrics.
- Batch and plate-layout diagnostics.
- Consensus nearest-neighbour stability across runs.
- MOA classifier calibration and uncertainty.
- SHAP/permutation attribution.
- Richer HTML reports with collapsible sections and embedded interactive plots.
- CellProfiler object-table merge helpers.
