# CPATK ML and nearest-neighbour method guide

This guide is bundled with CPATK so it can be included in HTML reports and exported with analysis results.

The central rule is simple: metadata validation, profile-building audits, preprocessing QC, replicate reproducibility and batch checks come first. PCA, nearest neighbours, clustering, MOA, SHAP and CLIPn are interpretation layers. They are useful only when the upstream QC supports them.

Use classical feature-space analysis as the default route. Treat CLIPn as optional and clearly provenance-dependent. A successful CLIPn backend run is not automatically a better biological representation than PCA, nearest neighbours, compound-level distance heatmaps or feature-space MOA.

The tab-separated guide file in this folder gives method-by-method guidance on what each method does, when it is useful, what results mean, when not to use it and the main caveats.
