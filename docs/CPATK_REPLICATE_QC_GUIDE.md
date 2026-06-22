# CPATK replicate QC guide

Version: 0.2.11 documentation expansion

## Why replicate QC matters

Replicate QC is one of the strongest reality checks in Cell Painting analysis. A UMAP can look biologically meaningful even when replicate profiles are inconsistent. Conversely, subtle phenotypes may be credible if replicate profiles group consistently and controls behave as expected.

Run replicate QC before interpreting clusters, nearest neighbours, MOA predictions, ML classifiers, SHAP explanations or CLIPn latent spaces.

## What counts as a replicate?

A replicate group should contain profiles that are expected to be similar.

Common replicate grouping columns:

```text
Metadata_Compound
Metadata_Dose
Metadata_Timepoint
Metadata_CellLine
Metadata_TreatmentDuration
```

Less specific grouping, such as MOA alone, is usually too broad for replicate QC. Different compounds with the same annotated MOA are not technical replicates.

## Basic replicate QC command

```bash
cpatk-stability \
  --input_table results/02_preprocess/preprocessed.tsv.gz \
  --output_dir results/04_replicate_qc_and_stability \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound,Metadata_MOA,Metadata_Dose,Metadata_Batch,cpd_type \
  --replicate_group_columns Metadata_Compound,Metadata_Dose \
  --n_neighbours 10 \
  --n_bootstraps 100 \
  --n_permutations 100 \
  --k_values 2,3,4,5,6,7,8,9,10 \
  --log_level INFO
```

## Outputs to inspect

```text
replicate_correlations.tsv
replicate_summary.tsv
neighbour_stability.tsv
cluster_stability.tsv
cluster_permutation.tsv
cluster_permutation_null.tsv
consensus_summary.tsv
stability_summary.xlsx
stability_report.html
```

## How to interpret replicate correlations

High replicate correlation suggests that profiles with the same compound/dose/timepoint produce similar morphology. Low replicate correlation can mean:

- the phenotype is weak;
- the dose is too low;
- the compound effect is heterogeneous;
- cells or staining were inconsistent;
- plate/batch/acquisition drift dominates;
- metadata merge or compound annotation is wrong;
- replicate grouping is too broad or incorrectly defined.

The exact acceptable threshold depends on assay, feature space and biological context. Treat replicate correlation as comparative evidence rather than a universal pass/fail value.

## Use replicate QC to compare preprocessing choices

Run replicate QC for each major preprocessing choice:

```text
baseline preprocessing
DMSO robust-z by plate
batch median-centred
future ComBat-corrected output
```

A better preprocessing strategy should usually:

- improve within-compound replicate consistency;
- reduce plate/batch dominance;
- preserve known positive/negative control separation;
- avoid collapsing biologically distinct references into one cloud;
- not create artificial separation by technical variables.

## Replicate QC across plates

For multi-plate projects, replicate QC is most informative when some compounds or controls are repeated across plates. If each compound appears on only one plate, it is difficult to distinguish compound biology from plate effects.

Recommended design features:

- DMSO/vehicle controls on every plate;
- shared positive controls on every plate;
- selected reference compounds repeated across plates;
- treatment classes distributed across plates where possible;
- randomised well positions where possible.

## Technical replicate versus biological replicate

Technical replicates share the same biological material and treatment but differ by well, field or imaging unit. Biological replicates differ by donor, batch, culture or independent experiment.

Do not mix these without labelling. A useful strategy is to run replicate QC at several levels:

```text
compound + dose within plate
compound + dose across plates
compound + dose + donor
positive controls within plate
DMSO controls within plate
```

## Neighbour stability

Neighbour stability checks whether the same profiles remain near each other when feature subsets are resampled. It is helpful for assessing nearest-neighbour claims.

Use it when:

- you plan to interpret top nearest neighbours;
- you compare compounds across CLIPn or classical spaces;
- you want confidence that neighbour relationships are not driven by a small number of unstable features.

Do not treat neighbour stability as mechanism proof. It is robustness evidence.

## Cluster stability

Cluster stability checks whether clusters are reproducible under sampling/permutation. Use it to avoid overinterpreting one visually pleasing UMAP or one K-means result.

A stable cluster structure should also make sense in:

- replicate correlations;
- known control separation;
- PCA/UMAP plots coloured by metadata;
- batch diagnostics;
- biological annotation.

## Replicate QC and CLIPn

Before using CLIPn, check replicate QC in the preprocessed input space. After CLIPn, inspect latent-space replicate consistency and nearest-neighbour behaviour. If replicate structure is poor before CLIPn, CLIPn should be treated as exploratory at best.

## Replicate QC and SHAP/feature attribution

Feature attribution is more meaningful when query compounds and their neighbours have consistent replicate profiles. If replicate QC is poor, SHAP may explain noise, batch or one outlier well rather than a reproducible phenotype.

## Reporting recommendations

For a manuscript or collaborator report, include:

- number of replicate groups;
- number of replicate pairs;
- median replicate correlation;
- replicate correlation distribution plot where available;
- whether controls are more consistent than test compounds;
- whether preprocessing improved replicate consistency;
- any groups removed or flagged because of poor reproducibility.

## Future replicate QC improvements

Useful future CPATK additions would be:

1. replicate correlation plots by compound/MOA/control class;
2. replicate-centroid distance summaries;
3. replicate QC thresholds with warning labels;
4. per-plate replicate QC comparisons;
5. replicate-aware nearest-neighbour summaries;
6. replicate-aware MOA validation;
7. group-aware ML cross-validation using compound ID to prevent leakage.
