# CPATK phenotype-labelled pseudo-anchor MOA guide

CPATK can generate pseudo-anchor clusters from Cell Painting profiles or CLIPn latent embeddings. From v0.2.14 onwards, those pseudo-anchor clusters can also be annotated using a curated compound-to-phenotype table.

This is useful when some compounds have known biological phenotypes, target classes or mechanism descriptions, but the full experiment is not labelled well enough for a fully supervised MOA classifier.

## When to use this

Use phenotype-labelled pseudo anchors when:

- you have a table mapping compounds to curated phenotype or mechanism labels;
- you want to interpret unsupervised morphology clusters;
- you want centroid scoring outputs to use conservative biological labels where the cluster is sufficiently supported;
- you still want unlabelled or mixed clusters to remain as pseudo-anchor identifiers.

Do not treat this as proof of mechanism. The label is inherited from known compounds present in the pseudo-anchor cluster. It should be interpreted alongside replicate QC, batch QC, nearest-neighbour evidence and visualisation.

## Input label table

The minimal table is tab-separated and has two columns:

```text
cpd_id	label
DHA	artemisinin-like parasite killing
KAE609	ATP4 inhibitor-like phenotype
```

Repeated rows are allowed and are audited. Multiple labels per compound are allowed and retained. If a cell contains a compound label such as `label one;label two`, CPATK will preserve that as one curated label unless you explicitly request splitting.

## Recommended command

```bash
cpatk-moa \
  --input_table results/02_preprocess/strategy_dmso_robust_z/preprocessed.tsv.gz \
  --output_dir results/09_moa \
  --id_column Metadata_Compound \
  --metadata_columns Metadata_Plate,Metadata_Well,Metadata_Compound \
  --make_pseudo_anchors \
  --pseudo_anchor_method bootstrap \
  --auto_k \
  --k_values 4,6,8,10,12 \
  --pseudo_anchor_label_table cpd_id_to_phenotype.tsv \
  --pseudo_anchor_label_id_column cpd_id \
  --pseudo_anchor_label_column label \
  --pseudo_anchor_final_moa_column moa_final \
  --pseudo_anchor_label_min_labelled_fraction 0.2 \
  --pseudo_anchor_label_min_dominant_fraction 0.5 \
  --aggregate_method median \
  --centroid_method median \
  --adaptive_shrinkage \
  --score_method cosine \
  --make_projection_plots \
  --projection both \
  --interactive
```

## Important outputs

```text
pseudo_anchor_phenotype_labels.tsv
```

Cleaned long-format phenotype label table after trimming blanks and removing exact duplicate rows.

```text
pseudo_anchor_phenotype_label_audit.tsv
```

Audit table reporting raw rows, duplicate rows removed, unique labelled identifiers and compounds with multiple labels.

```text
pseudo_anchor_phenotype_summary.tsv
```

Cluster-level interpretation table. Key columns include:

- `pseudo_moa`: unsupervised pseudo-anchor cluster ID;
- `moa_final`: final conservative label used for centroid scoring;
- `label_status`: whether the cluster was phenotype-labelled, unlabelled or weak/mixed;
- `n_compounds`: number of compounds in the pseudo-anchor cluster;
- `n_labelled_compounds`: number with labels;
- `labelled_fraction`: labelled compounds divided by all cluster members;
- `dominant_phenotype`: most common phenotype label among labelled compounds;
- `dominant_phenotype_fraction_of_labelled`: support for the dominant label among labelled members;
- `top_phenotype_labels`: compact pipe-separated label/count summary.

```text
pseudo_anchors.tsv
pseudo_anchor_clusters.tsv
moa_anchor_table.tsv
advanced_moa_top_predictions.tsv
```

These now carry the final interpreted `moa_final` column when a phenotype label table is supplied, unless `--annotate_pseudo_anchors_only` is used.

## Conservative labelling logic

A pseudo-anchor cluster receives the dominant phenotype as `moa_final` only when both conditions are met:

1. at least `--pseudo_anchor_label_min_labelled_fraction` of cluster compounds are labelled;
2. the dominant label accounts for at least `--pseudo_anchor_label_min_dominant_fraction` of labelled cluster compounds.

Otherwise, `moa_final` remains the pseudo-anchor ID, such as `PseudoMOA_0003`. This prevents one weakly labelled compound from naming a whole mixed cluster.

## Annotation-only mode

To write phenotype summaries but keep centroid scoring against unsupervised pseudo-anchor IDs, add:

```bash
--annotate_pseudo_anchors_only
```

## Splitting multi-label cells

Only use this when the label file is deliberately formatted with a known separator:

```bash
--pseudo_anchor_label_split_regex ';'
```

Do not use this if semicolons are part of a curated free-text label that should remain intact.
