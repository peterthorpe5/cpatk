# Malaria CPATK v0.2.14 full test workflow

This shell runs the malaria `ML-BE009` CellProfiler export through CPATK v0.2.14 and is designed as a real-world package validation workflow.

Compared with the previous v0.2.12 test shell, this version passes the optional compound-to-phenotype table into `cpatk-moa` so that pseudo-anchor clusters can be conservatively annotated with biological phenotype labels.

## Default paths

```bash
BASE_DIR=/home/pthorpe001/data/2025_jason_cell_painting/data/malaria
RAW_DIR=${BASE_DIR}/ML-BE009
RAW_METADATA=${BASE_DIR}/ML-BE009-kvp.csv
CLEANED_METADATA=${BASE_DIR}/ML-BE009-kvp_cleaned.csv
PHENOTYPE_LABEL_TABLE=${BASE_DIR}/cpd_id_to_phenotype.tsv
```

## Run

```bash
qsub run_malaria_cpatk_v0_2_14_full_test.sh
```

For a quick smoke test:

```bash
RUN_CLIPN=0 \
RUN_ML=0 \
RUN_EXPLAIN=0 \
RUN_MOA=1 \
STABILITY_BOOTSTRAPS=5 \
STABILITY_PERMUTATIONS=5 \
MOA_BOOTSTRAPS=5 \
MOA_PERMUTATIONS=10 \
qsub run_malaria_cpatk_v0_2_14_full_test.sh
```

## MOA phenotype labels

The script expects a phenotype label table with at least:

```text
cpd_id	label
```

These settings can be overridden:

```bash
PHENOTYPE_LABEL_TABLE=/path/to/cpd_id_to_phenotype.tsv \
PHENOTYPE_LABEL_ID_COLUMN=cpd_id \
PHENOTYPE_LABEL_COLUMN=label \
PSEUDO_ANCHOR_FINAL_MOA_COLUMN=moa_final \
qsub run_malaria_cpatk_v0_2_14_full_test.sh
```

Set this to annotate pseudo anchors without using phenotype-labelled `moa_final` for centroid scoring:

```bash
USE_PHENOTYPE_LABELS_FOR_MOA=0 qsub run_malaria_cpatk_v0_2_14_full_test.sh
```

## Key MOA outputs

```text
09_moa/pseudo_anchor_phenotype_labels.tsv
09_moa/pseudo_anchor_phenotype_label_audit.tsv
09_moa/pseudo_anchor_phenotype_summary.tsv
09_moa/pseudo_anchors.tsv
09_moa/advanced_moa_top_predictions.tsv
09_moa/moa_report.html
```

Inspect `pseudo_anchor_phenotype_summary.tsv` first. It shows which pseudo clusters were labelled, weakly labelled, mixed or unlabelled.
