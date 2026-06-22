# CPATK metadata and annotation guide

Version: 0.2.11 documentation expansion

## Core idea

Metadata handling is one of the highest-risk parts of Cell Painting analysis. CPATK therefore separates:

- assay/destination plate and well, used to merge CellProfiler profiles;
- source/library plate and well, used to merge compound annotations;
- compound identity and biological labels;
- technical covariates such as batch, date, instrument, site and acquisition order.

Do not allow source wells to become assay wells by accident.

## Preferred metadata columns

The safest input metadata should include:

```text
Metadata_Plate
Metadata_Well
Metadata_Compound
cpd_type
```

Recommended additional columns:

```text
Metadata_Source_Plate
Metadata_Source_Well
Metadata_MOA
Metadata_Dose
Metadata_Batch
Metadata_Run
Metadata_Date
Metadata_CellLine
Metadata_Donor
Metadata_Timepoint
Metadata_Replicate
```

## Assay well versus source well

Assay well means the well actually imaged by CellProfiler. This should match the Image table / CellProfiler well field.

Source well means the well in the source/library plate used by a robot, acoustic dispenser or plate handler. It is useful for annotation merging, but it is not the same as the imaged assay well.

Example:

```text
Assay_Plate_Barcode       Destination_Well       Source_Plate_Barcode       Source_Well       Compound
ASSAY_PLATE_01            B03                    LIBRARY_PLATE_07           A1                CMPD_001
```

In this case:

```text
--plate_column Assay_Plate_Barcode
--well_column Destination_Well
--source_plate_column Source_Plate_Barcode
--source_well_column Source_Well
```

## Well formatting

CPATK canonicalises simple wells to padded uppercase format:

```text
A1  -> A01
A01 -> A01
a1  -> A01
H12 -> H12
```

Raw values are preserved in audit columns such as:

```text
Metadata_Well__raw
Metadata_Source_Well__raw
```

This canonicalisation is intended only for well identity matching. It should not be used to infer that a source well is the same as an assay well.

## Basic metadata command

```bash
cpatk-metadata \
  --metadata_table metadata/raw_plate_map.tsv \
  --output_dir results/00_metadata_check \
  --plate_column Assay_Plate_Barcode \
  --well_column Destination_Well \
  --source_plate_column Source_Plate_Barcode \
  --source_well_column Source_Well \
  --duplicate_policy error \
  --log_level INFO
```

## Merging annotations

If compound annotations are keyed by source plate/well:

```bash
cpatk-metadata \
  --metadata_table metadata/raw_plate_map.tsv \
  --output_dir results/00_metadata_check \
  --plate_column Assay_Plate_Barcode \
  --well_column Destination_Well \
  --source_plate_column Source_Plate_Barcode \
  --source_well_column Source_Well \
  --annotation_tables metadata/compound_annotations.tsv \
  --annotation_source_plate_column Barcode \
  --annotation_source_well_column Well \
  --merge_keys Metadata_Source_Plate,Metadata_Source_Well \
  --duplicate_policy error \
  --log_level INFO
```

If annotations are keyed directly by compound ID, use compound ID as the merge key only after checking that compound IDs are unique in the annotation table.

## Duplicate policy

Use `--duplicate_policy error` by default. Duplicates should fail until reviewed.

Permissive policies should be used only after inspecting duplicate-key reports:

```text
error       fail if duplicate keys exist
identical   allow duplicates only if non-key values are identical
first       keep first row; use only after manual review
```

## Minimum QC checks

Before using `formatted_metadata.tsv`, inspect:

```text
metadata_summary.tsv
metadata_column_report.tsv
metadata_key_report.tsv
annotation_merge_report.tsv
formatted_metadata.tsv
metadata.log
```

Check:

- expected number of assay wells;
- no unexpected duplicate assay plate/well pairs;
- no source-well columns being used as assay wells;
- successful annotation merge rate;
- DMSO/control labels are correct;
- compound IDs are not missing for treated wells;
- control wells are not accidentally assigned compound IDs.

## Common failure modes

### Source well used as assay well

Symptom: metadata appears to merge but compounds are assigned to the wrong imaged wells.

Fix: explicitly set `--plate_column` and `--well_column` to the assay/destination columns.

### A1/A01 mismatch

Symptom: many unmatched wells despite apparently matching plate maps.

Fix: use `cpatk-metadata` to canonicalise wells and inspect raw/canonical columns.

### Plate barcode mismatch

Symptom: wells match but plates do not.

Fix: check whitespace, case, hidden characters and whether the assay plate or source plate is being used.

### Duplicate annotations

Symptom: one source well maps to multiple compounds or multiple names.

Fix: fail with `--duplicate_policy error`, inspect the duplicate report, then fix the library annotation table.
