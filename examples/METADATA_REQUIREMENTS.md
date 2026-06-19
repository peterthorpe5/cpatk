# CPATK metadata file requirements

A CPATK metadata file is usually a plate-map or annotation table with one row per assayed well. It can be TSV, CSV, TSV.GZ, CSV.GZ or Excel. TSV is preferred.

## Essential columns

The safest minimal metadata file contains these columns:

- `Metadata_Plate`: plate identifier. Essential when analysing more than one plate; strongly recommended even for one plate.
- `Metadata_Well`: well identifier, for example `A01`, `A1`, `P24`.
- `Metadata_Compound`: treatment or compound identifier. This can be a named compound, control label, siRNA, perturbation, CRISPR guide, etc.
- `cpd_type`: broad sample type, for example `control`, `positive_control`, `test`, `reference`, `known_toxic`, `unknown`.

## Strongly recommended columns

- `Metadata_MOA`: known or expected mode of action. Use `unknown` when not known. This is needed for supervised MOA classification.
- `Metadata_Dose`: dose or concentration.
- `Metadata_Batch`: batch, run, staining batch, imaging batch, site, or processing batch.
- `Replicate`: biological or technical replicate identifier.
- `Donor`, `CellLine`, `Timepoint`, `TreatmentDuration`: include when relevant.

## Important notes

CPATK has alias handling for messy files, so it can often recognise columns such as `Plate_Metadata`, `Well_Metadata`, `Source_Plate_Barcode`, `Source_Well`, `compound`, `Compound`, `COMPOUND_NAME`, `dose`, `concentration`, `moa`, and `Library`.

However, for new projects, the cleanest convention is to use the `Metadata_*` names above. Keep metadata as metadata. Do not encode metadata fields as numeric features unless they are intended to be analysed as biological measurements.

If several metadata rows match the same plate/well, CPATK keeps the first match and writes merge/audit reports. It is better to resolve duplicates before analysis.
