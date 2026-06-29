# CPATK v0.2.25 combined-profile key hardening

## Why this update was needed

A multi-dataset Cell Painting stress test failed after all per-dataset profile
builds had completed. The failure occurred in `cpatk-combine-profiles`, not in
object aggregation or metadata merging.

The failed command used this combined key:

```bash
Metadata_Profile_Source,Metadata_Plate,Metadata_Well
```

That key is too coarse for image-level CellProfiler profiles because a single
plate/well usually has multiple image rows. The correct combined key must
include image identity, normally `ImageNumber`, together with source and plate
provenance.

## Code changes

- The default `cpatk-combine-profiles` key now prefers source, plate and image
  identity where available.
- `ImageNumber` and other likely image/site identity columns are preserved as
  metadata during profile combining, even when the user supplies a coarser key.
- The combiner now writes `combined_key_candidate_report.tsv` on successful
  runs.
- Duplicate-key errors now include candidate unique key sets where CPATK can
  infer them.
- Documentation and example multi-plate commands now avoid well-only profile
  keys.

## Recommended image-level combined key

```bash
--key_columns Metadata_Profile_Source,Metadata_Plate,ImageNumber,Metadata_Well
```

`Metadata_Well` is retained because it is useful metadata, but it should not be
the only within-plate profile identity. For these datasets, `ImageNumber` comes
from the CellProfiler Image table and identifies the image-level profile row
within a plate/export.

## Interpretation

This update does not change object aggregation. The uploaded run logs showed
that object tables were already being safely stamped with `Metadata_Plate` from
the Image table using a unique `ImageNumber` mapping before aggregation. The
problem was the later stacking of completed profile tables across datasets.
