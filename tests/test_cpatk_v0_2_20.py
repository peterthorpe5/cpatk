"""Regression tests for CPATK v0.2.20 profile-building hardening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.merging import build_profiles_from_folder


class TestObjectImageKeyPropagation(unittest.TestCase):
    """Test propagation of image-level assay keys onto object tables."""

    def test_requested_plate_image_key_is_propagated_to_object_table(self) -> None:
        """Object tables without plate columns should use the Image table mapping when safe."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = pd.DataFrame(
                {
                    "Metadata_Plate": ["P1", "P2"],
                    "Metadata_Well": ["A01", "A01"],
                    "ImageNumber": [1, 2],
                    "Image_Intensity_Mean": [10.0, 20.0],
                }
            )
            cells = pd.DataFrame(
                {
                    "ImageNumber": [1, 1, 2, 2],
                    "ObjectNumber": [1, 2, 1, 2],
                    "AreaShape_Area": [1.0, 3.0, 10.0, 14.0],
                }
            )
            image.to_csv(tmp_path / "Image.csv", index=False)
            cells.to_csv(tmp_path / "Cells.csv", index=False)

            result = build_profiles_from_folder(
                input_dir=tmp_path,
                image_merge_keys="Metadata_Plate,ImageNumber",
            )
            profiles = result.profiles.sort_values("Metadata_Plate").reset_index(drop=True)
            self.assertEqual(profiles.shape[0], 2)
            self.assertIn("Cells__AreaShape_Area", profiles.columns)
            self.assertEqual(profiles.loc[0, "Cells__AreaShape_Area"], 2.0)
            self.assertEqual(profiles.loc[1, "Cells__AreaShape_Area"], 12.0)
            propagation = result.tables["object_key_propagation_report"]
            self.assertEqual(propagation.loc[0, "status"], "propagated")
            self.assertEqual(propagation.loc[0, "propagated_keys"], "Metadata_Plate")
            summary = dict(
                zip(
                    result.tables["profile_build_summary"]["item"],
                    result.tables["profile_build_summary"]["value"],
                )
            )
            self.assertEqual(int(summary["n_object_tables_with_image_key_propagation"]), 1)

    def test_ambiguous_image_number_mapping_fails_for_explicit_composite_key(self) -> None:
        """A repeated ImageNumber with multiple plates must not be guessed for object rows."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = pd.DataFrame(
                {
                    "Metadata_Plate": ["P1", "P2"],
                    "Metadata_Well": ["A01", "A01"],
                    "ImageNumber": [1, 1],
                    "Image_Intensity_Mean": [10.0, 20.0],
                }
            )
            cells = pd.DataFrame(
                {
                    "ImageNumber": [1, 1],
                    "ObjectNumber": [1, 2],
                    "AreaShape_Area": [1.0, 3.0],
                }
            )
            image.to_csv(tmp_path / "Image.csv", index=False)
            cells.to_csv(tmp_path / "Cells.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Cannot safely propagate"):
                build_profiles_from_folder(
                    input_dir=tmp_path,
                    image_merge_keys="Metadata_Plate,ImageNumber",
                )


if __name__ == "__main__":
    unittest.main()
