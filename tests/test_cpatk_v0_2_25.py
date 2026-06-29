"""Regression tests for CPATK v0.2.25 combined-profile keys."""

from __future__ import annotations

import unittest

import pandas as pd

from cpatk.profile_combining import combine_profile_tables, make_key_candidate_report


class TestCombinedProfileImageLevelKeys(unittest.TestCase):
    """Test image-level key handling during profile-table combining."""

    def _make_image_level_table(self, *, plate: str, feature_offset: float) -> pd.DataFrame:
        """Make a small image-level profile table with repeated wells."""
        return pd.DataFrame(
            {
                "Metadata_Plate": [plate, plate],
                "Metadata_Well": ["A01", "A01"],
                "ImageNumber": [1, 2],
                "Cells__AreaShape_Area": [1.0 + feature_offset, 2.0 + feature_offset],
            }
        )

    def test_default_combined_key_uses_image_identity(self) -> None:
        """Default combining should not treat repeated image rows per well as duplicates."""
        first = self._make_image_level_table(plate="P1", feature_offset=0.0)
        second = self._make_image_level_table(plate="P1", feature_offset=10.0)

        combined, reports = combine_profile_tables(
            profile_tables=[first, second],
            source_labels=["export_a", "export_b"],
            feature_join="union",
        )

        summary = dict(
            zip(
                reports["combine_profile_summary"]["item"],
                reports["combine_profile_summary"]["value"],
            )
        )
        self.assertEqual(combined.shape[0], 4)
        self.assertIn("Metadata_Profile_Source", combined.columns)
        self.assertIn("ImageNumber", str(summary["key_columns"]))
        self.assertIn("Metadata_Profile_Source", str(summary["key_columns"]))
        duplicate_report = reports["combined_duplicate_key_report"]
        self.assertEqual(duplicate_report.loc[0, "status"], "ok_unique_keys")
        self.assertIn("combined_key_candidate_report", reports)

    def test_explicit_image_level_combined_key_passes(self) -> None:
        """A source, plate and image key should uniquely identify image-level rows."""
        first = self._make_image_level_table(plate="P1", feature_offset=0.0)
        second = self._make_image_level_table(plate="P1", feature_offset=10.0)

        combined, reports = combine_profile_tables(
            profile_tables=[first, second],
            source_labels=["export_a", "export_b"],
            key_columns=["Metadata_Profile_Source", "Metadata_Plate", "ImageNumber"],
            feature_join="union",
        )

        self.assertEqual(combined.shape[0], 4)
        duplicate_report = reports["combined_duplicate_key_report"]
        self.assertEqual(duplicate_report.loc[0, "status"], "ok_unique_keys")

    def test_coarse_well_level_key_fails_with_image_number_suggestion(self) -> None:
        """Well-level keys should fail helpfully for image-level profiles."""
        first = self._make_image_level_table(plate="P1", feature_offset=0.0)
        second = self._make_image_level_table(plate="P1", feature_offset=10.0)

        with self.assertRaisesRegex(ValueError, "Candidate unique key.*ImageNumber"):
            combine_profile_tables(
                profile_tables=[first, second],
                source_labels=["export_a", "export_b"],
                key_columns=[
                    "Metadata_Profile_Source",
                    "Metadata_Plate",
                    "Metadata_Well",
                ],
                feature_join="union",
            )

    def test_key_candidate_report_marks_image_number_extension_unique(self) -> None:
        """The key candidate report should expose a usable image-level key."""
        data_frame = pd.DataFrame(
            {
                "Metadata_Profile_Source": ["export_a", "export_a"],
                "Metadata_Plate": ["P1", "P1"],
                "Metadata_Well": ["A01", "A01"],
                "ImageNumber": [1, 2],
                "Cells__AreaShape_Area": [1.0, 2.0],
            }
        )

        report = make_key_candidate_report(
            data_frame=data_frame,
            key_columns=[
                "Metadata_Profile_Source",
                "Metadata_Plate",
                "Metadata_Well",
            ],
        )
        unique_keys = report.loc[report["is_unique"], "key_columns"].tolist()
        self.assertTrue(any("ImageNumber" in key for key in unique_keys))


if __name__ == "__main__":
    unittest.main()
