"""Tests for CPATK v0.2.5 merge-first zero-row preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.merging import build_profiles_from_folder
from cpatk.preprocessing import calculate_all_zero_row_report, preprocess_profiles


class CpatkV025Tests(unittest.TestCase):
    """Coverage for all-zero row filtering and merge-before-preprocess order."""

    def test_calculate_all_zero_row_report_flags_observed_zero_rows(self) -> None:
        """Rows with only zero observed feature values should be flagged."""
        features = pd.DataFrame(
            {
                "f1": [0.0, 0.0, 1.0, None],
                "f2": [0.0, None, 0.0, None],
            }
        )
        metadata = pd.DataFrame({"Metadata_Well": ["A01", "A02", "A03", "A04"]})
        report = calculate_all_zero_row_report(features=features, metadata=metadata)
        self.assertEqual(report["all_zero_feature_row"].tolist(), [True, True, False, False])
        self.assertIn("Metadata_Well", report.columns)

    def test_preprocess_removes_all_zero_rows_before_imputation(self) -> None:
        """All-zero rows should be removed before missing-value imputation."""
        table = pd.DataFrame(
            {
                "Metadata_Well": ["A01", "A02", "A03"],
                "Intensity_MeanIntensity_DAPI": [0.0, 1.0, 2.0],
                "Texture_Contrast_DAPI": [0.0, 2.0, None],
                "AreaShape_Area": [0.0, 3.0, 4.0],
            }
        )
        result = preprocess_profiles(data_frame=table, remove_correlated=False)
        self.assertEqual(result["preprocessed"].shape[0], 2)
        self.assertEqual(int(result["all_zero_row_report"]["removed_by_all_zero_row_filter"].sum()), 1)
        retained_wells = result["preprocessed"]["Metadata_Well"].tolist()
        self.assertEqual(retained_wells, ["A02", "A03"])
        self.assertEqual(int(result["imputation_report"]["n_missing_after"].sum()), 0)

    def test_all_zero_filter_can_be_disabled(self) -> None:
        """Users should be able to disable all-zero row removal for unusual assays."""
        table = pd.DataFrame(
            {
                "Metadata_Well": ["A01", "A02", "A03"],
                "Intensity_MeanIntensity_DAPI": [0.0, 1.0, 2.0],
                "Texture_Contrast_DAPI": [0.0, 2.0, 3.0],
            }
        )
        result = preprocess_profiles(data_frame=table, remove_all_zero_rows=False, remove_correlated=False)
        self.assertEqual(result["preprocessed"].shape[0], 3)
        self.assertEqual(int(result["all_zero_row_report"]["all_zero_feature_row"].sum()), 1)
        self.assertEqual(int(result["all_zero_row_report"]["removed_by_all_zero_row_filter"].sum()), 0)

    def test_folder_build_then_preprocess_removes_only_fully_merged_zero_profiles(self) -> None:
        """Zero-row filtering should be based on the merged profile matrix, not one object table alone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = pd.DataFrame(
                {
                    "ImageNumber": [1, 2, 3],
                    "Row_Metadata": ["A", "A", "A"],
                    "Column_Metadata": [1, 2, 3],
                }
            )
            # Image 1 has zero Cell and zero Nuclei features and should be removed.
            # Image 2 has zero Cell but non-zero Nuclei and must be retained.
            # Image 3 has non-zero Cell and zero Nuclei and must be retained.
            cell = pd.DataFrame(
                {
                    "ImageNumber": [1, 1, 2, 2, 3, 3],
                    "ObjectNumber": [1, 2, 1, 2, 1, 2],
                    "Intensity_MeanIntensity_DAPI": [0, 0, 0, 0, 5, 7],
                }
            )
            nuclei = pd.DataFrame(
                {
                    "ImageNumber": [1, 1, 2, 2, 3, 3],
                    "ObjectNumber": [1, 2, 1, 2, 1, 2],
                    "Texture_Contrast_DAPI": [0, 0, 4, 6, 0, 0],
                }
            )
            metadata = pd.DataFrame(
                {
                    "Well_Metadata": ["A01", "A02", "A03"],
                    "Compound": ["zero", "nuclei_signal", "cell_signal"],
                }
            )
            image.to_csv(root / "Image.csv", index=False)
            cell.to_csv(root / "Cell.csv.gz", index=False)
            nuclei.to_csv(root / "Nuclei.tsv", sep="\t", index=False)
            metadata.to_csv(root / "plate_map.csv", index=False)

            build = build_profiles_from_folder(input_dir=root, metadata_table="plate_map.csv")
            self.assertEqual(build.profiles.shape[0], 3)
            self.assertIn("Cell__Intensity_MeanIntensity_DAPI", build.profiles.columns)
            self.assertIn("Nuclei__Texture_Contrast_DAPI", build.profiles.columns)

            result = preprocess_profiles(data_frame=build.profiles, remove_correlated=False)
            self.assertEqual(result["preprocessed"].shape[0], 2)
            self.assertEqual(result["preprocessed"]["Metadata_Well"].tolist(), ["A02", "A03"])
            self.assertEqual(int(result["all_zero_row_report"]["removed_by_all_zero_row_filter"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
