"""Tests for CPATK v0.2.12 multi-plate and batch hardening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.merging import build_profiles_from_folder
from cpatk.preprocessing import (
    calculate_reference_control_qc,
    combat_style_location_scale_correction,
    preprocess_profiles,
)
from cpatk.profile_combining import combine_profile_tables


class TestCompositeKeyProfileBuilding(unittest.TestCase):
    """Test native multi-plate profile building with repeated ImageNumber."""

    def test_build_profiles_uses_plate_image_composite_key(self) -> None:
        """Repeated ImageNumber values across plates should merge safely by composite key."""
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
            cell = pd.DataFrame(
                {
                    "Metadata_Plate": ["P1", "P1", "P2", "P2"],
                    "ImageNumber": [1, 1, 1, 1],
                    "ObjectNumber": [1, 2, 1, 2],
                    "AreaShape_Area": [1.0, 3.0, 10.0, 14.0],
                }
            )
            image.to_csv(tmp_path / "Image.csv", index=False)
            cell.to_csv(tmp_path / "Cell.csv", index=False)
            result = build_profiles_from_folder(input_dir=tmp_path)
            profiles = result.profiles.sort_values("Metadata_Plate").reset_index(drop=True)
            self.assertEqual(profiles.shape[0], 2)
            self.assertIn("Cell__AreaShape_Area", profiles.columns)
            self.assertEqual(profiles.loc[0, "Cell__AreaShape_Area"], 2.0)
            self.assertEqual(profiles.loc[1, "Cell__AreaShape_Area"], 12.0)

    def test_repeated_image_number_without_plate_fails(self) -> None:
        """Repeated ImageNumber without a shared plate key should fail rather than mis-merge."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = pd.DataFrame(
                {
                    "Metadata_Well": ["A01", "B01"],
                    "ImageNumber": [1, 1],
                    "Image_Intensity_Mean": [10.0, 20.0],
                }
            )
            cell = pd.DataFrame(
                {
                    "ImageNumber": [1, 1],
                    "ObjectNumber": [1, 2],
                    "AreaShape_Area": [1.0, 3.0],
                }
            )
            image.to_csv(tmp_path / "Image.csv", index=False)
            cell.to_csv(tmp_path / "Cell.csv", index=False)
            with self.assertRaisesRegex(ValueError, "ImageNumber is duplicated"):
                build_profiles_from_folder(input_dir=tmp_path)


class TestCombineProfiles(unittest.TestCase):
    """Test profile-table combining."""

    def test_combine_profiles_union_adds_source_and_keeps_features(self) -> None:
        """Combining two plates should add source labels and preserve feature union."""
        first = pd.DataFrame(
            {
                "Metadata_Plate": ["P1"],
                "Metadata_Well": ["A01"],
                "AreaShape_Area": [1.0],
            }
        )
        second = pd.DataFrame(
            {
                "Metadata_Plate": ["P2"],
                "Metadata_Well": ["A01"],
                "Texture_Info": [2.0],
            }
        )
        combined, reports = combine_profile_tables(
            profile_tables=[first, second],
            source_labels=["plate1", "plate2"],
            key_columns=["Metadata_Plate", "Metadata_Well"],
            feature_join="union",
        )
        self.assertEqual(combined.shape[0], 2)
        self.assertIn("Metadata_Profile_Source", combined.columns)
        self.assertIn("AreaShape_Area", combined.columns)
        self.assertIn("Texture_Info", combined.columns)
        self.assertIn("combine_profile_summary", reports)

    def test_combine_profiles_duplicate_keys_fail(self) -> None:
        """Duplicate combined profile keys should fail by default."""
        first = pd.DataFrame({"Metadata_Plate": ["P1"], "Metadata_Well": ["A01"], "AreaShape_Area": [1.0]})
        second = pd.DataFrame({"Metadata_Plate": ["P1"], "Metadata_Well": ["A01"], "AreaShape_Area": [2.0]})
        with self.assertRaisesRegex(ValueError, "not unique"):
            combine_profile_tables(
                profile_tables=[first, second],
                key_columns=["Metadata_Plate", "Metadata_Well"],
            )


class TestControlQcAndBatchCorrection(unittest.TestCase):
    """Test control QC and ComBat-style correction reports."""

    def test_reference_control_qc_reports_zero_mad_features(self) -> None:
        """Control QC should report weak DMSO/reference features before normalisation."""
        features = pd.DataFrame({"AreaShape_Area": [1.0, 1.0, 2.0], "Texture_Info": [5.0, 6.0, 7.0]})
        metadata = pd.DataFrame({"Metadata_Plate": ["P1", "P1", "P1"], "Metadata_Compound": ["DMSO", "DMSO", "Drug"]})
        report = calculate_reference_control_qc(
            features=features,
            metadata=metadata,
            reference_column="Metadata_Compound",
            reference_values=["DMSO"],
            group_columns=["Metadata_Plate"],
            method="robust_z",
        )
        self.assertEqual(report.loc[0, "n_reference_profiles"], 2)
        self.assertGreaterEqual(report.loc[0, "n_features_zero_or_near_zero_mad"], 1)

    def test_combat_style_location_scale_reduces_batch_mean_shift(self) -> None:
        """Batch correction should reduce a simple feature-wise batch mean shift."""
        features = pd.DataFrame({"AreaShape_Area": [1.0, 2.0, 3.0, 101.0, 102.0, 103.0]})
        metadata = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P1", "P2", "P2", "P2"],
                "Metadata_Compound": ["A", "B", "C", "A", "B", "C"],
            }
        )
        corrected, correction_report, confounding_report = combat_style_location_scale_correction(
            features=features,
            metadata=metadata,
            batch_column="Metadata_Plate",
            protected_columns=["Metadata_Compound"],
            method="combat_location_scale",
            min_batch_size=3,
        )
        before_delta = abs(features.iloc[:3, 0].mean() - features.iloc[3:, 0].mean())
        after_delta = abs(corrected.iloc[:3, 0].mean() - corrected.iloc[3:, 0].mean())
        self.assertLess(after_delta, before_delta)
        self.assertFalse(correction_report.empty)
        self.assertFalse(confounding_report.empty)

    def test_preprocess_writes_before_after_qc_tables(self) -> None:
        """Preprocessing should return control, batch and replicate QC audit tables."""
        data = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P1", "P2", "P2", "P2"],
                "Metadata_Well": ["A01", "A02", "A03", "A01", "A02", "A03"],
                "Metadata_Compound": ["DMSO", "DrugA", "DrugA", "DMSO", "DrugA", "DrugA"],
                "Metadata_Dose": [0, 1, 1, 0, 1, 1],
                "AreaShape_Area": [1.0, 2.0, 2.2, 11.0, 12.0, 12.2],
                "Texture_Info": [2.0, 3.0, 3.1, 22.0, 23.0, 23.1],
            }
        )
        result = preprocess_profiles(
            data_frame=data,
            reference_normalisation_method="robust_z",
            reference_column="Metadata_Compound",
            reference_values=["DMSO"],
            reference_group_columns=["Metadata_Plate"],
            batch_correction_method="combat_location_scale",
            batch_column="Metadata_Plate",
            batch_protect_columns=["Metadata_Compound"],
            batch_correction_min_batch_size=3,
            replicate_group_columns=["Metadata_Compound", "Metadata_Dose"],
            batch_report_columns=["Metadata_Plate"],
            max_feature_missing_fraction=1.0,
            max_sample_missing_fraction=1.0,
            max_absolute_correlation=1.0,
            scaling_method="none",
        )
        self.assertIn("reference_control_qc_before_normalisation", result)
        self.assertIn("batch_correction_report", result)
        self.assertIn("before_after_replicate_summary", result)
        self.assertIn("before_after_batch_pc_association", result)


if __name__ == "__main__":
    unittest.main()
