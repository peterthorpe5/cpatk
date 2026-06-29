"""Regression tests for CPATK v0.2.24 optional object trimming."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.merging import build_profiles_from_folder, trim_object_table_by_robust_distance


class TestObjectTrimming(unittest.TestCase):
    """Tests for optional robust object trimming before aggregation."""

    def test_trim_object_table_removes_extreme_object(self) -> None:
        """Robust-distance trimming should remove the most extreme object row."""
        table = pd.DataFrame(
            {
                "Metadata_Plate": ["P1"] * 5,
                "ImageNumber": [1] * 5,
                "ObjectNumber": [1, 2, 3, 4, 5],
                "AreaShape_Area": [1.0, 1.1, 1.2, 1.3, 50.0],
                "Intensity_MeanIntensity_DNA": [2.0, 2.1, 2.2, 2.3, 80.0],
            }
        )
        trimmed, summary, by_group = trim_object_table_by_robust_distance(
            data_frame=table,
            table_label="Cells",
            group_keys=["Metadata_Plate", "ImageNumber"],
            feature_columns=["AreaShape_Area", "Intensity_MeanIntensity_DNA"],
            keep_central_fraction=0.8,
        )
        self.assertEqual(trimmed.shape[0], 4)
        self.assertEqual(int(summary.loc[0, "n_objects_removed"]), 1)
        self.assertEqual(int(by_group.loc[0, "n_objects_removed"]), 1)
        self.assertNotIn(5, trimmed["ObjectNumber"].tolist())

    def test_build_profiles_reports_object_trimming(self) -> None:
        """Profile building should expose trimming summary tables when enabled."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "ImageNumber": [1],
                    "Metadata_Plate": ["P1"],
                    "Metadata_Well": ["A01"],
                }
            ).to_csv(root / "MyExpt_Image.csv", index=False)
            pd.DataFrame(
                {
                    "ImageNumber": [1, 1, 1, 1, 1],
                    "ObjectNumber": [1, 2, 3, 4, 5],
                    "AreaShape_Area": [1.0, 1.1, 1.2, 1.3, 50.0],
                    "Intensity_MeanIntensity_DNA": [2.0, 2.1, 2.2, 2.3, 80.0],
                }
            ).to_csv(root / "MyExpt_Cells.csv", index=False)
            result = build_profiles_from_folder(
                input_dir=root,
                output_dir=root / "out",
                image_merge_keys="Metadata_Plate,ImageNumber",
                trim_objects=True,
                trim_keep_central_fraction=0.8,
            )
        summary = result.tables["object_trimming_summary"]
        aggregation = result.tables["object_aggregation_report"]
        self.assertEqual(summary.loc[0, "status"], "enabled")
        self.assertEqual(int(summary.loc[0, "n_objects_removed"]), 1)
        self.assertEqual(int(aggregation.loc[0, "n_object_rows_removed_by_trimming"]), 1)
        self.assertIn("object_trimming_by_group", result.tables)


if __name__ == "__main__":
    unittest.main()
