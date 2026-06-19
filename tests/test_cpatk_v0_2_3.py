"""Unit tests for CPATK v0.2.3 folder profile building."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.io import read_table
from cpatk.merging import (
    build_profiles_from_folder,
    infer_table_role,
    inspect_folder_tables,
)
from cpatk.preprocessing import preprocess_profiles


class CpatkV023ProfileBuildTests(unittest.TestCase):
    """Test folder-level Cell Painting file merging and preprocessing."""

    def write_test_folder(self, base: Path) -> None:
        """Write a small CellProfiler-like folder with mixed input formats."""
        image = pd.DataFrame(
            {
                "ImageNumber": [1, 2, 3, 4],
                "Plate_Metadata": ["P1", "P1", "P1", "P1"],
                "Well_Metadata": ["A1", "A2", "B1", "B2"],
                "Count_Cell": [10, 8, 11, 7],
                "Mean_Cell_Intensity_MeanIntensity_DAPI": [1.0, 2.0, 1.2, 3.0],
                "Mean_Cell_Texture_Contrast_DAPI": [0.1, 0.4, 0.2, 0.8],
            }
        )
        image.to_csv(base / "Example_Image.csv", index=False)

        cells = pd.DataFrame(
            {
                "ImageNumber": [1, 1, 2, 2, 3, 3, 4, 4],
                "ObjectNumber": [1, 2, 1, 2, 1, 2, 1, 2],
                "AreaShape_Area": [10.0, 12.0, 20.0, 22.0, 11.0, None, 30.0, 32.0],
                "Intensity_MeanIntensity_FITC": [0.1, 0.2, 0.5, None, 0.2, 0.3, 0.9, 1.0],
                "ExecutionTime_01LoadData": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        cells.to_csv(base / "Example_Cell.tsv", sep="\t", index=False)

        nuclei = pd.DataFrame(
            {
                "ImageNumber": [1, 1, 2, 2, 3, 3, 4, 4],
                "ObjectNumber": [1, 2, 1, 2, 1, 2, 1, 2],
                "AreaShape_Eccentricity": [0.1, 0.2, 0.4, 0.45, 0.15, 0.2, 0.8, 0.85],
                "Texture_Contrast_DAPI": [1.0, 1.1, 2.0, 2.1, 1.2, 1.3, 3.0, 3.2],
            }
        )
        with gzip.open(base / "Example_Nuclei.csv.gz", "wt", encoding="utf-8") as handle:
            nuclei.to_csv(handle, index=False)

        metadata = pd.DataFrame(
            {
                "Plate_Metadata": ["P1", "P1", "P1", "P1"],
                "Well_Metadata": ["A01", "A02", "B01", "B02"],
                "Compound": ["DMSO", "DrugA", "DMSO", "DrugB"],
                "cpd_type": ["control", "kinase", "control", "tubulin"],
            }
        )
        metadata.to_csv(base / "metadata.tsv", sep="\t", index=False)

    def test_infer_table_roles(self) -> None:
        """Table role inference should identify image, object and metadata tables."""
        image_role, _ = infer_table_role(file_name="Image.csv", columns=["ImageNumber", "Metadata_Well"])
        object_role, _ = infer_table_role(file_name="Cell.csv", columns=["ImageNumber", "ObjectNumber", "AreaShape_Area"])
        meta_role, _ = infer_table_role(file_name="metadata.tsv", columns=["Plate_Metadata", "Well_Metadata", "Compound"])
        self.assertEqual(image_role, "image")
        self.assertEqual(object_role, "object")
        self.assertEqual(meta_role, "metadata")

    def test_inspect_folder_supports_csv_tsv_and_gz(self) -> None:
        """Folder inspection should handle CSV, TSV and gzipped CSV input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self.write_test_folder(base)
            inspection = inspect_folder_tables(input_dir=base)
            self.assertEqual(inspection.shape[0], 4)
            self.assertIn("object", set(inspection["inferred_role"]))
            self.assertIn("metadata", set(inspection["inferred_role"]))
            self.assertIn("image", set(inspection["inferred_role"]))

    def test_build_profiles_from_folder_aggregates_objects_and_merges_metadata(self) -> None:
        """Profile builder should aggregate object tables to ImageNumber and merge metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            out = base / "out"
            self.write_test_folder(base)
            result = build_profiles_from_folder(input_dir=base, output_dir=out)
            profiles = result.profiles
            self.assertEqual(profiles.shape[0], 4)
            self.assertIn("Metadata_Well", profiles.columns)
            self.assertIn("Metadata_Compound", profiles.columns)
            self.assertTrue(any(column.endswith("__AreaShape_Area") for column in profiles.columns))
            self.assertTrue(any(column.endswith("__Texture_Contrast_DAPI") for column in profiles.columns))
            self.assertTrue((out / "profile_build_report.html").exists())
            self.assertTrue((out / "profile_build_summary.xlsx").exists())
            self.assertIn("object_aggregation_report", result.tables)

    def test_built_profiles_can_be_preprocessed_and_imputed(self) -> None:
        """Merged profiles should flow into preprocessing and impute remaining missing values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self.write_test_folder(base)
            result = build_profiles_from_folder(input_dir=base)
            preprocessed = preprocess_profiles(
                data_frame=result.profiles,
                imputation_method="median",
                remove_correlated=False,
                max_feature_missing_fraction=0.75,
            )
            self.assertIn("imputation_report", preprocessed)
            self.assertEqual(int(preprocessed["preprocessed"].isna().sum().sum()), 0)
            self.assertGreater(preprocessed["preprocessed"].shape[1], 1)

    def test_read_table_handles_gzipped_tsv_and_csv(self) -> None:
        """Generic table reader should support compressed CSV and TSV inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            csv_gz = base / "table.csv.gz"
            tsv_gz = base / "table.tsv.gz"
            frame.to_csv(csv_gz, index=False)
            frame.to_csv(tsv_gz, sep="\t", index=False)
            self.assertEqual(read_table(path=csv_gz).shape, (2, 2))
            self.assertEqual(read_table(path=tsv_gz).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
