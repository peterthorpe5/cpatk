"""Tests for CPATK v0.2.10 release hardening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.merging import build_profiles_from_folder
from cpatk.metadata_validation import (
    choose_merge_keys,
    merge_annotation_tables,
    prepare_metadata_table,
    run_metadata_validation_workflow,
)
from cpatk.preprocessing import impute_features, preprocess_profiles
from cpatk.reporting import make_html_report
from cpatk.visualisation import run_visualisation_workflow


class TestReleaseBlockersV0210(unittest.TestCase):
    """Regression tests for release-hardening fixes."""

    def test_html_report_links_copied_html_assets_under_report_assets(self) -> None:
        """Interactive HTML assets copied into report_assets should have valid links."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plot_path = root / "interactive_plot.html"
            plot_path.write_text("<html><body>plot</body></html>", encoding="utf-8")
            report_path = root / "report.html"
            make_html_report(
                title="test",
                output_path=report_path,
                plot_paths=[plot_path],
                summary_tables={},
            )
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("report_assets/interactive_plot.html", text)
            self.assertTrue((root / "report_assets" / "interactive_plot.html").exists())

    def test_visualisation_workflow_returns_output_manifest(self) -> None:
        """Visualisation workflow should return paths it wrote."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.normal(size=(8, 4)), columns=["0", "1", "2", "3"])
        df.insert(0, "Metadata_Compound", [f"C{i}" for i in range(8)])
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "profiles.tsv"
            out_dir = Path(tmp) / "visualisation"
            df.to_csv(input_path, sep="\t", index=False)
            outputs = run_visualisation_workflow(
                input_table=input_path,
                output_dir=out_dir,
                id_column="Metadata_Compound",
                digit_named_latents=True,
                make_umap=False,
                make_heatmap=False,
                make_topology=False,
                interactive=False,
            )
            self.assertIn("visualisation_feature_columns", outputs)
            self.assertIn("pca_coordinates", outputs)
            self.assertTrue(outputs["pca_coordinates"].exists())

    def test_duplicate_image_rows_fail_by_default(self) -> None:
        """Duplicate ImageNumber rows should no longer be silently kept first."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = pd.DataFrame(
                {
                    "ImageNumber": [1, 1],
                    "Metadata_Plate": ["P1", "P1"],
                    "Metadata_Well": ["A01", "A01"],
                }
            )
            cell = pd.DataFrame(
                {
                    "ImageNumber": [1, 1],
                    "ObjectNumber": [1, 2],
                    "Intensity_MeanIntensity_DNA": [1.0, 2.0],
                }
            )
            image.to_csv(root / "Image.csv", index=False)
            cell.to_csv(root / "Cell.csv", index=False)
            with self.assertRaises(ValueError):
                build_profiles_from_folder(input_dir=root)

    def test_duplicate_metadata_keys_fail_by_default(self) -> None:
        """Duplicate external metadata keys should fail unless a policy allows them."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = pd.DataFrame(
                {
                    "ImageNumber": [1, 2],
                    "Metadata_Plate": ["P1", "P1"],
                    "Metadata_Well": ["A01", "A02"],
                }
            )
            cell = pd.DataFrame(
                {
                    "ImageNumber": [1, 2],
                    "ObjectNumber": [1, 1],
                    "Intensity_MeanIntensity_DNA": [1.0, 2.0],
                }
            )
            metadata = pd.DataFrame(
                {
                    "Metadata_Plate": ["P1", "P1", "P1"],
                    "Metadata_Well": ["A01", "A01", "A02"],
                    "Compound": ["A", "conflict", "B"],
                }
            )
            image.to_csv(root / "Image.csv", index=False)
            cell.to_csv(root / "Cell.csv", index=False)
            metadata.to_csv(root / "metadata.tsv", sep="\t", index=False)
            with self.assertRaises(ValueError):
                build_profiles_from_folder(input_dir=root, metadata_table="metadata.tsv")

    def test_knn_imputation_caps_neighbours_for_small_tables(self) -> None:
        """KNN imputation should cap n_neighbors to the available sample count."""
        features = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
        imputed = impute_features(features=features, method="knn", n_neighbors=10)
        self.assertEqual(imputed.shape, features.shape)
        self.assertFalse(imputed.isna().any().any())

    def test_reference_normalisation_uses_pre_imputation_controls(self) -> None:
        """Reference/control statistics should be calculated before imputation."""
        table = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P1"],
                "Metadata_Compound": ["DMSO", "DMSO", "Drug"],
                "Intensity_MeanIntensity_DNA": [0.0, np.nan, 10.0],
                "Texture_Contrast_DNA": [1.0, 1.0, 2.0],
            }
        )
        result = preprocess_profiles(
            data_frame=table,
            max_feature_missing_fraction=1.0,
            max_sample_missing_fraction=1.0,
            remove_all_zero_rows=False,
            remove_correlated=False,
            reference_normalisation_method="robust_z",
            reference_column="Metadata_Compound",
            reference_values=["DMSO"],
            reference_group_columns=["Metadata_Plate"],
            scaling_method="none",
        )
        report = result["reference_normalisation_report"]
        row = report.loc[report["feature"] == "Intensity_MeanIntensity_DNA"].iloc[0]
        self.assertAlmostEqual(float(row["centre"]), 0.0)
        self.assertIn("final_matrix_validation", result)
        self.assertEqual(result["final_matrix_validation"].loc[0, "status"], "ok")


class TestMetadataValidationV0210(unittest.TestCase):
    """Tests for step-one metadata validation and annotation merging."""

    def test_prepare_metadata_canonicalises_wells_and_legacy_aliases(self) -> None:
        """A1-style wells should become A01 and legacy columns should be added."""
        raw = pd.DataFrame(
            {
                "Plate": [" P1 "],
                "Well": ["a1"],
                "COMPOUND_NAME": ["DrugA"],
                "cpd_type": ["compound"],
            }
        )
        formatted, reports = prepare_metadata_table(data_frame=raw)
        self.assertEqual(formatted.loc[0, "Metadata_Well"], "A01")
        self.assertEqual(formatted.loc[0, "Well_Metadata"], "A01")
        self.assertEqual(formatted.loc[0, "cpd_id"], "DrugA")
        self.assertIn("metadata_alias_report", reports)

    def test_annotation_merge_uses_source_plate_and_source_well(self) -> None:
        """Messy source plate/well annotations should merge after canonicalisation."""
        metadata_raw = pd.DataFrame(
            {
                "Plate": ["Assay1"],
                "Well": ["B1"],
                "Source_Plate_Barcode": ["SRC1"],
                "Source_well": ["A1"],
                "cpd_id": ["OldName"],
            }
        )
        annotation_raw = pd.DataFrame(
            {
                "Barcode": ["SRC1"],
                "Well": ["A01"],
                "Target": ["Kinase"],
                "Pathway": ["Signalling"],
            }
        )
        metadata, _ = prepare_metadata_table(data_frame=metadata_raw)
        annotation, _ = prepare_metadata_table(
            data_frame=annotation_raw,
            source_plate_column="Barcode",
            source_well_column="Well",
        )
        keys = choose_merge_keys(left=metadata, right=annotation)
        self.assertEqual(keys, ["Metadata_Source_Plate", "Metadata_Source_Well"])
        merged, merge_report, dup_report = merge_annotation_tables(
            metadata=metadata,
            annotations=[annotation],
            annotation_labels=["annot"],
            duplicate_policy="error",
        )
        self.assertEqual(merged.loc[0, "Target"], "Kinase")
        self.assertEqual(int(merge_report.loc[0, "n_rows_without_annotation_match"]), 0)
        self.assertFalse(dup_report.empty)

    def test_metadata_validation_workflow_writes_formatted_metadata(self) -> None:
        """The step-one workflow should write formatted_metadata.tsv and reports."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = pd.DataFrame(
                {
                    "Plate": ["P1", "P1"],
                    "Well": ["A1", "A02"],
                    "cpd_id": ["DMSO", "DrugA"],
                    "cpd_type": ["control", "compound"],
                }
            )
            metadata_path = root / "metadata.csv"
            metadata.to_csv(metadata_path, index=False)
            out_dir = root / "out"
            result = run_metadata_validation_workflow(metadata_table=metadata_path, output_dir=out_dir)
            self.assertTrue((out_dir / "formatted_metadata.tsv").exists())
            self.assertTrue((out_dir / "metadata_validation_report.html").exists())
            self.assertEqual(result["formatted_metadata"].loc[0, "Metadata_Well"], "A01")


if __name__ == "__main__":
    unittest.main()
