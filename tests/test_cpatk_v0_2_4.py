"""Tests for CPATK v0.2.4 refinements."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.io import read_table
from cpatk.merging import build_profiles_from_folder
from cpatk.metadata import derive_well_from_row_column, standardise_metadata_aliases
from cpatk.moa import calculate_moa_separability, summarise_prediction_confidence
from cpatk.preprocessing import preprocess_profiles, replace_nonfinite_with_nan
from cpatk.qc import calculate_feature_qc, select_features_by_qc
from cpatk.reproducibility import (
    evaluate_kmeans_k_range,
    permutation_test_cluster_structure_detailed,
)


class CpatkV024Tests(unittest.TestCase):
    """Coverage for v0.2.4 preprocessing, merging and stability changes."""

    def setUp(self) -> None:
        """Create a small deterministic feature table."""
        rng = np.random.default_rng(42)
        cluster_a = rng.normal(loc=0.0, scale=0.2, size=(6, 4))
        cluster_b = rng.normal(loc=3.0, scale=0.2, size=(6, 4))
        self.features = pd.DataFrame(
            np.vstack([cluster_a, cluster_b]),
            columns=[f"Intensity_MeanIntensity_DAPI_{idx}" for idx in range(4)],
        )
        self.labels = pd.Series(["A"] * 6 + ["B"] * 6)

    def test_well_derivation_from_row_and_column(self) -> None:
        """Metadata_Well should be derived from row/column metadata."""
        table = pd.DataFrame({"Row_Metadata": ["a", "B"], "Column_Metadata": [1, "03"]})
        standardised, report = standardise_metadata_aliases(data_frame=table)
        self.assertIn("Metadata_Well", standardised.columns)
        self.assertEqual(standardised["Metadata_Well"].tolist(), ["A01", "B03"])
        self.assertIn("Metadata_Row", standardised.columns)
        self.assertGreaterEqual(report.shape[0], 1)

    def test_derive_well_preserves_existing_values(self) -> None:
        """Existing well values should not be overwritten by row/column derivation."""
        table = pd.DataFrame({"Metadata_Row": ["A"], "Metadata_Column": [2], "Metadata_Well": ["H12"]})
        derived, report = derive_well_from_row_column(data_frame=table)
        self.assertEqual(derived.loc[0, "Metadata_Well"], "H12")
        self.assertIn("n_values_created", report.columns)

    def test_replace_nonfinite_with_nan(self) -> None:
        """Infinite values should be converted to missing values before QC."""
        table = pd.DataFrame({"f1": [1.0, np.inf, 2.0], "f2": [0.0, -np.inf, 1.0]})
        cleaned, report = replace_nonfinite_with_nan(features=table)
        self.assertTrue(cleaned.isna().any().any())
        self.assertEqual(int(report["n_nonfinite_replaced"].sum()), 2)

    def test_zero_fraction_filter(self) -> None:
        """High-zero features should be removable when requested."""
        table = pd.DataFrame({"mostly_zero": [0, 0, 0, 1], "variable": [1, 2, 3, 4]})
        qc = calculate_feature_qc(features=table)
        selected, qc = select_features_by_qc(feature_qc=qc, max_zero_fraction=0.5)
        self.assertNotIn("mostly_zero", selected)
        self.assertIn("variable", selected)
        self.assertIn("pass_zero_fraction", qc.columns)

    def test_preprocess_reports_nonfinite_and_zero_filter(self) -> None:
        """Preprocessing should report non-finite values and honour zero filtering."""
        table = pd.DataFrame(
            {
                "Metadata_Well": ["A01", "A02", "A03", "A04"],
                "Intensity_MeanIntensity_DAPI": [1.0, 2.0, np.inf, 4.0],
                "Texture_Contrast_DAPI": [0.0, 0.0, 0.0, 1.0],
                "AreaShape_Area": [10.0, 11.0, 12.0, 13.0],
            }
        )
        result = preprocess_profiles(data_frame=table, max_zero_fraction=0.7, remove_correlated=False)
        self.assertIn("nonfinite_value_report", result)
        self.assertEqual(int(result["nonfinite_value_report"]["n_nonfinite_replaced"].sum()), 1)
        self.assertNotIn("Texture_Contrast_DAPI", result["retained_features"]["feature"].tolist())

    def test_folder_build_merges_multiple_metadata_tables(self) -> None:
        """Profile building should support more than one external metadata table."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = pd.DataFrame({"ImageNumber": [1, 2], "Row_Metadata": ["A", "A"], "Column_Metadata": [1, 2]})
            cell = pd.DataFrame(
                {
                    "ImageNumber": [1, 1, 2, 2],
                    "ObjectNumber": [1, 2, 1, 2],
                    "Intensity_MeanIntensity_DAPI": [1.0, 1.2, 2.0, 2.2],
                }
            )
            meta1 = pd.DataFrame({"Well_Metadata": ["A01", "A02"], "Compound": ["DMSO", "Drug"]})
            meta2 = pd.DataFrame({"Well_Metadata": ["A01", "A02"], "cpd_type": ["control", "test"]})
            image.to_csv(root / "Image.csv", index=False)
            cell.to_csv(root / "Cell.csv.gz", index=False)
            meta1.to_csv(root / "metadata1.tsv", sep="\t", index=False)
            meta2.to_csv(root / "metadata2.csv", index=False)
            result = build_profiles_from_folder(
                input_dir=root,
                metadata_table="metadata1.tsv,metadata2.csv",
            )
            self.assertIn("Metadata_Compound", result.profiles.columns)
            self.assertIn("Metadata_MOA", result.profiles.columns)
            self.assertIn("Cell__Intensity_MeanIntensity_DAPI", result.profiles.columns)
            self.assertEqual(result.tables["metadata_merge_report"].shape[0], 2)

    def test_read_table_uses_utf8_sig_for_csv(self) -> None:
        """CSV input with BOM should be read cleanly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.csv"
            path.write_text("\ufeffMetadata_Well,Intensity_MeanIntensity_DAPI\nA01,1\n", encoding="utf-8")
            table = read_table(path=path)
            self.assertEqual(table.columns[0], "Metadata_Well")

    def test_detailed_cluster_permutation_returns_null_table(self) -> None:
        """Detailed permutation testing should expose the null distribution."""
        summary, null_table = permutation_test_cluster_structure_detailed(
            features=self.features,
            n_clusters=2,
            n_permutations=5,
        )
        self.assertIn("empirical_p_value", summary.columns)
        self.assertEqual(null_table.shape[0], 5)

    def test_k_range_evaluation(self) -> None:
        """K-range evaluation should combine silhouette, permutation and bootstrap diagnostics."""
        table = evaluate_kmeans_k_range(
            features=self.features,
            k_values=[2, 3],
            n_bootstraps=3,
            n_permutations=3,
        )
        self.assertEqual(set(table["n_clusters"]), {2, 3})
        self.assertIn("mean_bootstrap_ari", table.columns)

    def test_moa_separability(self) -> None:
        """MOA separability should compare within- and between-class distances."""
        summary, null_table = calculate_moa_separability(
            features=self.features,
            labels=self.labels,
            n_permutations=5,
        )
        self.assertIn("observed_between_minus_within", summary.columns)
        self.assertEqual(null_table.shape[0], 5)

    def test_prediction_confidence_summary(self) -> None:
        """Prediction confidence summaries should work for probability and margin tables."""
        predictions = pd.DataFrame({"max_probability": [0.6, 0.9], "top1_similarity_margin": [0.1, 0.4]})
        summary = summarise_prediction_confidence(predictions=predictions)
        self.assertEqual(set(summary["confidence_metric"]), {"max_probability", "top1_similarity_margin"})


if __name__ == "__main__":
    unittest.main()
