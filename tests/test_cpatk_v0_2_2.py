"""Unit tests for CPATK v0.2.2 robustness upgrades."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.explainability import _shap_values_to_tables
from cpatk.features import assign_column_roles, infer_feature_columns, infer_metadata_columns
from cpatk.moa import (
    calculate_class_centroids,
    classify_by_knn,
    leave_one_out_centroid_validation,
    score_profiles_against_centroids,
)
from cpatk.preprocessing import (
    batch_center_features,
    normalise_features_to_reference,
    preprocess_profiles,
    winsorise_features,
)
from cpatk.reporting import make_html_report


class CpatkV022Tests(unittest.TestCase):
    """Test defensive preprocessing, MOA and SHAP helper improvements."""

    def make_table(self) -> pd.DataFrame:
        """Return a synthetic Cell Painting-like profile table."""
        return pd.DataFrame(
            {
                "Plate_Metadata": ["P1", "P1", "P1", "P1", "P2", "P2"],
                "Well_Metadata": ["A01", "A02", "B01", "B02", "A01", "B01"],
                "Compound": ["DMSO", "DrugA", "DMSO", "DrugB", "DMSO", "DrugA"],
                "MOA": ["control", "kinase", "control", "tubulin", "control", "kinase"],
                "Count_Cell": [100, 90, 110, 95, 105, 92],
                "ExecutionTime_01LoadData": [1, 1, 1, 1, 1, 1],
                "ImageNumber": [1, 2, 3, 4, 5, 6],
                "Intensity_MeanIntensity_DAPI": [1.0, 4.0, 1.1, 7.0, 1.2, 4.1],
                "Texture_Contrast_DAPI": [0.1, 0.8, 0.2, 1.2, 0.15, 0.75],
                "AreaShape_Area": [10.0, 15.0, 10.5, 18.0, 10.3, 15.2],
            }
        )

    def test_feature_inference_excludes_qc_and_provenance_by_default(self) -> None:
        """Feature inference should not treat counts or execution time as default morphology features."""
        table = self.make_table()
        metadata = infer_metadata_columns(data_frame=table)
        features = infer_feature_columns(data_frame=table, metadata_columns=metadata)
        self.assertIn("Intensity_MeanIntensity_DAPI", features)
        self.assertIn("Texture_Contrast_DAPI", features)
        self.assertNotIn("Count_Cell", features)
        self.assertNotIn("ExecutionTime_01LoadData", features)
        self.assertNotIn("ImageNumber", features)

    def test_column_role_report_is_auditable(self) -> None:
        """Column-role report should explain selected and excluded columns."""
        roles = assign_column_roles(data_frame=self.make_table())
        role_map = dict(zip(roles["column"], roles["role"]))
        self.assertEqual(role_map["Count_Cell"], "excluded_numeric_qc_or_provenance")
        self.assertEqual(role_map["ExecutionTime_01LoadData"], "excluded_numeric_qc_or_provenance")
        self.assertEqual(role_map["Intensity_MeanIntensity_DAPI"], "feature")

    def test_winsorisation_clips_extreme_values(self) -> None:
        """Winsorisation should clip extreme feature values and report it."""
        features = pd.DataFrame({"x": [1.0, 2.0, 100.0, 3.0]})
        clipped, report = winsorise_features(features=features, lower_quantile=0.0, upper_quantile=0.75)
        self.assertLess(float(clipped["x"].max()), 100.0)
        self.assertGreater(int(report.loc[0, "n_clipped_high"]), 0)

    def test_reference_normalisation_within_plate(self) -> None:
        """Reference normalisation should use DMSO controls within each plate."""
        table = self.make_table()
        metadata = table[["Plate_Metadata", "Compound"]]
        features = table[["Intensity_MeanIntensity_DAPI", "Texture_Contrast_DAPI"]]
        normalised, report = normalise_features_to_reference(
            features=features,
            metadata=metadata,
            reference_column="Compound",
            reference_values=["DMSO"],
            group_columns=["Plate_Metadata"],
            method="median_center",
            min_reference_profiles=1,
        )
        self.assertIn("method", report.columns)
        dmso_p1 = normalised.loc[[0, 2], "Intensity_MeanIntensity_DAPI"].median()
        self.assertAlmostEqual(float(dmso_p1), 0.0, places=6)

    def test_batch_centering(self) -> None:
        """Batch centering should centre each batch independently."""
        metadata = pd.DataFrame({"batch": ["A", "A", "B", "B"]})
        features = pd.DataFrame({"x": [1.0, 3.0, 10.0, 12.0]})
        centered, report = batch_center_features(
            features=features,
            metadata=metadata,
            batch_columns=["batch"],
            method="median_center",
        )
        self.assertAlmostEqual(float(centered.loc[[0, 1], "x"].median()), 0.0)
        self.assertEqual(set(report["status"]), {"ok"})

    def test_preprocess_returns_v022_audit_tables(self) -> None:
        """Preprocessing should include the new audit and normalisation tables."""
        result = preprocess_profiles(
            data_frame=self.make_table(),
            max_feature_missing_fraction=0.5,
            reference_normalisation_method="median_center",
            reference_column="Metadata_Compound",
            reference_values=["DMSO"],
            reference_group_columns=["Metadata_Plate"],
            batch_centering_method="none",
            remove_correlated=False,
        )
        for key in [
            "column_role_report",
            "preprocessing_decision_log",
            "preprocessing_config",
            "reference_normalisation_report",
            "imputed_unscaled_features_with_metadata",
        ]:
            self.assertIn(key, result)
        self.assertGreater(result["preprocessed"].shape[0], 0)

    def test_moa_centroid_scores_include_confidence_margin(self) -> None:
        """Centroid scores should include confidence and margin columns."""
        features = pd.DataFrame({"f1": [0, 0.1, 5, 5.1], "f2": [0, 0.2, 4.8, 5.0]})
        labels = pd.Series(["A", "A", "B", "B"])
        centroids, _ = calculate_class_centroids(features=features, labels=labels, min_class_size=1)
        scores = score_profiles_against_centroids(query_features=features, centroids=centroids, top_n=2)
        self.assertIn("softmax_confidence", scores.columns)
        self.assertIn("top1_similarity_margin", scores.columns)

    def test_knn_can_return_neighbour_table(self) -> None:
        """KNN classification should optionally return neighbour details."""
        features = pd.DataFrame({"f1": [0, 0.1, 5, 5.1], "f2": [0, 0.2, 4.8, 5.0]})
        labels = pd.Series(["A", "A", "B", "B"])
        predictions, neighbours = classify_by_knn(
            train_features=features,
            train_labels=labels,
            query_features=features,
            n_neighbors=2,
            return_neighbour_table=True,
        )
        self.assertIn("max_probability", predictions.columns)
        self.assertEqual(neighbours.groupby("query_index").size().iloc[0], 2)

    def test_leave_one_out_centroid_validation(self) -> None:
        """Leave-one-out centroid validation should produce a summary."""
        features = pd.DataFrame({"f1": [0, 0.1, 0.2, 5, 5.1, 5.2], "f2": [0, 0.2, 0.1, 4.8, 5.0, 5.1]})
        labels = pd.Series(["A", "A", "A", "B", "B", "B"])
        predictions, summary = leave_one_out_centroid_validation(features=features, labels=labels, min_class_size=2)
        self.assertFalse(predictions.empty)
        self.assertIn("accuracy", summary.columns)

    def test_shap_value_conversion_for_multiclass_arrays(self) -> None:
        """SHAP array conversion should support sample-feature-class arrays."""
        values = np.ones((4, 3, 2))
        global_table, class_table = _shap_values_to_tables(
            shap_values=values,
            feature_names=["a", "b", "c"],
            class_names=["A", "B"],
        )
        self.assertEqual(global_table.shape[0], 3)
        self.assertEqual(class_table["class_name"].nunique(), 2)

    def test_html_report_contains_cards_and_assets(self) -> None:
        """The richer HTML report should contain key summary cards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            report = make_html_report(
                title="CPATK test report",
                output_path=out / "report.html",
                summary_tables={"summary": pd.DataFrame({"item": ["rows"], "value": [10]})},
                narrative="Test narrative",
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("card-grid", text)
            self.assertIn("Table index", text)


if __name__ == "__main__":
    unittest.main()
