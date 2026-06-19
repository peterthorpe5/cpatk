"""Tests for CPATK v0.2.6 neighbourhood explanation upgrades."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.neighbourhood_explain import (
    benjamini_hochberg,
    calculate_neighbourhood_shap,
    calculate_query_background_statistics,
    calculate_two_sample_feature_statistics,
    clean_numeric_feature_matrix,
    group_importance_by_feature_family,
    make_binary_neighbourhood_dataset,
    parse_query_ids,
    plot_shap_outputs,
    plot_signed_feature_statistics,
    select_neighbour_ids,
)


class TestNeighbourhoodExplain(unittest.TestCase):
    """Test neighbourhood SHAP/statistical explanation helpers."""

    def setUp(self) -> None:
        """Create a deterministic small Cell Painting-like table."""
        self.metadata = pd.DataFrame(
            {
                "cpd_id": ["Q1", "Q1", "N1", "N1", "N2", "N2", "DMSO", "DMSO"],
                "cpd_type": ["compound", "compound", "compound", "compound", "compound", "compound", "DMSO", "DMSO"],
                "Metadata_Plate": ["P1"] * 8,
                "Metadata_Well": ["A01", "A02", "B01", "B02", "C01", "C02", "D01", "D02"],
            }
        )
        self.features = pd.DataFrame(
            {
                "Cells__Intensity_MeanIntensity_DAPI": [5.0, 5.2, 1.0, 1.2, 1.1, 1.3, 0.8, 0.9],
                "Nuclei__Texture_Contrast_DAPI": [4.0, 4.2, 1.0, 0.8, 1.1, 0.9, 0.7, 0.6],
                "Cytoplasm__AreaShape_Area": [2.0, 2.1, 1.9, 2.0, 1.8, 1.7, 1.9, 1.8],
                "ImageNumber": [1, 2, 3, 4, 5, 6, 7, 8],
                "Count_Cell": [100, 100, 90, 95, 85, 87, 80, 82],
            }
        )

    def test_benjamini_hochberg_monotonic_adjustment(self) -> None:
        """BH adjustment should preserve shape and bound q-values."""
        q_values = benjamini_hochberg(p_values=[0.001, 0.02, 0.5, 1.0])
        self.assertEqual(q_values.shape[0], 4)
        self.assertTrue(np.all(q_values >= 0.0))
        self.assertTrue(np.all(q_values <= 1.0))
        self.assertLess(q_values[0], q_values[-1])

    def test_clean_numeric_feature_matrix_excludes_qc_columns(self) -> None:
        """Feature cleaning should exclude ImageNumber and Count columns."""
        matrix, audit = clean_numeric_feature_matrix(features=self.features)
        self.assertIn("Cells__Intensity_MeanIntensity_DAPI", matrix.columns)
        self.assertNotIn("ImageNumber", matrix.columns)
        self.assertNotIn("Count_Cell", matrix.columns)
        excluded = audit.loc[audit["role"] == "excluded", "column"].tolist()
        self.assertIn("ImageNumber", excluded)

    def test_select_neighbour_ids_sorts_and_excludes_query(self) -> None:
        """Neighbour selection should respect distances and avoid self-neighbours."""
        nn = pd.DataFrame(
            {
                "query_id": ["Q1", "Q1", "Q1", "Q1"],
                "neighbour_id": ["Q1", "N2", "N1", "N3"],
                "distance": [0.0, 0.2, 0.1, 0.4],
            }
        )
        neighbours = select_neighbour_ids(neighbour_table=nn, query_id="Q1", n_neighbours=2)
        self.assertEqual(neighbours, ["N1", "N2"])

    def test_query_background_statistics(self) -> None:
        """Query-background tests should return per-feature statistics and q-values."""
        stats = calculate_query_background_statistics(
            metadata=self.metadata,
            features=self.features,
            id_column="cpd_id",
            query_id="Q1",
            background_column="cpd_type",
            background_values=["DMSO"],
            test="mw",
        )
        self.assertIn("q_value", stats.columns)
        self.assertGreaterEqual(stats.shape[0], 3)
        top = stats.sort_values("absolute_median_difference", ascending=False).iloc[0]
        self.assertIn("Intensity", top["feature"])

    def test_two_sample_stats_rejects_bad_masks(self) -> None:
        """Mismatched masks should fail clearly."""
        with self.assertRaises(ValueError):
            calculate_two_sample_feature_statistics(
                features=self.features,
                mask_a=[True, False],
                mask_b=[False, True],
                comparison_name="bad",
            )

    def test_make_binary_neighbourhood_dataset(self) -> None:
        """Binary query-vs-neighbour setup should produce target labels."""
        x, y, subset_metadata, audit = make_binary_neighbourhood_dataset(
            metadata=self.metadata,
            features=self.features,
            id_column="cpd_id",
            query_id="Q1",
            neighbour_ids=["N1", "N2"],
        )
        self.assertEqual(x.shape[0], 6)
        self.assertEqual(int(y.sum()), 2)
        self.assertEqual(subset_metadata.shape[0], 6)
        self.assertFalse(audit.empty)

    def test_neighbourhood_shap_outputs_tables(self) -> None:
        """Neighbourhood SHAP should return auditable tables when SHAP is available."""
        x, y, _, _ = make_binary_neighbourhood_dataset(
            metadata=self.metadata,
            features=self.features,
            id_column="cpd_id",
            query_id="Q1",
            neighbour_ids=["N1", "N2"],
        )
        result = calculate_neighbourhood_shap(
            x=x,
            y=y,
            query_id="Q1",
            n_top_features=3,
            max_background=6,
            max_explain=6,
        )
        status = result["status"]
        self.assertIn("status", status.columns)
        if status["status"].iloc[0] == "ok":
            self.assertFalse(result["top_features"].empty)
            self.assertEqual(result["top_features"].shape[0], 3)

    def test_plot_outputs_for_stats_and_shap(self) -> None:
        """Plot helpers should write files for feature stats and SHAP outputs."""
        stats = calculate_query_background_statistics(
            metadata=self.metadata,
            features=self.features,
            id_column="cpd_id",
            query_id="Q1",
            background_column="cpd_type",
            background_values=["DMSO"],
        )
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            paths = plot_signed_feature_statistics(stats_table=stats, output_path_base=tmp / "signed")
            self.assertTrue(any(path.exists() for path in paths))
            # Heavy SHAP rendering is integration-tested manually because SHAP/matplotlib
            # backends vary across environments.  The plotting helper should still be
            # safe when no SHAP matrix is available.
            shap_paths = plot_shap_outputs(
                shap_array=np.empty((0, 0)),
                explained_x=pd.DataFrame(),
                top_features=pd.DataFrame(),
                output_path_base=tmp / "shap",
                max_display=2,
                n_dependence=1,
            )
            self.assertEqual(shap_paths, [])

    def test_parse_query_ids_from_inline_and_file(self) -> None:
        """Query parsing should accept inline IDs and files."""
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "queries.txt"
            path.write_text("Q1\nN1\n", encoding="utf-8")
            parsed = parse_query_ids(query_ids=["N2,N3"], query_file=path)
        self.assertEqual(parsed, ["Q1", "N1", "N2", "N3"])

    def test_group_importance_by_family(self) -> None:
        """Feature-family grouping should summarise SHAP/importance scores."""
        importance = pd.DataFrame(
            {
                "feature": ["Cells__Intensity_Mean", "Nuclei__Texture_Contrast", "OtherFeature"],
                "mean_absolute_shap": [0.4, 0.2, 0.1],
            }
        )
        grouped = group_importance_by_feature_family(importance_table=importance)
        self.assertIn("Intensity", grouped["feature_family"].tolist())
        self.assertGreater(grouped["total_importance"].max(), 0)


if __name__ == "__main__":
    unittest.main()
