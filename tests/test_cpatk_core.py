"""Unit tests for CPATK core functionality."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.ai import check_backend_availability, make_ai_status_table
from cpatk.clustering import (
    calculate_silhouette_summary,
    run_agglomerative,
    run_dbscan,
    run_kmeans,
    summarise_clusters,
)
from cpatk.distances import (
    calculate_nearest_neighbours,
    calculate_pairwise_distance_matrix,
    summarise_neighbour_classes,
)
from cpatk.embedding import run_pca, run_umap_or_pca
from cpatk.features import (
    infer_feature_columns,
    infer_metadata_columns,
    make_column_inventory,
    parse_column_list,
    split_metadata_and_features,
    summarise_feature_matrix,
    validate_columns_present,
)
from cpatk.inspection import inspect_directory, inspect_table_file
from cpatk.io import (
    data_frame_to_html_table,
    list_supported_tables,
    read_table,
    sanitise_sheet_name,
    write_excel_workbook,
    write_table,
)
from cpatk.moa import (
    calculate_class_centroids,
    classify_by_knn,
    score_profiles_against_centroids,
    summarise_moa_predictions,
)
from cpatk.plotting import plot_embedding, plot_heatmap, plot_metric_by_group, write_interactive_embedding_html
from cpatk.preprocessing import (
    aggregate_profiles,
    impute_features,
    preprocess_profiles,
    remove_correlated_features,
    scale_features,
)
from cpatk.qc import (
    calculate_feature_qc,
    calculate_sample_qc,
    flag_profile_outliers,
    flag_samples_by_qc,
    robust_z_score,
    select_features_by_qc,
    summarise_qc,
)
from cpatk.reporting import make_html_report


class CpatkCoreTests(unittest.TestCase):
    """Test CPATK functions on small synthetic Cell Painting-like data."""

    def setUp(self) -> None:
        """Create synthetic data used by the tests."""
        self.table = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P1", "P1", "P2", "P2"],
                "Metadata_Well": ["A01", "A02", "B01", "B02", "A01", "B01"],
                "compound": ["DMSO", "DrugA", "DrugB", "DrugA", "DMSO", "DrugB"],
                "moa": ["control", "kinase", "tubulin", "kinase", "control", "tubulin"],
                "Cells_Intensity_Mean": [1.0, 2.0, 3.0, np.nan, 1.1, 2.9],
                "Cells_Texture_Info": [0.1, 0.2, 0.4, 0.21, 0.11, 0.39],
                "Nuclei_AreaShape_Area": [50, 55, 60, 54, 51, 59],
                "constant_feature": [1, 1, 1, 1, 1, 1],
            }
        )
        self.features = self.table[
            ["Cells_Intensity_Mean", "Cells_Texture_Info", "Nuclei_AreaShape_Area"]
        ].copy()
        self.metadata = self.table[["Metadata_Plate", "Metadata_Well", "compound", "moa"]].copy()

    def test_infer_metadata_columns_detects_metadata(self) -> None:
        metadata = infer_metadata_columns(data_frame=self.table)
        self.assertIn("Metadata_Plate", metadata)
        self.assertIn("Metadata_Well", metadata)
        self.assertIn("compound", metadata)

    def test_infer_feature_columns_detects_numeric_features(self) -> None:
        metadata = infer_metadata_columns(data_frame=self.table)
        features = infer_feature_columns(data_frame=self.table, metadata_columns=metadata)
        self.assertIn("Cells_Intensity_Mean", features)
        self.assertIn("Nuclei_AreaShape_Area", features)

    def test_split_metadata_and_features_returns_expected_shapes(self) -> None:
        metadata, features, metadata_names, feature_names = split_metadata_and_features(data_frame=self.table)
        self.assertEqual(metadata.shape[0], self.table.shape[0])
        self.assertGreaterEqual(features.shape[1], 3)
        self.assertIn("Metadata_Well", metadata_names)
        self.assertIn("Cells_Texture_Info", feature_names)

    def test_column_inventory_has_all_columns(self) -> None:
        inventory = make_column_inventory(data_frame=self.table)
        self.assertEqual(inventory.shape[0], self.table.shape[1])
        self.assertIn("missing_fraction", inventory.columns)

    def test_validate_columns_present_raises_for_missing(self) -> None:
        with self.assertRaises(ValueError):
            validate_columns_present(data_frame=self.table, required_columns=["not_here"])

    def test_parse_column_list_handles_multiple_separators(self) -> None:
        parsed = parse_column_list(value="a,b;c\td")
        self.assertEqual(parsed, ["a", "b", "c", "d"])

    def test_feature_matrix_summary(self) -> None:
        summary = summarise_feature_matrix(features=self.features)
        self.assertIn("feature", summary.columns)
        self.assertEqual(summary.shape[0], self.features.shape[1])

    def test_feature_qc_identifies_constant_feature(self) -> None:
        qc = calculate_feature_qc(features=self.table[["constant_feature"]])
        self.assertTrue(bool(qc.loc[0, "near_zero_variance"]))

    def test_select_features_by_qc_removes_missing_and_constant(self) -> None:
        qc = calculate_feature_qc(features=self.table[["Cells_Intensity_Mean", "constant_feature"]])
        selected, annotated = select_features_by_qc(feature_qc=qc, max_missing_fraction=0.5)
        self.assertIn("Cells_Intensity_Mean", selected)
        self.assertNotIn("constant_feature", selected)
        self.assertIn("feature_qc_pass", annotated.columns)

    def test_sample_qc_and_flags(self) -> None:
        qc = calculate_sample_qc(features=self.features, metadata=self.metadata)
        flagged = flag_samples_by_qc(sample_qc=qc, max_missing_fraction=0.5)
        self.assertIn("sample_qc_pass", flagged.columns)
        self.assertTrue(flagged["sample_qc_pass"].all())

    def test_robust_z_score_centres_values(self) -> None:
        z = robust_z_score(values=pd.Series([1, 2, 3, 100]))
        self.assertEqual(len(z), 4)
        self.assertGreater(z.iloc[-1], 10)

    def test_flag_profile_outliers_adds_columns(self) -> None:
        table = pd.DataFrame({"group": ["A", "A", "A", "A"], "metric": [1, 2, 3, 100]})
        flagged = flag_profile_outliers(
            data_frame=table,
            metric_columns=["metric"],
            group_columns=["group"],
            robust_z_threshold=5,
        )
        self.assertIn("metric_outlier", flagged.columns)

    def test_qc_summary(self) -> None:
        feature_qc = calculate_feature_qc(features=self.features)
        selected, feature_qc = select_features_by_qc(feature_qc=feature_qc)
        sample_qc = flag_samples_by_qc(sample_qc=calculate_sample_qc(features=self.features))
        summary = summarise_qc(feature_qc=feature_qc, sample_qc=sample_qc)
        self.assertIn("n_features_total", summary["item"].tolist())

    def test_impute_features_median(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        self.assertFalse(imputed.isna().any().any())

    def test_impute_features_knn(self) -> None:
        imputed = impute_features(features=self.features, method="knn", n_neighbors=2)
        self.assertFalse(imputed.isna().any().any())

    def test_scale_features_robust(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        scaled = scale_features(features=imputed, method="robust")
        self.assertEqual(scaled.shape, imputed.shape)

    def test_remove_correlated_features(self) -> None:
        features = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 1, 2, 2]})
        filtered, report = remove_correlated_features(features=features, max_absolute_correlation=0.99)
        self.assertLess(filtered.shape[1], features.shape[1])
        self.assertGreaterEqual(report.shape[0], 1)

    def test_aggregate_profiles(self) -> None:
        aggregated = aggregate_profiles(
            data_frame=self.table,
            group_columns=["Metadata_Plate", "Metadata_Well"],
            feature_columns=["Cells_Intensity_Mean", "Cells_Texture_Info"],
        )
        self.assertIn("n_objects", aggregated.columns)
        self.assertEqual(aggregated.shape[0], 6)

    def test_preprocess_profiles_returns_expected_outputs(self) -> None:
        result = preprocess_profiles(
            data_frame=self.table,
            max_feature_missing_fraction=0.5,
            remove_correlated=False,
        )
        self.assertIn("preprocessed", result)
        self.assertIn("feature_qc", result)
        self.assertIn("preprocessing_summary", result)
        self.assertGreater(result["preprocessed"].shape[0], 0)

    def test_pairwise_distance_matrix(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        matrix = calculate_pairwise_distance_matrix(features=imputed, metric="euclidean")
        self.assertEqual(matrix.shape[0], self.features.shape[0])
        self.assertEqual(matrix.shape[0], matrix.shape[1])

    def test_nearest_neighbours(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        matrix = calculate_pairwise_distance_matrix(features=imputed, metric="euclidean")
        neighbours = calculate_nearest_neighbours(distance_matrix=matrix, n_neighbours=2)
        self.assertEqual(neighbours.groupby("query_index").size().iloc[0], 2)

    def test_neighbour_class_summary(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        matrix = calculate_pairwise_distance_matrix(features=imputed, metric="euclidean")
        neighbours = calculate_nearest_neighbours(distance_matrix=matrix, n_neighbours=2)
        metadata = self.metadata.copy()
        summary = summarise_neighbour_classes(neighbours=neighbours, metadata=metadata, class_column="moa")
        self.assertIn("fraction_neighbours", summary.columns)

    def test_kmeans_clustering(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        clusters = run_kmeans(features=imputed, n_clusters=2)
        self.assertEqual(clusters.shape[0], imputed.shape[0])

    def test_agglomerative_clustering(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        clusters = run_agglomerative(features=imputed, n_clusters=2)
        self.assertEqual(clusters.shape[0], imputed.shape[0])

    def test_dbscan_clustering(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        clusters = run_dbscan(features=imputed, eps=100)
        self.assertIn("cluster", clusters.columns)

    def test_silhouette_summary(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        clusters = run_kmeans(features=imputed, n_clusters=2)
        summary = calculate_silhouette_summary(features=imputed, clusters=clusters["cluster"])
        self.assertIn("silhouette_score", summary.columns)

    def test_cluster_summary(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        clusters = run_kmeans(features=imputed, n_clusters=2)
        summary = summarise_clusters(metadata=self.metadata, clusters=clusters["cluster"], group_columns=["moa"])
        self.assertIn("n_profiles", summary.columns)

    def test_pca(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        scores, variance = run_pca(features=imputed, n_components=2)
        self.assertEqual(scores.shape[1], 2)
        self.assertEqual(variance.shape[0], 2)

    def test_umap_or_pca_returns_two_columns(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        embedding = run_umap_or_pca(features=imputed, n_components=2)
        self.assertEqual(embedding.shape[1], 2)

    def test_centroid_moa(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        centroids, summary = calculate_class_centroids(features=imputed, labels=self.metadata["moa"], min_class_size=1)
        self.assertGreaterEqual(centroids.shape[0], 3)
        self.assertIn("n_profiles", summary.columns)

    def test_centroid_scores(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        centroids, _ = calculate_class_centroids(features=imputed, labels=self.metadata["moa"], min_class_size=1)
        scores = score_profiles_against_centroids(query_features=imputed, centroids=centroids, top_n=2)
        self.assertEqual(scores.groupby("query_index").size().iloc[0], 2)

    def test_knn_moa(self) -> None:
        imputed = impute_features(features=self.features, method="median")
        predictions = classify_by_knn(
            train_features=imputed,
            train_labels=self.metadata["moa"],
            query_features=imputed,
            n_neighbors=1,
        )
        self.assertIn("predicted_class", predictions.columns)

    def test_moa_prediction_summary(self) -> None:
        predictions = pd.DataFrame({"predicted_class": ["a", "a", "b"]})
        summary = summarise_moa_predictions(predictions=predictions)
        self.assertIn("fraction", summary.columns)

    def test_io_tsv_parquet_excel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tsv = write_table(data_frame=self.table, path=tmp / "table.tsv")
            read_back = read_table(path=tsv)
            self.assertEqual(read_back.shape, self.table.shape)
            try:
                import pyarrow  # type: ignore  # noqa: F401
                parquet = write_table(data_frame=self.table, path=tmp / "table.parquet")
                read_parquet = read_table(path=parquet)
                self.assertEqual(read_parquet.shape, self.table.shape)
            except ImportError:
                pass
            xlsx = write_excel_workbook(tables={"summary": self.table}, path=tmp / "table.xlsx")
            self.assertTrue(xlsx.exists())

    def test_sanitise_sheet_name(self) -> None:
        name = sanitise_sheet_name(sheet_name="bad/name/that/is/very/very/very/long")
        self.assertLessEqual(len(name), 31)
        self.assertNotIn("/", name)

    def test_html_table(self) -> None:
        html = data_frame_to_html_table(data_frame=self.table, max_rows=2)
        self.assertIn("<table>", html)
        self.assertIn("Metadata_Plate", html)

    def test_list_supported_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_table(data_frame=self.table, path=tmp / "table.tsv")
            inventory = list_supported_tables(input_dir=tmp)
            self.assertEqual(inventory.shape[0], 1)

    def test_inspection_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path = write_table(data_frame=self.table, path=tmp / "table.tsv")
            result = inspect_table_file(path=path)
            self.assertIn("column_inventory", result)
            directory = inspect_directory(input_dir=tmp)
            self.assertIn("file_inventory", directory)

    def test_static_plotting_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imputed = impute_features(features=self.features, method="median")
            embedding, _ = run_pca(features=imputed, n_components=2)
            paths = plot_embedding(
                embedding=embedding,
                metadata=self.metadata,
                x_column="PC1",
                y_column="PC2",
                colour_column="moa",
                output_path_base=tmp / "embedding",
            )
            self.assertTrue(all(path.exists() for path in paths))
            heat = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["C", "D"])
            heat_paths = plot_heatmap(matrix=heat, output_path_base=tmp / "heat", title="Heat", value_label="value")
            self.assertTrue(all(path.exists() for path in heat_paths))
            metric_paths = plot_metric_by_group(
                data_frame=pd.DataFrame({"group": ["a", "a", "b"], "metric": [1, 2, 3]}),
                group_column="group",
                metric_column="metric",
                output_path_base=tmp / "metric",
            )
            self.assertTrue(all(path.exists() for path in metric_paths))

    def test_interactive_plotting_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imputed = impute_features(features=self.features, method="median")
            embedding, _ = run_pca(features=imputed, n_components=2)
            path = write_interactive_embedding_html(
                embedding=embedding,
                metadata=self.metadata,
                x_column="PC1",
                y_column="PC2",
                colour_column="moa",
                output_path=tmp / "interactive.html",
            )
            self.assertTrue(path is None or path.exists())

    def test_html_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            report = make_html_report(
                title="Test report",
                output_path=tmp / "report.html",
                summary_tables={"table": self.table},
                narrative="Test narrative",
            )
            self.assertTrue(report.exists())
            self.assertIn("Test report", report.read_text(encoding="utf-8"))

    def test_ai_status(self) -> None:
        status = check_backend_availability(backend_name="definitely_missing_backend")
        self.assertFalse(status.available)
        table = make_ai_status_table(backend_name="definitely_missing_backend")
        self.assertIn("available", table.columns)


if __name__ == "__main__":
    unittest.main()
