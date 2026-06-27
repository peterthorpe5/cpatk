"""Unit tests for CPATK v0.2.0 functionality."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.batch import (
    calculate_batch_centroid_distances,
    cross_validated_batch_prediction,
    calculate_metadata_association_with_pcs,
)
from cpatk.clipn_adapter import (
    ClipnAdapterConfig,
    align_dataset_features,
    check_clipn_backend,
    load_clipn_config,
    run_clipn_adapter,
    save_clipn_config,
)
from cpatk.explainability import (
    calculate_permutation_feature_importance,
    calculate_shap_importance,
    group_feature_importance_by_family,
)
from cpatk.layout import (
    add_plate_position_columns,
    normalise_well_name,
    run_plate_layout_diagnostics,
    summarise_layout_axis,
    summarise_plate_metric,
)
from cpatk.ml import (
    build_classifier,
    compare_moa_models,
    cross_validate_classifier,
    train_predict_classifier,
)
from cpatk.plotting import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_variance,
    set_publication_theme,
    write_interactive_heatmap_html,
)
from cpatk.reproducibility import (
    bootstrap_cluster_stability,
    bootstrap_neighbour_stability,
    calculate_neighbour_sets,
    calculate_replicate_correlations,
    consensus_clustering,
    permutation_test_cluster_structure,
    summarise_replicate_correlations,
)
from cpatk.reporting import default_methods_text, make_html_report


class CpatkV02Tests(unittest.TestCase):
    """Test v0.2.0 diagnostics, ML, explainability and reporting."""

    def setUp(self) -> None:
        """Create synthetic profile data with separable MOA and batch labels."""
        rng = np.random.default_rng(seed=7)
        records = []
        features = []
        classes = ["control", "kinase", "tubulin"]
        wells = ["A01", "A02", "B01", "B02", "C01", "C02"]
        for class_index, class_name in enumerate(classes):
            centre = np.zeros(8)
            centre[class_index] = 3.0
            centre[class_index + 3] = -2.0
            for replicate in range(6):
                records.append(
                    {
                        "Metadata_Plate": "P1" if replicate < 3 else "P2",
                        "Metadata_Well": wells[(class_index * 2 + replicate) % len(wells)],
                        "compound": f"{class_name}_{replicate % 2}",
                        "moa": class_name,
                        "batch": "B1" if replicate % 2 == 0 else "B2",
                        "replicate_group": class_name,
                    }
                )
                features.append(centre + rng.normal(loc=0.0, scale=0.25, size=8))
        self.metadata = pd.DataFrame.from_records(records)
        feature_columns = [f"Cells_Intensity_Mean_F{index}" for index in range(4)] + [
            f"Cells_Texture_Info_F{index}" for index in range(4)
        ]
        self.features = pd.DataFrame(data=np.vstack(features), columns=feature_columns)
        self.table = pd.concat([self.metadata, self.features], axis=1)

    def test_normalise_well_name(self) -> None:
        self.assertEqual(normalise_well_name(value="a1"), "A01")
        self.assertEqual(normalise_well_name(value="B12"), "B12")

    def test_add_plate_position_columns(self) -> None:
        table = add_plate_position_columns(data_frame=self.table, well_column="Metadata_Well")
        self.assertIn("plate_row", table.columns)
        self.assertIn("plate_column", table.columns)

    def test_summarise_plate_metric(self) -> None:
        matrix = summarise_plate_metric(
            data_frame=self.table,
            metric_column="Cells_Intensity_Mean_F0",
            well_column="Metadata_Well",
        )
        self.assertGreaterEqual(matrix.shape[0], 1)
        self.assertGreaterEqual(matrix.shape[1], 1)

    def test_summarise_layout_axis(self) -> None:
        summary = summarise_layout_axis(
            data_frame=self.table,
            metric_columns=["Cells_Intensity_Mean_F0"],
            axis_columns=["moa", "batch"],
        )
        self.assertIn("layout_axis", summary.columns)
        self.assertIn("median_Cells_Intensity_Mean_F0", summary.columns)

    def test_run_plate_layout_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_plate_layout_diagnostics(
                data_frame=self.table,
                output_dir=Path(tmpdir),
                well_column="Metadata_Well",
                metric_columns=["Cells_Intensity_Mean_F0"],
                grouping_columns=["moa"],
            )
            self.assertIn("plate_layout_table", result)
            self.assertIn("layout_axis_summary", result)

    def test_replicate_correlations(self) -> None:
        pairs = calculate_replicate_correlations(
            features=self.features,
            metadata=self.metadata,
            replicate_group_columns=["replicate_group"],
        )
        summary = summarise_replicate_correlations(
            replicate_correlations=pairs,
            group_columns=["replicate_group"],
        )
        self.assertGreater(pairs.shape[0], 0)
        self.assertIn("median_correlation", summary.columns)

    def test_neighbour_sets(self) -> None:
        sets = calculate_neighbour_sets(features=self.features, n_neighbours=3)
        self.assertEqual(len(sets), self.features.shape[0])
        self.assertLessEqual(max(len(item) for item in sets), 3)

    def test_bootstrap_neighbour_stability(self) -> None:
        stability = bootstrap_neighbour_stability(
            features=self.features,
            n_neighbours=3,
            n_bootstraps=3,
            feature_fraction=0.75,
        )
        self.assertIn("mean_neighbour_jaccard", stability.columns)

    def test_consensus_clustering(self) -> None:
        matrix, summary = consensus_clustering(
            features=self.features,
            n_clusters=3,
            n_bootstraps=3,
            sample_fraction=0.8,
        )
        self.assertEqual(matrix.shape[0], self.features.shape[0])
        self.assertIn("mean_pair_consensus", summary.columns)

    def test_permutation_test_cluster_structure(self) -> None:
        result = permutation_test_cluster_structure(
            features=self.features,
            n_clusters=3,
            n_permutations=3,
        )
        self.assertIn("empirical_p_value", result.columns)

    def test_bootstrap_cluster_stability(self) -> None:
        result = bootstrap_cluster_stability(
            features=self.features,
            n_clusters=3,
            n_bootstraps=3,
        )
        self.assertIn("mean_adjusted_rand_index", result.columns)

    def test_batch_centroid_distances(self) -> None:
        distances = calculate_batch_centroid_distances(
            features=self.features,
            metadata=self.metadata,
            batch_column="batch",
        )
        self.assertIn("distance", distances.columns)
        self.assertEqual(distances["batch_1"].nunique(), 2)

    def test_pc_metadata_association(self) -> None:
        association = calculate_metadata_association_with_pcs(
            features=self.features,
            metadata=self.metadata,
            columns_to_test=["batch", "moa"],
            n_components=3,
        )
        self.assertIn("eta_squared", association.columns)

    def test_batch_prediction(self) -> None:
        prediction = cross_validated_batch_prediction(
            features=self.features,
            metadata=self.metadata,
            batch_column="batch",
            n_splits=3,
        )
        self.assertIn("balanced_accuracy", prediction.columns)

    def test_build_classifier_supported_models(self) -> None:
        for model_name in ["knn", "random_forest", "extra_trees", "gradient_boosting", "logistic_regression"]:
            model = build_classifier(model_name=model_name)
            self.assertIsNotNone(model)

    def test_cross_validate_classifier(self) -> None:
        summary, predictions, confusion = cross_validate_classifier(
            features=self.features,
            labels=self.metadata["moa"],
            model_name="random_forest",
            n_splits=3,
        )
        self.assertIn("balanced_accuracy", summary.columns)
        self.assertIn("predicted_class", predictions.columns)
        self.assertIn("true_class", confusion.columns)

    def test_train_predict_classifier(self) -> None:
        predictions, model = train_predict_classifier(
            train_features=self.features,
            train_labels=self.metadata["moa"],
            query_features=self.features.head(3),
            model_name="random_forest",
        )
        self.assertEqual(predictions.shape[0], 3)
        self.assertTrue(hasattr(model, "predict"))

    def test_compare_moa_models(self) -> None:
        summary, predictions = compare_moa_models(
            features=self.features,
            labels=self.metadata["moa"],
            model_names=["knn", "random_forest"],
            n_splits=3,
        )
        self.assertEqual(summary.shape[0], 2)
        self.assertIn("model_name", predictions.columns)

    def test_permutation_feature_importance(self) -> None:
        importance, summary = calculate_permutation_feature_importance(
            features=self.features,
            labels=self.metadata["moa"],
            model_name="random_forest",
            n_repeats=2,
            test_size=0.4,
        )
        self.assertIn("permutation_importance_mean", importance.columns)
        self.assertIn("balanced_accuracy", summary.columns)

    def test_group_feature_importance_by_family(self) -> None:
        importance = pd.DataFrame(
            {
                "feature": ["Intensity_A", "Intensity_B", "Texture_A"],
                "permutation_importance_mean": [0.2, 0.1, 0.05],
            }
        )
        grouped = group_feature_importance_by_family(importance_table=importance)
        self.assertIn("feature_family", grouped.columns)
        self.assertIn("total_importance", grouped.columns)

    def test_shap_importance_status(self) -> None:
        importance, status = calculate_shap_importance(
            features=self.features,
            labels=self.metadata["moa"],
            model_name="random_forest",
            max_background=8,
            max_explain=6,
        )
        self.assertIn("status", status.columns)
        self.assertIn("feature", importance.columns)

    def test_clipn_backend_missing(self) -> None:
        status = check_clipn_backend(backend_module="definitely_missing_clipn_backend")
        self.assertFalse(bool(status["available"].iloc[0]))

    def test_align_dataset_features(self) -> None:
        datasets = {"a": self.table, "b": self.table.copy()}
        aligned, summary = align_dataset_features(
            datasets=datasets,
            feature_columns=self.features.columns.tolist(),
        )
        self.assertEqual(set(aligned.keys()), {"a", "b"})
        self.assertIn("n_shared_features", summary.columns)

    def test_clipn_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config = ClipnAdapterConfig(backend_module="clipn", feature_columns=["a", "b"])
            save_clipn_config(config=config, path=path)
            loaded = load_clipn_config(path=path)
            self.assertEqual(loaded.feature_columns, ["a", "b"])

    def test_run_clipn_adapter_without_backend(self) -> None:
        config = ClipnAdapterConfig(
            backend_module="definitely_missing_clipn_backend",
            feature_columns=self.features.columns.tolist(),
        )
        result = run_clipn_adapter(
            datasets={"dataset1": self.table, "dataset2": self.table.copy()},
            config=config,
        )
        self.assertIn("clipn_status", result)
        self.assertIn("clipn_feature_summary", result)

    def test_publication_plot_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            set_publication_theme(font_size=9)
            pca_var = pd.DataFrame(
                {"component": ["PC1", "PC2"], "explained_variance_ratio": [0.7, 0.2]}
            )
            paths = plot_pca_variance(explained_variance=pca_var, output_path_base=tmp / "variance")
            self.assertTrue(all(path.exists() for path in paths))
            importance = pd.DataFrame({"feature": ["a", "b"], "score": [0.5, 0.2]})
            imp_paths = plot_feature_importance(
                importance_table=importance,
                value_column="score",
                output_path_base=tmp / "importance",
            )
            self.assertTrue(all(path.exists() for path in imp_paths))
            confusion = pd.DataFrame({"true_class": ["a", "b"], "a": [2, 1], "b": [0, 3]})
            cm_paths = plot_confusion_matrix(confusion_table=confusion, output_path_base=tmp / "confusion")
            self.assertTrue(all(path.exists() for path in cm_paths))

    def test_interactive_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["c", "d"])
            path = write_interactive_heatmap_html(
                matrix=matrix,
                output_path=Path(tmpdir) / "heatmap.html",
                title="Heatmap",
            )
            self.assertTrue(path is None or path.exists())

    def test_richer_html_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = make_html_report(
                title="CPATK v0.2 report",
                output_path=Path(tmpdir) / "report.html",
                summary_tables={"metadata": self.metadata.head()},
                narrative="Synthetic report.",
                methods_text=default_methods_text(),
                warnings=["Synthetic warning."],
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("Methods explained", text)
            self.assertIn("Synthetic warning", text)


if __name__ == "__main__":
    unittest.main()
