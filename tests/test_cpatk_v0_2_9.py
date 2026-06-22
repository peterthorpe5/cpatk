"""Tests for CPATK v0.2.9 QC, visualisation and neighbour analysis."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.neighbour_analysis import (
    _try_gaussian_kde_density,
    evaluate_neighbour_overlap,
    normalise_neighbour_long_table,
    rank_biased_overlap,
    run_neighbour_workflow,
    shared_neighbour_table,
)
from cpatk.qc_drift import (
    compute_drift_statistics,
    infer_compartment_from_name,
    run_drift_qc,
    select_drift_features,
)
from cpatk.visualisation import (
    build_knn_topology,
    calculate_pca,
    draw_clustered_heatmap,
    l2_norm_summary,
    run_visualisation_workflow,
    select_feature_columns,
    zscore_matrix,
)


class TestVisualisationV029(unittest.TestCase):
    """Tests for visualisation utilities."""

    def test_select_feature_columns_excludes_metadata(self) -> None:
        """Feature selection should exclude known metadata and provenance."""
        df = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1"],
                "cpd_id": ["A", "B"],
                "ImageNumber": [1, 2],
                "0": [0.1, 0.2],
                "1": [0.3, 0.4],
                "Intensity_MeanIntensity_DNA": [10.0, 11.0],
            }
        )
        cols = select_feature_columns(data_frame=df, digit_named_latents=True)
        self.assertEqual(cols, ["0", "1"])
        cols_all = select_feature_columns(data_frame=df)
        self.assertIn("Intensity_MeanIntensity_DNA", cols_all)
        self.assertNotIn("ImageNumber", cols_all)

    def test_l2_norm_summary_counts_zero_norms(self) -> None:
        """Latent norm summary should report zero vectors."""
        features = pd.DataFrame({"a": [3.0, 0.0], "b": [4.0, 0.0]})
        summary = l2_norm_summary(features=features)
        self.assertEqual(int(summary.loc[0, "n_vectors"]), 2)
        self.assertEqual(int(summary.loc[0, "n_zero_norm"]), 1)
        self.assertAlmostEqual(float(summary.loc[0, "max_norm"]), 5.0)

    def test_pca_and_heatmap_outputs(self) -> None:
        """PCA and heatmap utilities should produce expected tables and plots."""
        rng = np.random.default_rng(1)
        features = pd.DataFrame(rng.normal(size=(8, 5)), index=[f"C{i}" for i in range(8)])
        coords, variance = calculate_pca(features=features, n_components=2)
        self.assertEqual(coords.shape, (8, 2))
        self.assertEqual(variance.shape[0], 2)
        with tempfile.TemporaryDirectory() as tmp:
            ordered, row_order, col_order, plots = draw_clustered_heatmap(
                matrix=features,
                output_path_base=Path(tmp) / "heatmap",
                title="test heatmap",
            )
            self.assertEqual(ordered.shape, features.shape)
            self.assertEqual(row_order.shape[0], features.shape[0])
            self.assertEqual(col_order.shape[0], features.shape[1])
            self.assertTrue(any(path.exists() for path in plots))

    def test_knn_topology_has_edges(self) -> None:
        """kNN topology should return nodes and edges."""
        features = pd.DataFrame(
            np.eye(5),
            index=[f"C{i}" for i in range(5)],
        )
        metadata = pd.DataFrame({"compound": [f"C{i}" for i in range(5)], "group": ["A", "A", "B", "B", "B"]})
        nodes, edges, coords = build_knn_topology(
            features=features,
            metadata=metadata,
            id_column="compound",
            colour_column="group",
            n_neighbours=2,
        )
        self.assertEqual(nodes.shape[0], 5)
        self.assertGreater(edges.shape[0], 0)
        self.assertEqual(coords.shape[0], 5)

    def test_run_visualisation_workflow(self) -> None:
        """The visualisation workflow should write report outputs."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.normal(size=(10, 4)), columns=["0", "1", "2", "3"])
        df.insert(0, "Metadata_Compound", [f"C{i}" for i in range(10)])
        df.insert(1, "Metadata_MOA", ["A"] * 5 + ["B"] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "latent.tsv"
            df.to_csv(input_path, sep="\t", index=False)
            run_visualisation_workflow(
                input_table=input_path,
                output_dir=Path(tmp) / "vis",
                id_column="Metadata_Compound",
                colour_columns=["Metadata_MOA"],
                digit_named_latents=True,
                make_umap=False,
                make_phate=False,
                interactive=False,
            )
            self.assertTrue((Path(tmp) / "vis" / "latent_norm_summary.tsv").exists())
            self.assertTrue((Path(tmp) / "vis" / "topology_edges.tsv").exists())

    def test_zscore_row_mode(self) -> None:
        """Row z-scoring should centre each row."""
        matrix = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0]})
        zed = zscore_matrix(matrix=matrix, mode="row")
        self.assertTrue(np.allclose(zed.mean(axis=1), 0.0))


class TestDriftQCV029(unittest.TestCase):
    """Tests for per-compartment drift QC."""

    def test_infer_compartment_from_name(self) -> None:
        """Compartment names should be inferred from CellProfiler filenames."""
        self.assertEqual(infer_compartment_from_name(filename="Plate1_Nuclei.csv.gz"), "Nuclei")
        self.assertEqual(infer_compartment_from_name(filename="Plate1_Cytoplasm.csv.gz"), "Cytoplasm")
        self.assertEqual(infer_compartment_from_name(filename="Plate1_Cell.csv.gz"), "Cell")

    def test_drift_statistics_detect_trend(self) -> None:
        """A monotonically drifting feature should have positive rho."""
        image = np.repeat(np.arange(1, 21), 5)
        df = pd.DataFrame(
            {
                "ImageNumber": image,
                "Intensity_MeanIntensity_DNA": image.astype(float) + np.linspace(0, 0.1, image.size),
                "AreaShape_Area": np.ones(image.size),
            }
        )
        features = select_drift_features(data_frame=df)
        stats = compute_drift_statistics(
            data_frame=df,
            image_col="ImageNumber",
            feature_columns=features,
            min_points=10,
        )
        row = stats.loc[stats["feature"] == "Intensity_MeanIntensity_DNA"].iloc[0]
        self.assertGreater(float(row["spearman_rho"]), 0.9)

    def test_run_drift_qc_writes_outputs(self) -> None:
        """Drift QC should process a small object table and write a report."""
        image = np.repeat(np.arange(1, 16), 4)
        df = pd.DataFrame(
            {
                "ImageNumber": image,
                "ObjectNumber": np.arange(image.size),
                "Intensity_MeanIntensity_DNA": image.astype(float),
                "Texture_Info_DNA": np.sin(image),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            df.to_csv(input_dir / "Example_Cell.csv", index=False)
            out_dir = Path(tmp) / "out"
            run_drift_qc(
                input_dir=input_dir,
                output_dir=out_dir,
                min_points=10,
                plot_top_n=1,
            )
            self.assertTrue((out_dir / "Cell" / "drift_statistics.tsv").exists())
            self.assertTrue((out_dir / "drift_qc_report.html").exists())


class TestNeighbourAnalysisV029(unittest.TestCase):
    """Tests for nearest-neighbour utilities."""

    def test_kde_density_is_optional(self) -> None:
        """KDE density colouring should degrade gracefully if SciPy cannot import."""
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "scipy.stats":
                raise ImportError("simulated scipy.stats import failure")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            density = _try_gaussian_kde_density(
                x=np.asarray([0.1, 0.2, 0.3]),
                y=np.asarray([0.2, 0.3, 0.4]),
            )
        self.assertIsNone(density)

    def test_normalise_long_table(self) -> None:
        """Column aliases should be normalised."""
        df = pd.DataFrame(
            {
                "QueryID": ["A"],
                "NeighbourID": ["B"],
                "Distance": [0.2],
            }
        )
        out = normalise_neighbour_long_table(data_frame=df)
        self.assertEqual(list(out.columns), ["query_id", "neighbour_id", "distance"])

    def test_rank_biased_overlap(self) -> None:
        """Identical ranked lists should have higher RBO than disjoint lists."""
        same = rank_biased_overlap(first=["A", "B", "C"], second=["A", "B", "C"], depth=3)
        different = rank_biased_overlap(first=["A", "B", "C"], second=["X", "Y", "Z"], depth=3)
        self.assertGreater(same, different)

    def test_evaluate_neighbour_overlap(self) -> None:
        """Overlap evaluation should summarise runs."""
        base = pd.DataFrame(
            {
                "query_id": ["A", "A", "B", "B"],
                "neighbour_id": ["B", "C", "A", "C"],
                "rank": [1, 2, 1, 2],
                "distance": [0.1, 0.2, 0.1, 0.3],
            }
        )
        run = pd.DataFrame(
            {
                "query_id": ["A", "A", "B", "B"],
                "neighbour_id": ["B", "D", "A", "D"],
                "rank": [1, 2, 1, 2],
                "distance": [0.1, 0.25, 0.1, 0.35],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "base.tsv"
            run_path = Path(tmp) / "run.tsv"
            base.to_csv(base_path, sep="\t", index=False)
            run.to_csv(run_path, sep="\t", index=False)
            item, summary = evaluate_neighbour_overlap(
                baseline_path=base_path,
                run_paths=[run_path],
                k="2",
            )
            self.assertEqual(item.shape[0], 2)
            self.assertEqual(summary.shape[0], 1)
            self.assertGreater(float(summary.loc[0, "mean_jaccard"]), 0.0)

    def test_shared_neighbour_table(self) -> None:
        """Shared neighbour table should align similarities for compound pairs."""
        nn = pd.DataFrame(
            {
                "query_id": ["A", "A", "B", "B"],
                "neighbour_id": ["X", "Y", "X", "Z"],
                "distance": [0.2, 0.4, 0.3, 0.5],
            }
        )
        shared = shared_neighbour_table(neighbours=nn, first_compound="A", second_compound="B")
        self.assertEqual(shared.shape[0], 1)
        self.assertEqual(shared.loc[0, "neighbour_id"], "X")

    def test_run_neighbour_workflow_writes_report(self) -> None:
        """Neighbour workflow should write HTML and TSV outputs."""
        nn = pd.DataFrame(
            {
                "query_id": ["A", "A", "B", "B", "C", "C"],
                "neighbour_id": ["B", "C", "A", "C", "A", "B"],
                "distance": [0.1, 0.2, 0.1, 0.4, 0.2, 0.4],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nn.tsv"
            nn.to_csv(path, sep="\t", index=False)
            out = Path(tmp) / "out"
            run_neighbour_workflow(
                output_dir=out,
                input_neighbours=path,
                compounds=["A", "B"],
            )
            self.assertTrue((out / "neighbour_analysis_report.html").exists())
            self.assertTrue((out / "shared_neighbours_A_vs_B.tsv").exists())


if __name__ == "__main__":
    unittest.main()
