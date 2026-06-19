"""Tests for CPATK v0.2.8 advanced MOA utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.moa_advanced import (
    anchor_permutation_test,
    bootstrap_kmeans_stability,
    build_moa_centroids,
    make_pseudo_anchors,
    pairwise_distance_outputs,
    prepare_embedding_matrix,
    score_against_moa_centroids,
    score_matrix_table,
)
from cpatk.cli import moa as moa_cli


class TestAdvancedMoaUtilities(unittest.TestCase):
    """Unit tests for advanced MOA helper functions."""

    def setUp(self) -> None:
        """Create a small synthetic embedding table."""
        rows = []
        rng = np.random.default_rng(7)
        for moa_name, centre in [
            ("A", np.array([1.0, 0.0, 0.0, 0.0])),
            ("B", np.array([0.0, 1.0, 0.0, 0.0])),
            ("C", np.array([0.0, 0.0, 1.0, 0.0])),
        ]:
            for compound_index in range(4):
                compound_id = f"{moa_name}{compound_index}"
                for replicate in range(2):
                    noise = rng.normal(0, 0.02, size=4)
                    vector = centre + noise
                    rows.append(
                        {
                            "cpd_id": compound_id,
                            "known_moa": moa_name,
                            "Plate_Metadata": "P1",
                            "Well_Metadata": f"A{compound_index + 1:02d}",
                            "f0": vector[0],
                            "f1": vector[1],
                            "f2": vector[2],
                            "f3": vector[3],
                        }
                    )
        self.table = pd.DataFrame(rows)
        self.features = ["f0", "f1", "f2", "f3"]

    def test_prepare_embedding_matrix_aggregates_replicates(self) -> None:
        """Replicate rows should aggregate to one row per compound."""
        emb, matrix, features, summary = prepare_embedding_matrix(
            table=self.table,
            id_column="cpd_id",
            feature_columns=self.features,
            metadata_columns=["known_moa", "Plate_Metadata", "Well_Metadata"],
            aggregate_method="median",
        )
        self.assertEqual(emb.shape[0], 12)
        self.assertEqual(matrix.shape, (12, 4))
        self.assertEqual(features, self.features)
        self.assertTrue((summary["n_replicate_rows"] == 2).all())

    def test_bootstrap_kmeans_stability_returns_k_selection(self) -> None:
        """Bootstrap stability should evaluate valid k values."""
        emb, matrix, _, _ = prepare_embedding_matrix(
            table=self.table,
            id_column="cpd_id",
            feature_columns=self.features,
            metadata_columns=["known_moa"],
        )
        del emb
        best_k, stability = bootstrap_kmeans_stability(
            values=matrix,
            k_values=[2, 3, 4],
            n_bootstraps=3,
            subsample_fraction=0.8,
            random_state=1,
        )
        self.assertIn(best_k, {2, 3, 4})
        self.assertIn("mean_bootstrap_ari", stability.columns)
        self.assertGreaterEqual(stability.shape[0], 3)

    def test_make_pseudo_anchors_outputs_anchor_tables(self) -> None:
        """Pseudo-anchor generation should return assignments and summaries."""
        anchors, summary, clusters, k_selection = make_pseudo_anchors(
            table=self.table,
            id_column="cpd_id",
            feature_columns=self.features,
            metadata_columns=["known_moa"],
            n_clusters=3,
            auto_k=False,
            bootstrap=False,
            random_state=2,
        )
        self.assertEqual(anchors.shape[0], 12)
        self.assertIn("pseudo_moa", anchors.columns)
        self.assertIn("n_compounds", summary.columns)
        self.assertIn("cluster", clusters.columns)
        self.assertFalse(k_selection.empty)

    def test_centroid_scoring_with_subcentroids(self) -> None:
        """Centroid scoring should produce top predictions and a score matrix."""
        emb, _, features, _ = prepare_embedding_matrix(
            table=self.table,
            id_column="cpd_id",
            feature_columns=self.features,
            metadata_columns=["known_moa"],
        )
        anchors = emb[["cpd_id", "known_moa"]].rename(columns={"known_moa": "moa"})
        centroids, centroid_summary = build_moa_centroids(
            embedding_table=emb,
            anchors=anchors,
            id_column="cpd_id",
            moa_column="moa",
            feature_columns=features,
            n_subcentroids=2,
            shrinkage=0.05,
            adaptive_shrinkage=True,
            random_state=3,
        )
        self.assertIn("centroid_id", centroids.columns)
        self.assertTrue((centroid_summary["n_members"] >= 1).any())
        long_scores, top = score_against_moa_centroids(
            embedding_table=emb,
            centroid_table=centroids,
            id_column="cpd_id",
            feature_columns=features,
            score_method="cosine",
            top_n=2,
        )
        self.assertEqual(top.shape[0], emb.shape[0])
        self.assertIn("top1_score_margin", top.columns)
        self.assertGreaterEqual(long_scores["rank"].max(), 2)
        matrix = score_matrix_table(
            embedding_table=emb,
            centroid_table=centroids,
            id_column="cpd_id",
            feature_columns=features,
        )
        self.assertIn("cpd_id", matrix.columns)
        self.assertGreater(matrix.shape[1], 2)

    def test_csls_and_anchor_permutation(self) -> None:
        """CSLS scoring and anchor permutation should be finite on small data."""
        emb, _, features, _ = prepare_embedding_matrix(
            table=self.table,
            id_column="cpd_id",
            feature_columns=self.features,
            metadata_columns=["known_moa"],
        )
        anchors = emb[["cpd_id", "known_moa"]].rename(columns={"known_moa": "moa"})
        centroids, _ = build_moa_centroids(
            embedding_table=emb,
            anchors=anchors,
            id_column="cpd_id",
            moa_column="moa",
            feature_columns=features,
            random_state=4,
        )
        _, top = score_against_moa_centroids(
            embedding_table=emb,
            centroid_table=centroids,
            id_column="cpd_id",
            feature_columns=features,
            score_method="csls",
            csls_k=2,
            top_n=1,
        )
        self.assertTrue(np.isfinite(top["moa_score"]).all())
        perm_summary, perm_null = anchor_permutation_test(
            embedding_table=emb,
            anchors=anchors,
            id_column="cpd_id",
            moa_column="moa",
            feature_columns=features,
            n_permutations=3,
            random_state=5,
        )
        self.assertEqual(perm_summary.shape[0], emb.shape[0])
        self.assertEqual(perm_null["permutation_index"].nunique(), 3)
        self.assertTrue(((perm_summary["empirical_p_value"] >= 0) & (perm_summary["empirical_p_value"] <= 1)).all())

    def test_pairwise_distance_outputs(self) -> None:
        """Pairwise distance helper should return matrices and neighbours."""
        emb, _, features, _ = prepare_embedding_matrix(
            table=self.table,
            id_column="cpd_id",
            feature_columns=self.features,
            metadata_columns=["known_moa"],
        )
        outputs = pairwise_distance_outputs(
            embedding_table=emb,
            id_column="cpd_id",
            feature_columns=features,
            metrics=["cosine", "spearman"],
            top_n=2,
        )
        self.assertIn("pairwise_distance_cosine", outputs)
        self.assertIn("nearest_neighbours_spearman", outputs)
        self.assertEqual(outputs["nearest_neighbours_cosine"].groupby("cpd_id").size().min(), 2)


class TestMoaCliV028(unittest.TestCase):
    """Smoke tests for the upgraded command-line MOA workflow."""

    def test_cli_generates_advanced_outputs(self) -> None:
        """The CLI should run pseudo-anchor and advanced MOA scoring."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "profiles.tsv"
            out_dir = tmp_path / "out"
            table = pd.DataFrame(
                {
                    "cpd_id": [f"C{i}" for i in range(10)],
                    "known_moa": ["A"] * 5 + ["B"] * 5,
                    "f0": [1, 1.1, 1.0, 0.9, 1.2, 0, 0.1, -0.1, 0.0, 0.2],
                    "f1": [0, 0.1, -0.1, 0.0, 0.2, 1, 1.1, 0.9, 1.0, 1.2],
                    "f2": np.linspace(0, 1, 10),
                }
            )
            table.to_csv(data_path, sep="\t", index=False)
            parser = moa_cli.build_parser()
            args = parser.parse_args(
                [
                    "--input_table",
                    str(data_path),
                    "--output_dir",
                    str(out_dir),
                    "--class_column",
                    "known_moa",
                    "--id_column",
                    "cpd_id",
                    "--make_pseudo_anchors",
                    "--pseudo_anchor_method",
                    "simple",
                    "--n_clusters",
                    "2",
                    "--n_permutations",
                    "2",
                    "--top_n",
                    "2",
                    "--disable_html_report",
                ]
            )
            original_parse_args = moa_cli.build_parser
            try:
                moa_cli.build_parser = lambda: parser
                # Patch parse_args to return our prepared namespace.
                parser.parse_args = lambda: args  # type: ignore[assignment]
                moa_cli.main()
            finally:
                moa_cli.build_parser = original_parse_args
            self.assertTrue((out_dir / "advanced_moa_top_predictions.tsv").exists())
            self.assertTrue((out_dir / "pseudo_anchors.tsv").exists())
            self.assertTrue((out_dir / "moa_summary.xlsx").exists())


if __name__ == "__main__":
    unittest.main()
