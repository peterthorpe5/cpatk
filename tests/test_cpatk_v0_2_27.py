"""Tests for CPATK v0.2.27 thread-control additions."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.batch import cross_validated_batch_prediction
from cpatk.cli.clipn import build_parser as build_latent_parser
from cpatk.clipn_adapter import ClipnAdapterConfig
from cpatk.contrastive import NativeContrastiveConfig, fit_native_contrastive_backend
from cpatk.distances import calculate_pairwise_distance_matrix
from cpatk.ml import build_classifier, cross_validate_classifier
from cpatk.threading_utils import configure_threading, normalise_thread_count


class TestThreadControls(unittest.TestCase):
    """Test safe multi-threading controls added in v0.2.27."""

    def test_thread_count_is_normalised(self) -> None:
        """Invalid or small thread counts should resolve to a positive value."""
        self.assertEqual(normalise_thread_count(value=0, default=2), 1)
        self.assertEqual(normalise_thread_count(value="bad", default=3), 3)
        self.assertEqual(normalise_thread_count(value="4", default=1), 4)

    def test_configure_threading_sets_environment(self) -> None:
        """Thread configuration should set common native-library variables."""
        threads = configure_threading(n_threads=2, use_threadpoolctl=False)
        self.assertEqual(threads, 2)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
        self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")

    def test_pairwise_distances_accept_threads(self) -> None:
        """Pairwise distance calculation should accept an n_jobs argument."""
        features = pd.DataFrame(
            {
                "a": [0.0, 1.0, 2.0],
                "b": [1.0, 1.0, 0.0],
            }
        )
        distances = calculate_pairwise_distance_matrix(
            features=features,
            metric="euclidean",
            n_jobs=2,
        )
        self.assertEqual(distances.shape, (3, 3))
        self.assertTrue(np.isclose(float(distances.iloc[0, 0]), 0.0))

    def test_estimators_accept_n_jobs(self) -> None:
        """Supported tree estimators should receive the requested n_jobs."""
        model = build_classifier(model_name="random_forest", n_jobs=2)
        self.assertEqual(model.n_jobs, 2)
        model = build_classifier(model_name="extra_trees", n_jobs=3)
        self.assertEqual(model.n_jobs, 3)

    def test_cli_and_configs_include_threads(self) -> None:
        """Latent CLI and config objects should expose thread settings."""
        args = build_latent_parser().parse_args(
            [
                "--dataset",
                "a=/tmp/a.tsv",
                "--output_dir",
                "/tmp/out",
                "--threads",
                "4",
            ]
        )
        self.assertEqual(args.threads, 4)
        self.assertEqual(ClipnAdapterConfig(n_threads=5).n_threads, 5)
        self.assertEqual(NativeContrastiveConfig(n_threads=6).n_threads, 6)

    def test_cross_validation_thread_arguments_run(self) -> None:
        """ML and batch diagnostic functions should run with n_jobs > 1."""
        features = pd.DataFrame(
            {
                "f1": [0, 0.1, 0.2, 5, 5.1, 5.2],
                "f2": [0, 0.2, 0.1, 4, 4.2, 4.1],
            }
        )
        labels = pd.Series(["a", "a", "a", "b", "b", "b"])
        summary, predictions, confusion = cross_validate_classifier(
            features=features,
            labels=labels,
            n_splits=2,
            n_jobs=2,
        )
        self.assertEqual(summary["n_splits"].iloc[0], 2)
        self.assertEqual(predictions.shape[0], 6)
        self.assertIn("true_class", confusion.columns)
        metadata = pd.DataFrame({"batch": labels})
        batch_summary = cross_validated_batch_prediction(
            features=features,
            metadata=metadata,
            batch_column="batch",
            n_splits=2,
            n_jobs=2,
        )
        self.assertEqual(batch_summary["status"].iloc[0], "tested")

    def test_native_contrastive_records_threads(self) -> None:
        """Native contrastive training should record the configured thread count."""
        cleaned = {
            "dataset_a": pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.1],
                        [0.1, 0.0],
                        [2.0, 2.1],
                        [2.1, 2.0],
                    ],
                    dtype=float,
                )
            ),
        }
        metadata = pd.DataFrame(
            {
                "Dataset": ["dataset_a"] * 4,
                "Sample": [0, 1, 2, 3],
                "cpd_id": ["x", "x", "y", "y"],
            }
        )
        result = fit_native_contrastive_backend(
            cleaned=cleaned,
            metadata=metadata,
            config=NativeContrastiveConfig(
                latent_dim=2,
                hidden_dims=[8],
                epochs=1,
                batch_size=4,
                validation_fraction=0.0,
                n_threads=2,
            ),
        )
        self.assertEqual(int(result.training_summary["n_threads"].iloc[0]), 2)
        self.assertEqual(result.latent_table.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
