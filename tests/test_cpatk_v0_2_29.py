"""Tests for CPATK v0.2.29 synthetic latent benchmarking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cpatk.synthetic_latent import (
    LatentBenchmarkConfig,
    SyntheticCellPaintingConfig,
    calculate_embedding_retrieval_metrics,
    generate_synthetic_cell_painting_profiles,
    infer_synthetic_feature_columns,
    run_synthetic_latent_benchmark,
)

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class SyntheticLatentBenchmarkTests(unittest.TestCase):
    """Validate synthetic benchmark generation and execution."""

    def test_generate_synthetic_profiles_has_expected_shape_and_metadata(self) -> None:
        """Synthetic generator should create feature and metadata columns."""
        config = SyntheticCellPaintingConfig(
            n_compounds=6,
            n_moa_classes=3,
            n_batches=2,
            n_datasets=2,
            replicates_per_compound_dataset=2,
            n_features=20,
            n_informative_features=8,
            random_state=11,
        )
        profiles, truth, scenario = generate_synthetic_cell_painting_profiles(
            config=config,
        )
        self.assertEqual(profiles.shape[0], 24)
        self.assertEqual(truth.shape[0], 24)
        self.assertIn("cpd_id", profiles.columns)
        self.assertIn("cpd_type", profiles.columns)
        self.assertIn("Metadata_Plate", profiles.columns)
        feature_columns = infer_synthetic_feature_columns(profiles=profiles)
        self.assertEqual(len(feature_columns), 20)
        self.assertTrue(np.isfinite(profiles[feature_columns].dropna().to_numpy()).all())
        self.assertIn("n_profiles", set(scenario["item"].astype(str)))

    def test_retrieval_metrics_report_expected_columns(self) -> None:
        """Embedding retrieval metrics should include compound and batch rates."""
        config = SyntheticCellPaintingConfig(
            n_compounds=5,
            n_moa_classes=2,
            n_batches=2,
            n_datasets=2,
            replicates_per_compound_dataset=2,
            n_features=12,
            n_informative_features=6,
            random_state=7,
        )
        profiles, _, _ = generate_synthetic_cell_painting_profiles(config=config)
        metrics, neighbours = calculate_embedding_retrieval_metrics(
            embedding=profiles,
            n_neighbours=3,
            method_name="raw_scaled_features",
        )
        self.assertFalse(metrics.empty)
        self.assertFalse(neighbours.empty)
        self.assertIn("top1_same_compound_rate", set(metrics["metric"].astype(str)))
        self.assertIn("top1_same_batch_rate", set(metrics["metric"].astype(str)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
    def test_small_synthetic_benchmark_runs_native_contrastive(self) -> None:
        """A small synthetic benchmark should run the native backend and write outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "benchmark"
            config = LatentBenchmarkConfig(
                output_dir=output_dir,
                scenarios=["clean_biology"],
                n_compounds=8,
                n_moa_classes=2,
                n_batches=2,
                n_datasets=2,
                replicates_per_compound_dataset=2,
                n_features=24,
                n_informative_features=10,
                latent_dim=4,
                epochs=3,
                batch_size=16,
                steps_per_epoch=1,
                validation_fraction=0.0,
                hidden_dims=[16],
                n_neighbours=3,
                threads=1,
                random_state=17,
            )
            outputs = run_synthetic_latent_benchmark(config=config)
            self.assertIn("synthetic_metric_summary", outputs)
            metrics = outputs["synthetic_metric_summary"]
            self.assertIn("cpatk_contrastive", set(metrics["method"].astype(str)))
            self.assertTrue((output_dir / "clean_biology" / "cpatk_contrastive_latent.tsv.gz").exists())
            self.assertTrue((output_dir / "synthetic_pass_fail_summary.tsv").exists())


if __name__ == "__main__":
    unittest.main()
