"""Tests for CPATK v0.2.30 synthetic benchmark generalisation metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpatk.synthetic_latent import (
    LatentBenchmarkConfig,
    SyntheticCellPaintingConfig,
    calculate_validation_to_train_retrieval_metrics,
    generate_synthetic_cell_painting_profiles,
    run_synthetic_latent_benchmark,
)

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class SyntheticGeneralisationMetricTests(unittest.TestCase):
    """Validate held-out retrieval metrics for the synthetic benchmark."""

    def test_validation_to_train_metrics_are_reported(self) -> None:
        """Validation-to-train retrieval should avoid all-row self-neighbour bias."""
        config = SyntheticCellPaintingConfig(
            n_compounds=6,
            n_moa_classes=3,
            n_batches=2,
            n_datasets=2,
            replicates_per_compound_dataset=4,
            n_features=20,
            n_informative_features=8,
            random_state=23,
        )
        profiles, _, _ = generate_synthetic_cell_painting_profiles(config=config)
        metrics, neighbours, split_report = calculate_validation_to_train_retrieval_metrics(
            embedding=profiles,
            validation_fraction=0.25,
            random_state=23,
            n_neighbours=2,
            method_name="raw_scaled_features",
        )
        observed_metrics = set(metrics["metric"].astype(str))
        self.assertIn("validation_to_train_top1_same_compound_rate", observed_metrics)
        self.assertIn("validation_to_train_n_validation_rows", observed_metrics)
        self.assertFalse(neighbours.empty)
        self.assertFalse(split_report.empty)
        self.assertTrue(set(neighbours["query_split"].astype(str)).issubset({"validation"}))
        self.assertTrue(set(neighbours["neighbour_split"].astype(str)).issubset({"train"}))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
    def test_negative_control_uses_validation_to_train_pass_fail(self) -> None:
        """The synthetic score table should include held-out pass/fail checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "benchmark"
            config = LatentBenchmarkConfig(
                output_dir=output_dir,
                scenarios=["no_biology_negative_control"],
                n_compounds=8,
                n_moa_classes=2,
                n_batches=2,
                n_datasets=2,
                replicates_per_compound_dataset=4,
                n_features=24,
                n_informative_features=10,
                latent_dim=4,
                epochs=3,
                batch_size=32,
                steps_per_epoch=1,
                validation_fraction=0.25,
                hidden_dims=[16],
                n_neighbours=3,
                threads=1,
                random_state=19,
            )
            outputs = run_synthetic_latent_benchmark(config=config)
            metrics = outputs["synthetic_metric_summary"]
            checks = outputs["synthetic_pass_fail_summary"]
            self.assertIn(
                "validation_to_train_top1_same_compound_rate",
                set(metrics["metric"].astype(str)),
            )
            self.assertIn(
                "native_validation_to_train_top1_same_compound_rate",
                set(checks["check"].astype(str)),
            )


if __name__ == "__main__":
    unittest.main()
