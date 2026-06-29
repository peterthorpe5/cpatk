"""Tests for CPATK v0.2.26 native contrastive backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.cli.clipn import build_parser
from cpatk.clipn_adapter import ClipnAdapterConfig, run_clipn_workflow
from cpatk.contrastive import (
    NativeContrastiveConfig,
    fit_native_contrastive_backend,
    sample_positive_batch,
    supervised_contrastive_loss,
)

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - only for broken environments
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for native contrastive tests.")
class TestNativeContrastiveBackend(unittest.TestCase):
    """Test the CPATK-native contrastive embedding backend."""

    def _make_cleaned_data(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Create two small repeated-compound profile matrices."""
        rng = np.random.default_rng(seed=123)
        rows = []
        cleaned = {}
        for dataset in ["reference", "query"]:
            features = []
            samples = []
            sample_id = 0
            for compound_index, compound in enumerate(["A", "B", "C"]):
                centre = np.zeros(6, dtype=float)
                centre[compound_index] = 3.0
                for _ in range(6):
                    values = centre + rng.normal(loc=0.0, scale=0.15, size=6)
                    features.append(values)
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Sample": sample_id,
                            "cpd_id": compound,
                            "cpd_type": "compound",
                            "Plate_Metadata": "P1",
                            "Well_Metadata": f"A{sample_id + 1:02d}",
                        }
                    )
                    samples.append(sample_id)
                    sample_id += 1
            cleaned[dataset] = pd.DataFrame(
                features,
                columns=[f"feature_{index}" for index in range(6)],
                index=samples,
            )
        metadata = pd.DataFrame.from_records(rows)
        return cleaned, metadata


    def test_default_backend_is_cpatk_contrastive(self) -> None:
        """Default latent backend should be CPATK-native, not published CLIPn."""
        config = ClipnAdapterConfig()
        self.assertEqual(config.backend_module, "cpatk_contrastive")
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "demo=/tmp/demo.tsv",
                "--output_dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.backend_module, "cpatk_contrastive")

    def test_supervised_contrastive_loss_is_positive(self) -> None:
        """Loss should be finite and positive for a valid positive-pair batch."""
        embeddings = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        loss = supervised_contrastive_loss(
            embeddings=embeddings,
            labels=labels,
            temperature=0.1,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_positive_batch_contains_repeated_labels(self) -> None:
        """The sampler should construct batches with repeated labels."""
        rng = np.random.default_rng(seed=1)
        labels = np.array([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int64)
        batch = sample_positive_batch(
            labels=labels,
            indices=np.arange(labels.shape[0]),
            batch_size=6,
            positives_per_label=2,
            rng=rng,
        )
        sampled = labels[batch]
        counts = pd.Series(sampled).value_counts()
        self.assertTrue((counts >= 2).any())

    def test_native_backend_returns_latent_and_training_tables(self) -> None:
        """Native backend should return latent, training and audit tables."""
        cleaned, metadata = self._make_cleaned_data()
        config = NativeContrastiveConfig(
            latent_dim=4,
            hidden_dims=[16],
            epochs=6,
            batch_size=12,
            validation_fraction=0.2,
            early_stopping_patience=4,
            eval_batches=1,
            random_state=7,
            positive_column="cpd_id",
            device="cpu",
        )
        result = fit_native_contrastive_backend(
            cleaned=cleaned,
            metadata=metadata,
            config=config,
        )
        self.assertEqual(result.latent_table.shape[0], metadata.shape[0])
        self.assertIn("latent_1", result.latent_table.columns)
        self.assertFalse(result.training_loss.empty)
        self.assertFalse(result.training_summary.empty)
        self.assertIn("nearest_neighbour_same_positive_label_rate", result.training_summary.columns)
        self.assertIn("usable_for_contrastive_training", result.positive_label_report.columns)

    def test_run_clipn_workflow_can_use_native_backend(self) -> None:
        """The existing latent CLI workflow should support cpatk_contrastive."""
        rng = np.random.default_rng(seed=99)
        datasets = {}
        for dataset in ["reference", "query"]:
            records = []
            for compound_index, compound in enumerate(["A", "B"]):
                centre = np.zeros(4)
                centre[compound_index] = 2.0
                for replicate in range(2):
                    values = centre + rng.normal(scale=0.1, size=4)
                    record = {
                        "cpd_id": compound,
                        "cpd_type": "compound",
                        "Plate_Metadata": "P1",
                        "Well_Metadata": f"B{replicate + 1:02d}",
                    }
                    record.update({f"feature_{idx}": values[idx] for idx in range(4)})
                    records.append(record)
            datasets[dataset] = pd.DataFrame.from_records(records)
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ClipnAdapterConfig(
                latent_dim=3,
                epochs=5,
                validation_fraction=0.2,
                early_stopping_patience=3,
                native_hidden_dims=[12],
                native_batch_size=4,
                native_eval_batches=1,
                native_steps_per_epoch=2,
                native_device="cpu",
                native_positive_column="cpd_id",
            )
            output = run_clipn_workflow(
                datasets=datasets,
                output_dir=Path(temp_dir),
                config=config,
            )
            self.assertIn("clipn_latent", output)
            self.assertIn("cpatk_contrastive_positive_label_report", output)
            loss_table = output["clipn_training_loss"]
            self.assertTrue((loss_table["n_train_steps"] == 2).all())
            self.assertTrue((Path(temp_dir) / "clipn_latent.tsv.gz").exists())
            self.assertTrue((Path(temp_dir) / "cpatk_contrastive_positive_label_report.tsv").exists())


if __name__ == "__main__":
    unittest.main()
