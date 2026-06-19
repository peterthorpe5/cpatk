"""Tests for CPATK v0.2.7 CLIPn workflow upgrades."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from cpatk.clipn_adapter import (
    ClipnAdapterConfig,
    align_dataset_features,
    clean_impute_and_scale_aligned,
    run_clipn_workflow,
    standardise_clipn_metadata,
)


class CpatkV027ClipnTests(unittest.TestCase):
    """Tests for the improved CLIPn adapter."""

    def _dataset(self, offset: float = 0.0) -> pd.DataFrame:
        """Create a small synthetic preprocessed Cell Painting table."""
        return pd.DataFrame(
            {
                "COMPOUND_NAME": ["DMSO", "A", "B", "C"],
                "MOA": ["control", "moa1", "moa1", "moa2"],
                "Plate": ["P1", "P1", "P1", "P1"],
                "Well": ["A1", "A2", "A3", "A4"],
                "Library": ["L", "L", "L", "L"],
                "ImageNumber": [1, 2, 3, 4],
                "feature_a": [1.0 + offset, 2.0 + offset, np.nan, 4.0 + offset],
                "feature_b": [2.0 + offset, 3.0 + offset, 4.0 + offset, np.inf],
                "feature_c": [5.0 + offset, 6.0 + offset, 7.0 + offset, 8.0 + offset],
            }
        )

    def test_standardise_clipn_metadata_from_aliases(self) -> None:
        """Alias metadata should be copied into canonical CLIPn fields."""
        table, report = standardise_clipn_metadata(
            data_frame=self._dataset(),
            dataset_name="reference",
        )
        self.assertIn("cpd_id", table.columns)
        self.assertIn("cpd_type", table.columns)
        self.assertIn("Plate_Metadata", table.columns)
        self.assertIn("Well_Metadata", table.columns)
        self.assertEqual(table.loc[0, "Well_Metadata"], "A01")
        self.assertIn("Dataset", table.columns)
        self.assertIn("Sample", table.columns)
        self.assertTrue((report["dataset"] == "reference").any())

    def test_align_dataset_features_excludes_metadata_and_uses_intersection(self) -> None:
        """Feature alignment should keep shared biological numeric features only."""
        ref, _ = standardise_clipn_metadata(data_frame=self._dataset(), dataset_name="ref")
        query, _ = standardise_clipn_metadata(data_frame=self._dataset(1.0), dataset_name="query")
        query = query.drop(columns=["feature_c"])
        aligned, summary, feature_report = align_dataset_features(
            datasets={"ref": ref, "query": query},
            return_feature_report=True,
        )
        self.assertIn("feature_a", aligned["ref"].columns)
        self.assertIn("feature_b", aligned["query"].columns)
        self.assertNotIn("ImageNumber", aligned["ref"].columns)
        self.assertNotIn("Sample", aligned["ref"].columns)
        self.assertEqual(summary["n_shared_features"].iloc[0], 2)
        self.assertIn("feature_c", feature_report["feature"].tolist())
        self.assertFalse(
            bool(feature_report.loc[feature_report["feature"] == "feature_c", "in_shared_intersection"].iloc[0])
        )

    def test_clean_impute_and_scale_aligned_removes_nonfinite_and_imputes(self) -> None:
        """CLIPn preprocessing should clean infinity, impute and scale shared features."""
        ref, _ = standardise_clipn_metadata(data_frame=self._dataset(), dataset_name="ref")
        query, _ = standardise_clipn_metadata(data_frame=self._dataset(1.0), dataset_name="query")
        aligned, _summary, _feature_report = align_dataset_features(
            datasets={"ref": ref, "query": query},
            return_feature_report=True,
        )
        config = ClipnAdapterConfig(imputation_method="median", scaling_method="robust")
        cleaned, preprocess_summary = clean_impute_and_scale_aligned(
            aligned=aligned,
            metadata={"ref": ref, "query": query},
            config=config,
        )
        for block in cleaned.values():
            self.assertFalse(block.isna().any().any())
            self.assertTrue(np.isfinite(block.to_numpy()).all())
        items = dict(zip(preprocess_summary["item"], preprocess_summary["value"]))
        self.assertGreaterEqual(int(items["missing_values_before_imputation"]), 1)
        self.assertEqual(int(items["missing_values_after_imputation"]), 0)

    def test_run_clipn_workflow_without_backend_writes_audit_report(self) -> None:
        """Unavailable CLIPn backends should not crash the audit workflow."""
        ref, _ = standardise_clipn_metadata(data_frame=self._dataset(), dataset_name="ref")
        query, _ = standardise_clipn_metadata(data_frame=self._dataset(1.0), dataset_name="query")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            config = ClipnAdapterConfig(
                backend_module="definitely_missing_clipn_backend_for_test",
                allow_pca_fallback=True,
                latent_dim=2,
                n_neighbours=2,
            )
            result = run_clipn_workflow(
                datasets={"ref": ref, "query": query},
                output_dir=out_dir,
                config=config,
            )
            self.assertIn("clipn_status", result)
            self.assertIn("clipn_latent", result)
            self.assertTrue((out_dir / "clipn_report.html").exists())
            self.assertTrue((out_dir / "clipn_summary.xlsx").exists())
            self.assertTrue((out_dir / "clipn_latent.tsv.gz").exists())

    def test_run_clipn_workflow_with_fake_backend(self) -> None:
        """A compatible backend should be fitted and projected into a latent table."""
        ref, _ = standardise_clipn_metadata(data_frame=self._dataset(), dataset_name="ref")
        query, _ = standardise_clipn_metadata(data_frame=self._dataset(1.0), dataset_name="query")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            module_path = tmp_path / "fakeclipn.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    import numpy as np

                    class CLIPn:
                        def __init__(self, X, y, latent_dim=2):
                            self.latent_dim = latent_dim
                        def fit(self, X, y, lr=1e-5, epochs=3):
                            return [3.0, 2.0, 1.0]
                        def predict(self, X):
                            out = {}
                            for key, matrix in X.items():
                                arr = np.asarray(matrix, dtype=float)
                                first = arr[:, :1]
                                second = arr[:, 1:2] if arr.shape[1] > 1 else first
                                out[key] = np.hstack([first, second])
                            return out
                    """
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, str(tmp_path))
            try:
                out_dir = tmp_path / "out"
                config = ClipnAdapterConfig(
                    backend_module="fakeclipn",
                    latent_dim=2,
                    epochs=3,
                    n_neighbours=2,
                )
                result = run_clipn_workflow(
                    datasets={"ref": ref, "query": query},
                    output_dir=out_dir,
                    config=config,
                )
                self.assertEqual(result["clipn_run_status"]["backend_run"].iloc[0], "success")
                self.assertIn("clipn_training_loss", result)
                self.assertIn("clipn_latent", result)
                self.assertIn("latent_1", result["clipn_latent"].columns)
                self.assertIn("nearest_neighbours", result)
            finally:
                if str(tmp_path) in sys.path:
                    sys.path.remove(str(tmp_path))


if __name__ == "__main__":
    unittest.main()
