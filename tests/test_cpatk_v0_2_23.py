"""Regression tests for CPATK v0.2.23 protected features and CLIPn training policy."""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from cpatk.clipn_adapter import ClipnAdapterConfig, run_clipn_workflow, standardise_clipn_metadata
from cpatk.preprocessing import preprocess_profiles, remove_correlated_features
from cpatk.cli.preprocess import _load_protected_features


class TestProtectedFeatureFiltering(unittest.TestCase):
    """Tests for user-protected feature behaviour."""

    def test_protected_feature_survives_variance_qc_when_usable(self) -> None:
        """A usable requested feature should be retained despite low variance."""
        table = pd.DataFrame(
            {
                "Metadata_Plate": ["P1"] * 5,
                "Metadata_Well": ["A01", "A02", "A03", "A04", "A05"],
                "Metadata_Compound": ["a", "b", "c", "d", "e"],
                "low_variance_feature": [1.0, 1.0, 1.0, 1.0, 1.0001],
                "informative_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = preprocess_profiles(
            data_frame=table,
            metadata_columns=["Metadata_Plate", "Metadata_Well", "Metadata_Compound"],
            min_feature_variance=0.01,
            max_feature_missing_fraction=1.0,
            max_sample_missing_fraction=1.0,
            remove_correlated=False,
            protected_features=["low_variance_feature"],
        )
        retained = set(result["retained_features"]["feature"])
        self.assertIn("low_variance_feature", retained)
        audit = result["protected_feature_audit"]
        self.assertEqual(audit.loc[0, "final_status"], "protected_retained")
        selection = result["feature_selection_report"].set_index("feature")
        self.assertTrue(bool(selection.loc["low_variance_feature", "protected_feature"]))
        self.assertIn("protected", selection.loc["low_variance_feature", "selection_reason"])

    def test_protected_feature_survives_correlation_filter(self) -> None:
        """Protected features should not be removed as redundant correlation hits."""
        features = pd.DataFrame(
            {
                "protected_low_variance": [1, 2, 3, 4, 5],
                "unprotected_high_variance": [10, 20, 30, 40, 50],
                "independent": [1, 3, 2, 5, 4],
            }
        )
        filtered, report = remove_correlated_features(
            features=features,
            max_absolute_correlation=0.99,
            correlation_method="spearman",
            correlation_filter_strategy="variance",
            protected_features=["protected_low_variance"],
        )
        self.assertIn("protected_low_variance", filtered.columns)
        self.assertNotIn("unprotected_high_variance", filtered.columns)
        self.assertEqual(report.loc[0, "retained_feature"], "protected_low_variance")

    def test_protected_feature_loader_combines_inline_and_file(self) -> None:
        """Protected feature names can come from CLI text and a one-per-line file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protected.txt"
            path.write_text("# comment\nf2\n\nf3\n", encoding="utf-8")
            features = _load_protected_features(inline_features="f1,f2", feature_file=str(path))
        self.assertEqual(features, ["f1", "f2", "f3"])


class TestClipnTrainingPolicy(unittest.TestCase):
    """Tests for CLIPn training-control audit behaviour."""

    @staticmethod
    def _dataset(offset: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cpd_id": ["a", "b", "c", "d"],
                "cpd_type": ["class1", "class1", "class2", "class2"],
                "Plate_Metadata": ["P1"] * 4,
                "f1": [1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset],
                "f2": [2.0 + offset, 3.0 + offset, 4.0 + offset, 5.0 + offset],
            }
        )

    def test_clipn_early_stopping_policy_is_reported(self) -> None:
        """Chunked training-loss early stopping should write a summary table."""
        ref, _ = standardise_clipn_metadata(data_frame=self._dataset(), dataset_name="ref")
        query, _ = standardise_clipn_metadata(data_frame=self._dataset(1.0), dataset_name="query")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            module_path = tmp_path / "plateauclipn.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    import numpy as np

                    class CLIPn:
                        def __init__(self, X, y, latent_dim=2):
                            self.latent_dim = latent_dim
                        def fit(self, X, y, lr=1e-5, epochs=2):
                            return [1.0 for _ in range(epochs)]
                        def predict(self, X):
                            out = {}
                            for key, matrix in X.items():
                                arr = np.asarray(matrix, dtype=float)
                                out[key] = arr[:, :self.latent_dim]
                            return out
                    """
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, str(tmp_path))
            try:
                config = ClipnAdapterConfig(
                    backend_module="plateauclipn",
                    latent_dim=2,
                    epochs=20,
                    early_stopping=True,
                    early_stopping_patience=3,
                    early_stopping_chunk_size=2,
                    n_neighbours=2,
                )
                result = run_clipn_workflow(
                    datasets={"ref": ref, "query": query},
                    output_dir=tmp_path / "out",
                    config=config,
                )
                summary = result["clipn_training_summary"]
                self.assertEqual(summary.loc[0, "training_policy"], "chunked_training_loss_early_stopping")
                self.assertEqual(summary.loc[0, "stopping_reason"], "early_stopping_training_loss_plateau")
                self.assertLess(int(summary.loc[0, "reported_loss_rows"]), 20)
                self.assertIn("clipn_backend_provenance", result)
            finally:
                if str(tmp_path) in sys.path:
                    sys.path.remove(str(tmp_path))


if __name__ == "__main__":
    unittest.main()
