"""Regression tests for CPATK v0.2.32 latent validation updates."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cpatk.clipn_adapter import (
    calculate_compound_holdout_embedding_metrics,
    diagnose_feature_alignment_quality,
)
from cpatk.synthetic_latent import score_synthetic_scenario


class TestCompoundHoldoutMetrics(unittest.TestCase):
    """Tests for whole-compound holdout validation diagnostics."""

    def test_holdout_internal_retrieval_is_reported(self) -> None:
        """Held-out replicate cohesion metrics are calculated from latent rows."""
        latent = pd.DataFrame(
            {
                "latent_1": [1.0, 0.95, -1.0, -0.95, 0.0, 0.1],
                "latent_2": [0.0, 0.05, 0.0, -0.05, 1.0, 0.9],
                "cpd_id": ["A", "A", "B", "B", "C", "D"],
                "cpd_type": ["M1", "M1", "M2", "M2", "M3", "M4"],
                "Dataset": ["d1", "d2", "d1", "d2", "d1", "d1"],
                "Metadata_Plate": ["p1", "p2", "p1", "p2", "p1", "p1"],
            }
        )
        summary, neighbours = calculate_compound_holdout_embedding_metrics(
            latent_table=latent,
            holdout_values=["A", "B"],
            holdout_column="cpd_id",
            label_column="cpd_type",
            threads=1,
            repeat_index=1,
        )
        metrics = dict(zip(summary["metric"], summary["value"]))
        self.assertEqual(metrics["n_holdout_groups"], 2.0)
        self.assertEqual(metrics["n_holdout_rows"], 4.0)
        self.assertGreaterEqual(
            metrics["heldout_internal_top1_same_compound_rate"],
            0.5,
        )
        self.assertFalse(neighbours.empty)
        self.assertIn("comparison", neighbours.columns)

    def test_missing_holdout_column_reports_status(self) -> None:
        """A missing holdout column produces a status table rather than crashing."""
        latent = pd.DataFrame({"latent_1": [0.0, 1.0], "latent_2": [1.0, 0.0]})
        summary, neighbours = calculate_compound_holdout_embedding_metrics(
            latent_table=latent,
            holdout_values=["A"],
            holdout_column="cpd_id",
            label_column="cpd_type",
        )
        self.assertTrue(neighbours.empty)
        self.assertEqual(summary.loc[0, "metric"], "compound_holdout_status")


class TestFeatureAlignmentWarnings(unittest.TestCase):
    """Tests for feature-block/missing-compartment warnings."""

    def test_low_shared_feature_fraction_warns(self) -> None:
        """A small shared intersection relative to the union is warned."""
        summary = pd.DataFrame(
            {
                "dataset": ["d1", "d2"],
                "n_candidate_features": [100, 100],
                "n_shared_features": [50, 50],
                "n_missing_from_union": [50, 50],
            }
        )
        warnings = diagnose_feature_alignment_quality(feature_summary=summary)
        self.assertIn("warning", set(warnings["severity"]))
        self.assertIn(
            "low_shared_feature_fraction",
            set(warnings["diagnostic"]),
        )

    def test_complete_alignment_is_info(self) -> None:
        """Complete shared-feature alignment does not trigger warnings."""
        summary = pd.DataFrame(
            {
                "dataset": ["d1", "d2"],
                "n_candidate_features": [100, 100],
                "n_shared_features": [100, 100],
                "n_missing_from_union": [0, 0],
            }
        )
        warnings = diagnose_feature_alignment_quality(feature_summary=summary)
        self.assertEqual(set(warnings["severity"]), {"info"})


class TestSyntheticScoringMessages(unittest.TestCase):
    """Tests for clearer synthetic benchmark decision wording."""

    def test_no_biology_high_batch_message_is_not_biology_present(self) -> None:
        """Negative-control wording should not call the scenario biology-present."""
        metrics = pd.DataFrame(
            [
                {
                    "method": "cpatk_contrastive",
                    "metric": "validation_to_train_top1_same_compound_rate",
                    "value": 0.02,
                },
                {
                    "method": "cpatk_contrastive",
                    "metric": "validation_to_train_top1_same_dataset_rate",
                    "value": 0.95,
                },
                {
                    "method": "cpatk_contrastive",
                    "metric": "validation_to_train_top1_same_batch_rate",
                    "value": 0.98,
                },
                {
                    "method": "cpatk_contrastive",
                    "metric": "top1_same_compound_rate",
                    "value": 0.10,
                },
            ]
        )
        result = score_synthetic_scenario(
            metrics=metrics,
            scenario_name="no_biology_high_batch_negative_control",
        )
        joined_messages = " ".join(result["message"].astype(str))
        self.assertIn("No-biology", joined_messages)
        self.assertNotIn("Biology-present scenarios", joined_messages)


if __name__ == "__main__":
    unittest.main()
