"""Regression tests for CPATK v0.2.22 correlation filtering."""

from __future__ import annotations

import unittest

import pandas as pd

from cpatk.preprocessing import preprocess_profiles, remove_correlated_features


class CpatkV022CorrelationFilteringTests(unittest.TestCase):
    """Tests for Spearman/variance-prioritised correlation filtering."""

    def test_spearman_correlation_filters_monotonic_nonlinear_features(self) -> None:
        """Spearman should catch monotonic redundancy missed by strict Pearson."""
        features = pd.DataFrame(
            {
                "linear": [1, 2, 3, 4, 5],
                "quadratic": [1, 4, 9, 16, 25],
                "unrelated": [5, 1, 4, 2, 3],
            }
        )
        filtered, report = remove_correlated_features(
            features=features,
            max_absolute_correlation=0.999,
            correlation_method="spearman",
            correlation_filter_strategy="variance",
        )
        self.assertEqual(filtered.shape[1], 2)
        self.assertEqual(report.loc[0, "correlation_method"], "spearman")
        self.assertEqual(report.loc[0, "correlation_filter_strategy"], "variance")

    def test_variance_strategy_keeps_highest_variance_representative(self) -> None:
        """The default strategy should keep the most variable correlated feature."""
        features = pd.DataFrame(
            {
                "low_variance": [1, 2, 3, 4, 5],
                "high_variance": [10, 20, 30, 40, 50],
                "independent": [1, 3, 2, 5, 4],
            }
        )
        filtered, report = remove_correlated_features(
            features=features,
            max_absolute_correlation=0.99,
            correlation_method="spearman",
            correlation_filter_strategy="variance",
        )
        self.assertIn("high_variance", filtered.columns)
        self.assertNotIn("low_variance", filtered.columns)
        self.assertEqual(report.loc[0, "retained_feature"], "high_variance")
        self.assertGreater(report.loc[0, "retained_variance"], report.loc[0, "removed_variance"])

    def test_preprocess_records_correlation_method_and_strategy(self) -> None:
        """Preprocessing outputs should audit the correlation method and strategy."""
        table = pd.DataFrame(
            {
                "Metadata_Plate": ["P1"] * 5,
                "Metadata_Well": ["A01", "A02", "A03", "A04", "A05"],
                "Metadata_Compound": ["a", "b", "c", "d", "e"],
                "linear": [1, 2, 3, 4, 5],
                "quadratic": [1, 4, 9, 16, 25],
                "independent": [5, 1, 4, 2, 3],
            }
        )
        result = preprocess_profiles(
            data_frame=table,
            metadata_columns=["Metadata_Plate", "Metadata_Well", "Metadata_Compound"],
            max_absolute_correlation=0.999,
            min_feature_variance=0.0,
            max_feature_missing_fraction=1.0,
            max_sample_missing_fraction=1.0,
            correlation_method="spearman",
            correlation_filter_strategy="variance",
        )
        config = result["preprocessing_config"]
        config_values = dict(zip(config["parameter"], config["value"]))
        self.assertEqual(config_values["correlation_method"], "spearman")
        self.assertEqual(config_values["correlation_filter_strategy"], "variance")


if __name__ == "__main__":
    unittest.main()
