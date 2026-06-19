"""Unit tests for CPATK v0.2.1 preprocessing upgrades."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.metadata import (
    canonicalise_well_value,
    drop_unnamed_index_columns,
    normalise_column_names,
    standardise_metadata_aliases,
)
from cpatk.plotting import (
    plot_feature_family_summary,
    plot_feature_qc_status,
    plot_feature_variance_histogram,
    plot_missingness_histogram,
    plot_preprocessing_retention,
)
from cpatk.preprocessing import (
    add_missingness_indicators,
    impute_features,
    preprocess_profiles,
    summarise_feature_families,
    summarise_imputation,
)
from cpatk.reporting import make_html_report


class CpatkV021Tests(unittest.TestCase):
    """Tests for preprocessing robustness and reporting additions."""

    def make_messy_table(self) -> pd.DataFrame:
        """Return a small mixed metadata/feature table."""
        return pd.DataFrame(
            {
                "\ufeffPlate_Metadata ": ["P1", "P1", "P2", "P2"],
                "Well_Metadata": ["A1", "A02", "B1", "B02"],
                "cpd_id": ["A", "A", "B", "B"],
                "cpd_type": ["known", "known", "novel", "novel"],
                "Unnamed: 0": [0, 1, 2, 3],
                "AreaShape_Area_Cell": [1.0, np.nan, 2.0, 2.2],
                "Intensity_MeanIntensity_DAPI": [0.5, 0.6, np.nan, 0.9],
                "Texture_Contrast_DAPI": [3.0, 3.1, 4.0, 4.2],
                "constant_feature": [1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_normalise_column_names_removes_bom_and_spaces(self) -> None:
        """Column-name cleanup should remove BOM characters and spaces."""
        table, report = normalise_column_names(data_frame=self.make_messy_table())
        self.assertIn("Plate_Metadata", table.columns)
        self.assertTrue(report.loc[0, "changed"])

    def test_canonicalise_well_value(self) -> None:
        """Well names should be normalised where possible."""
        self.assertEqual(canonicalise_well_value(value="A1"), "A01")
        self.assertEqual(canonicalise_well_value(value="b02"), "B02")

    def test_metadata_aliases_are_created(self) -> None:
        """Common metadata aliases should be standardised."""
        table, _ = normalise_column_names(data_frame=self.make_messy_table())
        standardised, report = standardise_metadata_aliases(data_frame=table)
        self.assertIn("Metadata_Plate", standardised.columns)
        self.assertIn("Metadata_Well", standardised.columns)
        self.assertEqual(standardised.loc[0, "Metadata_Well"], "A01")
        self.assertIn("created", set(report["action"]))

    def test_drop_unnamed_index_columns(self) -> None:
        """Likely accidental index columns should be removable."""
        table, report = drop_unnamed_index_columns(data_frame=self.make_messy_table())
        self.assertNotIn("Unnamed: 0", table.columns)
        self.assertTrue(report.loc[0, "dropped"])

    def test_group_median_imputation(self) -> None:
        """Group-median imputation should use metadata groups."""
        features = pd.DataFrame({"x": [1.0, np.nan, 10.0, np.nan]})
        metadata = pd.DataFrame({"plate": ["A", "A", "B", "B"]})
        imputed = impute_features(
            features=features,
            method="group_median",
            metadata=metadata,
            group_columns=["plate"],
        )
        self.assertEqual(float(imputed.loc[1, "x"]), 1.0)
        self.assertEqual(float(imputed.loc[3, "x"]), 10.0)

    def test_missingness_indicators(self) -> None:
        """Missingness indicators should be generated for missing features."""
        features = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, 3.0]})
        indicators, report = add_missingness_indicators(features=features)
        self.assertEqual(indicators.shape[1], 1)
        self.assertEqual(report.loc[0, "source_feature"], "x")

    def test_summarise_imputation(self) -> None:
        """Imputation reporting should compare before and after matrices."""
        before = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        after = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        report = summarise_imputation(before=before, after=after, method="median")
        self.assertEqual(int(report.loc[0, "n_missing_before"]), 1)
        self.assertEqual(int(report.loc[0, "n_missing_after"]), 0)

    def test_preprocess_profiles_v021_outputs(self) -> None:
        """Preprocessing should return the expanded v0.2.1 report tables."""
        result = preprocess_profiles(
            data_frame=self.make_messy_table(),
            imputation_method="median",
            add_missing_indicators=True,
            max_feature_missing_fraction=0.6,
            max_absolute_correlation=0.999,
        )
        expected = {
            "preprocessed",
            "imputation_report",
            "missingness_indicator_report",
            "metadata_alias_report",
            "column_name_report",
            "dropped_index_column_report",
            "feature_family_summary",
        }
        self.assertTrue(expected.issubset(result.keys()))
        self.assertIn("Metadata_Well", result["preprocessed"].columns)

    def test_feature_family_summary(self) -> None:
        """Feature family summaries should classify common CP features."""
        summary = summarise_feature_families(
            feature_names=["Intensity_MeanIntensity_DAPI", "Texture_Contrast_DAPI", "AreaShape_Area_Cell"]
        )
        self.assertIn("Intensity", set(summary["feature_family"]))
        self.assertIn("Texture", set(summary["feature_family"]))

    def test_preprocessing_plots_and_report(self) -> None:
        """Preprocessing plots and HTML report should be writable."""
        result = preprocess_profiles(data_frame=self.make_messy_table(), max_feature_missing_fraction=0.6)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            paths = []
            paths.extend(
                plot_missingness_histogram(
                    qc_table=result["feature_qc"],
                    missing_column="missing_fraction",
                    output_path_base=out / "feature_missingness",
                    title="Feature missingness",
                )
            )
            paths.extend(
                plot_feature_variance_histogram(
                    feature_qc=result["feature_qc"],
                    output_path_base=out / "feature_variance",
                )
            )
            paths.extend(
                plot_feature_qc_status(
                    feature_qc=result["feature_qc"],
                    output_path_base=out / "feature_status",
                )
            )
            paths.extend(
                plot_preprocessing_retention(
                    summary=result["preprocessing_summary"],
                    output_path_base=out / "retention",
                )
            )
            paths.extend(
                plot_feature_family_summary(
                    feature_family_summary=result["feature_family_summary"],
                    output_path_base=out / "families",
                )
            )
            report = make_html_report(
                title="Test preprocessing report",
                output_path=out / "report.html",
                summary_tables={"summary": result["preprocessing_summary"]},
                plot_paths=paths,
            )
            self.assertTrue(report.exists())
            self.assertGreater(len(paths), 0)


if __name__ == "__main__":
    unittest.main()
