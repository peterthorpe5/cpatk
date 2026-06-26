"""Regression tests for CPATK v0.2.17 CLIPn/report hardening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.clipn_adapter import (
    ClipnAdapterConfig,
    apply_clipn_zero_policy,
    clean_impute_and_scale_aligned,
)
from cpatk.reporting import make_html_report


class TestClipnZeroPolicy(unittest.TestCase):
    """Tests for CLIPn literal-zero handling."""

    def test_keep_policy_audits_but_keeps_literal_zeros(self) -> None:
        """The default policy should audit zeros but keep them for CLIPn."""
        table = pd.DataFrame(
            {
                "Dataset": ["a", "a", "b"],
                "Sample": [0, 1, 0],
                "f1": [0.0, 1.0, 2.0],
                "f2": [3.0, 0.0, 4.0],
            }
        )
        config = ClipnAdapterConfig(zero_policy="keep")
        cleaned, features, report = apply_clipn_zero_policy(
            table=table,
            feature_cols=["f1", "f2"],
            config=config,
        )
        self.assertEqual(features, ["f1", "f2"])
        self.assertEqual(int(cleaned[features].eq(0.0).sum().sum()), 2)
        values = dict(zip(report["item"], report["value"]))
        self.assertEqual(values["literal_zero_values_before_policy"], 2)
        self.assertEqual(values["literal_zero_values_after_policy"], 2)
        self.assertEqual(values["literal_zero_values_changed_by_policy"], 0)

    def test_drop_rows_policy_can_fail_loudly(self) -> None:
        """Strict row dropping should explain when it removes every sample."""
        table = pd.DataFrame(
            {
                "Dataset": ["a", "b"],
                "Sample": [0, 0],
                "f1": [0.0, 1.0],
                "f2": [2.0, 0.0],
            }
        )
        config = ClipnAdapterConfig(zero_policy="drop_rows")
        with self.assertRaisesRegex(ValueError, "removed every sample"):
            apply_clipn_zero_policy(
                table=table,
                feature_cols=["f1", "f2"],
                config=config,
            )

    def test_clean_impute_scale_applies_zero_policy_after_imputation(self) -> None:
        """Zeros and missing values are handled after the alignment/imputation step."""
        aligned = {
            "a": pd.DataFrame({"f1": [0.0, 1.0], "f2": [float("nan"), 2.0]}),
            "b": pd.DataFrame({"f1": [3.0, 4.0], "f2": [0.0, 5.0]}),
        }
        metadata = {
            "a": pd.DataFrame({"Metadata_Compound": ["x", "x"]}),
            "b": pd.DataFrame({"Metadata_Compound": ["y", "y"]}),
        }
        config = ClipnAdapterConfig(
            imputation_method="median",
            scaling_method="none",
            zero_policy="keep",
        )
        cleaned, summary = clean_impute_and_scale_aligned(
            aligned=aligned,
            metadata=metadata,
            config=config,
        )
        combined = pd.concat(cleaned.values(), ignore_index=True)
        self.assertGreaterEqual(int(combined.eq(0.0).sum().sum()), 1)
        summary_values = dict(zip(summary["item"], summary["value"]))
        self.assertEqual(summary_values["zero_policy"], "keep")
        self.assertGreaterEqual(int(summary_values["literal_zero_values_after_policy"]), 1)


class TestReportPolishing(unittest.TestCase):
    """Tests for report wording and interpretability additions."""

    def test_report_uses_summary_not_executive_summary(self) -> None:
        """Generated reports should avoid the unwanted executive-summary wording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.html"
            make_html_report(
                title="CPATK test report",
                output_path=output,
                summary_tables={
                    "Metadata validation summary": pd.DataFrame(
                        {"item": ["n_rows"], "value": [3]}
                    )
                },
                narrative="Plain summary text.",
            )
            html_text = output.read_text(encoding="utf-8")
        self.assertIn("<h2>Summary</h2>", html_text)
        self.assertNotIn("Executive summary", html_text)
        self.assertIn("How to read this report", html_text)
        self.assertIn("Checks that assay plate/well metadata", html_text)


if __name__ == "__main__":
    unittest.main()

class TestQueryBackgroundShapCli(unittest.TestCase):
    """Tests for query-vs-control explanation outputs."""

    def test_query_background_shap_writes_per_query_report(self) -> None:
        """cpatk-explain should report a query-vs-DMSO explanation."""
        from cpatk.cli.explain import main as explain_main
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            profiles = pd.DataFrame(
                {
                    "Metadata_Compound": ["DrugA", "DrugA", "DMSO", "DMSO"],
                    "cpd_type": ["compound", "compound", "DMSO", "DMSO"],
                    "f1": [5.0, 6.0, 0.0, 0.0],
                    "f2": [1.0, 1.2, 3.0, 3.1],
                    "f3": [2.0, 2.1, 2.0, 2.1],
                }
            )
            input_table = temp_path / "profiles.tsv"
            profiles.to_csv(input_table, sep="\t", index=False)
            output_dir = temp_path / "explain"
            argv = [
                "cpatk-explain",
                "--input_table",
                str(input_table),
                "--output_dir",
                str(output_dir),
                "--metadata_columns",
                "Metadata_Compound,cpd_type",
                "--id_column",
                "Metadata_Compound",
                "--query_ids",
                "DrugA",
                "--run_query_background_shap",
                "--background_column",
                "Metadata_Compound",
                "--background_values",
                "DMSO",
                "--n_top_features",
                "2",
                "--disable_html_report",
            ]
            with patch("sys.argv", argv):
                explain_main()
            query_dir = output_dir / "query_neighbourhoods" / "DrugA"
            self.assertTrue((query_dir / "query_vs_background_top_shap_features.tsv").exists())
            self.assertTrue((query_dir / "query_vs_background_shap_status.tsv").exists())
            self.assertTrue((query_dir / "query_explanation_report.html").exists())
