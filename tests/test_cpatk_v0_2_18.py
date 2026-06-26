"""Regression tests for CPATK v0.2.18 review hardening."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cpatk.cli.explain import main as explain_main
from cpatk.clipn_adapter import ClipnAdapterConfig, apply_clipn_zero_policy
from cpatk.logging_utils import configure_logging


class TestClipnPolicyReview(unittest.TestCase):
    """Tests for clarified CLIPn missing/zero behaviour."""

    def test_default_clipn_zero_policy_keeps_real_zeros(self) -> None:
        """The reviewed default keeps real zeros and records an audit."""
        table = pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 0.0]})
        cleaned, features, report = apply_clipn_zero_policy(
            table=table,
            feature_cols=["f1", "f2"],
            config=ClipnAdapterConfig(zero_policy="keep"),
        )
        values = dict(zip(report["item"], report["value"]))
        self.assertEqual(features, ["f1", "f2"])
        self.assertEqual(int(cleaned.eq(0.0).sum().sum()), 2)
        self.assertEqual(values["literal_zero_values_changed_by_policy"], 0)
        self.assertEqual(values["literal_zero_values_after_policy"], 2)


class TestLoggingHandlerCleanup(unittest.TestCase):
    """Tests for repeated logger setup without leaking file handlers."""

    def test_configure_logging_closes_previous_file_handlers(self) -> None:
        """Reconfiguring the CPATK logger should close old file handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            first_log = Path(temp_dir) / "first.log"
            second_log = Path(temp_dir) / "second.log"
            logger = configure_logging(log_file=first_log, logger_name="cpatk_test_cleanup")
            old_handlers = list(logger.handlers)
            self.assertTrue(any(isinstance(h, logging.FileHandler) for h in old_handlers))
            logger = configure_logging(log_file=second_log, logger_name="cpatk_test_cleanup")
            for handler in old_handlers:
                if isinstance(handler, logging.FileHandler):
                    self.assertTrue(handler.stream is None or handler.stream.closed)
            logger.info("new logger works")
            self.assertTrue(second_log.exists())


class TestQueryBackgroundSummary(unittest.TestCase):
    """Tests for clearer query-vs-control reporting metadata."""

    def test_query_background_summary_records_requested_contrast(self) -> None:
        """The explanation summary should state that query-vs-background was requested."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            profiles = pd.DataFrame(
                {
                    "Metadata_Compound": ["DrugA", "DrugA", "DMSO", "DMSO"],
                    "f1": [5.0, 6.0, 0.0, 0.0],
                    "f2": [1.0, 1.2, 3.0, 3.1],
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
                "Metadata_Compound",
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
            summary = pd.read_csv(
                output_dir / "query_neighbourhoods" / "query_neighbourhood_summary.tsv",
                sep="\t",
            )
        self.assertTrue(bool(summary.loc[0, "query_background_shap_requested"]))
        self.assertEqual(summary.loc[0, "background_column"], "Metadata_Compound")
        self.assertEqual(summary.loc[0, "background_values"], "DMSO")


if __name__ == "__main__":
    unittest.main()
