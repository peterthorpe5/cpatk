"""Regression tests for CPATK v0.2.21 reporting polish."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cpatk.cli.explain import main as explain_main
from cpatk.cli.preprocess import _write_result_tables
from cpatk.cli.report import main as report_main
from cpatk.clipn_adapter import collect_clipn_backend_provenance
from cpatk.method_guidance import load_ml_nn_method_guide, ml_nn_method_guide_html
from cpatk.preprocessing import _summarise_stage_batch_association
from cpatk.strategy_selection import summarise_preprocessing_strategies


class TestMethodGuide(unittest.TestCase):
    """Tests for the bundled ML/NN method guide."""

    def test_bundled_method_guide_loads_and_mentions_core_methods(self) -> None:
        """The package should always ship a reusable method guide."""
        guide = load_ml_nn_method_guide()
        self.assertIn("method", guide.columns)
        self.assertIn("best_use_case", guide.columns)
        methods = set(guide["method"].astype(str))
        self.assertIn("Nearest neighbours", methods)
        self.assertIn("CLIPn", methods)
        html = ml_nn_method_guide_html()
        self.assertIn("ML and nearest-neighbour method guide", html)
        self.assertIn("CLIPn", html)


class TestStrategySelection(unittest.TestCase):
    """Tests for normalisation/preprocessing strategy comparison."""

    def _write_strategy(self, root: Path, name: str, replicate: float, batch_eta: float) -> None:
        strategy = root / name
        strategy.mkdir(parents=True)
        pd.DataFrame(
            {
                "item": [
                    "n_rows_input",
                    "n_rows_passing_qc",
                    "n_features_input",
                    "n_features_after_correlation_filter",
                ],
                "value": [100, 95, 50, 25],
            }
        ).to_csv(strategy / "preprocessing_summary.tsv", sep="\t", index=False)
        pd.DataFrame([{"context": "matrix", "status": "ok"}]).to_csv(
            strategy / "final_matrix_validation.tsv", sep="\t", index=False
        )
        pd.DataFrame([{"group": "P1", "status": "ok"}]).to_csv(
            strategy / "control_qc_before_normalisation.tsv", sep="\t", index=False
        )
        pd.DataFrame(
            [
                {
                    "stage": "after_batch_correction",
                    "Metadata_Compound": "DrugA",
                    "median_correlation": replicate,
                }
            ]
        ).to_csv(strategy / "before_after_replicate_summary.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "stage": "after_batch_correction",
                    "metadata_column": "Metadata_Plate",
                    "component": "PC1",
                    "status": "tested",
                    "eta_squared": batch_eta,
                },
                {
                    "stage": "after_batch_correction",
                    "metadata_column": "Metadata_Compound",
                    "component": "PC1",
                    "status": "tested",
                    "eta_squared": 0.4,
                },
            ]
        ).to_csv(strategy / "before_after_batch_pc_association.tsv", sep="\t", index=False)

    def test_strategy_summary_ranks_candidate_strategy(self) -> None:
        """A compact strategy summary should be created from run outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_strategy(root, "good", replicate=0.5, batch_eta=0.05)
            self._write_strategy(root, "batchy", replicate=0.4, batch_eta=0.6)
            summary = summarise_preprocessing_strategies(strategy_root=root)
        self.assertEqual(summary.iloc[0]["strategy"], "good")
        self.assertIn("selection_score_heuristic", summary.columns)
        self.assertIn("recommendation", summary.columns)


class TestBatchAssociationReporting(unittest.TestCase):
    """Tests for one-variable-at-a-time PC association reporting."""

    def test_single_group_batch_column_is_not_reported_as_tested(self) -> None:
        """Single-level metadata cannot be meaningfully tested for PC association."""
        features = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [2.0, 1.0, 0.0]})
        metadata = pd.DataFrame({"Metadata_Plate": ["P1", "P1", "P1"]})
        result = _summarise_stage_batch_association(
            stage="after_batch_correction",
            features=features,
            metadata=metadata,
            batch_report_columns=["Metadata_Plate"],
        )
        self.assertTrue((result["status"] == "not_testable_single_group").all())
        self.assertTrue(result["eta_squared"].isna().all())


class TestLargeReplicateOutputs(unittest.TestCase):
    """Tests for compressed large pairwise replicate outputs."""

    def test_pairwise_replicate_correlations_are_written_compressed(self) -> None:
        """The full pairwise replicate table should not be written as a huge plain TSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            result = {
                "preprocessed": pd.DataFrame({"Metadata_Compound": ["A"], "f1": [1.0]}),
                "before_after_replicate_correlations": pd.DataFrame(
                    {"stage": ["before"], "replicate_group": ["A"], "correlation": [1.0]}
                ),
                "preprocessing_summary": pd.DataFrame({"item": ["x"], "value": [1]}),
            }
            _write_result_tables(result=result, output_dir=output, logger=None)
            self.assertTrue((output / "before_after_replicate_correlations.tsv.gz").exists())
            self.assertFalse((output / "before_after_replicate_correlations.tsv").exists())


class TestClipnProvenance(unittest.TestCase):
    """Tests for CLIPn backend provenance output."""

    def test_clipn_backend_provenance_has_required_fields(self) -> None:
        """CLIPn reports should record whether output is backend or fallback-derived."""
        provenance = collect_clipn_backend_provenance(
            backend_module="clipn",
            backend_run="success",
            pca_fallback_used=False,
            loss_table=pd.DataFrame({"epoch": [1, 2], "loss": [2.0, 1.5]}),
        )
        for column in [
            "backend_module",
            "backend_run",
            "pca_fallback_used",
            "torch_version",
            "torch_cuda_available",
            "training_loss_rows",
            "final_training_loss",
        ]:
            self.assertIn(column, provenance.columns)
        self.assertEqual(int(provenance.loc[0, "training_loss_rows"]), 2)


class TestExplainSkipsBackgroundQuery(unittest.TestCase):
    """Tests for skipping DMSO-vs-DMSO style explanations."""

    def test_query_background_shap_skips_query_that_is_background(self) -> None:
        """A background/control query should be skipped rather than compared with itself."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            profiles = pd.DataFrame(
                {
                    "Metadata_Compound": ["DMSO", "DMSO", "DrugA", "DrugA"],
                    "f1": [0.0, 0.1, 5.0, 5.2],
                    "f2": [1.0, 1.1, 2.0, 2.2],
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
                "DMSO",
                "--run_query_background_shap",
                "--background_column",
                "Metadata_Compound",
                "--background_values",
                "DMSO",
                "--disable_html_report",
            ]
            with patch("sys.argv", argv):
                explain_main()
            status = pd.read_csv(
                output_dir
                / "query_neighbourhoods"
                / "DMSO"
                / "query_vs_background_shap_status.tsv",
                sep="\t",
            )
        self.assertEqual(status.loc[0, "status"], "skipped_query_is_background")


class TestReportStrategyAndGuide(unittest.TestCase):
    """Tests for report-level strategy and guide integration."""

    def test_report_exports_method_guide_and_strategy_summary(self) -> None:
        """cpatk-report should be able to pull in strategy comparison and guide files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strategy = root / "strategies" / "baseline"
            strategy.mkdir(parents=True)
            pd.DataFrame(
                {
                    "item": ["n_rows_input", "n_rows_passing_qc", "n_features_input", "n_features_after_correlation_filter"],
                    "value": [10, 9, 5, 4],
                }
            ).to_csv(strategy / "preprocessing_summary.tsv", sep="\t", index=False)
            pd.DataFrame([{"context": "matrix", "status": "ok"}]).to_csv(
                strategy / "final_matrix_validation.tsv", sep="\t", index=False
            )
            output_html = root / "report.html"
            argv = [
                "cpatk-report",
                "--output_html",
                str(output_html),
                "--strategy_root",
                str(root / "strategies"),
                "--export_method_guide",
            ]
            with patch("sys.argv", argv):
                report_main()
            html = output_html.read_text(encoding="utf-8")
            self.assertIn("Normalisation strategy comparison", html)
            self.assertIn("ML and nearest-neighbour method guide", html)
            self.assertTrue((root / "normalisation_strategy_comparison.tsv").exists())
            self.assertTrue((root / "method_guides" / "ml_nn_method_guide.tsv").exists())


if __name__ == "__main__":
    unittest.main()
