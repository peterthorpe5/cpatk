"""Tests for CPATK v0.2.14 phenotype-labelled pseudo anchors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cpatk.moa_advanced import (
    label_pseudo_anchor_clusters,
    make_pseudo_anchors,
    normalise_phenotype_label_table,
)
from cpatk.cli import moa as moa_cli


class TestPhenotypeLabelledPseudoAnchors(unittest.TestCase):
    """Test phenotype label handling for pseudo-anchor MOA interpretation."""

    def setUp(self) -> None:
        """Create a small deterministic profile table and phenotype map."""
        self.table = pd.DataFrame(
            {
                "cpd_id": [f"C{i}" for i in range(8)],
                "f0": [1.0, 1.1, 0.9, 1.0, 0.0, 0.1, -0.1, 0.0],
                "f1": [0.0, 0.1, -0.1, 0.0, 1.0, 1.1, 0.9, 1.0],
                "f2": np.linspace(0.0, 1.0, 8),
            }
        )
        self.labels = pd.DataFrame(
            {
                "cpd_id": ["C0", "C1", "C2", "C4", "C5", "C5", "C6"],
                "label": [
                    "phenotype_alpha",
                    "phenotype_alpha",
                    "phenotype_alpha",
                    "phenotype_beta",
                    "phenotype_beta",
                    "phenotype_beta",
                    "phenotype_beta",
                ],
            }
        )

    def test_normalise_phenotype_label_table_audits_duplicates(self) -> None:
        """Duplicate label rows should be removed and audited."""
        clean, audit = normalise_phenotype_label_table(
            label_table=self.labels,
            id_column="cpd_id",
            label_column="label",
        )
        self.assertEqual(clean.shape[0], 6)
        audit_map = dict(zip(audit["metric"], audit["value"]))
        self.assertEqual(audit_map["exact_duplicate_rows_removed"], 1)
        self.assertEqual(audit_map["unique_labelled_ids"], 6)

    def test_label_pseudo_anchor_clusters_adds_moa_final(self) -> None:
        """Phenotype labels should annotate confident pseudo-anchor clusters."""
        anchors, _, clusters, _ = make_pseudo_anchors(
            table=self.table,
            id_column="cpd_id",
            feature_columns=["f0", "f1", "f2"],
            n_clusters=2,
            auto_k=False,
            bootstrap=False,
            random_state=3,
        )
        clean, _ = normalise_phenotype_label_table(
            label_table=self.labels,
            id_column="cpd_id",
            label_column="label",
        )
        labelled_anchors, labelled_clusters, summary = label_pseudo_anchor_clusters(
            anchors=anchors,
            clusters=clusters,
            label_table=clean,
            id_column="cpd_id",
            label_column="label",
            min_labelled_fraction=0.2,
            min_dominant_fraction=0.5,
        )
        self.assertIn("moa_final", labelled_anchors.columns)
        self.assertIn("moa_final", labelled_clusters.columns)
        self.assertIn("top_phenotype_labels", summary.columns)
        self.assertTrue(summary["moa_final"].notna().all())
        self.assertTrue((summary["n_labelled_compounds"] > 0).any())

    def test_moa_cli_uses_phenotype_labelled_pseudo_anchors(self) -> None:
        """The MOA CLI should write phenotype-labelled pseudo-anchor outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profiles_path = tmp_path / "profiles.tsv"
            labels_path = tmp_path / "labels.tsv"
            output_dir = tmp_path / "moa"
            self.table.to_csv(profiles_path, sep="\t", index=False)
            self.labels.to_csv(labels_path, sep="\t", index=False)
            parser = moa_cli.build_parser()
            args = parser.parse_args(
                [
                    "--input_table",
                    str(profiles_path),
                    "--output_dir",
                    str(output_dir),
                    "--id_column",
                    "cpd_id",
                    "--make_pseudo_anchors",
                    "--pseudo_anchor_method",
                    "simple",
                    "--n_clusters",
                    "2",
                    "--pseudo_anchor_label_table",
                    str(labels_path),
                    "--pseudo_anchor_label_id_column",
                    "cpd_id",
                    "--pseudo_anchor_label_column",
                    "label",
                    "--n_permutations",
                    "2",
                    "--disable_html_report",
                ]
            )
            original_build_parser = moa_cli.build_parser
            try:
                moa_cli.build_parser = lambda: parser
                parser.parse_args = lambda: args  # type: ignore[assignment]
                moa_cli.main()
            finally:
                moa_cli.build_parser = original_build_parser

            pseudo_anchors = pd.read_csv(output_dir / "pseudo_anchors.tsv", sep="\t")
            summary = pd.read_csv(output_dir / "pseudo_anchor_phenotype_summary.tsv", sep="\t")
            top_predictions = pd.read_csv(output_dir / "advanced_moa_top_predictions.tsv", sep="\t")
            self.assertIn("moa_final", pseudo_anchors.columns)
            self.assertIn("dominant_phenotype", summary.columns)
            self.assertIn("predicted_moa", top_predictions.columns)
            self.assertGreaterEqual(summary.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
