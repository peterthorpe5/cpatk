"""Regression tests for CPATK v0.2.11 metadata and CLIPn hardening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpatk.clipn_adapter import (
    ClipnAdapterConfig,
    align_dataset_features,
    clean_impute_and_scale_aligned,
    split_single_dataset_by_group,
    standardise_clipn_metadata,
)
from cpatk.metadata_validation import (
    choose_merge_keys,
    prepare_metadata_table,
    run_metadata_validation_workflow,
)


class TestMetadataWellSafetyV0211(unittest.TestCase):
    """Tests for assay-vs-source plate/well safety."""

    def test_source_columns_are_not_promoted_to_primary_assay_keys(self) -> None:
        """Source wells must remain source keys, not CellProfiler assay keys."""
        raw = pd.DataFrame(
            {
                "Source_Plate_Barcode": ["SRC1"],
                "Source_Well": ["A1"],
                "COMPOUND_NAME": ["DrugA"],
            }
        )
        formatted, _ = prepare_metadata_table(data_frame=raw)
        self.assertNotIn("Metadata_Plate", formatted.columns)
        self.assertNotIn("Metadata_Well", formatted.columns)
        self.assertEqual(formatted.loc[0, "Metadata_Source_Plate"], "SRC1")
        self.assertEqual(formatted.loc[0, "Metadata_Source_Well"], "A01")

    def test_explicit_assay_and_source_keys_are_kept_separate(self) -> None:
        """Users can explicitly identify assay and source key columns."""
        raw = pd.DataFrame(
            {
                "AssayPlate": ["ASSAY1"],
                "AssayWell": ["B1"],
                "Source_Plate_Barcode": ["SRC1"],
                "Source_Well": ["A1"],
                "cpd_id": ["DrugA"],
            }
        )
        formatted, reports = prepare_metadata_table(
            data_frame=raw,
            plate_column="AssayPlate",
            well_column="AssayWell",
            source_plate_column="Source_Plate_Barcode",
            source_well_column="Source_Well",
            require_assay_keys=True,
        )
        self.assertEqual(formatted.loc[0, "Metadata_Plate"], "ASSAY1")
        self.assertEqual(formatted.loc[0, "Metadata_Well"], "B01")
        self.assertEqual(formatted.loc[0, "Metadata_Source_Plate"], "SRC1")
        self.assertEqual(formatted.loc[0, "Metadata_Source_Well"], "A01")
        self.assertEqual(formatted.loc[0, "Metadata_Well__raw"], "B1")
        self.assertIn("explicit_key_column_report", reports)

    def test_workflow_requires_assay_keys_by_default(self) -> None:
        """The step-one workflow should fail if only robot/source keys exist."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "metadata.tsv"
            pd.DataFrame(
                {
                    "Source_Plate_Barcode": ["SRC1"],
                    "Source_Well": ["A1"],
                    "cpd_id": ["DrugA"],
                }
            ).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "assay key columns"):
                run_metadata_validation_workflow(
                    metadata_table=path,
                    output_dir=root / "out",
                )

    def test_annotation_source_merge_requires_explicit_ambiguous_well(self) -> None:
        """Annotation tables using generic Well need explicit source-well mapping."""
        metadata_raw = pd.DataFrame(
            {
                "Plate": ["ASSAY1"],
                "Well": ["B1"],
                "Source_Plate_Barcode": ["SRC1"],
                "Source_Well": ["A1"],
            }
        )
        annotation_raw = pd.DataFrame({"Barcode": ["SRC1"], "Well": ["A01"], "Target": ["Kinase"]})
        metadata, _ = prepare_metadata_table(data_frame=metadata_raw, require_assay_keys=True)
        annotation_without_explicit, _ = prepare_metadata_table(data_frame=annotation_raw)
        with self.assertRaises(ValueError):
            choose_merge_keys(left=metadata, right=annotation_without_explicit)
        annotation, _ = prepare_metadata_table(
            data_frame=annotation_raw,
            source_plate_column="Barcode",
            source_well_column="Well",
        )
        self.assertEqual(
            choose_merge_keys(left=metadata, right=annotation),
            ["Metadata_Source_Plate", "Metadata_Source_Well"],
        )


class TestClipnInputSafetyV0211(unittest.TestCase):
    """Tests for CLIPn-specific input safeguards."""

    @staticmethod
    def _table() -> pd.DataFrame:
        """Create a small CLIPn-like table with compounds and features."""
        return pd.DataFrame(
            {
                "cpd_id": ["A", "A", "B", "B", "C", "C"],
                "cpd_type": ["known", "known", "known", "known", "query", "query"],
                "Plate_Metadata": ["P1"] * 6,
                "Well_Metadata": ["A01", "A02", "A03", "A04", "A05", "A06"],
                "Feature_1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
                "Feature_2": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                "All_Zero_Feature": [0.0] * 6,
            }
        )

    def test_single_dataset_can_be_split_by_compound(self) -> None:
        """Single tables can be split into two datasets without splitting compounds."""
        datasets, report = split_single_dataset_by_group(
            data_frame=self._table(),
            group_column="cpd_id",
            random_state=1,
            split_names=("reference_like", "query_like"),
        )
        self.assertEqual(set(datasets), {"reference_like", "query_like"})
        self.assertFalse(datasets["reference_like"].empty)
        self.assertFalse(datasets["query_like"].empty)
        self.assertEqual(report["cpd_id"].nunique(), 3)
        assigned = report.set_index("cpd_id")["assigned_dataset"].to_dict()
        for name, table in datasets.items():
            for compound in table["cpd_id"].unique():
                self.assertEqual(assigned[compound], name)

    def test_clipn_alignment_requires_two_datasets(self) -> None:
        """CLIPn feature alignment should reject a single unsplit dataset."""
        table, _ = standardise_clipn_metadata(data_frame=self._table(), dataset_name="only")
        with self.assertRaisesRegex(ValueError, "at least two"):
            align_dataset_features(datasets={"only": table})

    def test_clipn_preprocessing_removes_all_zero_rows_and_features(self) -> None:
        """Zero-only rows/features are removed before CLIPn fitting."""
        ref = pd.DataFrame(
            {
                "Feature_1": [1.0, 2.0, 0.0],
                "Feature_2": [1.5, 2.5, 0.0],
                "All_Zero_Feature": [0.0, 0.0, 0.0],
            }
        )
        query = pd.DataFrame(
            {
                "Feature_1": [3.0, 4.0],
                "Feature_2": [3.5, 4.5],
                "All_Zero_Feature": [0.0, 0.0],
            }
        )
        config = ClipnAdapterConfig(scaling_method="none", imputation_method="median")
        cleaned, summary = clean_impute_and_scale_aligned(
            aligned={"ref": ref, "query": query},
            metadata={},
            config=config,
        )
        self.assertNotIn("All_Zero_Feature", cleaned["ref"].columns)
        self.assertEqual(int(summary.loc[summary["item"] == "all_zero_rows_dropped", "value"].iloc[0]), 1)
        self.assertEqual(int(summary.loc[summary["item"] == "zero_only_features_dropped", "value"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
