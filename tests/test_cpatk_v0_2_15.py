"""Regression tests for CPATK v0.2.15 sidecar and drift hardening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from cpatk.inspection import inspect_directory
from cpatk.io import is_ignored_sidecar_path, list_supported_tables, write_table
from cpatk.merging import discover_table_files
from cpatk.qc_drift import safe_spearmanr


class TestSidecarDiscoveryV0215(unittest.TestCase):
    """Tests for ignoring macOS and hidden sidecar files."""

    def test_ignored_sidecar_path_detects_hidden_files(self) -> None:
        """Hidden and AppleDouble files should be ignored during discovery."""
        self.assertTrue(is_ignored_sidecar_path(path="._table.tsv"))
        self.assertTrue(is_ignored_sidecar_path(path=".hidden/table.tsv"))
        self.assertTrue(is_ignored_sidecar_path(path="~$workbook.xlsx"))
        self.assertFalse(is_ignored_sidecar_path(path="table.tsv"))

    def test_list_supported_tables_skips_appledouble_files(self) -> None:
        """Supported-table discovery should not include macOS sidecars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_table(
                data_frame=pd.DataFrame({"a": [1], "b": [2]}),
                path=tmp / "table.tsv",
            )
            (tmp / "._table.tsv").write_bytes(b"\x00\xb0not a UTF-8 table")
            inventory = list_supported_tables(input_dir=tmp)
            self.assertEqual(inventory.shape[0], 1)
            self.assertEqual(inventory.loc[0, "file_name"], "table.tsv")

    def test_inspect_directory_ignores_appledouble_files(self) -> None:
        """Inspection should skip AppleDouble files rather than crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_table(
                data_frame=pd.DataFrame({"a": [1], "b": [2]}),
                path=tmp / "table.tsv",
            )
            (tmp / "._table.tsv").write_bytes(b"\x00\xb0not a UTF-8 table")
            result = inspect_directory(input_dir=tmp)
            self.assertIn("inspection_failure_report", result)
            self.assertEqual(result["file_summary"].shape[0], 1)
            self.assertEqual(result["file_summary"].loc[0, "n_rows"], 1)

    def test_profile_discovery_skips_appledouble_files(self) -> None:
        """Profile-building discovery should skip sidecar files too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_table(
                data_frame=pd.DataFrame({"ImageNumber": [1], "Feature": [2.0]}),
                path=tmp / "Image.tsv",
            )
            (tmp / "._Image.tsv").write_bytes(b"\x00\xb0not a table")
            inventory = discover_table_files(input_dir=tmp)
            self.assertEqual(inventory.shape[0], 1)
            self.assertEqual(inventory.loc[0, "file_name"], "Image.tsv")


class TestDriftImportFallbackV0215(unittest.TestCase):
    """Tests for avoiding hard failure when SciPy stats is unavailable."""

    def test_safe_spearmanr_falls_back_without_scipy(self) -> None:
        """A pandas/numpy Spearman fallback should be available."""
        with mock.patch.dict("sys.modules", {"scipy.stats": None}):
            rho, p_value = safe_spearmanr(
                x=np.asarray([1, 2, 3, 4], dtype=float),
                y=np.asarray([1, 2, 3, 4], dtype=float),
            )
        self.assertAlmostEqual(rho, 1.0)
        self.assertTrue(np.isnan(p_value))


if __name__ == "__main__":
    unittest.main()
