"""Tests for CPATK v0.2.28 batch heatmap threading fix."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cpatk.cli import batch as batch_cli
from cpatk.plotting import plot_heatmap


class TestBatchHeatmapThreadingFix(unittest.TestCase):
    """Test the threaded batch heatmap path fixed in v0.2.28."""

    def test_plot_heatmap_accepts_n_jobs_argument(self) -> None:
        """Heatmap plotting should accept n_jobs for CLI compatibility."""
        matrix = pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]],
            index=["batch_a", "batch_b"],
            columns=["batch_a", "batch_b"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir) / "heatmap"
            paths = plot_heatmap(
                matrix=matrix,
                output_path_base=output_base,
                title="Batch heatmap",
                value_label="Distance",
                n_jobs=2,
            )
            self.assertTrue(paths)
            for path in paths:
                self.assertTrue(path.exists())

    def test_batch_cli_threads_complete_heatmap_plotting(self) -> None:
        """cpatk-batch --threads should complete through heatmap plotting."""
        data_frame = pd.DataFrame(
            {
                "Metadata_Batch": ["a", "a", "a", "b", "b", "b"],
                "Metadata_Compound": ["x", "x", "y", "x", "y", "y"],
                "Feature_1": [0.0, 0.1, 0.2, 3.0, 3.1, 3.2],
                "Feature_2": [1.0, 1.1, 1.2, 4.0, 4.1, 4.2],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_table = temp_path / "input.tsv"
            output_dir = temp_path / "batch"
            data_frame.to_csv(input_table, sep="\t", index=False)
            argv = [
                "cpatk-batch",
                "--input_table",
                str(input_table),
                "--output_dir",
                str(output_dir),
                "--metadata_columns",
                "Metadata_Batch,Metadata_Compound",
                "--batch_column",
                "Metadata_Batch",
                "--columns_to_test",
                "Metadata_Batch,Metadata_Compound",
                "--threads",
                "2",
            ]
            with patch("sys.argv", argv):
                batch_cli.main()
            self.assertTrue((output_dir / "batch_centroid_distances.tsv").exists())
            self.assertTrue((output_dir / "batch_centroid_distance_heatmap.svg").exists())
            self.assertTrue((output_dir / "batch.log").exists())


if __name__ == "__main__":
    unittest.main()
