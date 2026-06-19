"""Command-line profile builder for folders of Cell Painting exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list
from cpatk.logging_utils import configure_logging
from cpatk.merging import build_profiles_from_folder


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build an analysis-ready profile table from a folder containing "
            "Cell Painting Image/Object/metadata exports."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing CSV, TSV, gzipped, Parquet or Excel tables.")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged profiles and audit reports.")
    parser.add_argument("--image_table", default=None, help="Optional explicit Image/profile backbone table path.")
    parser.add_argument(
        "--object_tables",
        default=None,
        help="Optional comma-separated object table paths. If omitted, object tables are inferred.",
    )
    parser.add_argument("--metadata_table", default=None, help="Optional external metadata/platemap table path.")
    parser.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    parser.add_argument("--aggregate_statistic", default="median", choices=["median", "mean"])
    parser.add_argument("--include_qc_numeric_features", action="store_true")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    """Run the command-line profile builder."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "build_profiles.log", log_level=args.log_level)
    object_tables = parse_column_list(value=args.object_tables)
    build_profiles_from_folder(
        input_dir=args.input_dir,
        output_dir=output_dir,
        recursive=args.recursive,
        image_table=args.image_table,
        object_tables=object_tables,
        metadata_table=args.metadata_table,
        aggregate_statistic=args.aggregate_statistic,
        include_qc_numeric_features=args.include_qc_numeric_features,
        logger=logger,
    )
    logger.info("CPATK profile building complete")


if __name__ == "__main__":
    main()
