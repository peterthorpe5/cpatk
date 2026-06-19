"""Command-line plate-layout diagnostics for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.layout import run_plate_layout_diagnostics
from cpatk.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run plate-layout diagnostics.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--well_column", default="Well_Metadata")
    parser.add_argument("--metric_columns", required=True)
    parser.add_argument("--grouping_columns", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "plate_layout.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)
    result = run_plate_layout_diagnostics(
        data_frame=data_frame,
        output_dir=output_dir,
        well_column=args.well_column,
        metric_columns=parse_column_list(value=args.metric_columns) or [],
        grouping_columns=parse_column_list(value=args.grouping_columns),
        logger=logger,
    )
    for name, table in result.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=result, path=output_dir / "plate_layout_diagnostics.xlsx", logger=logger)
    logger.info("CPATK plate-layout diagnostics complete")


if __name__ == "__main__":
    main()
