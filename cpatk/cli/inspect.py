"""Command-line inspection workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.inspection import inspect_directory
from cpatk.io import write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser object.
    """
    parser = argparse.ArgumentParser(description="Inspect Cell Painting tables.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        log_file=output_dir / "inspect.log",
        log_level=args.log_level,
    )
    result = inspect_directory(input_dir=args.input_dir, logger=logger)
    for name, table in result.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=result, path=output_dir / "inspection_summary.xlsx", logger=logger)
    logger.info("CPATK inspection complete")


if __name__ == "__main__":
    main()
