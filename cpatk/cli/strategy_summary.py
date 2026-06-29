"""Command-line preprocessing strategy comparison for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.logging_utils import configure_logging
from cpatk.strategy_selection import write_preprocessing_strategy_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Summarise CPATK preprocessing/normalisation strategies."
    )
    parser.add_argument("--strategy_root", required=True)
    parser.add_argument("--output_table", required=True)
    parser.add_argument("--batch_column", default="Metadata_Plate")
    parser.add_argument("--compound_column", default="Metadata_Compound")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_table = Path(args.output_table)
    logger = configure_logging(log_file=output_table.with_suffix(".log"), log_level=args.log_level)
    write_preprocessing_strategy_summary(
        strategy_root=Path(args.strategy_root),
        output_path=output_table,
        batch_column=args.batch_column,
        compound_column=args.compound_column,
        logger=logger,
    )
    logger.info("CPATK preprocessing strategy summary written: %s", output_table)


if __name__ == "__main__":
    main()
