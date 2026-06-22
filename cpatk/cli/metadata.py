"""Command-line metadata validation workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list
from cpatk.logging_utils import configure_logging
from cpatk.metadata_validation import run_metadata_validation_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Step 1 metadata validation and formatting for Cell Painting projects. "
            "Writes formatted_metadata.tsv plus audit reports before profile building."
        )
    )
    parser.add_argument("--metadata_table", required=True, help="Raw metadata or plate-map table.")
    parser.add_argument("--output_dir", required=True, help="Output directory for formatted metadata and reports.")
    parser.add_argument(
        "--annotation_tables",
        default=None,
        help="Optional comma-separated annotation tables to merge into the metadata.",
    )
    parser.add_argument(
        "--merge_keys",
        default=None,
        help="Optional comma-separated merge keys, for example Metadata_Source_Plate,Metadata_Source_Well.",
    )
    parser.add_argument(
        "--duplicate_policy",
        default="error",
        choices=["error", "identical", "first"],
        help="How to handle duplicate annotation keys. Default: error.",
    )
    parser.add_argument(
        "--allow_well_only",
        action="store_true",
        help="Allow unsafe well-only merges when no plate/source-plate key is available.",
    )
    parser.add_argument(
        "--plate_column",
        default=None,
        help="Explicit assay plate column in the main metadata table. Use this when the file also contains source/robot plate columns.",
    )
    parser.add_argument(
        "--well_column",
        default=None,
        help="Explicit assay well column in the main metadata table. This is the well column expected to match CellProfiler output.",
    )
    parser.add_argument(
        "--source_plate_column",
        default=None,
        help="Explicit source-library/robot plate column in the main metadata table.",
    )
    parser.add_argument(
        "--source_well_column",
        default=None,
        help="Explicit source-library/robot well column in the main metadata table.",
    )
    parser.add_argument(
        "--annotation_plate_column",
        default=None,
        help="Explicit assay plate column in annotation tables, if different from the main metadata table.",
    )
    parser.add_argument(
        "--annotation_well_column",
        default=None,
        help="Explicit assay well column in annotation tables, if different from the main metadata table.",
    )
    parser.add_argument(
        "--annotation_source_plate_column",
        default=None,
        help="Explicit source-library/robot plate column in annotation tables.",
    )
    parser.add_argument(
        "--annotation_source_well_column",
        default=None,
        help="Explicit source-library/robot well column in annotation tables.",
    )
    parser.add_argument(
        "--no_require_assay_keys",
        action="store_true",
        help="Do not require the main metadata table to contain assay plate/well keys. Mainly for annotation-only audits.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    """Run the metadata validation command."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "metadata_validation.log", log_level=args.log_level)
    annotation_tables = [Path(path) for path in parse_column_list(value=args.annotation_tables) or []]
    merge_keys = parse_column_list(value=args.merge_keys)
    run_metadata_validation_workflow(
        metadata_table=Path(args.metadata_table),
        output_dir=output_dir,
        annotation_tables=annotation_tables,
        merge_keys=merge_keys,
        duplicate_policy=args.duplicate_policy,
        allow_well_only=args.allow_well_only,
        plate_column=args.plate_column,
        well_column=args.well_column,
        source_plate_column=args.source_plate_column,
        source_well_column=args.source_well_column,
        annotation_plate_column=args.annotation_plate_column,
        annotation_well_column=args.annotation_well_column,
        annotation_source_plate_column=args.annotation_source_plate_column,
        annotation_source_well_column=args.annotation_source_well_column,
        require_assay_keys=not args.no_require_assay_keys,
        logger=logger,
    )
    logger.info("CPATK metadata validation complete: %s", output_dir)


if __name__ == "__main__":
    main()
