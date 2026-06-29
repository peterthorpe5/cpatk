"""Command-line workflow for combining CPATK profile tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list
from cpatk.logging_utils import configure_logging
from cpatk.profile_combining import run_combine_profiles_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple already-built CPATK profile tables into one joint table. "
            "This is the recommended route for independent multi-plate CellProfiler exports."
        )
    )
    parser.add_argument("--profile_tables", required=True, help="Comma-separated profile tables to combine.")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined profiles and reports.")
    parser.add_argument("--source_labels", default=None, help="Optional comma-separated source labels, one per table.")
    parser.add_argument(
        "--key_columns",
        default=None,
        help=(
            "Optional comma-separated unique profile keys. For image-level "
            "profiles, use a key that includes image identity, for example "
            "Metadata_Profile_Source,Metadata_Plate,ImageNumber. If omitted, "
            "CPATK prefers source, plate and image identity where available."
        ),
    )
    parser.add_argument("--feature_join", default="union", choices=["union", "intersection"])
    parser.add_argument("--duplicate_policy", default="error", choices=["error", "allow"])
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    """Run the combine-profiles command."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "combine_profiles.log", log_level=args.log_level)
    run_combine_profiles_workflow(
        profile_paths=[Path(path) for path in parse_column_list(value=args.profile_tables) or []],
        output_dir=output_dir,
        source_labels=parse_column_list(value=args.source_labels),
        key_columns=parse_column_list(value=args.key_columns),
        feature_join=args.feature_join,
        duplicate_policy=args.duplicate_policy,
        logger=logger,
    )
    logger.info("CPATK profile combining complete")


if __name__ == "__main__":
    main()
