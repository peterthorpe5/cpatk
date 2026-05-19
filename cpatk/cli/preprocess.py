"""Command-line preprocessing workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.preprocessing import aggregate_profiles, preprocess_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser object.
    """
    parser = argparse.ArgumentParser(description="Preprocess generic Cell Painting profiles.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--aggregate_by", default=None)
    parser.add_argument("--aggregate_statistic", default="median")
    parser.add_argument("--imputation_method", default="median")
    parser.add_argument("--scaling_method", default="robust")
    parser.add_argument("--max_feature_missing_fraction", type=float, default=0.2)
    parser.add_argument("--max_sample_missing_fraction", type=float, default=0.5)
    parser.add_argument("--max_absolute_correlation", type=float, default=0.95)
    parser.add_argument("--disable_correlation_filter", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "preprocess.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata_columns = parse_column_list(value=args.metadata_columns)
    feature_columns = parse_column_list(value=args.feature_columns)

    if args.aggregate_by:
        group_columns = parse_column_list(value=args.aggregate_by)
        if feature_columns is None:
            from cpatk.features import infer_metadata_columns, infer_feature_columns

            metadata_columns = infer_metadata_columns(data_frame=data_frame)
            feature_columns = infer_feature_columns(data_frame=data_frame, metadata_columns=metadata_columns)
        data_frame = aggregate_profiles(
            data_frame=data_frame,
            group_columns=group_columns or [],
            feature_columns=feature_columns,
            statistic=args.aggregate_statistic,
        )
        write_table(data_frame=data_frame, path=output_dir / "aggregated_profiles.parquet", logger=logger)

    result = preprocess_profiles(
        data_frame=data_frame,
        metadata_columns=metadata_columns,
        feature_columns=feature_columns,
        max_feature_missing_fraction=args.max_feature_missing_fraction,
        max_sample_missing_fraction=args.max_sample_missing_fraction,
        remove_correlated=not args.disable_correlation_filter,
        max_absolute_correlation=args.max_absolute_correlation,
        imputation_method=args.imputation_method,
        scaling_method=args.scaling_method,
        logger=logger,
    )
    for name, table in result.items():
        if name == "preprocessed":
            try:
                write_table(data_frame=table, path=output_dir / "preprocessed.parquet", logger=logger)
            except ImportError as exc:
                logger.warning("Parquet writing unavailable; writing TSV.GZ fallback: %s", exc)
                write_table(data_frame=table, path=output_dir / "preprocessed.tsv.gz", logger=logger)
        else:
            write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=result, path=output_dir / "preprocessing_summary.xlsx", logger=logger)
    logger.info("CPATK preprocessing complete")


if __name__ == "__main__":
    main()
