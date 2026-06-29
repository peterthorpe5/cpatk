"""Command-line batch and domain-shift diagnostics for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.batch import calculate_batch_centroid_distances, cross_validated_batch_prediction, calculate_metadata_association_with_pcs
from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.plotting import plot_heatmap
from cpatk.threading_utils import configure_threading


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run CPATK batch/domain diagnostics.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--batch_column", required=True)
    parser.add_argument("--columns_to_test", default=None)
    parser.add_argument("--threads", type=int, default=1, help="Thread count for supported batch-prediction operations.")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "batch.log", log_level=args.log_level)
    threads = configure_threading(n_threads=args.threads, logger=logger)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata, features, _, _ = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        additional_metadata_columns=[args.batch_column],
    )
    centroid_distances = calculate_batch_centroid_distances(
        features=features,
        metadata=metadata,
        batch_column=args.batch_column,
    )
    prediction = cross_validated_batch_prediction(
        features=features,
        metadata=metadata,
        batch_column=args.batch_column,
        logger=logger,
        n_jobs=threads,
    )
    columns_to_test = parse_column_list(value=args.columns_to_test) or [args.batch_column]
    pc_association = calculate_metadata_association_with_pcs(
        features=features,
        metadata=metadata,
        columns_to_test=columns_to_test,
    )
    tables = {
        "batch_centroid_distances": centroid_distances,
        "batch_prediction": prediction,
        "pc_metadata_association": pc_association,
    }
    for name, table in tables.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=tables, path=output_dir / "batch_diagnostics.xlsx", logger=logger)
    matrix = centroid_distances.pivot(index="batch_1", columns="batch_2", values="distance")
    plot_heatmap(
        matrix=matrix,
        output_path_base=output_dir / "batch_centroid_distance_heatmap",
        title="Batch centroid distances",
        value_label="Distance",
        logger=logger,
        n_jobs=threads,
    )
    logger.info("CPATK batch diagnostics complete")


if __name__ == "__main__":
    main()
