"""Command-line MOA workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.moa import calculate_class_centroids, score_profiles_against_centroids, summarise_moa_predictions


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser object.
    """
    parser = argparse.ArgumentParser(description="Run centroid-based MOA classification.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_column", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--min_class_size", type=int, default=2)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "moa.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata, features, _, _ = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        additional_metadata_columns=[args.class_column],
    )
    if args.class_column not in metadata.columns:
        raise ValueError(f"Class column is missing from metadata: {args.class_column}")
    centroids, class_summary = calculate_class_centroids(
        features=features,
        labels=metadata[args.class_column],
        min_class_size=args.min_class_size,
    )
    scores = score_profiles_against_centroids(
        query_features=features,
        centroids=centroids,
        metric=args.metric,
        top_n=args.top_n,
    )
    predictions = scores.loc[scores["rank"] == 1, ["query_index", "predicted_class", "distance", "similarity"]]
    prediction_summary = summarise_moa_predictions(predictions=predictions)
    write_table(data_frame=centroids.reset_index(), path=output_dir / "moa_centroids.tsv", logger=logger)
    write_table(data_frame=class_summary, path=output_dir / "moa_class_summary.tsv", logger=logger)
    write_table(data_frame=scores, path=output_dir / "moa_centroid_scores.tsv", logger=logger)
    write_table(data_frame=predictions, path=output_dir / "moa_top_predictions.tsv", logger=logger)
    write_table(data_frame=prediction_summary, path=output_dir / "moa_prediction_summary.tsv", logger=logger)
    write_excel_workbook(
        tables={
            "class_summary": class_summary,
            "top_predictions": predictions,
            "prediction_summary": prediction_summary,
            "centroid_scores": scores.head(5000),
        },
        path=output_dir / "moa_summary.xlsx",
        logger=logger,
    )
    logger.info("CPATK MOA analysis complete")


if __name__ == "__main__":
    main()
