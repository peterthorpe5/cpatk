"""Command-line MOA workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.moa import (
    calculate_class_centroids,
    classify_by_knn,
    leave_one_out_centroid_validation,
    score_profiles_against_centroids,
    summarise_moa_predictions,
)
from cpatk.plotting import plot_prediction_confidence
from cpatk.reporting import default_methods_text, make_html_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run centroid and KNN MOA classification.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_column", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--min_class_size", type=int, default=2)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--run_knn", action="store_true")
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--disable_html_report", action="store_true")
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
    labels = metadata[args.class_column]
    centroids, class_summary = calculate_class_centroids(
        features=features,
        labels=labels,
        min_class_size=args.min_class_size,
    )
    scores = score_profiles_against_centroids(
        query_features=features,
        centroids=centroids,
        metric=args.metric,
        top_n=args.top_n,
    )
    predictions = scores.loc[scores["rank"] == 1, :].copy()
    predictions = predictions.merge(
        metadata.reset_index().rename(columns={"index": "query_index"}),
        on="query_index",
        how="left",
    )
    prediction_summary = summarise_moa_predictions(predictions=predictions)
    loo_predictions, loo_summary = leave_one_out_centroid_validation(
        features=features,
        labels=labels,
        min_class_size=args.min_class_size,
        metric=args.metric,
    )
    tables = {
        "moa_centroids": centroids.reset_index(),
        "moa_class_summary": class_summary,
        "moa_centroid_scores": scores,
        "moa_top_predictions": predictions,
        "moa_prediction_summary": prediction_summary,
        "centroid_leave_one_out_predictions": loo_predictions,
        "centroid_leave_one_out_summary": loo_summary,
    }
    plot_paths = []
    plot_paths.extend(
        plot_prediction_confidence(
            predictions=predictions,
            confidence_column="softmax_confidence",
            output_path_base=output_dir / "centroid_prediction_confidence",
            logger=logger,
        )
    )
    if args.run_knn:
        knn_predictions, knn_neighbours = classify_by_knn(
            train_features=features,
            train_labels=labels,
            query_features=features,
            n_neighbors=args.n_neighbors,
            return_neighbour_table=True,
        )
        knn_predictions = knn_predictions.merge(
            metadata.reset_index().rename(columns={"index": "query_index"}),
            on="query_index",
            how="left",
        )
        tables["knn_predictions"] = knn_predictions
        tables["knn_neighbours"] = knn_neighbours
        plot_paths.extend(
            plot_prediction_confidence(
                predictions=knn_predictions,
                confidence_column="max_probability",
                output_path_base=output_dir / "knn_prediction_confidence",
                title="KNN prediction confidence distribution",
                logger=logger,
            )
        )
    for name, table in tables.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(
        tables={key: value.head(5000) for key, value in tables.items()},
        path=output_dir / "moa_summary.xlsx",
        logger=logger,
    )
    if not args.disable_html_report:
        make_html_report(
            title="CPATK MOA analysis report",
            output_path=output_dir / "moa_report.html",
            summary_tables={key: value.head(200) for key, value in tables.items()},
            plot_paths=plot_paths,
            narrative="CPATK ran centroid-based MOA scoring with confidence margins and optional KNN classification.",
            methods_text=default_methods_text(),
            warnings=["MOA predictions should be interpreted with replicate consistency, training-class size and cross-validation performance."],
        )
    logger.info("CPATK MOA analysis complete")


if __name__ == "__main__":
    main()
