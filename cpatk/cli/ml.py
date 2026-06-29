"""Command-line supervised ML classifier workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.ml import compare_moa_models, cross_validate_classifier
from cpatk.plotting import plot_confusion_matrix, plot_model_summary
from cpatk.reporting import default_methods_text, make_html_report
from cpatk.threading_utils import configure_threading


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run CPATK supervised ML classification.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_column", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--model_name", default="random_forest")
    parser.add_argument("--compare_models", action="store_true")
    parser.add_argument("--models", default="knn,random_forest,extra_trees,gradient_boosting,logistic_regression,linear_svm")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1, help="Thread count for supported estimators and cross-validation.")
    parser.add_argument("--disable_html_report", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "ml.log", log_level=args.log_level)
    threads = configure_threading(n_threads=args.threads, logger=logger)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata, features, _, _ = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        additional_metadata_columns=[args.class_column],
    )
    labels = metadata[args.class_column]
    plot_paths = []
    if args.compare_models:
        model_names = parse_column_list(value=args.models)
        summary, predictions = compare_moa_models(
            features=features,
            labels=labels,
            model_names=model_names,
            n_splits=args.n_splits,
            logger=logger,
            n_jobs=threads,
        )
        tables = {"model_comparison_summary": summary, "model_comparison_predictions": predictions}
        plot_paths.extend(
            plot_model_summary(
                summary=summary,
                metric_column="balanced_accuracy",
                output_path_base=output_dir / "model_comparison_balanced_accuracy",
                title="MOA classifier balanced accuracy",
                logger=logger,
            )
        )
    else:
        summary, predictions, confusion = cross_validate_classifier(
            features=features,
            labels=labels,
            model_name=args.model_name,
            n_splits=args.n_splits,
            logger=logger,
            n_jobs=threads,
        )
        tables = {"classifier_summary": summary, "classifier_predictions": predictions, "confusion_matrix": confusion}
        plot_paths.extend(plot_confusion_matrix(confusion_table=confusion, output_path_base=output_dir / "confusion_matrix", logger=logger))
    for name, table in tables.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=tables, path=output_dir / "ml_classifier_summary.xlsx", logger=logger)
    if not args.disable_html_report:
        make_html_report(
            title="CPATK supervised MOA classifier report",
            output_path=output_dir / "ml_classifier_report.html",
            summary_tables=tables,
            plot_paths=plot_paths,
            narrative="CPATK evaluated supervised MOA classifiers using cross-validation.",
            methods_text=default_methods_text(),
            warnings=["Classifier performance should be interpreted cautiously for small or imbalanced MOA classes."],
        )
    logger.info("CPATK ML workflow complete")


if __name__ == "__main__":
    main()
