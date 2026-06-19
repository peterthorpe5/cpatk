"""Command-line feature attribution workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.explainability import (
    calculate_permutation_feature_importance,
    calculate_shap_importance_detailed,
    group_feature_importance_by_family,
)
from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.plotting import plot_feature_importance
from cpatk.reporting import default_methods_text, make_html_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run CPATK feature attribution.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_column", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--model_name", default="random_forest")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--include_shap", action="store_true")
    parser.add_argument("--max_shap_background", type=int, default=200)
    parser.add_argument("--max_shap_explain", type=int, default=200)
    parser.add_argument("--disable_html_report", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "explain.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata, features, _, _ = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        additional_metadata_columns=[args.class_column],
    )
    labels = metadata[args.class_column]
    permutation, permutation_summary = calculate_permutation_feature_importance(
        features=features,
        labels=labels,
        model_name=args.model_name,
        n_repeats=args.n_repeats,
        logger=logger,
    )
    family = group_feature_importance_by_family(
        importance_table=permutation,
        value_column="permutation_importance_mean",
    )
    tables = {
        "permutation_importance": permutation,
        "permutation_summary": permutation_summary,
        "feature_family_importance": family,
    }
    plot_paths = []
    plot_paths.extend(
        plot_feature_importance(
            importance_table=permutation,
            value_column="permutation_importance_mean",
            output_path_base=output_dir / "permutation_feature_importance",
            title="Permutation feature importance",
            logger=logger,
        )
    )
    if args.include_shap:
        shap_importance, shap_class_importance, shap_status = calculate_shap_importance_detailed(
            features=features,
            labels=labels,
            model_name=args.model_name,
            max_background=args.max_shap_background,
            max_explain=args.max_shap_explain,
            logger=logger,
        )
        tables["shap_importance"] = shap_importance
        tables["shap_class_importance"] = shap_class_importance
        tables["shap_status"] = shap_status
        if not shap_importance.empty:
            plot_paths.extend(
                plot_feature_importance(
                    importance_table=shap_importance,
                    value_column="mean_absolute_shap",
                    output_path_base=output_dir / "shap_feature_importance",
                    title="SHAP global feature importance",
                    logger=logger,
                )
            )
    for name, table in tables.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=tables, path=output_dir / "feature_attribution_summary.xlsx", logger=logger)
    if not args.disable_html_report:
        warnings = [
            "Feature attribution explains the fitted model, not biological causality. Review model performance before interpreting feature rankings."
        ]
        make_html_report(
            title="CPATK feature attribution report",
            output_path=output_dir / "feature_attribution_report.html",
            summary_tables=tables,
            plot_paths=plot_paths,
            narrative="CPATK calculated model-based feature attribution using permutation importance and optional SHAP.",
            methods_text=default_methods_text(),
            warnings=warnings,
        )
    logger.info("CPATK feature attribution workflow complete")


if __name__ == "__main__":
    main()
