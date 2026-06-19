"""Command-line preprocessing workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from cpatk.features import infer_feature_columns, infer_metadata_columns, parse_column_list
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.merging import build_profiles_from_folder
from cpatk.logging_utils import configure_logging
from cpatk.plotting import (
    plot_all_zero_row_summary,
    plot_column_role_summary,
    plot_correlation_filter_summary,
    plot_feature_family_summary,
    plot_feature_qc_status,
    plot_feature_variance_histogram,
    plot_imputation_missingness_top_features,
    plot_missingness_histogram,
    plot_preprocessing_retention,
)
from cpatk.preprocessing import aggregate_profiles, preprocess_profiles
from cpatk.reporting import default_methods_text, make_html_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Preprocess generic Cell Painting profiles or build profiles from a folder first.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_table", default=None, help="Analysis-ready profile table to preprocess.")
    input_group.add_argument("--input_dir", default=None, help="Folder of Cell Painting Image/Object/metadata files to merge then preprocess.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--additional_metadata_columns", default=None)
    parser.add_argument("--aggregate_by", default=None)
    parser.add_argument("--image_table", default=None, help="When --input_dir is used, optional explicit Image/profile backbone table path.")
    parser.add_argument("--object_tables", default=None, help="When --input_dir is used, optional comma-separated object table paths.")
    parser.add_argument("--metadata_table", default=None, help="When --input_dir is used, optional external metadata/platemap table path.")
    parser.add_argument("--recursive", action="store_true", help="When --input_dir is used, search recursively for input tables.")
    parser.add_argument("--aggregate_statistic", default="median", choices=["median", "mean"])
    parser.add_argument(
        "--imputation_method",
        default="median",
        choices=["median", "mean", "zero", "knn", "group_median", "group_mean"],
    )
    parser.add_argument("--imputation_group_columns", default=None)
    parser.add_argument("--add_missing_indicators", action="store_true")
    parser.add_argument("--minimum_missing_indicator_fraction", type=float, default=0.0)
    parser.add_argument("--include_qc_numeric_features", action="store_true")
    parser.add_argument("--winsorise_lower_quantile", type=float, default=None)
    parser.add_argument("--winsorise_upper_quantile", type=float, default=None)
    parser.add_argument("--reference_normalisation_method", default="none", choices=["none", "robust_z", "median_center", "zscore"])
    parser.add_argument("--reference_column", default=None)
    parser.add_argument("--reference_values", default=None)
    parser.add_argument("--reference_group_columns", default=None)
    parser.add_argument("--batch_centering_method", default="none", choices=["none", "median_center", "mean_center"])
    parser.add_argument("--batch_centering_columns", default=None)
    parser.add_argument("--max_missing_indicators", type=int, default=500)
    parser.add_argument("--scaling_method", default="robust", choices=["robust", "standard", "minmax", "none"])
    parser.add_argument("--max_feature_missing_fraction", type=float, default=0.2)
    parser.add_argument("--max_sample_missing_fraction", type=float, default=0.5)
    parser.add_argument("--max_absolute_correlation", type=float, default=0.95)
    parser.add_argument("--max_zero_fraction", type=float, default=1.0, help="Maximum allowed fraction of exact zero values per feature; 1.0 disables this filter.")
    parser.add_argument("--disable_all_zero_row_filter", action="store_true", help="Disable removal of rows whose observed retained feature values are all zero. By default this filter runs after all input files have been merged and before imputation.")
    parser.add_argument("--all_zero_row_tolerance", type=float, default=0.0, help="Absolute tolerance for treating a feature value as zero in the all-zero row filter.")
    parser.add_argument("--disable_correlation_filter", action="store_true")
    parser.add_argument("--disable_metadata_standardisation", action="store_true")
    parser.add_argument("--keep_unnamed_index_columns", action="store_true")
    parser.add_argument("--disable_plots", action="store_true")
    parser.add_argument("--disable_html_report", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def _write_result_tables(*, result: Dict[str, pd.DataFrame], output_dir: Path, logger) -> None:
    """Write preprocessing result tables."""
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


def _make_preprocessing_plots(*, result: Dict[str, pd.DataFrame], output_dir: Path, logger) -> List[Path]:
    """Create preprocessing QC plots."""
    plot_dir = output_dir / "plots"
    plot_paths: List[Path] = []
    plot_jobs = [
        lambda: plot_missingness_histogram(
            qc_table=result["feature_qc"],
            missing_column="missing_fraction",
            output_path_base=plot_dir / "feature_missingness_histogram",
            title="Feature missingness before imputation",
            logger=logger,
        ),
        lambda: plot_missingness_histogram(
            qc_table=result["sample_qc"],
            missing_column="missing_fraction",
            output_path_base=plot_dir / "sample_missingness_histogram",
            title="Sample/profile missingness before imputation",
            logger=logger,
        ),
        lambda: plot_feature_variance_histogram(
            feature_qc=result["feature_qc"],
            output_path_base=plot_dir / "feature_variance_histogram",
            logger=logger,
        ),
        lambda: plot_feature_qc_status(
            feature_qc=result["feature_qc"],
            output_path_base=plot_dir / "feature_qc_status",
            logger=logger,
        ),
        lambda: plot_preprocessing_retention(
            summary=result["preprocessing_summary"],
            output_path_base=plot_dir / "preprocessing_retention_summary",
            logger=logger,
        ),
        lambda: plot_feature_family_summary(
            feature_family_summary=result["feature_family_summary"],
            output_path_base=plot_dir / "retained_feature_family_summary",
            logger=logger,
        ),
        lambda: plot_column_role_summary(
            column_role_report=result["column_role_report"],
            output_path_base=plot_dir / "column_role_summary",
            logger=logger,
        ),
        lambda: plot_imputation_missingness_top_features(
            imputation_report=result["imputation_report"],
            output_path_base=plot_dir / "top_missing_features_before_imputation",
            logger=logger,
        ),
        lambda: plot_all_zero_row_summary(
            all_zero_row_report=result.get("all_zero_row_report", pd.DataFrame()),
            output_path_base=plot_dir / "all_zero_feature_row_qc",
            logger=logger,
        ),
        lambda: plot_correlation_filter_summary(
            correlation_report=result["correlation_filter_report"],
            output_path_base=plot_dir / "correlation_filter_removed_feature_distribution",
            logger=logger,
        ),
    ]
    for job in plot_jobs:
        try:
            written = job()
            plot_paths.extend(written or [])
        except Exception as exc:  # defensive CLI reporting
            logger.warning("Skipping one preprocessing plot because it failed: %s", exc)
    plot_index = pd.DataFrame(
        {"plot_path": [str(path) for path in plot_paths], "plot_name": [path.stem for path in plot_paths]}
    )
    write_table(data_frame=plot_index, path=output_dir / "preprocessing_plot_index.tsv", logger=logger)
    return plot_paths


def _write_html_report(
    *,
    result: Dict[str, pd.DataFrame],
    output_dir: Path,
    plot_paths: List[Path],
    logger,
) -> Path:
    """Write the preprocessing HTML report."""
    summary = result["preprocessing_summary"]
    summary_dict = dict(zip(summary["item"].astype(str), summary["value"].astype(str)))
    narrative = (
        "CPATK preprocessing completed. The workflow separated metadata from numeric Cell Painting features, "
        "standardised common metadata aliases where possible, removed features and samples failing missingness/variance QC, "
        "imputed remaining missing values, scaled the retained feature matrix and optionally removed highly correlated features. "
        f"Rows retained: {summary_dict.get('n_rows_passing_qc', 'NA')} of {summary_dict.get('n_rows_input', 'NA')}. "
        f"Features retained after final filtering: {summary_dict.get('n_features_after_correlation_filter', 'NA')} "
        f"from {summary_dict.get('n_features_input', 'NA')} detected input features."
    )
    warnings = []
    if int(float(summary_dict.get("n_missing_feature_values_before_imputation", "0"))) > 0:
        warnings.append(
            "Missing feature values were present and were imputed. Review the imputation report and missingness plots before interpreting downstream analyses."
        )
    if int(float(summary_dict.get("n_all_zero_feature_rows_removed", "0"))) > 0:
        warnings.append(
            "One or more profiles were removed because all observed retained feature values were zero. This filter is applied only after all CellProfiler tables have been merged and before imputation/correlation filtering."
        )
    if result["dropped_index_column_report"].get("dropped", pd.Series(dtype=bool)).any():
        warnings.append("One or more likely accidental CSV index columns were dropped before preprocessing.")
    report_path = output_dir / "preprocessing_report.html"
    report = make_html_report(
        title="CPATK preprocessing summary report",
        output_path=report_path,
        summary_tables={
            "Preprocessing summary": result["preprocessing_summary"],
            "Feature QC": result["feature_qc"],
            "Sample QC": result["sample_qc"],
            "All-zero row report": result.get("all_zero_row_report", pd.DataFrame()),
            "Imputation report": result["imputation_report"],
            "Non-finite value report": result.get("nonfinite_value_report", pd.DataFrame()),
            "Metadata alias report": result["metadata_alias_report"],
            "Dropped index column report": result["dropped_index_column_report"],
            "Retained feature families": result["feature_family_summary"],
            "Column role report": result["column_role_report"],
            "Preprocessing decisions": result["preprocessing_decision_log"],
            "Preprocessing config": result["preprocessing_config"],
            "Reference normalisation report": result["reference_normalisation_report"].head(2000),
            "Batch centering report": result["batch_centering_report"].head(2000),
        },
        plot_paths=plot_paths,
        narrative=narrative,
        methods_text=default_methods_text(),
        warnings=warnings,
    )
    write_table(
        data_frame=pd.DataFrame.from_records(
            [{"item": "preprocessing_report", "path": str(report), "n_plots_linked": len(plot_paths)}]
        ),
        path=output_dir / "report_generation_summary.tsv",
        logger=logger,
    )
    return report


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "preprocess.log", log_level=args.log_level)
    if args.input_dir:
        logger.info("Building profiles from input directory before preprocessing: %s", args.input_dir)
        profile_build_dir = output_dir / "00_profile_build"
        profile_build = build_profiles_from_folder(
            input_dir=args.input_dir,
            output_dir=profile_build_dir,
            recursive=args.recursive,
            image_table=args.image_table,
            object_tables=parse_column_list(value=args.object_tables),
            metadata_table=args.metadata_table,
            aggregate_statistic=args.aggregate_statistic,
            include_qc_numeric_features=args.include_qc_numeric_features,
            logger=logger,
        )
        data_frame = profile_build.profiles
    else:
        data_frame = read_table(path=args.input_table, logger=logger)
    metadata_columns = parse_column_list(value=args.metadata_columns)
    feature_columns = parse_column_list(value=args.feature_columns)
    additional_metadata_columns = parse_column_list(value=args.additional_metadata_columns)
    imputation_group_columns = parse_column_list(value=args.imputation_group_columns)
    reference_values = parse_column_list(value=args.reference_values)
    reference_group_columns = parse_column_list(value=args.reference_group_columns)
    batch_centering_columns = parse_column_list(value=args.batch_centering_columns)

    if args.aggregate_by:
        group_columns = parse_column_list(value=args.aggregate_by)
        if feature_columns is None:
            inferred_metadata = infer_metadata_columns(
                data_frame=data_frame,
                additional_metadata_columns=additional_metadata_columns,
            )
            feature_columns = infer_feature_columns(data_frame=data_frame, metadata_columns=inferred_metadata)
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
        additional_metadata_columns=additional_metadata_columns,
        max_feature_missing_fraction=args.max_feature_missing_fraction,
        max_sample_missing_fraction=args.max_sample_missing_fraction,
        remove_correlated=not args.disable_correlation_filter,
        max_absolute_correlation=args.max_absolute_correlation,
        max_zero_fraction=args.max_zero_fraction,
        remove_all_zero_rows=not args.disable_all_zero_row_filter,
        all_zero_row_tolerance=args.all_zero_row_tolerance,
        imputation_method=args.imputation_method,
        imputation_group_columns=imputation_group_columns,
        add_missing_indicators=args.add_missing_indicators,
        max_missing_indicators=args.max_missing_indicators,
        minimum_missing_indicator_fraction=args.minimum_missing_indicator_fraction,
        include_qc_numeric_features=args.include_qc_numeric_features,
        winsorise_lower_quantile=args.winsorise_lower_quantile,
        winsorise_upper_quantile=args.winsorise_upper_quantile,
        reference_normalisation_method=args.reference_normalisation_method,
        reference_column=args.reference_column,
        reference_values=reference_values,
        reference_group_columns=reference_group_columns,
        batch_centering_method=args.batch_centering_method,
        batch_centering_columns=batch_centering_columns,
        scaling_method=args.scaling_method,
        standardise_metadata=not args.disable_metadata_standardisation,
        drop_unnamed_indexes=not args.keep_unnamed_index_columns,
        logger=logger,
    )
    _write_result_tables(result=result, output_dir=output_dir, logger=logger)
    plot_paths: List[Path] = []
    if not args.disable_plots:
        plot_paths = _make_preprocessing_plots(result=result, output_dir=output_dir, logger=logger)
    if not args.disable_html_report:
        report_path = _write_html_report(result=result, output_dir=output_dir, plot_paths=plot_paths, logger=logger)
        logger.info("Wrote preprocessing HTML report: %s", report_path)
    logger.info("CPATK preprocessing complete")


if __name__ == "__main__":
    main()
