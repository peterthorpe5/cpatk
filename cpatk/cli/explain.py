"""Command-line feature attribution and neighbourhood explanation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cpatk.explainability import (
    calculate_permutation_feature_importance,
    calculate_shap_importance_detailed,
    group_feature_importance_by_family,
)
from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.neighbourhood_explain import (
    calculate_neighbourhood_shap,
    calculate_query_background_statistics,
    clean_numeric_feature_matrix,
    group_importance_by_feature_family,
    make_binary_neighbourhood_dataset,
    parse_query_ids,
    plot_shap_outputs,
    plot_signed_feature_statistics,
    select_neighbour_ids,
)
from cpatk.plotting import plot_feature_importance
from cpatk.reporting import default_methods_text, make_html_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run CPATK feature attribution and neighbourhood explanation.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--class_column",
        default=None,
        help="Known class/MOA column for global supervised feature attribution.",
    )
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--model_name", default="random_forest")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--include_shap", action="store_true")
    parser.add_argument("--max_shap_background", type=int, default=200)
    parser.add_argument("--max_shap_explain", type=int, default=200)

    parser.add_argument("--id_column", default="cpd_id", help="Profile identifier column for query/neighbour analysis.")
    parser.add_argument("--query_ids", nargs="+", default=None, help="One or more query IDs; comma-separated values are also accepted.")
    parser.add_argument("--query_file", default=None, help="Optional text/CSV/TSV file containing query IDs.")
    parser.add_argument("--nn_file", default=None, help="Nearest-neighbour table with cpd_id/query_id and neighbour_id columns.")
    parser.add_argument("--n_neighbours", type=int, default=5)
    parser.add_argument("--run_neighbourhood_shap", action="store_true")
    parser.add_argument(
        "--run_query_background_shap",
        action="store_true",
        help=(
            "Run a local SHAP-style binary explanation for each query ID versus "
            "the configured background/control values, for example compound vs DMSO."
        ),
    )
    parser.add_argument("--run_feature_tests", action="store_true")
    parser.add_argument("--background_column", default="cpd_type")
    parser.add_argument("--background_values", default="DMSO,control,negative_control")
    parser.add_argument("--test", choices=["mw", "ks"], default="mw")
    parser.add_argument("--n_top_features", type=int, default=20)
    parser.add_argument("--n_dependence_plots", type=int, default=5)

    parser.add_argument("--disable_html_report", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def _run_global_attribution(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
    logger,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Run global supervised permutation/SHAP feature attribution."""
    global_dir = output_dir / "global_supervised_attribution"
    global_dir.mkdir(parents=True, exist_ok=True)
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
    tables: dict[str, pd.DataFrame] = {
        "global_permutation_importance": permutation,
        "global_permutation_summary": permutation_summary,
        "global_feature_family_importance": family,
    }
    plot_paths: list[Path] = []
    plot_paths.extend(
        plot_feature_importance(
            importance_table=permutation,
            value_column="permutation_importance_mean",
            output_path_base=global_dir / "permutation_feature_importance",
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
        tables["global_shap_importance"] = shap_importance
        tables["global_shap_class_importance"] = shap_class_importance
        tables["global_shap_status"] = shap_status
        if not shap_importance.empty:
            plot_paths.extend(
                plot_feature_importance(
                    importance_table=shap_importance,
                    value_column="mean_absolute_shap",
                    output_path_base=global_dir / "shap_feature_importance",
                    title="SHAP global feature importance",
                    logger=logger,
                )
            )
    for name, table in tables.items():
        write_table(data_frame=table, path=global_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=tables, path=global_dir / "global_feature_attribution_summary.xlsx", logger=logger)
    return tables, plot_paths


def _run_query_neighbourhoods(
    *,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    logger,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Run query-vs-background statistics and query-vs-neighbour SHAP."""
    queries = parse_query_ids(query_ids=args.query_ids, query_file=args.query_file)
    if not queries:
        return {}, []
    if args.id_column not in metadata.columns:
        raise ValueError(f"id_column '{args.id_column}' is not present in metadata. Add it to --metadata_columns if needed.")
    neighbour_table = read_table(path=args.nn_file, logger=logger) if args.nn_file else None
    background_values = [value.strip() for value in str(args.background_values).split(",") if value.strip()]
    all_tables: dict[str, pd.DataFrame] = {}
    all_plots: list[Path] = []
    summary_records = []

    for query_id in queries:
        query_dir = output_dir / "query_neighbourhoods" / str(query_id)
        query_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running query-level explanation for %s", query_id)
        neighbours: list[str] = []
        query_report_tables: dict[str, pd.DataFrame] = {}
        query_report_plots: list[Path] = []
        if neighbour_table is not None:
            neighbours = select_neighbour_ids(
                neighbour_table=neighbour_table,
                query_id=query_id,
                n_neighbours=args.n_neighbours,
            )
            pd.DataFrame({"query_id": query_id, "neighbour_id": neighbours}).to_csv(
                query_dir / "selected_neighbours.tsv",
                sep="\t",
                index=False,
            )
        if args.run_feature_tests:
            try:
                stats = calculate_query_background_statistics(
                    metadata=metadata,
                    features=features.loc[:, feature_columns],
                    id_column=args.id_column,
                    query_id=query_id,
                    background_column=args.background_column,
                    background_values=background_values,
                    test=args.test,
                )
                write_table(data_frame=stats, path=query_dir / "query_vs_background_feature_statistics.tsv", logger=logger)
                top_increased = stats.sort_values(
                    ["median_difference_group_a_minus_group_b", "q_value"],
                    ascending=[False, True],
                ).head(args.n_top_features)
                top_decreased = stats.sort_values(
                    ["median_difference_group_a_minus_group_b", "q_value"],
                    ascending=[True, True],
                ).head(args.n_top_features)
                write_table(data_frame=top_increased, path=query_dir / "top_query_increased_features.tsv", logger=logger)
                write_table(data_frame=top_decreased, path=query_dir / "top_query_decreased_features.tsv", logger=logger)
                all_tables[f"{query_id}_feature_stats"] = stats.head(200)
                query_report_tables["Query vs background feature statistics"] = stats.head(200)
                feature_stat_plots = plot_signed_feature_statistics(
                    stats_table=stats,
                    output_path_base=query_dir / "query_vs_background_signed_feature_shifts",
                    top_n=args.n_top_features,
                    logger=logger,
                )
                all_plots.extend(feature_stat_plots)
                query_report_plots.extend(feature_stat_plots)
            except Exception as exc:
                logger.warning("Feature tests failed for %s: %s", query_id, exc)
                write_table(
                    data_frame=pd.DataFrame.from_records([{"query_id": query_id, "status": "failed", "message": str(exc)}]),
                    path=query_dir / "query_vs_background_feature_statistics_status.tsv",
                    logger=logger,
                )
        if args.run_query_background_shap:
            try:
                ids = metadata[args.id_column].map(lambda value: str(value).strip().lower())
                query_key = str(query_id).strip().lower()
                query_mask = ids == query_key
                accepted_background = {str(value).strip().lower() for value in background_values}
                if args.background_column in metadata.columns:
                    background_mask = metadata[args.background_column].map(
                        lambda value: str(value).strip().lower()
                    ).isin(accepted_background)
                else:
                    background_mask = ~query_mask
                combined_mask = (query_mask | background_mask).to_numpy()
                if int(query_mask.sum()) == 0:
                    raise ValueError(f"No profiles found for query_id: {query_id}")
                if int(background_mask.sum()) == 0:
                    raise ValueError("No background/control profiles available for SHAP comparison.")
                subset_features = features.loc[combined_mask, :].reset_index(drop=True)
                subset_metadata = metadata.loc[combined_mask, :].reset_index(drop=True)
                clean_x, column_audit = clean_numeric_feature_matrix(
                    features=subset_features,
                    explicit_feature_columns=feature_columns,
                )
                y = subset_metadata[args.id_column].map(lambda value: str(value).strip().lower()).eq(query_key).astype(int)
                if y.nunique() < 2:
                    raise ValueError("Query-background SHAP requires both query and background profiles.")
                write_table(data_frame=subset_metadata, path=query_dir / "query_vs_background_metadata.tsv", logger=logger)
                write_table(data_frame=column_audit, path=query_dir / "query_vs_background_column_audit.tsv", logger=logger)
                shap_result = calculate_neighbourhood_shap(
                    x=clean_x,
                    y=y,
                    query_id=f"{query_id}_vs_background",
                    n_top_features=args.n_top_features,
                    max_background=args.max_shap_background,
                    max_explain=args.max_shap_explain,
                    n_jobs=1,
                    logger=logger,
                )
                top_features = shap_result["top_features"]  # type: ignore[index]
                low_features = shap_result["low_contribution_features"]  # type: ignore[index]
                shap_values = shap_result["sample_feature_shap_values"]  # type: ignore[index]
                shap_status = shap_result["status"]  # type: ignore[index]
                write_table(data_frame=top_features, path=query_dir / "query_vs_background_top_shap_features.tsv", logger=logger)
                write_table(data_frame=low_features, path=query_dir / "query_vs_background_low_contribution_shap_features.tsv", logger=logger)
                write_table(data_frame=shap_values, path=query_dir / "query_vs_background_sample_feature_shap_values.tsv.gz", logger=logger)
                write_table(data_frame=shap_status, path=query_dir / "query_vs_background_shap_status.tsv", logger=logger)
                family = group_importance_by_feature_family(importance_table=top_features, value_column="mean_absolute_shap")
                write_table(data_frame=family, path=query_dir / "query_vs_background_shap_feature_family_summary.tsv", logger=logger)
                all_tables[f"{query_id}_vs_background_shap_top_features"] = top_features
                all_tables[f"{query_id}_vs_background_shap_status"] = shap_status
                query_report_tables["Query vs background SHAP top features"] = top_features
                query_report_tables["Query vs background SHAP status"] = shap_status
                query_report_tables["Query vs background SHAP feature-family summary"] = family
                background_plots = plot_shap_outputs(
                    shap_array=shap_result["shap_array"],  # type: ignore[arg-type,index]
                    explained_x=shap_result["explained_x"],  # type: ignore[arg-type,index]
                    top_features=top_features,
                    output_path_base=query_dir / "query_vs_background_shap",
                    max_display=args.n_top_features,
                    n_dependence=args.n_dependence_plots,
                    logger=logger,
                )
                all_plots.extend(background_plots)
                query_report_plots.extend(background_plots)
                write_excel_workbook(
                    tables={
                        "top_shap_features": top_features,
                        "low_contribution_features": low_features,
                        "shap_status": shap_status,
                        "feature_family_summary": family,
                    },
                    path=query_dir / "query_vs_background_shap_summary.xlsx",
                    logger=logger,
                )
            except Exception as exc:
                logger.warning("Query-vs-background SHAP failed for %s: %s", query_id, exc)
                status = pd.DataFrame.from_records(
                    [{"query_id": query_id, "status": "failed", "message": str(exc)}]
                )
                write_table(data_frame=status, path=query_dir / "query_vs_background_shap_status.tsv", logger=logger)
                query_report_tables["Query vs background SHAP status"] = status

        if args.run_neighbourhood_shap:
            if not neighbours:
                logger.warning("Skipping neighbourhood SHAP for %s because no neighbours were selected.", query_id)
            else:
                try:
                    x, y, subset_metadata, column_audit = make_binary_neighbourhood_dataset(
                        metadata=metadata,
                        features=features,
                        id_column=args.id_column,
                        query_id=query_id,
                        neighbour_ids=neighbours,
                        feature_columns=feature_columns,
                    )
                    write_table(data_frame=subset_metadata, path=query_dir / "query_neighbourhood_metadata.tsv", logger=logger)
                    write_table(data_frame=column_audit, path=query_dir / "query_neighbourhood_column_audit.tsv", logger=logger)
                    shap_result = calculate_neighbourhood_shap(
                        x=x,
                        y=y,
                        query_id=query_id,
                        n_top_features=args.n_top_features,
                        max_background=args.max_shap_background,
                        max_explain=args.max_shap_explain,
                        n_jobs=1,
                        logger=logger,
                    )
                    top_features = shap_result["top_features"]  # type: ignore[index]
                    low_features = shap_result["low_contribution_features"]  # type: ignore[index]
                    shap_values = shap_result["sample_feature_shap_values"]  # type: ignore[index]
                    shap_status = shap_result["status"]  # type: ignore[index]
                    write_table(data_frame=top_features, path=query_dir / "top_shap_features_driving_query_difference.tsv", logger=logger)
                    write_table(data_frame=low_features, path=query_dir / "low_contribution_shap_features.tsv", logger=logger)
                    write_table(data_frame=shap_values, path=query_dir / "sample_feature_shap_values.tsv.gz", logger=logger)
                    write_table(data_frame=shap_status, path=query_dir / "neighbourhood_shap_status.tsv", logger=logger)
                    family = group_importance_by_feature_family(importance_table=top_features, value_column="mean_absolute_shap")
                    write_table(data_frame=family, path=query_dir / "top_shap_feature_family_summary.tsv", logger=logger)
                    all_tables[f"{query_id}_shap_top_features"] = top_features
                    all_tables[f"{query_id}_shap_status"] = shap_status
                    query_report_tables["Neighbourhood SHAP top features"] = top_features
                    query_report_tables["Neighbourhood SHAP status"] = shap_status
                    query_report_tables["Neighbourhood SHAP feature-family summary"] = family
                    neighbourhood_plots = plot_shap_outputs(
                        shap_array=shap_result["shap_array"],  # type: ignore[arg-type,index]
                        explained_x=shap_result["explained_x"],  # type: ignore[arg-type,index]
                        top_features=top_features,
                        output_path_base=query_dir / "neighbourhood_shap",
                        max_display=args.n_top_features,
                        n_dependence=args.n_dependence_plots,
                        logger=logger,
                    )
                    all_plots.extend(neighbourhood_plots)
                    query_report_plots.extend(neighbourhood_plots)
                    write_excel_workbook(
                        tables={
                            "top_shap_features": top_features,
                            "low_contribution_features": low_features,
                            "shap_status": shap_status,
                            "feature_family_summary": family,
                            "selected_neighbours": pd.DataFrame({"query_id": query_id, "neighbour_id": neighbours}),
                        },
                        path=query_dir / "query_neighbourhood_explanation_summary.xlsx",
                        logger=logger,
                    )
                except Exception as exc:
                    logger.warning("Neighbourhood SHAP failed for %s: %s", query_id, exc)
                    status = pd.DataFrame.from_records(
                        [{"query_id": query_id, "status": "failed", "message": str(exc)}]
                    )
                    write_table(
                        data_frame=status,
                        path=query_dir / "neighbourhood_shap_status.tsv",
                        logger=logger,
                    )
                    query_report_tables["Neighbourhood SHAP status"] = status

        if query_report_tables:
            make_html_report(
                title=f"CPATK feature explanation report: {query_id}",
                output_path=query_dir / "query_explanation_report.html",
                summary_tables=query_report_tables,
                plot_paths=query_report_plots,
                narrative=(
                    f"Feature-level explanation for {query_id}. This report includes any available "
                    "query-vs-control statistics, query-vs-control SHAP output and local "
                    "neighbourhood SHAP output."
                ),
                methods_text=default_methods_text(),
                warnings=[
                    "This report explains features associated with a contrast; it does not prove causality.",
                    "For default CPATK stress-test shells, query-vs-background uses DMSO/control values when available.",
                ],
            )

        summary_records.append(
            {
                "query_id": query_id,
                "n_selected_neighbours": len(neighbours),
                "feature_tests_requested": bool(args.run_feature_tests),
                "query_background_shap_requested": bool(args.run_query_background_shap),
                "neighbourhood_shap_requested": bool(args.run_neighbourhood_shap),
                "background_column": args.background_column,
                "background_values": ",".join(background_values),
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    if not summary.empty:
        all_tables["query_neighbourhood_summary"] = summary
        write_table(data_frame=summary, path=output_dir / "query_neighbourhoods" / "query_neighbourhood_summary.tsv", logger=logger)
    return all_tables, all_plots


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "explain.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)

    extra_metadata = [column for column in [args.class_column, args.id_column, args.background_column] if column]
    metadata, features, metadata_columns, feature_columns = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        additional_metadata_columns=extra_metadata,
    )
    tables: dict[str, pd.DataFrame] = {}
    plot_paths: list[Path] = []

    if args.class_column:
        if args.class_column not in metadata.columns:
            raise ValueError(f"class_column '{args.class_column}' is not present after metadata/feature splitting.")
        global_tables, global_plots = _run_global_attribution(
            features=features,
            labels=metadata[args.class_column],
            args=args,
            output_dir=output_dir,
            logger=logger,
        )
        tables.update(global_tables)
        plot_paths.extend(global_plots)

    query_tables, query_plots = _run_query_neighbourhoods(
        metadata=metadata,
        features=features,
        feature_columns=feature_columns,
        args=args,
        output_dir=output_dir,
        logger=logger,
    )
    tables.update(query_tables)
    plot_paths.extend(query_plots)

    if not tables:
        raise ValueError("Nothing to explain. Provide --class_column and/or --query_ids/--query_file.")

    write_excel_workbook(tables=tables, path=output_dir / "feature_explanation_summary.xlsx", logger=logger)
    if not args.disable_html_report:
        warnings = [
            "Feature attribution explains model behaviour and per-feature statistical differences. It does not prove biological causality.",
            "Neighbourhood SHAP is a local query-vs-neighbour explanation and should be interpreted with replicate consistency, preprocessing QC, batch effects and known biology.",
        ]
        make_html_report(
            title="CPATK feature explanation report",
            output_path=output_dir / "feature_explanation_report.html",
            summary_tables=tables,
            plot_paths=plot_paths,
            narrative="CPATK calculated supervised and/or query-neighbourhood feature explanations.",
            methods_text=default_methods_text(),
            warnings=warnings,
        )
    logger.info("CPATK feature explanation workflow complete")


if __name__ == "__main__":
    main()
