"""Command-line MOA workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
from cpatk.moa_advanced import (
    anchor_permutation_test,
    build_moa_centroids,
    label_pseudo_anchor_clusters,
    make_pseudo_anchors,
    normalise_phenotype_label_table,
    pairwise_distance_outputs,
    plot_k_selection,
    plot_prediction_score_distribution,
    plot_projection,
    plot_score_heatmap,
    prepare_embedding_matrix,
    project_embedding_with_centroids,
    score_against_moa_centroids,
    score_matrix_table,
)
from cpatk.plotting import plot_prediction_confidence
from cpatk.reporting import default_methods_text, make_html_report


def _parse_k_values(value: str | None) -> list[int] | None:
    """Parse comma-separated K values."""
    if value is None or not str(value).strip():
        return None
    out = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out or None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run MOA classification, pseudo-anchor generation and centroid "
            "scoring for generic Cell Painting features or CLIPn embeddings."
        )
    )
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--class_column",
        default=None,
        help=(
            "Known MOA/class column. If provided, CPATK runs supervised "
            "centroid/KNN validation and can use this as the anchor MOA column."
        ),
    )
    parser.add_argument("--id_column", default="cpd_id")
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--min_class_size", type=int, default=2)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--run_knn", action="store_true")
    parser.add_argument("--n_neighbors", type=int, default=5)

    parser.add_argument(
        "--anchor_table",
        default=None,
        help=(
            "Optional TSV/CSV/Parquet/Excel table containing id_column and an "
            "MOA/anchor column. If omitted, --class_column or "
            "--make_pseudo_anchors can supply anchors."
        ),
    )
    parser.add_argument("--anchor_moa_column", default="pseudo_moa")
    parser.add_argument("--make_pseudo_anchors", action="store_true")
    parser.add_argument(
        "--pseudo_anchor_method",
        choices=["simple", "bootstrap"],
        default="bootstrap",
    )
    parser.add_argument("--n_clusters", type=int, default=30)
    parser.add_argument("--auto_k", action="store_true")
    parser.add_argument("--k_values", default="8,12,16,24,32")
    parser.add_argument("--n_bootstraps", type=int, default=50)
    parser.add_argument("--subsample_fraction", type=float, default=0.8)

    parser.add_argument(
        "--pseudo_anchor_label_table",
        default=None,
        help=(
            "Optional compound-to-phenotype table used to annotate pseudo-anchor "
            "clusters and create a conservative final MOA label column."
        ),
    )
    parser.add_argument("--pseudo_anchor_label_id_column", default="cpd_id")
    parser.add_argument("--pseudo_anchor_label_column", default="label")
    parser.add_argument(
        "--pseudo_anchor_label_split_regex",
        default=None,
        help=(
            "Optional regular expression for splitting multi-label phenotype cells. "
            "Omit to preserve each cell as one curated label."
        ),
    )
    parser.add_argument("--pseudo_anchor_final_moa_column", default="moa_final")
    parser.add_argument("--pseudo_anchor_label_min_labelled_fraction", type=float, default=0.2)
    parser.add_argument("--pseudo_anchor_label_min_dominant_fraction", type=float, default=0.5)
    parser.add_argument("--pseudo_anchor_label_top_n", type=int, default=3)
    parser.add_argument(
        "--annotate_pseudo_anchors_only",
        action="store_true",
        help=(
            "Annotate pseudo-anchor clusters with phenotype labels but keep "
            "pseudo_moa as the centroid-scoring label."
        ),
    )

    parser.add_argument(
        "--aggregate_method",
        choices=["median", "mean"],
        default="median",
    )
    parser.add_argument(
        "--centroid_method",
        choices=["median", "mean"],
        default="median",
    )
    parser.add_argument("--n_subcentroids", type=int, default=1)
    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--adaptive_shrinkage", action="store_true")
    parser.add_argument("--adaptive_c", type=float, default=0.5)
    parser.add_argument("--adaptive_max", type=float, default=0.3)
    parser.add_argument(
        "--score_method",
        choices=["cosine", "csls"],
        default="cosine",
    )
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument(
        "--score_collapse",
        choices=["max", "mean", "median"],
        default="max",
    )
    parser.add_argument("--n_permutations", type=int, default=100)
    parser.add_argument("--distance_metrics", default="cosine,spearman")
    parser.add_argument("--make_projection_plots", action="store_true")
    parser.add_argument(
        "--projection",
        choices=["pca", "umap", "both"],
        default="both",
    )
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--disable_html_report", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def _write_tables(
    *,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    logger,
) -> None:
    """Write named tables as TSV files."""
    for name, table in tables.items():
        if table is None:
            continue
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)


def _normalise_anchor_columns(
    *,
    anchors: pd.DataFrame,
    id_column: str,
    moa_column: str,
) -> pd.DataFrame:
    """Return a clean two-column anchor table."""
    if id_column not in anchors.columns:
        raise KeyError(f"Anchor table is missing identifier column: {id_column}")
    if moa_column not in anchors.columns:
        raise KeyError(f"Anchor table is missing MOA column: {moa_column}")
    out = anchors[[id_column, moa_column]].dropna().copy()
    out[id_column] = out[id_column].astype(str).str.strip()
    out[moa_column] = out[moa_column].astype(str).str.strip()
    out = out.loc[out[id_column].ne("") & out[moa_column].ne(""), :]
    out = out.drop_duplicates(subset=[id_column, moa_column])
    return out


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "moa.log", log_level=args.log_level)
    logger.info("Starting CPATK MOA workflow")
    logger.info("Arguments: %s", vars(args))

    data_frame = read_table(path=args.input_table, logger=logger)
    additional_metadata = [args.id_column]
    if args.class_column:
        additional_metadata.append(args.class_column)
    metadata, features, metadata_columns, feature_columns = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        additional_metadata_columns=additional_metadata,
    )
    if args.id_column not in metadata.columns:
        if args.id_column in data_frame.columns:
            metadata[args.id_column] = data_frame[args.id_column].astype(str)
        else:
            metadata[args.id_column] = data_frame.index.astype(str)
            logger.warning(
                "Identifier column '%s' not found; using row index as identifiers.",
                args.id_column,
            )
    labels = metadata[args.class_column] if args.class_column and args.class_column in metadata.columns else None
    logger.info("Detected %d metadata columns and %d feature columns.", len(metadata_columns), len(feature_columns))

    tables: dict[str, pd.DataFrame] = {
        "moa_feature_audit": pd.DataFrame(
            {
                "column": [*metadata_columns, *feature_columns],
                "role": ["metadata"] * len(metadata_columns) + ["feature"] * len(feature_columns),
            }
        )
    }
    plot_paths: list[Path] = []
    warnings = [
        "MOA predictions should be interpreted with replicate consistency, anchor quality, class size and validation performance.",
    ]

    # ------------------------------------------------------------------
    # Backwards-compatible supervised known-MOA workflow.
    # ------------------------------------------------------------------
    if labels is not None:
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
        tables.update(
            {
                "moa_centroids": centroids.reset_index(),
                "moa_class_summary": class_summary,
                "moa_centroid_scores": scores,
                "moa_top_predictions": predictions,
                "moa_prediction_summary": prediction_summary,
                "centroid_leave_one_out_predictions": loo_predictions,
                "centroid_leave_one_out_summary": loo_summary,
            }
        )
        plot_paths.extend(
            plot_prediction_confidence(
                predictions=predictions,
                confidence_column="softmax_confidence",
                output_path_base=plot_dir / "centroid_prediction_confidence",
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
                    output_path_base=plot_dir / "knn_prediction_confidence",
                    title="KNN prediction confidence distribution",
                    logger=logger,
                )
            )

    # ------------------------------------------------------------------
    # Advanced generic anchor/centroid workflow.
    # ------------------------------------------------------------------
    combined = pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    emb, _, feature_columns, aggregation_summary = prepare_embedding_matrix(
        table=combined,
        id_column=args.id_column,
        feature_columns=feature_columns,
        metadata_columns=metadata_columns,
        aggregate_method=args.aggregate_method,
        normalise=True,
    )
    tables["moa_aggregation_summary"] = aggregation_summary
    anchors = None
    anchor_moa_column = args.anchor_moa_column

    if args.anchor_table:
        anchors_raw = read_table(path=args.anchor_table, logger=logger)
        anchors = _normalise_anchor_columns(
            anchors=anchors_raw,
            id_column=args.id_column,
            moa_column=anchor_moa_column,
        )
        logger.info("Loaded %d anchors from %s.", anchors.shape[0], args.anchor_table)
    elif args.make_pseudo_anchors:
        k_values = _parse_k_values(args.k_values)
        anchors, anchor_summary, clusters, k_selection = make_pseudo_anchors(
            table=combined,
            id_column=args.id_column,
            feature_columns=feature_columns,
            metadata_columns=metadata_columns,
            aggregate_method=args.aggregate_method,
            n_clusters=args.n_clusters,
            auto_k=args.auto_k,
            k_values=k_values,
            bootstrap=args.pseudo_anchor_method == "bootstrap",
            n_bootstraps=args.n_bootstraps,
            subsample_fraction=args.subsample_fraction,
            random_state=args.random_state,
        )
        anchor_moa_column = "pseudo_moa"
        if args.pseudo_anchor_label_table:
            label_raw = read_table(path=args.pseudo_anchor_label_table, logger=logger)
            label_table, label_audit = normalise_phenotype_label_table(
                label_table=label_raw,
                id_column=args.pseudo_anchor_label_id_column,
                label_column=args.pseudo_anchor_label_column,
                split_regex=args.pseudo_anchor_label_split_regex,
            )
            label_table = label_table.rename(
                columns={
                    args.pseudo_anchor_label_id_column: args.id_column,
                    args.pseudo_anchor_label_column: "phenotype_label",
                }
            )
            anchors, clusters, label_summary = label_pseudo_anchor_clusters(
                anchors=anchors,
                clusters=clusters,
                label_table=label_table,
                id_column=args.id_column,
                label_column="phenotype_label",
                pseudo_column="pseudo_moa",
                final_column=args.pseudo_anchor_final_moa_column,
                min_labelled_fraction=args.pseudo_anchor_label_min_labelled_fraction,
                min_dominant_fraction=args.pseudo_anchor_label_min_dominant_fraction,
                top_n_labels=args.pseudo_anchor_label_top_n,
            )
            tables["pseudo_anchor_phenotype_labels"] = label_table
            tables["pseudo_anchor_phenotype_label_audit"] = label_audit
            tables["pseudo_anchor_phenotype_summary"] = label_summary
            if not args.annotate_pseudo_anchors_only:
                anchor_moa_column = args.pseudo_anchor_final_moa_column
            warnings.append(
                "Pseudo-anchor phenotype labels are interpretive annotations based on the "
                "supplied compound-to-phenotype table. Weakly labelled or mixed clusters "
                "retain their pseudo-anchor identifiers as final labels."
            )
            logger.info(
                "Annotated pseudo-anchors with phenotype labels from %s; scoring column: %s.",
                args.pseudo_anchor_label_table,
                anchor_moa_column,
            )
        tables["pseudo_anchors"] = anchors
        tables["pseudo_anchor_summary"] = anchor_summary
        tables["pseudo_anchor_clusters"] = clusters
        tables["pseudo_anchor_k_selection"] = k_selection
        plot_paths.extend(plot_k_selection(table=k_selection, output_path=plot_dir / "pseudo_anchor_k_selection.pdf"))
    elif labels is not None:
        anchors = metadata[[args.id_column, args.class_column]].dropna().copy()
        anchors = anchors.rename(columns={args.class_column: "known_moa_anchor"})
        anchor_moa_column = "known_moa_anchor"
        anchors = _normalise_anchor_columns(
            anchors=anchors,
            id_column=args.id_column,
            moa_column=anchor_moa_column,
        )
        logger.info("Using known class labels as anchors (%d rows).", anchors.shape[0])

    if anchors is not None and not anchors.empty:
        tables["moa_anchor_table"] = anchors
        advanced_centroids, advanced_centroid_summary = build_moa_centroids(
            embedding_table=emb,
            anchors=anchors,
            id_column=args.id_column,
            moa_column=anchor_moa_column,
            feature_columns=feature_columns,
            centroid_method=args.centroid_method,
            n_subcentroids=args.n_subcentroids,
            shrinkage=args.shrinkage,
            adaptive_shrinkage=args.adaptive_shrinkage,
            adaptive_c=args.adaptive_c,
            adaptive_max=args.adaptive_max,
            random_state=args.random_state,
        )
        long_scores, top_predictions = score_against_moa_centroids(
            embedding_table=emb,
            centroid_table=advanced_centroids,
            id_column=args.id_column,
            feature_columns=feature_columns,
            score_method=args.score_method,
            csls_k=args.csls_k,
            top_n=args.top_n,
            collapse=args.score_collapse,
        )
        matrix = score_matrix_table(
            embedding_table=emb,
            centroid_table=advanced_centroids,
            id_column=args.id_column,
            feature_columns=feature_columns,
            score_method=args.score_method,
            csls_k=args.csls_k,
            collapse=args.score_collapse,
        )
        tables.update(
            {
                "advanced_moa_centroids": advanced_centroids,
                "advanced_moa_centroid_summary": advanced_centroid_summary,
                "advanced_moa_scores_long": long_scores,
                "advanced_moa_top_predictions": top_predictions,
                "advanced_moa_score_matrix": matrix,
            }
        )
        plot_paths.extend(
            plot_prediction_score_distribution(
                predictions=top_predictions,
                output_path=plot_dir / "advanced_moa_prediction.pdf",
            )
        )
        plot_paths.extend(
            plot_score_heatmap(
                score_matrix=matrix,
                id_column=args.id_column,
                output_path=plot_dir / "advanced_moa_score_heatmap.pdf",
            )
        )
        if args.n_permutations > 0:
            perm_summary, perm_null = anchor_permutation_test(
                embedding_table=emb,
                anchors=anchors,
                id_column=args.id_column,
                moa_column=anchor_moa_column,
                feature_columns=feature_columns,
                centroid_method=args.centroid_method,
                n_subcentroids=args.n_subcentroids,
                shrinkage=args.shrinkage,
                adaptive_shrinkage=args.adaptive_shrinkage,
                score_method=args.score_method,
                csls_k=args.csls_k,
                collapse=args.score_collapse,
                n_permutations=args.n_permutations,
                random_state=args.random_state,
            )
            tables["advanced_moa_permutation_summary"] = perm_summary
            tables["advanced_moa_permutation_null"] = perm_null
        distance_metrics = [item.strip() for item in args.distance_metrics.split(",") if item.strip()]
        distance_tables = pairwise_distance_outputs(
            embedding_table=emb,
            id_column=args.id_column,
            feature_columns=feature_columns,
            metrics=distance_metrics,
            top_n=args.top_n,
        )
        tables.update(distance_tables)
        if args.make_projection_plots:
            methods = ["pca", "umap"] if args.projection == "both" else [args.projection]
            colour_series = None
            if anchor_moa_column in anchors.columns:
                colour_lookup = anchors.drop_duplicates(args.id_column).set_index(args.id_column)[anchor_moa_column]
                colour_series = emb[args.id_column].astype(str).map(colour_lookup)
            for method in methods:
                try:
                    comp_coords, cent_coords, all_coords = project_embedding_with_centroids(
                        embedding_table=emb,
                        centroid_table=advanced_centroids,
                        id_column=args.id_column,
                        feature_columns=feature_columns,
                        method=method,
                        random_state=args.random_state,
                        umap_n_neighbors=args.umap_n_neighbors,
                        umap_min_dist=args.umap_min_dist,
                    )
                    tables[f"advanced_moa_{method}_coordinates"] = all_coords
                    plot_paths.extend(
                        plot_projection(
                            compound_coords=comp_coords,
                            centroid_coords=cent_coords,
                            id_column=args.id_column,
                            method=method,
                            output_path=plot_dir / f"advanced_moa_{method}.pdf",
                            colour_series=colour_series,
                            interactive=args.interactive,
                        )
                    )
                except Exception as exc:
                    logger.warning("%s projection failed: %s", method, exc)
    else:
        warnings.append(
            "No MOA anchors were supplied or generated, so advanced centroid scoring was skipped."
        )

    _write_tables(tables=tables, output_dir=output_dir, logger=logger)
    write_excel_workbook(
        tables={key: value.head(5000) for key, value in tables.items() if isinstance(value, pd.DataFrame)},
        path=output_dir / "moa_summary.xlsx",
        logger=logger,
    )
    if not args.disable_html_report:
        make_html_report(
            title="CPATK MOA analysis report",
            output_path=output_dir / "moa_report.html",
            summary_tables={key: value.head(200) for key, value in tables.items() if isinstance(value, pd.DataFrame)},
            plot_paths=plot_paths,
            narrative=(
                "CPATK ran generic MOA analysis including supervised centroid/KNN "
                "classification where labels were available, and advanced "
                "anchor-based centroid scoring where anchors or pseudo-anchors were supplied."
            ),
            methods_text=default_methods_text(),
            warnings=warnings,
        )
    logger.info("CPATK MOA analysis complete")


if __name__ == "__main__":
    main()
