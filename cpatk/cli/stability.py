"""Command-line reproducibility and stability workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.plotting import plot_heatmap
from cpatk.reporting import make_html_report
from cpatk.reproducibility import (
    bootstrap_cluster_stability,
    bootstrap_neighbour_stability,
    calculate_replicate_correlations,
    consensus_clustering,
    evaluate_kmeans_k_range,
    permutation_test_cluster_structure_detailed,
    summarise_replicate_correlations,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run CPATK reproducibility and stability checks.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--replicate_group_columns", default=None)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--k_values", default=None, help="Optional comma-separated K values, e.g. 2,3,4,5,6.")
    parser.add_argument("--n_bootstraps", type=int, default=50)
    parser.add_argument("--n_permutations", type=int, default=100)
    parser.add_argument("--n_neighbours", type=int, default=10)
    parser.add_argument("--sample_fraction", type=float, default=0.8)
    parser.add_argument("--feature_fraction", type=float, default=0.8)
    parser.add_argument("--disable_html_report", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "stability.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata, features, _, _ = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
    )
    replicate_columns = parse_column_list(value=args.replicate_group_columns)
    tables = {}
    if replicate_columns:
        replicate_pairs = calculate_replicate_correlations(
            features=features,
            metadata=metadata,
            replicate_group_columns=replicate_columns,
        )
        replicate_summary = summarise_replicate_correlations(
            replicate_correlations=replicate_pairs,
            group_columns=replicate_columns,
        )
        tables["replicate_correlations"] = replicate_pairs
        tables["replicate_summary"] = replicate_summary
    neighbour_stability = bootstrap_neighbour_stability(
        features=features,
        n_neighbours=args.n_neighbours,
        n_bootstraps=args.n_bootstraps,
        feature_fraction=args.feature_fraction,
        logger=logger,
    )
    cluster_permutation, cluster_permutation_null = permutation_test_cluster_structure_detailed(
        features=features,
        n_clusters=args.n_clusters,
        n_permutations=args.n_permutations,
        logger=logger,
    )
    cluster_stability = bootstrap_cluster_stability(
        features=features,
        n_clusters=args.n_clusters,
        n_bootstraps=args.n_bootstraps,
        sample_fraction=args.sample_fraction,
        logger=logger,
    )
    consensus_matrix, consensus_summary = consensus_clustering(
        features=features,
        n_clusters=args.n_clusters,
        n_bootstraps=args.n_bootstraps,
        sample_fraction=args.sample_fraction,
        logger=logger,
    )
    tables.update(
        {
            "neighbour_stability": neighbour_stability,
            "cluster_permutation": cluster_permutation,
            "cluster_permutation_null": cluster_permutation_null,
            "cluster_stability": cluster_stability,
            "consensus_summary": consensus_summary,
        }
    )
    k_values = parse_column_list(value=args.k_values)
    if k_values:
        tables["cluster_k_range_evaluation"] = evaluate_kmeans_k_range(
            features=features,
            k_values=[int(value) for value in k_values],
            n_bootstraps=max(5, min(args.n_bootstraps, 30)),
            n_permutations=max(5, min(args.n_permutations, 50)),
            sample_fraction=args.sample_fraction,
            logger=logger,
        )
    for name, table in tables.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_table(data_frame=consensus_matrix, path=output_dir / "consensus_matrix.tsv.gz", index=True, logger=logger)
    write_excel_workbook(tables=tables, path=output_dir / "stability_summary.xlsx", logger=logger)
    plot_paths = []
    if consensus_matrix.shape[0] <= 250:
        plot_paths.extend(
            plot_heatmap(
                matrix=consensus_matrix,
                output_path_base=output_dir / "consensus_matrix_heatmap",
                title="Consensus co-clustering matrix",
                value_label="Consensus",
                logger=logger,
            )
        )
    if not args.disable_html_report:
        warnings = [
            "Cluster permutation tests evaluate whether the observed clustering is stronger than a feature-shuffled null; they do not prove a uniquely correct biological cluster count.",
            "Bootstrap and consensus summaries should be interpreted together with replicate reproducibility and known control behaviour.",
        ]
        make_html_report(
            title="CPATK stability and reproducibility report",
            output_path=output_dir / "stability_report.html",
            summary_tables=tables,
            plot_paths=plot_paths,
            narrative=(
                "CPATK assessed replicate reproducibility, nearest-neighbour stability, K-means cluster stability, "
                "feature-permutation evidence for cluster structure and consensus co-clustering."
            ),
            methods_text=(
                "Neighbour stability was estimated by repeatedly subsampling feature columns and comparing nearest-neighbour sets by Jaccard overlap. "
                "Cluster stability was estimated by sample subsampling and adjusted Rand index against full-data K-means labels. "
                "The cluster permutation test shuffled each feature independently across profiles to break coordinated morphology while preserving marginal feature distributions."
            ),
            warnings=warnings,
        )
    logger.info("CPATK stability workflow complete")


if __name__ == "__main__":
    main()
