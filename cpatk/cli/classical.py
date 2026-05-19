"""Command-line classical analysis workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cpatk.clustering import calculate_silhouette_summary, run_kmeans, summarise_clusters
from cpatk.distances import calculate_nearest_neighbours, calculate_pairwise_distance_matrix
from cpatk.embedding import run_pca, run_umap_or_pca
from cpatk.features import parse_column_list, split_metadata_and_features
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.plotting import plot_embedding, write_interactive_embedding_html


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser object.
    """
    parser = argparse.ArgumentParser(description="Run non-AI classical Cell Painting analysis.")
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--id_column", default=None)
    parser.add_argument("--colour_column", default=None)
    parser.add_argument("--cluster_group_columns", default=None)
    parser.add_argument("--distance_metric", default="cosine")
    parser.add_argument("--n_neighbours", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "classical.log", log_level=args.log_level)
    data_frame = read_table(path=args.input_table, logger=logger)
    metadata, features, _, _ = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
    )

    pca_scores, pca_variance = run_pca(features=features, n_components=2)
    embedding = run_umap_or_pca(features=features, n_components=2, logger=logger)
    distance_matrix = calculate_pairwise_distance_matrix(features=features, metric=args.distance_metric)
    neighbours = calculate_nearest_neighbours(
        distance_matrix=distance_matrix,
        metadata=metadata,
        id_column=args.id_column,
        n_neighbours=args.n_neighbours,
    )
    clusters = run_kmeans(features=features, n_clusters=args.n_clusters)
    cluster_summary = summarise_clusters(
        metadata=metadata,
        clusters=clusters["cluster"],
        group_columns=parse_column_list(value=args.cluster_group_columns),
    )
    silhouette = calculate_silhouette_summary(features=features, clusters=clusters["cluster"])

    write_table(data_frame=pca_scores, path=output_dir / "pca_scores.tsv", logger=logger)
    write_table(data_frame=pca_variance, path=output_dir / "pca_explained_variance.tsv", logger=logger)
    write_table(data_frame=embedding, path=output_dir / "embedding.tsv", logger=logger)
    write_table(data_frame=distance_matrix, path=output_dir / "pairwise_distances.tsv.gz", index=True, logger=logger)
    write_table(data_frame=neighbours, path=output_dir / "nearest_neighbours.tsv", logger=logger)
    write_table(data_frame=clusters, path=output_dir / "clusters.tsv", logger=logger)
    write_table(data_frame=cluster_summary, path=output_dir / "cluster_summary.tsv", logger=logger)
    write_table(data_frame=silhouette, path=output_dir / "cluster_silhouette_summary.tsv", logger=logger)

    plot_embedding(
        embedding=embedding,
        metadata=metadata,
        x_column=embedding.columns[0],
        y_column=embedding.columns[1],
        colour_column=args.colour_column,
        output_path_base=output_dir / "embedding_static",
        logger=logger,
    )
    write_interactive_embedding_html(
        embedding=embedding,
        metadata=metadata,
        x_column=embedding.columns[0],
        y_column=embedding.columns[1],
        colour_column=args.colour_column,
        output_path=output_dir / "embedding_interactive.html",
        logger=logger,
    )
    write_excel_workbook(
        tables={
            "pca_variance": pca_variance,
            "nearest_neighbours": neighbours.head(5000),
            "clusters": clusters,
            "cluster_summary": cluster_summary,
            "silhouette": silhouette,
        },
        path=output_dir / "classical_analysis_summary.xlsx",
        logger=logger,
    )
    logger.info("CPATK classical analysis complete")


if __name__ == "__main__":
    main()
