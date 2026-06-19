"""Command-line visualisation workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.logging_utils import configure_logging
from cpatk.reporting import make_html_report
from cpatk.visualisation import run_visualisation_workflow
from cpatk.io import read_table


def _parse_csv(value: str | None) -> list[str]:
    """Parse a comma-separated CLI value."""
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create PCA/UMAP/heatmap/topology visualisations from CPATK tables."
    )
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_columns", default="")
    parser.add_argument("--id_column", default="")
    parser.add_argument("--colour_columns", default="")
    parser.add_argument("--latent_prefix", default=None)
    parser.add_argument("--digit_named_latents", action="store_true")
    parser.add_argument("--aggregate_by_id", action="store_true")
    parser.add_argument("--aggregate_method", choices=["median", "mean"], default="median")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--no_umap", action="store_true")
    parser.add_argument("--phate", action="store_true")
    parser.add_argument("--no_heatmap", action="store_true")
    parser.add_argument("--no_topology", action="store_true")
    parser.add_argument("--no_interactive", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run visualisation CLI."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "visualise.log", log_level=args.log_level)
    run_visualisation_workflow(
        input_table=Path(args.input_table),
        output_dir=output_dir,
        metadata_columns=_parse_csv(args.metadata_columns),
        id_column=args.id_column or None,
        colour_columns=_parse_csv(args.colour_columns),
        latent_prefix=args.latent_prefix,
        digit_named_latents=args.digit_named_latents,
        aggregate_by_id=args.aggregate_by_id,
        aggregate_method=args.aggregate_method,
        make_pca=not args.no_pca,
        make_umap=not args.no_umap,
        make_phate=args.phate,
        make_heatmap=not args.no_heatmap,
        make_topology=not args.no_topology,
        interactive=not args.no_interactive,
        logger=logger,
    )
    tables = {}
    for path in [
        output_dir / "latent_norm_summary.tsv",
        output_dir / "pca_variance.tsv",
        output_dir / "visualisation_feature_columns.tsv",
        output_dir / "topology_nodes.tsv",
        output_dir / "topology_edges.tsv",
    ]:
        if path.exists():
            tables[path.stem] = read_table(path=path, logger=logger)
    plot_paths = sorted((output_dir / "plots").glob("*.svg"))[:100]
    plot_paths.extend(sorted((output_dir / "plots").glob("*.html"))[:30])
    make_html_report(
        title="CPATK visualisation report",
        output_path=output_dir / "visualisation_report.html",
        summary_tables=tables,
        plot_paths=plot_paths,
        narrative="PCA, optional UMAP/PHATE, latent norms, heatmaps and topology plots were generated.",
    )
    logger.info("Visualisation workflow complete: %s", output_dir)


if __name__ == "__main__":
    main()
