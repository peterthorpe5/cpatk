"""Command-line report-generation workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.io import read_table
from cpatk.logging_utils import configure_logging
from cpatk.method_guidance import export_ml_nn_method_guide
from cpatk.reporting import default_methods_text, discover_plot_paths, make_html_report
from cpatk.strategy_selection import summarise_preprocessing_strategies


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Create a CPATK HTML summary report.")
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--title", default="CPATK Cell Painting analysis report")
    parser.add_argument("--table", action="append", default=[])
    parser.add_argument("--plot", action="append", default=[])
    parser.add_argument("--narrative", default=None)
    parser.add_argument("--warning", action="append", default=[])
    parser.add_argument(
        "--disable_auto_discover_plots",
        action="store_true",
        help="Do not automatically add SVG/PDF/PNG/HTML outputs below the report directory.",
    )
    parser.add_argument(
        "--auto_plot_root",
        default=None,
        help="Directory to scan for plots. Defaults to the output HTML parent directory.",
    )
    parser.add_argument(
        "--max_auto_plots",
        type=int,
        default=200,
        help="Maximum number of auto-discovered plot/output files to include.",
    )
    parser.add_argument(
        "--max_table_rows",
        type=int,
        default=50,
        help="Maximum preview rows to show for each table in the HTML report.",
    )
    parser.add_argument(
        "--strategy_root",
        default=None,
        help="Optional preprocessing strategy-comparison root to summarise in the report.",
    )
    parser.add_argument(
        "--strategy_batch_column",
        default="Metadata_Plate",
        help="Batch column used for strategy-comparison scoring.",
    )
    parser.add_argument(
        "--strategy_compound_column",
        default="Metadata_Compound",
        help="Compound/treatment column used for strategy-comparison scoring.",
    )
    parser.add_argument(
        "--export_method_guide",
        action="store_true",
        help="Write the bundled ML/NN method guide beside the report.",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser


def parse_named_paths(*, values: list[str]) -> dict[str, Path]:
    """Parse name=path arguments."""
    parsed = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        parsed[name] = Path(path)
    return parsed


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_html = Path(args.output_html)
    logger = configure_logging(log_file=output_html.with_suffix(".log"), log_level=args.log_level)
    table_paths = parse_named_paths(values=args.table)
    tables = {}
    for name, path in table_paths.items():
        tables[name] = read_table(path=path, logger=logger)

    if args.strategy_root:
        strategy_summary = summarise_preprocessing_strategies(
            strategy_root=Path(args.strategy_root),
            batch_column=args.strategy_batch_column,
            compound_column=args.strategy_compound_column,
            logger=logger,
        )
        strategy_path = output_html.parent / "normalisation_strategy_comparison.tsv"
        strategy_summary.to_csv(strategy_path, sep="\t", index=False)
        tables["Normalisation strategy comparison"] = strategy_summary
        table_paths["Normalisation strategy comparison"] = strategy_path

    if args.export_method_guide:
        export_ml_nn_method_guide(output_dir=output_html.parent / "method_guides")

    explicit_plot_paths = [Path(value) for value in args.plot]
    auto_plot_paths: list[Path] = []
    if not args.disable_auto_discover_plots:
        auto_root = Path(args.auto_plot_root) if args.auto_plot_root else output_html.parent
        auto_plot_paths = discover_plot_paths(
            root_dir=auto_root,
            output_html=output_html,
            max_plots=args.max_auto_plots,
        )
        logger.info("Auto-discovered %s plot/output files for report.", len(auto_plot_paths))

    seen = set()
    plot_paths = []
    for path in [*explicit_plot_paths, *auto_plot_paths]:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        plot_paths.append(path)

    make_html_report(
        title=args.title,
        output_path=output_html,
        summary_tables=tables,
        table_paths=table_paths,
        plot_paths=plot_paths,
        narrative=args.narrative,
        methods_text=default_methods_text(),
        warnings=args.warning,
        max_table_rows=args.max_table_rows,
    )
    logger.info("CPATK report written: %s", output_html)


if __name__ == "__main__":
    main()
