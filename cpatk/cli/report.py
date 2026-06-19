"""Command-line report-generation workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.io import read_table
from cpatk.logging_utils import configure_logging
from cpatk.reporting import default_methods_text, make_html_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Create a CPATK HTML summary report.")
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--title", default="CPATK Cell Painting analysis report")
    parser.add_argument("--table", action="append", default=[])
    parser.add_argument("--plot", action="append", default=[])
    parser.add_argument("--narrative", default=None)
    parser.add_argument("--warning", action="append", default=[])
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
    tables = {}
    for name, path in parse_named_paths(values=args.table).items():
        tables[name] = read_table(path=path, logger=logger)
    plot_paths = [Path(value) for value in args.plot]
    make_html_report(
        title=args.title,
        output_path=output_html,
        summary_tables=tables,
        plot_paths=plot_paths,
        narrative=args.narrative,
        methods_text=default_methods_text(),
        warnings=args.warning,
    )
    logger.info("CPATK report written: %s", output_html)


if __name__ == "__main__":
    main()
