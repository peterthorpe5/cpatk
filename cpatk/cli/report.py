"""Command-line HTML report workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.io import read_table, write_table
from cpatk.logging_utils import configure_logging
from cpatk.reporting import make_html_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser object.
    """
    parser = argparse.ArgumentParser(description="Create a generic CPATK HTML report.")
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--title", default="CPATK Cell Painting analysis report")
    parser.add_argument("--narrative", default=None)
    parser.add_argument("--table", action="append", default=[])
    parser.add_argument("--plot", action="append", default=[])
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_html = Path(args.output_html)
    logger = configure_logging(log_file=output_html.with_suffix(".log"), log_level=args.log_level)
    tables = {}
    for table_path in args.table:
        path = Path(table_path)
        tables[path.stem] = read_table(path=path, logger=logger)
    report_path = make_html_report(
        title=args.title,
        output_path=output_html,
        summary_tables=tables,
        plot_paths=[Path(path) for path in args.plot],
        narrative=args.narrative,
    )
    summary = __import__("pandas").DataFrame.from_records(
        [{"report_path": str(report_path), "n_tables": len(tables), "n_plots": len(args.plot)}]
    )
    write_table(data_frame=summary, path=output_html.with_name("report_generation_summary.tsv"), logger=logger)
    logger.info("CPATK report complete")


if __name__ == "__main__":
    main()
