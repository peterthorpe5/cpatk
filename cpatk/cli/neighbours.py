"""Command-line nearest-neighbour analysis for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.logging_utils import configure_logging
from cpatk.neighbour_analysis import run_neighbour_workflow


def _parse_csv(value: str | None) -> list[str]:
    """Parse comma-separated strings."""
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        description="Create nearest-neighbour plots and overlap summaries."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_neighbours", default=None)
    parser.add_argument("--compounds", default="", help="Comma-separated compounds to plot; first two are compared.")
    parser.add_argument("--baseline_neighbours", default=None)
    parser.add_argument("--run_neighbours", action="append", default=[])
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--k", default="auto")
    parser.add_argument("--include_ties_at_k", action="store_true")
    parser.add_argument("--no_rbo", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run nearest-neighbour CLI."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "neighbour_analysis.log", log_level=args.log_level)
    logger.info("Starting nearest-neighbour analysis.")
    run_neighbour_workflow(
        output_dir=output_dir,
        input_neighbours=Path(args.input_neighbours) if args.input_neighbours else None,
        compounds=_parse_csv(args.compounds),
        baseline_neighbours=Path(args.baseline_neighbours) if args.baseline_neighbours else None,
        run_neighbours=[Path(path) for path in args.run_neighbours],
        top_n=args.top_n,
        k=args.k,
        include_ties_at_k=args.include_ties_at_k,
        with_rbo=not args.no_rbo,
    )
    logger.info("Nearest-neighbour analysis complete: %s", output_dir)


if __name__ == "__main__":
    main()
