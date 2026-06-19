"""Command-line per-compartment acquisition drift QC for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.logging_utils import configure_logging
from cpatk.qc_drift import run_drift_qc


def _parse_csv(value: str | None) -> list[str]:
    """Parse comma-separated strings."""
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run per-compartment object-level acquisition drift QC."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_col", default="ImageNumber")
    parser.add_argument("--include_glob", action="append", default=[])
    parser.add_argument("--feature_columns", default="")
    parser.add_argument("--max_features", type=int, default=200)
    parser.add_argument("--plot_top_n", type=int, default=8)
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    """Run drift QC CLI."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "drift_qc.log", log_level=args.log_level)
    run_drift_qc(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        image_col=args.image_col,
        include_globs=args.include_glob or None,
        feature_columns=_parse_csv(args.feature_columns) or None,
        max_features=args.max_features,
        plot_top_n=args.plot_top_n,
        min_points=args.min_points,
        logger=logger,
    )
    logger.info("Drift QC complete: %s", output_dir)


if __name__ == "__main__":
    main()
