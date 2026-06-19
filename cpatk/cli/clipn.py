"""Command-line optional CLIPn adapter workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.clipn_adapter import ClipnAdapterConfig, check_clipn_backend, run_clipn_adapter, save_clipn_config
from cpatk.features import parse_column_list
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run the optional CPATK CLIPn adapter.")
    parser.add_argument("--dataset", action="append", required=True, help="Named dataset as name=path. Can be repeated.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend_module", default="clipn")
    parser.add_argument("--model_class", default=None)
    parser.add_argument("--fit_method", default="fit")
    parser.add_argument("--predict_method", default="predict")
    parser.add_argument("--transform_method", default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser


def parse_named_datasets(*, values: list[str]) -> dict[str, Path]:
    """Parse name=path dataset arguments."""
    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Dataset argument must use name=path syntax: {value}")
        name, path = value.split("=", 1)
        datasets[name] = Path(path)
    return datasets


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=output_dir / "clipn_adapter.log", log_level=args.log_level)
    dataset_paths = parse_named_datasets(values=args.dataset)
    datasets = {name: read_table(path=path, logger=logger) for name, path in dataset_paths.items()}
    config = ClipnAdapterConfig(
        backend_module=args.backend_module,
        model_class=args.model_class,
        fit_method=args.fit_method,
        predict_method=args.predict_method,
        transform_method=args.transform_method,
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
    )
    save_clipn_config(config=config, path=output_dir / "clipn_adapter_config.json")
    result = run_clipn_adapter(datasets=datasets, config=config, logger=logger)
    for name, table in result.items():
        write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=result, path=output_dir / "clipn_adapter_summary.xlsx", logger=logger)
    logger.info("CPATK CLIPn adapter workflow complete")


if __name__ == "__main__":
    main()
