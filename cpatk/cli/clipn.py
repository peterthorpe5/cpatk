"""Command-line optional CLIPn workflow for CPATK."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpatk.clipn_adapter import (
    ClipnAdapterConfig,
    load_clipn_datasets,
    read_datasets_manifest,
    run_clipn_workflow,
    split_single_dataset_by_group,
)
from cpatk.features import parse_column_list
from cpatk.io import read_table
from cpatk.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the optional CPATK CLIPn workflow. Inputs should usually be "
            "preprocessed CPATK profile tables, one per dataset/reference/query."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--datasets_csv",
        default=None,
        help="CSV/TSV/CSV.GZ/TSV.GZ manifest with columns dataset and path.",
    )
    input_group.add_argument(
        "--dataset",
        action="append",
        help="Named dataset as name=path. Can be repeated.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment", default="cpatk_clipn")
    parser.add_argument(
        "--mode",
        choices=["integrate_all", "reference_only"],
        default="integrate_all",
        help="Train on all datasets or train on references and project all datasets.",
    )
    parser.add_argument(
        "--reference_names",
        default=None,
        help="Comma-separated reference dataset names for --mode reference_only.",
    )
    parser.add_argument("--backend_module", default="clipn")
    parser.add_argument("--model_class", default=None)
    parser.add_argument("--fit_method", default="fit")
    parser.add_argument("--predict_method", default="predict")
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=300, help="Maximum CLIPn training epochs. Used as a hard cap when early stopping is enabled.")
    parser.add_argument(
        "--clipn_early_stopping",
        action="store_true",
        help="Train in chunks and stop when reported training loss has plateaued. This monitors training loss, not validation loss, because common CLIPn backends do not expose a validation callback.",
    )
    parser.add_argument("--clipn_patience", type=int, default=20, help="Reported loss rows without improvement before stopping when --clipn_early_stopping is enabled.")
    parser.add_argument("--clipn_min_delta", type=float, default=1e-4, help="Minimum loss improvement required to reset CLIPn early-stopping patience.")
    parser.add_argument("--clipn_epoch_chunk_size", type=int, default=10, help="Epochs per backend fit call when --clipn_early_stopping is enabled.")
    parser.add_argument("--clipn_validation_fraction", type=float, default=0.0, help="Recorded for provenance. Current generic CLIPn backend uses training-loss monitoring because it lacks a validation hook.")
    parser.add_argument(
        "--imputation_method",
        choices=["none", "median", "mean", "knn"],
        default="median",
    )
    parser.add_argument(
        "--imputation_group_columns",
        default="Dataset,Plate_Metadata",
        help="Comma-separated group columns for grouped median/mean imputation.",
    )
    parser.add_argument("--max_feature_missing_fraction", type=float, default=0.3)
    parser.add_argument("--max_sample_missing_fraction", type=float, default=0.8)
    parser.add_argument(
        "--scaling_method",
        choices=["none", "robust", "standard"],
        default="robust",
    )
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--metadata_columns", default=None)
    parser.add_argument("--id_column", default="cpd_id")
    parser.add_argument("--label_column", default="cpd_type")
    parser.add_argument("--n_neighbours", type=int, default=15)
    parser.add_argument(
        "--distance_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
    )
    parser.add_argument(
        "--disable_latent_l2_normalisation",
        action="store_true",
        help="Do not row-wise L2-normalise latent vectors before diagnostics.",
    )
    parser.add_argument(
        "--allow_pca_fallback",
        action="store_true",
        help=(
            "If CLIPn is unavailable or fails, produce PCA fallback latent output for "
            "debugging only. This is clearly labelled and is not CLIPn."
        ),
    )
    parser.add_argument("--save_model", default=None, help="Optional pickle path for fitted backend model.")
    parser.add_argument(
        "--split_single_dataset_by_column",
        default=None,
        help=(
            "If exactly one dataset is supplied, split it into two CLIPn datasets by this "
            "grouping column, usually cpd_id or Metadata_Compound. Compound groups are not split."
        ),
    )
    parser.add_argument(
        "--single_dataset_split_names",
        default="split_a,split_b",
        help="Comma-separated names for --split_single_dataset_by_column output datasets.",
    )
    parser.add_argument(
        "--single_dataset_split_seed",
        type=int,
        default=42,
        help="Random seed for reproducible single-dataset splitting.",
    )
    parser.add_argument(
        "--drop_rows_with_any_zero",
        action="store_true",
        help=(
            "Strict legacy/debug option: drop any row containing at least one literal zero "
            "after imputation/scaling. This is usually too aggressive for Cell Painting; "
            "CLIPn can accept literal zeros, so the default is to keep them."
        ),
    )
    parser.add_argument(
        "--clipn_zero_policy",
        choices=["drop_rows", "keep", "error"],
        default="keep",
        help=(
            "How to handle literal zeros after imputation/scaling before CLIPn. "
            "keep leaves real zeros untouched and records an audit; this is the default. "
            "drop_rows/error are mainly for legacy/debugging checks."
        ),
    )
    parser.add_argument(
        "--clipn_zero_epsilon",
        type=float,
        default=1e-8,
        help="Deprecated compatibility argument retained for old scripts; CPATK does not epsilon-replace zeros.",
    )
    parser.add_argument(
        "--keep_all_zero_rows",
        action="store_true",
        help="Do not remove all-zero CLIPn rows. Not recommended.",
    )
    parser.add_argument(
        "--keep_all_zero_features",
        action="store_true",
        help="Do not remove all-zero CLIPn features. Not recommended.",
    )
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


def _dataset_paths_from_args(args: argparse.Namespace, logger) -> dict[str, Path]:
    """Resolve dataset paths from CLI arguments."""
    if args.datasets_csv:
        manifest = read_datasets_manifest(path=args.datasets_csv, logger=logger)
        base_dir = Path(args.datasets_csv).resolve().parent
        paths = {}
        for _, row in manifest.iterrows():
            path = Path(str(row["path"]))
            if not path.exists():
                candidate = base_dir / path
                path = candidate if candidate.exists() else path
            paths[str(row["dataset"])] = path
        return paths
    return parse_named_datasets(values=args.dataset or [])


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        log_file=output_dir / "clipn_adapter.log",
        log_level=args.log_level,
    )
    logger.info("Starting CPATK CLIPn workflow")
    logger.info("Arguments: %s", vars(args))
    dataset_paths = _dataset_paths_from_args(args, logger)
    datasets, alias_report = load_clipn_datasets(dataset_paths=dataset_paths, logger=logger)
    split_report = None
    if len(datasets) == 1 and args.split_single_dataset_by_column:
        original_name = next(iter(datasets))
        split_names = parse_column_list(value=args.single_dataset_split_names) or ["split_a", "split_b"]
        if len(split_names) != 2:
            raise ValueError("--single_dataset_split_names must provide exactly two names.")
        datasets, split_report = split_single_dataset_by_group(
            data_frame=datasets[original_name],
            group_column=args.split_single_dataset_by_column,
            random_state=args.single_dataset_split_seed,
            split_names=(split_names[0], split_names[1]),
        )
        logger.info(
            "Split single CLIPn dataset '%s' into %s using group column '%s'.",
            original_name,
            ",".join(split_names),
            args.split_single_dataset_by_column,
        )
    if not alias_report.empty:
        alias_report.to_csv(output_dir / "clipn_metadata_alias_report.tsv", sep="\t", index=False)
    if split_report is not None:
        split_report.to_csv(output_dir / "clipn_single_dataset_split_report.tsv", sep="\t", index=False)
    config = ClipnAdapterConfig(
        backend_module=args.backend_module,
        model_class=args.model_class,
        fit_method=args.fit_method,
        predict_method=args.predict_method,
        reference_names=parse_column_list(value=args.reference_names),
        metadata_columns=parse_column_list(value=args.metadata_columns),
        feature_columns=parse_column_list(value=args.feature_columns),
        id_column=args.id_column,
        label_column=args.label_column,
        mode=args.mode,
        latent_dim=args.latent_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        early_stopping=args.clipn_early_stopping,
        early_stopping_patience=args.clipn_patience,
        early_stopping_min_delta=args.clipn_min_delta,
        early_stopping_chunk_size=args.clipn_epoch_chunk_size,
        validation_fraction=args.clipn_validation_fraction,
        imputation_method=args.imputation_method,
        imputation_group_columns=parse_column_list(value=args.imputation_group_columns)
        or ["Dataset", "Plate_Metadata"],
        max_feature_missing_fraction=args.max_feature_missing_fraction,
        max_sample_missing_fraction=args.max_sample_missing_fraction,
        scaling_method=args.scaling_method,
        normalise_latent=not args.disable_latent_l2_normalisation,
        n_neighbours=args.n_neighbours,
        distance_metric=args.distance_metric,
        remove_all_zero_rows=not args.keep_all_zero_rows,
        remove_all_zero_features=not args.keep_all_zero_features,
        drop_rows_with_any_zero=args.drop_rows_with_any_zero,
        zero_policy=args.clipn_zero_policy,
        zero_epsilon=args.clipn_zero_epsilon,
        allow_pca_fallback=args.allow_pca_fallback,
    )
    run_clipn_workflow(
        datasets=datasets,
        output_dir=output_dir,
        config=config,
        logger=logger,
        save_model_path=Path(args.save_model) if args.save_model else None,
    )
    logger.info("CPATK CLIPn workflow complete")


if __name__ == "__main__":
    main()
