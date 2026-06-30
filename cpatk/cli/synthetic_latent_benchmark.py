"""Command line interface for CPATK synthetic latent benchmarking."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cpatk.synthetic_latent import (
    COMPREHENSIVE_SYNTHETIC_SCENARIOS,
    QUICK_SYNTHETIC_SCENARIOS,
    SCENARIO_PRESETS,
    run_synthetic_latent_benchmark_from_cli,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate controlled synthetic Cell Painting-like profiles and "
            "benchmark CPATK-native contrastive latent learning against PCA/raw features."
        )
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for benchmark tables.")
    parser.add_argument(
        "--benchmark_mode",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help=(
            "Benchmark size. quick uses the four core scenarios and one seed; "
            "standard keeps the v0.2.30-style core scenarios; comprehensive "
            "uses the expanded stress grid and repeated seeds."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help=(
            "Optional comma-separated scenarios to run. If empty, scenarios are "
            "chosen from --benchmark_mode. Known values: " + ",".join(sorted(SCENARIO_PRESETS))
        ),
    )
    parser.add_argument(
        "--seed_values",
        default="",
        help=(
            "Optional comma-separated random seeds. If empty, quick/standard use "
            "--random_state only; comprehensive uses 42,101,202,303,404."
        ),
    )
    parser.add_argument("--n_compounds", type=int, default=36)
    parser.add_argument("--n_moa_classes", type=int, default=6)
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--n_datasets", type=int, default=2)
    parser.add_argument("--replicates_per_compound_dataset", type=int, default=3)
    parser.add_argument("--n_features", type=int, default=160)
    parser.add_argument("--n_informative_features", type=int, default=60)
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=4,
        help="Sampled contrastive mini-batches per epoch. Use 0 for automatic sizing.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--hidden_dims", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_neighbours", type=int, default=5)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NSLOTS", os.environ.get("THREADS", "1"))))
    parser.add_argument("--skip_native_contrastive", action="store_true")
    parser.add_argument("--skip_pca", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def parse_csv_strings(*, value: str) -> list[str]:
    """Parse comma-separated strings."""
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_csv_ints(*, value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(item) for item in parse_csv_strings(value=value)]


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    if args.scenarios:
        scenarios = parse_csv_strings(value=args.scenarios)
    elif args.benchmark_mode == "comprehensive":
        scenarios = list(COMPREHENSIVE_SYNTHETIC_SCENARIOS)
    else:
        scenarios = list(QUICK_SYNTHETIC_SCENARIOS)
    if args.seed_values:
        seed_values = parse_csv_ints(value=args.seed_values)
    elif args.benchmark_mode == "comprehensive":
        seed_values = [42, 101, 202, 303, 404]
    else:
        seed_values = []
    run_synthetic_latent_benchmark_from_cli(
        output_dir=Path(args.output_dir),
        scenarios=scenarios,
        n_compounds=args.n_compounds,
        n_moa_classes=args.n_moa_classes,
        n_batches=args.n_batches,
        n_datasets=args.n_datasets,
        replicates_per_compound_dataset=args.replicates_per_compound_dataset,
        n_features=args.n_features,
        n_informative_features=args.n_informative_features,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        validation_fraction=args.validation_fraction,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        hidden_dims=parse_csv_ints(value=args.hidden_dims),
        dropout=args.dropout,
        random_state=args.random_state,
        seed_values=seed_values,
        benchmark_mode=args.benchmark_mode,
        n_neighbours=args.n_neighbours,
        threads=args.threads,
        skip_native_contrastive=args.skip_native_contrastive,
        skip_pca=args.skip_pca,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
