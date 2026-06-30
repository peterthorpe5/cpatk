"""Synthetic latent-benchmark utilities for CPATK.

The functions in this module generate controlled Cell Painting-like profile
matrices and benchmark latent embeddings against known ground truth.  The goal
is to test when CPATK-native contrastive learning improves compound retrieval,
and when it should be treated cautiously because batch/source structure still
wins.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from cpatk.clipn_adapter import ClipnAdapterConfig, calculate_latent_diagnostics
from cpatk.contrastive import NativeContrastiveConfig, fit_native_contrastive_backend
from cpatk.io import write_excel_workbook, write_table
from cpatk.logging_utils import configure_logging
from cpatk.threading_utils import configure_threading

try:  # pragma: no cover - import failure tested indirectly by graceful handling
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import RobustScaler, StandardScaler

    SKLEARN_SYNTHETIC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    PCA = None
    NearestNeighbors = None
    RobustScaler = None
    StandardScaler = None
    silhouette_score = None
    SKLEARN_SYNTHETIC_IMPORT_ERROR = exc


@dataclass
class SyntheticCellPaintingConfig:
    """Configuration for synthetic Cell Painting profile generation."""

    scenario_name: str = "batch_confounded_biology"
    n_compounds: int = 48
    n_moa_classes: int = 8
    n_batches: int = 4
    n_datasets: int = 2
    replicates_per_compound_dataset: int = 4
    n_features: int = 240
    n_informative_features: int = 80
    n_latent_biology_factors: int = 6
    n_latent_batch_factors: int = 4
    biology_strength: float = 2.0
    moa_strength: float = 0.8
    batch_strength: float = 1.5
    dataset_strength: float = 1.2
    replicate_noise: float = 0.8
    feature_noise: float = 0.25
    nonlinear_fraction: float = 0.10
    missing_fraction: float = 0.01
    random_state: int = 42
    metadata_prefix: str = "Metadata"


@dataclass
class LatentBenchmarkConfig:
    """Configuration for synthetic latent benchmark execution."""

    output_dir: Path
    scenarios: list[str] = field(
        default_factory=lambda: [
            "clean_biology",
            "batch_confounded_biology",
            "weak_biology",
            "no_biology_negative_control",
        ]
    )
    n_compounds: int = 36
    n_moa_classes: int = 6
    n_batches: int = 4
    n_datasets: int = 2
    replicates_per_compound_dataset: int = 3
    n_features: int = 160
    n_informative_features: int = 60
    latent_dim: int = 12
    epochs: int = 60
    batch_size: int = 128
    steps_per_epoch: Optional[int] = 4
    validation_fraction: float = 0.15
    learning_rate: float = 1e-3
    temperature: float = 0.10
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.10
    random_state: int = 42
    n_neighbours: int = 5
    threads: int = 1
    run_native_contrastive: bool = True
    run_pca: bool = True


SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "clean_biology": {
        "biology_strength": 2.5,
        "moa_strength": 1.0,
        "batch_strength": 0.5,
        "dataset_strength": 0.4,
        "replicate_noise": 0.65,
        "feature_noise": 0.20,
    },
    "batch_confounded_biology": {
        "biology_strength": 1.9,
        "moa_strength": 0.8,
        "batch_strength": 2.2,
        "dataset_strength": 1.5,
        "replicate_noise": 0.85,
        "feature_noise": 0.25,
    },
    "weak_biology": {
        "biology_strength": 0.75,
        "moa_strength": 0.35,
        "batch_strength": 1.2,
        "dataset_strength": 0.9,
        "replicate_noise": 1.05,
        "feature_noise": 0.35,
    },
    "no_biology_negative_control": {
        "biology_strength": 0.0,
        "moa_strength": 0.0,
        "batch_strength": 1.3,
        "dataset_strength": 1.0,
        "replicate_noise": 1.0,
        "feature_noise": 0.30,
    },
}


def require_sklearn_for_synthetic(*, purpose: str) -> None:
    """Raise a clear error if the scikit-learn stack is unavailable."""
    if SKLEARN_SYNTHETIC_IMPORT_ERROR is not None:
        raise ImportError(
            "The SciPy/scikit-learn stack could not be imported for "
            f"{purpose}. Install scikit-learn in the CPATK environment."
        ) from SKLEARN_SYNTHETIC_IMPORT_ERROR


def config_for_scenario(
    *,
    scenario_name: str,
    base_config: SyntheticCellPaintingConfig,
) -> SyntheticCellPaintingConfig:
    """Return a synthetic-generation config with scenario-specific settings."""
    if scenario_name not in SCENARIO_PRESETS:
        known = ", ".join(sorted(SCENARIO_PRESETS))
        raise ValueError(f"Unknown synthetic scenario '{scenario_name}'. Known scenarios: {known}")
    values = asdict(base_config)
    values.update(SCENARIO_PRESETS[scenario_name])
    values["scenario_name"] = scenario_name
    return SyntheticCellPaintingConfig(**values)


def generate_synthetic_cell_painting_profiles(
    *,
    config: SyntheticCellPaintingConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a controlled Cell Painting-like profile table.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Profile table, ground-truth table and scenario-parameter table.
    """
    rng = np.random.default_rng(seed=int(config.random_state))
    _validate_synthetic_config(config=config)
    feature_names = [f"Cells_Texture_Synthetic_{idx + 1:04d}" for idx in range(config.n_features)]
    informative_features = feature_names[: int(config.n_informative_features)]
    noise_features = feature_names[int(config.n_informative_features) :]

    compound_ids = [f"CPD_{idx + 1:04d}" for idx in range(config.n_compounds)]
    moa_labels = [f"MOA_{(idx % config.n_moa_classes) + 1:02d}" for idx in range(config.n_compounds)]
    datasets = [f"dataset_{idx + 1}" for idx in range(config.n_datasets)]
    batches = [f"plate_{idx + 1:02d}" for idx in range(config.n_batches)]

    biology_factors = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(config.n_compounds, config.n_latent_biology_factors),
    )
    moa_factors = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(config.n_moa_classes, config.n_latent_biology_factors),
    )
    for compound_index, moa_label in enumerate(moa_labels):
        moa_index = int(moa_label.split("_")[1]) - 1
        biology_factors[compound_index, :] = (
            config.biology_strength * biology_factors[compound_index, :]
            + config.moa_strength * moa_factors[moa_index, :]
        )

    biology_loadings = rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(config.n_latent_biology_factors),
        size=(config.n_latent_biology_factors, config.n_informative_features),
    )
    batch_loadings = rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(config.n_latent_batch_factors),
        size=(config.n_latent_batch_factors, config.n_features),
    )
    batch_factors = rng.normal(
        loc=0.0,
        scale=float(config.batch_strength),
        size=(config.n_batches, config.n_latent_batch_factors),
    )
    dataset_offsets = rng.normal(
        loc=0.0,
        scale=float(config.dataset_strength),
        size=(config.n_datasets, config.n_features),
    )

    rows = []
    ground_truth_rows = []
    image_number = 1
    for compound_index, compound_id in enumerate(compound_ids):
        moa_label = moa_labels[compound_index]
        for dataset_index, dataset_name in enumerate(datasets):
            for replicate in range(int(config.replicates_per_compound_dataset)):
                batch_index = int((compound_index + replicate + dataset_index) % config.n_batches)
                batch_name = batches[batch_index]
                values = np.zeros(config.n_features, dtype=float)
                values[: config.n_informative_features] += (
                    biology_factors[compound_index, :] @ biology_loadings
                )
                values += dataset_offsets[dataset_index, :]
                values += batch_factors[batch_index, :] @ batch_loadings
                values += rng.normal(loc=0.0, scale=float(config.replicate_noise), size=config.n_features)
                values += rng.normal(loc=0.0, scale=float(config.feature_noise), size=config.n_features)
                if config.nonlinear_fraction > 0:
                    n_nonlinear = int(round(config.n_informative_features * config.nonlinear_fraction))
                    if n_nonlinear > 0:
                        nonlinear_indices = np.arange(n_nonlinear)
                        values[nonlinear_indices] = np.tanh(values[nonlinear_indices]) * 2.0
                if config.missing_fraction > 0:
                    missing_mask = rng.random(config.n_features) < float(config.missing_fraction)
                    values[missing_mask] = np.nan
                well = _index_to_well(index=image_number)
                row = {
                    "Dataset": dataset_name,
                    "Sample": image_number - 1,
                    "ImageNumber": image_number,
                    "Metadata_Profile_Source": dataset_name,
                    "Metadata_Plate": batch_name,
                    "Metadata_Well": well,
                    "Metadata_Compound": compound_id,
                    "cpd_id": compound_id,
                    "cpd_type": moa_label,
                    "synthetic_scenario": config.scenario_name,
                    "synthetic_batch": batch_name,
                    "synthetic_replicate": replicate + 1,
                }
                row.update({feature: value for feature, value in zip(feature_names, values)})
                rows.append(row)
                ground_truth_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Sample": image_number - 1,
                        "ImageNumber": image_number,
                        "cpd_id": compound_id,
                        "cpd_type": moa_label,
                        "synthetic_batch": batch_name,
                        "synthetic_scenario": config.scenario_name,
                        "n_informative_features": int(config.n_informative_features),
                        "n_noise_features": int(len(noise_features)),
                    }
                )
                image_number += 1
    profiles = pd.DataFrame.from_records(rows)
    truth = pd.DataFrame.from_records(ground_truth_rows)
    scenario_table = pd.DataFrame(
        [
            {"item": key, "value": value}
            for key, value in asdict(config).items()
        ]
    )
    scenario_table = pd.concat(
        [
            scenario_table,
            pd.DataFrame(
                [
                    {
                        "item": "n_profiles",
                        "value": int(profiles.shape[0]),
                    },
                    {
                        "item": "n_metadata_columns",
                        "value": int(profiles.shape[1] - len(feature_names)),
                    },
                    {
                        "item": "n_feature_columns",
                        "value": int(len(feature_names)),
                    },
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    if logger is not None:
        logger.info(
            "Generated synthetic scenario %s with %d rows and %d features.",
            config.scenario_name,
            profiles.shape[0],
            len(feature_names),
        )
    return profiles, truth, scenario_table


def _validate_synthetic_config(*, config: SyntheticCellPaintingConfig) -> None:
    """Validate synthetic generation settings."""
    if config.n_compounds < 2:
        raise ValueError("Synthetic benchmark requires at least two compounds.")
    if config.n_moa_classes < 1:
        raise ValueError("Synthetic benchmark requires at least one MOA class.")
    if config.n_features < 4:
        raise ValueError("Synthetic benchmark requires at least four features.")
    if config.n_informative_features < 1 or config.n_informative_features > config.n_features:
        raise ValueError("n_informative_features must be between 1 and n_features.")
    if config.n_datasets < 1:
        raise ValueError("Synthetic benchmark requires at least one dataset.")
    if config.replicates_per_compound_dataset < 2:
        raise ValueError("At least two replicates per compound/dataset are required.")


def _index_to_well(*, index: int) -> str:
    """Convert a one-based profile index into a 384-well style well code."""
    rows = "ABCDEFGHIJKLMNOP"
    zero_index = int(index - 1)
    row = rows[zero_index % len(rows)]
    column = (zero_index // len(rows)) % 24 + 1
    return f"{row}{column:02d}"


def split_profiles_by_dataset(
    *,
    profiles: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Split a synthetic profile table into CPATK latent input datasets."""
    if "Dataset" not in profiles.columns:
        raise ValueError("Synthetic profiles must contain a Dataset column.")
    datasets = {
        str(dataset): block.reset_index(drop=True).copy()
        for dataset, block in profiles.groupby("Dataset", sort=False)
    }
    manifest = pd.DataFrame(
        [
            {"dataset": str(dataset), "n_rows": int(block.shape[0])}
            for dataset, block in datasets.items()
        ]
    )
    return datasets, manifest


def infer_synthetic_feature_columns(*, profiles: pd.DataFrame) -> list[str]:
    """Return numeric synthetic feature columns from a profile table."""
    return [
        str(column)
        for column in profiles.select_dtypes(include=[np.number]).columns
        if str(column).startswith("Cells_Texture_Synthetic_")
    ]


def make_cleaned_feature_blocks(
    *,
    datasets: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Robust-scale synthetic features and return blocks for native contrastive."""
    require_sklearn_for_synthetic(purpose="synthetic latent preprocessing")
    combined = []
    for dataset_name, table in datasets.items():
        block = table.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
        block.insert(0, "Dataset", str(dataset_name))
        block.insert(1, "Sample", np.arange(block.shape[0], dtype=int))
        combined.append(block)
    merged = pd.concat(combined, ignore_index=True, sort=False)
    merged.loc[:, feature_columns] = merged[feature_columns].replace([np.inf, -np.inf], np.nan)
    missing_before = int(merged[feature_columns].isna().sum().sum())
    fill_values = merged[feature_columns].median(axis=0)
    merged.loc[:, feature_columns] = merged[feature_columns].fillna(fill_values)
    scaler = RobustScaler()
    merged.loc[:, feature_columns] = scaler.fit_transform(merged[feature_columns])
    cleaned = {}
    for dataset_name in datasets:
        block = merged.loc[merged["Dataset"].eq(str(dataset_name)), ["Sample", *feature_columns]].copy()
        cleaned[str(dataset_name)] = block.set_index("Sample")
    summary = pd.DataFrame.from_records(
        [
            {"item": "n_features", "value": int(len(feature_columns))},
            {"item": "missing_values_before_imputation", "value": missing_before},
            {"item": "imputation_method", "value": "global_median"},
            {"item": "scaling_method", "value": "robust"},
        ]
    )
    metadata = _metadata_for_native_backend(datasets=datasets)
    return cleaned, metadata, summary


def _metadata_for_native_backend(*, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create row-aligned metadata for native contrastive benchmarking."""
    metadata_columns = [
        "cpd_id",
        "cpd_type",
        "Metadata_Compound",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Profile_Source",
        "synthetic_batch",
        "synthetic_scenario",
    ]
    records = []
    for dataset_name, table in datasets.items():
        meta = pd.DataFrame(
            {
                "Dataset": str(dataset_name),
                "Sample": np.arange(table.shape[0], dtype=int),
            }
        )
        for column in metadata_columns:
            if column in table.columns:
                meta[column] = table[column].to_numpy()
        records.append(meta)
    return pd.concat(records, ignore_index=True, sort=False)


def calculate_embedding_retrieval_metrics(
    *,
    embedding: pd.DataFrame,
    id_column: str = "cpd_id",
    label_column: str = "cpd_type",
    dataset_column: str = "Dataset",
    batch_column: str = "synthetic_batch",
    n_neighbours: int = 5,
    metric: str = "cosine",
    threads: int = 1,
    method_name: str = "embedding",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate retrieval and leakage metrics for an embedding table."""
    require_sklearn_for_synthetic(purpose="synthetic embedding retrieval metrics")
    latent_columns = _numeric_embedding_columns(embedding=embedding)
    if len(latent_columns) < 1 or embedding.shape[0] < 3:
        status = pd.DataFrame.from_records(
            [
                {
                    "method": method_name,
                    "metric": "status",
                    "value": np.nan,
                    "message": "Insufficient embedding data.",
                }
            ]
        )
        return status, pd.DataFrame()
    X = embedding.loc[:, latent_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    n_query_neighbours = min(int(max(1, n_neighbours)) + 1, int(X.shape[0]))
    nn_model = NearestNeighbors(
        n_neighbors=n_query_neighbours,
        metric=metric,
        n_jobs=max(1, int(threads)),
    )
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X, return_distance=True)
    rows = []
    for query_index in range(X.shape[0]):
        rank = 0
        for neighbour_index, distance in zip(indices[query_index], distances[query_index]):
            if int(neighbour_index) == int(query_index):
                continue
            rank += 1
            row = {
                "method": method_name,
                "query_index": int(query_index),
                "neighbour_index": int(neighbour_index),
                "rank": int(rank),
                "distance": float(distance),
            }
            for column in [id_column, label_column, dataset_column, batch_column]:
                if column in embedding.columns:
                    row[f"Query_{column}"] = embedding.iloc[query_index].get(column)
                    row[f"Neighbour_{column}"] = embedding.iloc[neighbour_index].get(column)
                    row[f"same_{column}"] = bool(
                        str(row[f"Query_{column}"]) == str(row[f"Neighbour_{column}"])
                    )
            rows.append(row)
            if rank >= int(n_neighbours):
                break
    neighbours = pd.DataFrame.from_records(rows)
    first_neighbour = neighbours.loc[neighbours["rank"].eq(1)].copy()
    records = []
    for column, metric_name in [
        (id_column, "top1_same_compound_rate"),
        (label_column, "top1_same_moa_rate"),
        (dataset_column, "top1_same_dataset_rate"),
        (batch_column, "top1_same_batch_rate"),
    ]:
        flag = f"same_{column}"
        if flag in first_neighbour.columns:
            records.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "value": float(first_neighbour[flag].mean()),
                }
            )
    for column, metric_name in [
        (id_column, "mean_topk_same_compound_rate"),
        (label_column, "mean_topk_same_moa_rate"),
        (dataset_column, "mean_topk_same_dataset_rate"),
        (batch_column, "mean_topk_same_batch_rate"),
    ]:
        flag = f"same_{column}"
        if flag in neighbours.columns:
            records.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "value": float(neighbours[flag].mean()),
                }
            )
    for column in [id_column, label_column, dataset_column, batch_column]:
        if column in embedding.columns and embedding[column].nunique(dropna=True) >= 2:
            values = embedding[column].fillna("missing").astype(str)
            if values.value_counts().min() >= 2:
                try:
                    score = silhouette_score(X, values, metric=metric)
                    records.append(
                        {
                            "method": method_name,
                            "metric": f"silhouette_{column}",
                            "value": float(score),
                        }
                    )
                except Exception:
                    continue
    records.append(
        {
            "method": method_name,
            "metric": "n_embedding_rows",
            "value": float(embedding.shape[0]),
        }
    )
    records.append(
        {
            "method": method_name,
            "metric": "n_embedding_dimensions",
            "value": float(len(latent_columns)),
        }
    )
    return pd.DataFrame.from_records(records), neighbours


def _numeric_embedding_columns(*, embedding: pd.DataFrame) -> list[str]:
    """Infer numeric embedding columns from a benchmark table."""
    preferred = [column for column in embedding.columns if str(column).startswith("latent_")]
    if preferred:
        return preferred
    synthetic = [column for column in embedding.columns if str(column).startswith("Cells_Texture_Synthetic_")]
    if synthetic:
        return synthetic
    excluded = {
        "Dataset",
        "Sample",
        "ImageNumber",
        "cpd_id",
        "cpd_type",
        "Metadata_Compound",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Profile_Source",
        "synthetic_batch",
        "synthetic_scenario",
        "synthetic_replicate",
    }
    return [
        str(column)
        for column in embedding.select_dtypes(include=[np.number]).columns
        if str(column) not in excluded
    ]


def build_pca_embedding(
    *,
    profiles: pd.DataFrame,
    feature_columns: Sequence[str],
    n_components: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a PCA embedding from synthetic profiles."""
    require_sklearn_for_synthetic(purpose="synthetic PCA benchmark")
    X = profiles.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(axis=0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = PCA(n_components=min(int(n_components), X_scaled.shape[1]), random_state=int(random_state))
    scores = model.fit_transform(X_scaled)
    latent_columns = [f"latent_{idx + 1}" for idx in range(scores.shape[1])]
    output = pd.DataFrame(scores, columns=latent_columns)
    for column in [
        "Dataset",
        "Sample",
        "cpd_id",
        "cpd_type",
        "Metadata_Compound",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Profile_Source",
        "synthetic_batch",
        "synthetic_scenario",
    ]:
        if column in profiles.columns:
            output[column] = profiles[column].to_numpy()
    explained = pd.DataFrame(
        {
            "component": latent_columns,
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    )
    return output, explained


def run_synthetic_latent_benchmark(
    *,
    config: LatentBenchmarkConfig,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Run CPATK synthetic latent benchmarks across configured scenarios."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    threads = configure_threading(n_threads=config.threads, logger=logger)
    config.threads = int(threads)
    all_metrics = []
    all_pass_fail = []
    all_training = []
    all_quality = []
    scenario_tables = []
    for scenario_index, scenario_name in enumerate(config.scenarios):
        scenario_dir = config.output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        base_config = SyntheticCellPaintingConfig(
            scenario_name=scenario_name,
            n_compounds=config.n_compounds,
            n_moa_classes=config.n_moa_classes,
            n_batches=config.n_batches,
            n_datasets=config.n_datasets,
            replicates_per_compound_dataset=config.replicates_per_compound_dataset,
            n_features=config.n_features,
            n_informative_features=config.n_informative_features,
            random_state=int(config.random_state + scenario_index * 997),
        )
        synthetic_config = config_for_scenario(
            scenario_name=scenario_name,
            base_config=base_config,
        )
        profiles, truth, scenario_table = generate_synthetic_cell_painting_profiles(
            config=synthetic_config,
            logger=logger,
        )
        feature_columns = infer_synthetic_feature_columns(profiles=profiles)
        write_table(data_frame=profiles, path=scenario_dir / "synthetic_profiles.tsv.gz", logger=logger)
        write_table(data_frame=truth, path=scenario_dir / "synthetic_ground_truth.tsv", logger=logger)
        write_table(data_frame=scenario_table, path=scenario_dir / "synthetic_scenario_config.tsv", logger=logger)
        scenario_table.insert(0, "scenario", scenario_name)
        scenario_tables.append(scenario_table)

        raw_embedding = profiles.loc[:, feature_columns].copy()
        for column in [
            "Dataset",
            "Sample",
            "cpd_id",
            "cpd_type",
            "Metadata_Compound",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_Profile_Source",
            "synthetic_batch",
            "synthetic_scenario",
        ]:
            if column in profiles.columns:
                raw_embedding[column] = profiles[column].to_numpy()
        raw_metrics, raw_neighbours = calculate_embedding_retrieval_metrics(
            embedding=raw_embedding,
            n_neighbours=config.n_neighbours,
            threads=config.threads,
            method_name="raw_scaled_features",
        )
        raw_metrics.insert(0, "scenario", scenario_name)
        raw_neighbours.insert(0, "scenario", scenario_name)
        all_metrics.append(raw_metrics)
        write_table(data_frame=raw_neighbours, path=scenario_dir / "raw_feature_neighbours.tsv.gz", logger=logger)

        if config.run_pca:
            pca_embedding, explained = build_pca_embedding(
                profiles=profiles,
                feature_columns=feature_columns,
                n_components=config.latent_dim,
                random_state=config.random_state,
            )
            pca_metrics, pca_neighbours = calculate_embedding_retrieval_metrics(
                embedding=pca_embedding,
                n_neighbours=config.n_neighbours,
                threads=config.threads,
                method_name="pca",
            )
            pca_metrics.insert(0, "scenario", scenario_name)
            pca_neighbours.insert(0, "scenario", scenario_name)
            all_metrics.append(pca_metrics)
            write_table(data_frame=pca_embedding, path=scenario_dir / "pca_latent.tsv.gz", logger=logger)
            write_table(data_frame=explained, path=scenario_dir / "pca_explained_variance.tsv", logger=logger)
            write_table(data_frame=pca_neighbours, path=scenario_dir / "pca_neighbours.tsv.gz", logger=logger)

        if config.run_native_contrastive:
            native_tables = _run_native_for_scenario(
                profiles=profiles,
                feature_columns=feature_columns,
                scenario_name=scenario_name,
                scenario_dir=scenario_dir,
                config=config,
                logger=logger,
            )
            all_metrics.append(native_tables["metrics"])
            all_training.append(native_tables["training_summary"])
            all_quality.append(native_tables["latent_quality"])

        scenario_metrics = pd.concat(all_metrics, ignore_index=True, sort=False)
        scenario_subset = scenario_metrics.loc[scenario_metrics["scenario"].eq(scenario_name)].copy()
        scenario_pass_fail = score_synthetic_scenario(
            metrics=scenario_subset,
            scenario_name=scenario_name,
        )
        all_pass_fail.append(scenario_pass_fail)
        write_table(data_frame=scenario_subset, path=scenario_dir / "scenario_metric_summary.tsv", logger=logger)
        write_table(data_frame=scenario_pass_fail, path=scenario_dir / "scenario_pass_fail.tsv", logger=logger)

    outputs = {
        "synthetic_metric_summary": pd.concat(all_metrics, ignore_index=True, sort=False),
        "synthetic_pass_fail_summary": pd.concat(all_pass_fail, ignore_index=True, sort=False),
        "synthetic_scenario_configs": pd.concat(scenario_tables, ignore_index=True, sort=False),
    }
    if all_training:
        outputs["native_contrastive_training_summary"] = pd.concat(
            all_training,
            ignore_index=True,
            sort=False,
        )
    if all_quality:
        outputs["native_contrastive_quality_summary"] = pd.concat(
            all_quality,
            ignore_index=True,
            sort=False,
        )
    for name, table in outputs.items():
        write_table(data_frame=table, path=config.output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(
        tables=outputs,
        path=config.output_dir / "synthetic_latent_benchmark_summary.xlsx",
        logger=logger,
    )
    return outputs


def _run_native_for_scenario(
    *,
    profiles: pd.DataFrame,
    feature_columns: Sequence[str],
    scenario_name: str,
    scenario_dir: Path,
    config: LatentBenchmarkConfig,
    logger: Optional[logging.Logger],
) -> dict[str, pd.DataFrame]:
    """Run the CPATK-native contrastive backend for one scenario."""
    datasets, manifest = split_profiles_by_dataset(profiles=profiles)
    cleaned, metadata, preprocessing_summary = make_cleaned_feature_blocks(
        datasets=datasets,
        feature_columns=feature_columns,
    )
    write_table(data_frame=manifest, path=scenario_dir / "native_input_manifest.tsv", logger=logger)
    write_table(data_frame=preprocessing_summary, path=scenario_dir / "native_preprocessing_summary.tsv", logger=logger)
    native_config = NativeContrastiveConfig(
        latent_dim=int(config.latent_dim),
        hidden_dims=list(config.hidden_dims),
        dropout=float(config.dropout),
        learning_rate=float(config.learning_rate),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        temperature=float(config.temperature),
        validation_fraction=float(config.validation_fraction),
        steps_per_epoch=config.steps_per_epoch,
        random_state=int(config.random_state),
        positive_column="cpd_id",
        n_threads=int(config.threads),
    )
    result = fit_native_contrastive_backend(
        cleaned=cleaned,
        metadata=metadata,
        config=native_config,
        logger=logger,
    )
    latent = result.latent_table.copy()
    latent_metrics, latent_neighbours = calculate_embedding_retrieval_metrics(
        embedding=latent,
        n_neighbours=config.n_neighbours,
        threads=config.threads,
        method_name="cpatk_contrastive",
    )
    latent_metrics.insert(0, "scenario", scenario_name)
    latent_neighbours.insert(0, "scenario", scenario_name)
    training_summary = result.training_summary.copy()
    training_summary.insert(0, "scenario", scenario_name)
    training_loss = result.training_loss.copy()
    training_loss.insert(0, "scenario", scenario_name)
    positive_report = result.positive_label_report.copy()
    positive_report.insert(0, "scenario", scenario_name)
    split_report = result.split_report.copy()
    split_report.insert(0, "scenario", scenario_name)
    clipn_config = ClipnAdapterConfig(
        backend_module="cpatk_contrastive",
        id_column="cpd_id",
        label_column="cpd_type",
        n_neighbours=int(config.n_neighbours),
        n_threads=int(config.threads),
    )
    diagnostic_tables = calculate_latent_diagnostics(
        latent_table=latent,
        config=clipn_config,
    )
    diagnostic_summary = diagnostic_tables.get("latent_diagnostic_summary", pd.DataFrame()).copy()
    if not diagnostic_summary.empty:
        diagnostic_summary.insert(0, "scenario", scenario_name)
        diagnostic_summary.insert(1, "method", "cpatk_contrastive")
    write_table(data_frame=latent, path=scenario_dir / "cpatk_contrastive_latent.tsv.gz", logger=logger)
    write_table(data_frame=training_loss, path=scenario_dir / "cpatk_contrastive_training_loss.tsv", logger=logger)
    write_table(data_frame=training_summary, path=scenario_dir / "cpatk_contrastive_training_summary.tsv", logger=logger)
    write_table(data_frame=positive_report, path=scenario_dir / "cpatk_contrastive_positive_label_report.tsv", logger=logger)
    write_table(data_frame=split_report, path=scenario_dir / "cpatk_contrastive_split_report.tsv", logger=logger)
    write_table(data_frame=latent_neighbours, path=scenario_dir / "cpatk_contrastive_neighbours.tsv.gz", logger=logger)
    if not diagnostic_summary.empty:
        write_table(data_frame=diagnostic_summary, path=scenario_dir / "cpatk_contrastive_latent_diagnostics.tsv", logger=logger)
    return {
        "metrics": latent_metrics,
        "training_summary": training_summary,
        "latent_quality": diagnostic_summary,
    }


def score_synthetic_scenario(
    *,
    metrics: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Score whether synthetic benchmark behaviour is broadly sensible."""
    pivot = metrics.pivot_table(
        index="method",
        columns="metric",
        values="value",
        aggfunc="first",
    )
    rows = []
    native_top1 = _metric_value(pivot=pivot, method="cpatk_contrastive", metric="top1_same_compound_rate")
    pca_top1 = _metric_value(pivot=pivot, method="pca", metric="top1_same_compound_rate")
    raw_top1 = _metric_value(pivot=pivot, method="raw_scaled_features", metric="top1_same_compound_rate")
    native_dataset = _metric_value(pivot=pivot, method="cpatk_contrastive", metric="top1_same_dataset_rate")
    native_batch = _metric_value(pivot=pivot, method="cpatk_contrastive", metric="top1_same_batch_rate")
    if scenario_name == "no_biology_negative_control":
        passed = bool(np.isnan(native_top1) or native_top1 < 0.20)
        message = (
            "Negative control should not create strong compound retrieval when no biology was simulated."
        )
    elif scenario_name == "weak_biology":
        passed = bool(np.isfinite(native_top1) and native_top1 >= 0.20)
        message = "Weak biology scenario should show modest, not necessarily strong, retrieval."
    else:
        passed = bool(np.isfinite(native_top1) and native_top1 >= max(0.40, pca_top1 - 0.05, raw_top1 - 0.05))
        message = "Biology-present scenarios should give useful compound retrieval without major regression versus PCA/raw."
    rows.append(
        {
            "scenario": scenario_name,
            "check": "native_top1_same_compound_rate",
            "passed": passed,
            "value": native_top1,
            "comparison_pca": pca_top1,
            "comparison_raw": raw_top1,
            "message": message,
        }
    )
    leakage_passed = bool(
        not np.isfinite(native_dataset)
        or native_dataset <= 0.90
        or scenario_name == "no_biology_negative_control"
    )
    rows.append(
        {
            "scenario": scenario_name,
            "check": "native_dataset_leakage_not_extreme",
            "passed": leakage_passed,
            "value": native_dataset,
            "comparison_pca": np.nan,
            "comparison_raw": np.nan,
            "message": "Same-dataset nearest-neighbour rate should not dominate in ordinary biology scenarios.",
        }
    )
    rows.append(
        {
            "scenario": scenario_name,
            "check": "native_batch_leakage_recorded",
            "passed": True,
            "value": native_batch,
            "comparison_pca": np.nan,
            "comparison_raw": np.nan,
            "message": "Batch leakage is recorded as a diagnostic rather than a hard failure in synthetic stress tests.",
        }
    )
    return pd.DataFrame.from_records(rows)


def _metric_value(
    *,
    pivot: pd.DataFrame,
    method: str,
    metric: str,
) -> float:
    """Safely fetch a metric value from a method-by-metric table."""
    if method not in pivot.index or metric not in pivot.columns:
        return float("nan")
    value = pivot.loc[method, metric]
    try:
        return float(value)
    except Exception:
        return float("nan")


def run_synthetic_latent_benchmark_from_cli(
    *,
    output_dir: Path,
    scenarios: Sequence[str],
    n_compounds: int,
    n_moa_classes: int,
    n_batches: int,
    n_datasets: int,
    replicates_per_compound_dataset: int,
    n_features: int,
    n_informative_features: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    validation_fraction: float,
    learning_rate: float,
    temperature: float,
    hidden_dims: Sequence[int],
    dropout: float,
    random_state: int,
    n_neighbours: int,
    threads: int,
    skip_native_contrastive: bool,
    skip_pca: bool,
    log_level: str,
) -> dict[str, pd.DataFrame]:
    """Run the synthetic benchmark using command-line compatible arguments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        log_file=output_dir / "synthetic_latent_benchmark.log",
        log_level=log_level,
    )
    config = LatentBenchmarkConfig(
        output_dir=output_dir,
        scenarios=[str(item) for item in scenarios],
        n_compounds=int(n_compounds),
        n_moa_classes=int(n_moa_classes),
        n_batches=int(n_batches),
        n_datasets=int(n_datasets),
        replicates_per_compound_dataset=int(replicates_per_compound_dataset),
        n_features=int(n_features),
        n_informative_features=int(n_informative_features),
        latent_dim=int(latent_dim),
        epochs=int(epochs),
        batch_size=int(batch_size),
        steps_per_epoch=(None if int(steps_per_epoch) <= 0 else int(steps_per_epoch)),
        validation_fraction=float(validation_fraction),
        learning_rate=float(learning_rate),
        temperature=float(temperature),
        hidden_dims=[int(value) for value in hidden_dims],
        dropout=float(dropout),
        random_state=int(random_state),
        n_neighbours=int(n_neighbours),
        threads=int(threads),
        run_native_contrastive=not bool(skip_native_contrastive),
        run_pca=not bool(skip_pca),
    )
    write_table(
        data_frame=pd.DataFrame([asdict(config)]),
        path=output_dir / "synthetic_benchmark_configuration.tsv",
        logger=logger,
    )
    return run_synthetic_latent_benchmark(config=config, logger=logger)
