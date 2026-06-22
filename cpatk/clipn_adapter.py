"""Optional CLIPn workflow adapter for CPATK.

This module keeps CLIPn integration optional.  It provides a reproducible and
well-logged route for preparing several preprocessed Cell Painting datasets,
harmonising features, running CLIPn when a compatible backend is installed, and
writing latent-space diagnostics.  Classical CPATK workflows remain available
when CLIPn is not installed.

The adapter is deliberately defensive because CLIPn APIs vary between local
installations.  The default backend compatibility targets the commonly used
``clipn.model.CLIPn`` API where the model is constructed as ``CLIPn(X, y,
latent_dim=...)``, fitted with ``fit(X, y, lr=..., epochs=...)`` and projected
with ``predict(X)``.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from cpatk.embedding import run_pca, run_umap_or_pca
from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.metadata import canonicalise_well_value, normalise_column_names
from cpatk.plotting import plot_embedding, plot_pca_variance
from cpatk.reporting import make_html_report

TECHNICAL_COLUMNS = {
    "ImageNumber",
    "ObjectNumber",
    "Number_Object_Number",
    "TableNumber",
    "row_number",
}

METADATA_ALIASES = {
    "cpd_id": (
        "cpd_id",
        "Metadata_Compound",
        "Metadata_cpd_id",
        "compound",
        "Compound",
        "compound_id",
        "COMPOUND_NAME",
        "compound_name",
        "Name",
        "name",
        "pert_iname",
    ),
    "cpd_type": (
        "cpd_type",
        "Metadata_MOA",
        "MOA",
        "moa",
        "Mechanism",
        "mechanism",
        "Class",
        "class",
        "mode_of_action",
        "ModeOfAction",
    ),
    "Library": (
        "Library",
        "library",
        "Metadata_Batch",
        "Batch",
        "batch",
        "Dataset",
        "experiment",
    ),
    "Plate_Metadata": (
        "Plate_Metadata",
        "Metadata_Plate",
        "Plate",
        "plate",
        "Image_Metadata_Plate",
        "Assay_Plate_Barcode",
        "AssayPlateBarcode",
        "Destination_Plate_Barcode",
        "DestinationPlateBarcode",
    ),
    "Well_Metadata": (
        "Well_Metadata",
        "Metadata_Well",
        "Well",
        "well",
        "Image_Metadata_Well",
        "Assay_Well",
        "AssayWell",
        "Destination_Well",
        "DestinationWell",
    ),
}

METADATA_NAME_PATTERNS = re.compile(
    r"""(?ix)
    ( ^metadata(_|$)
    | _metadata$
    | ^dataset$
    | ^sample$
    | ^library$
    | ^plate(_|$)|_plate$
    | ^well(_|$)|_well$
    | ^cpd(_|$)|^compound(_|$)|^compound$|^name$
    | ^moa$|mechanism|mode_of_action|^class$
    | concentration|dose|timepoint|batch|replicate
    | ^filename_|^pathname_|^url_|^md5digest_
    | ^executiontime_|^group_|^imagename_|^series_
    | ^imagenumber$|^objectnumber$|^number_object_number$|^tablenumber$
    )
    """
)


@dataclass
class ClipnAdapterConfig:
    """Configuration for a CLIPn adapter run."""

    backend_module: str = "clipn"
    model_class: Optional[str] = None
    fit_method: str = "fit"
    predict_method: str = "predict"
    transform_method: Optional[str] = None
    reference_names: Optional[list[str]] = None
    dataset_column: str = "Dataset"
    sample_column: str = "Sample"
    id_column: str = "cpd_id"
    label_column: str = "cpd_type"
    metadata_columns: Optional[list[str]] = None
    feature_columns: Optional[list[str]] = None
    mode: str = "integrate_all"
    latent_dim: int = 20
    learning_rate: float = 1e-5
    epochs: int = 300
    imputation_method: str = "median"
    imputation_group_columns: list[str] = field(default_factory=lambda: ["Dataset", "Plate_Metadata"])
    max_feature_missing_fraction: float = 0.3
    max_sample_missing_fraction: float = 0.8
    scaling_method: str = "robust"
    normalise_latent: bool = True
    random_state: int = 42
    n_neighbours: int = 15
    distance_metric: str = "cosine"
    remove_all_zero_rows: bool = True
    remove_all_zero_features: bool = True
    drop_rows_with_any_zero: bool = False
    allow_pca_fallback: bool = False


def check_clipn_backend(*, backend_module: str = "clipn") -> pd.DataFrame:
    """Check whether a CLIPn backend module can be imported."""
    try:
        module = importlib.import_module(name=backend_module)
        return pd.DataFrame.from_records(
            [
                {
                    "backend_module": backend_module,
                    "available": True,
                    "module_file": getattr(module, "__file__", "unknown"),
                    "message": "Backend imported successfully.",
                }
            ]
        )
    except Exception as exc:
        return pd.DataFrame.from_records(
            [
                {
                    "backend_module": backend_module,
                    "available": False,
                    "module_file": "",
                    "message": str(exc),
                }
            ]
        )


def save_clipn_config(*, config: ClipnAdapterConfig, path: Path) -> Path:
    """Save CLIPn adapter configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data=json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def load_clipn_config(*, path: Path) -> ClipnAdapterConfig:
    """Load CLIPn adapter configuration from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClipnAdapterConfig(**data)


def save_model_pickle(*, model: object, path: Path) -> Path:
    """Save a Python model object using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="wb") as handle:
        pickle.dump(obj=model, file=handle)
    return path


def load_model_pickle(*, path: Path) -> object:
    """Load a Python model object using pickle."""
    with Path(path).open(mode="rb") as handle:
        return pickle.load(file=handle)


def _find_first_column(data_frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column found by case-insensitive matching."""
    lookup = {str(column).strip().lower(): column for column in data_frame.columns}
    for candidate in candidates:
        hit = lookup.get(str(candidate).strip().lower())
        if hit is not None:
            return str(hit)
    return None


def standardise_clipn_metadata(
    *,
    data_frame: pd.DataFrame,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise common metadata columns needed by CLIPn workflows.

    The function is non-destructive: original columns are kept and canonical
    aliases are added when needed.
    """
    output, column_report = normalise_column_names(data_frame=data_frame)
    records = []
    for canonical, aliases in METADATA_ALIASES.items():
        if canonical in output.columns:
            records.append(
                {
                    "dataset": dataset_name,
                    "canonical_column": canonical,
                    "source_column": canonical,
                    "action": "already_present",
                    "n_non_missing": int(output[canonical].notna().sum()),
                }
            )
            continue
        source = _find_first_column(output, aliases)
        if source is not None:
            output[canonical] = output[source]
            records.append(
                {
                    "dataset": dataset_name,
                    "canonical_column": canonical,
                    "source_column": source,
                    "action": "created_from_alias",
                    "n_non_missing": int(output[canonical].notna().sum()),
                }
            )
        elif canonical == "Library":
            output[canonical] = dataset_name
            records.append(
                {
                    "dataset": dataset_name,
                    "canonical_column": canonical,
                    "source_column": "dataset_name",
                    "action": "filled_from_dataset_name",
                    "n_non_missing": int(output[canonical].notna().sum()),
                }
            )
        elif canonical == "cpd_type":
            output[canonical] = "unknown"
            records.append(
                {
                    "dataset": dataset_name,
                    "canonical_column": canonical,
                    "source_column": "constant_unknown",
                    "action": "filled_unknown",
                    "n_non_missing": int(output[canonical].notna().sum()),
                }
            )
        else:
            records.append(
                {
                    "dataset": dataset_name,
                    "canonical_column": canonical,
                    "source_column": "",
                    "action": "missing",
                    "n_non_missing": 0,
                }
            )
    if "Well_Metadata" in output.columns:
        output["Well_Metadata"] = output["Well_Metadata"].map(
            lambda value: canonicalise_well_value(value=value)
        )
    output["Dataset"] = dataset_name
    output["Sample"] = np.arange(output.shape[0], dtype=int)
    if logger is not None:
        logger.info("[%s] Standardised metadata for CLIPn: %s", dataset_name, output.shape)
    alias_report = pd.concat(
        [column_report.assign(dataset=dataset_name), pd.DataFrame.from_records(records)],
        ignore_index=True,
        sort=False,
    )
    return output, alias_report


def split_single_dataset_by_group(
    *,
    data_frame: pd.DataFrame,
    group_column: str,
    random_state: int = 42,
    split_names: tuple[str, str] = ("split_a", "split_b"),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Split one table into two CLIPn datasets without splitting compounds.

    Parameters
    ----------
    data_frame:
        Input table to split.
    group_column:
        Column used to keep related rows together, usually ``cpd_id`` or
        ``Metadata_Compound``.
    random_state:
        Random seed for reproducible group assignment.
    split_names:
        Names to use for the two output datasets.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], pandas.DataFrame]
        Two datasets and a group-to-split report.

    Raises
    ------
    ValueError
        If the grouping column is missing or has fewer than two groups.
    """
    if group_column not in data_frame.columns:
        raise ValueError(f"Cannot split single CLIPn dataset; missing group column: {group_column}")
    groups = sorted(data_frame[group_column].dropna().astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("Cannot split single CLIPn dataset; at least two non-missing groups are required.")
    rng = np.random.default_rng(seed=int(random_state))
    shuffled = np.asarray(groups, dtype=object)
    rng.shuffle(shuffled)
    midpoint = max(1, int(math.ceil(len(shuffled) / 2)))
    split_a_groups = set(map(str, shuffled[:midpoint]))
    assignments = {
        str(group): split_names[0] if str(group) in split_a_groups else split_names[1]
        for group in groups
    }
    report = pd.DataFrame(
        {
            group_column: list(assignments.keys()),
            "assigned_dataset": list(assignments.values()),
        }
    )
    split_series = data_frame[group_column].astype(str).map(assignments)
    datasets = {
        name: data_frame.loc[split_series == name].reset_index(drop=True).copy()
        for name in split_names
    }
    empty = [name for name, table in datasets.items() if table.empty]
    if empty:
        raise ValueError(f"Single-dataset split created empty datasets: {empty}")
    return datasets, report


def validate_clipn_dataset_count(*, datasets: Mapping[str, pd.DataFrame]) -> None:
    """Require at least two non-empty datasets for CLIPn integration."""
    non_empty = [name for name, table in datasets.items() if not table.empty]
    if len(non_empty) < 2:
        raise ValueError(
            "CLIPn integration requires at least two non-empty datasets. "
            "Provide a manifest/repeated --dataset entries, or split one table by compound using "
            "--split_single_dataset_by_column."
        )


def remove_zero_only_clipn_profiles(
    *,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: ClipnAdapterConfig,
) -> tuple[pd.DataFrame, list[str], int, int]:
    """Remove zero-only rows/features before CLIPn modelling.

    Literal zero values can be introduced by valid preprocessing and scaling, so
    the default policy removes only all-zero feature columns and all-zero rows.
    The optional ``drop_rows_with_any_zero`` mode is deliberately off by default
    because it can be very destructive for scaled Cell Painting profiles.
    """
    output = table.copy()
    features = list(feature_cols)
    dropped_features: list[str] = []
    rows_dropped_all_zero = 0
    rows_dropped_any_zero = 0
    if features and config.remove_all_zero_features:
        zero_feature_mask = output[features].fillna(0.0).eq(0.0).all(axis=0)
        dropped_features = zero_feature_mask[zero_feature_mask].index.astype(str).tolist()
        features = [feature for feature in features if feature not in dropped_features]
        output = output.drop(columns=dropped_features, errors="ignore")
    if features and config.remove_all_zero_rows:
        zero_row_mask = output[features].fillna(0.0).eq(0.0).all(axis=1)
        rows_dropped_all_zero = int(zero_row_mask.sum())
        output = output.loc[~zero_row_mask].copy()
    if features and config.drop_rows_with_any_zero:
        any_zero_mask = output[features].eq(0.0).any(axis=1)
        rows_dropped_any_zero = int(any_zero_mask.sum())
        output = output.loc[~any_zero_mask].copy()
    return (
        output,
        [feature for feature in features if feature in output.columns],
        dropped_features,
        rows_dropped_all_zero,
        rows_dropped_any_zero,
    )


def read_datasets_manifest(*, path: Path | str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read a dataset manifest with columns ``dataset`` and ``path``."""
    manifest = read_table(path=path, logger=logger)
    lowered = {str(column).lower(): column for column in manifest.columns}
    rename = {}
    if "dataset" not in manifest.columns and "name" in lowered:
        rename[lowered["name"]] = "dataset"
    if "path" not in manifest.columns and "file" in lowered:
        rename[lowered["file"]] = "path"
    if rename:
        manifest = manifest.rename(columns=rename)
    missing = [column for column in ["dataset", "path"] if column not in manifest.columns]
    if missing:
        raise ValueError(f"Dataset manifest is missing required columns: {missing}")
    manifest = manifest.loc[:, ["dataset", "path"]].copy()
    manifest["dataset"] = manifest["dataset"].astype(str).str.strip()
    manifest["path"] = manifest["path"].astype(str).str.strip()
    return manifest


def load_clipn_datasets(
    *,
    dataset_paths: Mapping[str, Path | str],
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load and metadata-standardise named datasets for CLIPn."""
    datasets: dict[str, pd.DataFrame] = {}
    reports = []
    for name, path in dataset_paths.items():
        table = read_table(path=Path(path), logger=logger)
        table, report = standardise_clipn_metadata(
            data_frame=table,
            dataset_name=str(name),
            logger=logger,
        )
        datasets[str(name)] = table
        reports.append(report)
    alias_report = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return datasets, alias_report


def is_metadata_like_column(column: str) -> bool:
    """Return whether a column should be excluded from CLIPn features."""
    if column in TECHNICAL_COLUMNS:
        return True
    return bool(METADATA_NAME_PATTERNS.search(str(column).lower()))


def infer_clipn_feature_columns(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    """Infer numeric CLIPn features while excluding metadata and QC columns."""
    metadata_set = {str(column) for column in (metadata_columns or [])}
    features = []
    for column in data_frame.select_dtypes(include=[np.number]).columns:
        if column in metadata_set or is_metadata_like_column(str(column)):
            continue
        features.append(str(column))
    return features


def align_dataset_features(
    *,
    datasets: Mapping[str, pd.DataFrame],
    feature_columns: Optional[Sequence[str]] = None,
    metadata_columns: Optional[Sequence[str]] = None,
    return_feature_report: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame] | tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Align multiple datasets to a shared numeric feature set."""
    if not datasets:
        raise ValueError("At least one dataset is required.")
    validate_clipn_dataset_count(datasets=datasets)
    feature_sets = {}
    for name, table in datasets.items():
        if feature_columns is None:
            inferred = infer_clipn_feature_columns(
                data_frame=table,
                metadata_columns=metadata_columns,
            )
        else:
            inferred = [column for column in feature_columns if column in table.columns]
        feature_sets[name] = set(inferred)
    union = set().union(*feature_sets.values()) if feature_sets else set()
    intersection = set.intersection(*feature_sets.values()) if feature_sets else set()
    if feature_columns is None:
        features = sorted(intersection)
    else:
        features = [column for column in feature_columns if column in intersection]
    if not features:
        raise ValueError("No shared numeric feature columns were found across datasets.")
    aligned = {
        name: table.loc[:, features].apply(pd.to_numeric, errors="coerce").copy()
        for name, table in datasets.items()
    }
    summary = pd.DataFrame.from_records(
        [
            {
                "dataset": name,
                "n_rows": int(datasets[name].shape[0]),
                "n_original_columns": int(datasets[name].shape[1]),
                "n_candidate_features": int(len(feature_sets[name])),
                "n_shared_features": int(len(features)),
                "n_missing_from_union": int(len(union - feature_sets[name])),
            }
            for name in datasets
        ]
    )
    feature_report = pd.DataFrame.from_records(
        [
            {
                "feature": feature,
                "in_shared_intersection": feature in intersection,
                **{f"present_in_{name}": feature in values for name, values in feature_sets.items()},
            }
            for feature in sorted(union)
        ]
    )
    if return_feature_report:
        return aligned, summary, feature_report
    return aligned, summary


def clean_impute_and_scale_aligned(
    *,
    aligned: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, pd.DataFrame],
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Clean non-finite values, remove sparse rows/features, impute and scale."""
    combined = []
    for name, features in aligned.items():
        block = features.copy()
        block.insert(0, "Dataset", name)
        block.insert(1, "Sample", np.arange(block.shape[0], dtype=int))
        combined.append(block)
    table = pd.concat(combined, ignore_index=True, sort=False)
    feature_cols = [c for c in table.columns if c not in {"Dataset", "Sample"}]
    table.loc[:, feature_cols] = table[feature_cols].replace([np.inf, -np.inf], np.nan)
    extreme_mask = table[feature_cols].abs() > 1e10
    n_extreme = int(extreme_mask.sum().sum())
    if n_extreme:
        table.loc[:, feature_cols] = table[feature_cols].where(~extreme_mask, np.nan)
    missing_fraction = table[feature_cols].isna().mean(axis=0)
    missingness_dropped_features = missing_fraction[
        missing_fraction > config.max_feature_missing_fraction
    ].index.tolist()
    feature_cols = [c for c in feature_cols if c not in missingness_dropped_features]
    row_missing_fraction = table[feature_cols].isna().mean(axis=1) if feature_cols else pd.Series(1.0, index=table.index)
    row_keep = row_missing_fraction <= config.max_sample_missing_fraction
    dropped_rows = int((~row_keep).sum())
    table = table.loc[row_keep, ["Dataset", "Sample", *feature_cols]].copy()
    (
        table,
        feature_cols,
        zero_only_dropped_features,
        rows_dropped_all_zero,
        rows_dropped_any_zero,
    ) = remove_zero_only_clipn_profiles(
        table=table,
        feature_cols=feature_cols,
        config=config,
    )

    if not feature_cols:
        raise ValueError("No CLIPn features remained after missingness/zero filtering.")
    n_missing_before = int(table[feature_cols].isna().sum().sum()) if feature_cols else 0
    if config.imputation_method == "none":
        pass
    elif config.imputation_method == "knn":
        n_neighbours = min(5, max(1, int(table.shape[0]) - 1))
        imputer = KNNImputer(n_neighbors=n_neighbours)
        table.loc[:, feature_cols] = imputer.fit_transform(table[feature_cols])
    elif config.imputation_method in {"median", "mean"}:
        strategy = config.imputation_method
        group_columns = [c for c in config.imputation_group_columns if c in table.columns]
        if group_columns:
            for _, idx in table.groupby(group_columns, dropna=False, sort=False).groups.items():
                block = table.loc[idx, feature_cols]
                fill_values = block.median(axis=0) if strategy == "median" else block.mean(axis=0)
                table.loc[idx, feature_cols] = block.fillna(fill_values)
        fill_values = table[feature_cols].median(axis=0) if strategy == "median" else table[feature_cols].mean(axis=0)
        table.loc[:, feature_cols] = table[feature_cols].fillna(fill_values)
    else:
        raise ValueError(f"Unsupported imputation_method: {config.imputation_method}")
    n_missing_after = int(table[feature_cols].isna().sum().sum()) if feature_cols else 0

    if config.scaling_method != "none" and feature_cols:
        scaler_cls = RobustScaler if config.scaling_method == "robust" else StandardScaler
        scaler = scaler_cls()
        table.loc[:, feature_cols] = scaler.fit_transform(table[feature_cols])
    cleaned: dict[str, pd.DataFrame] = {}
    for name in aligned:
        block = table.loc[table["Dataset"] == name, ["Sample", *feature_cols]].copy()
        block = block.set_index("Sample")
        cleaned[name] = block
    summary = pd.DataFrame.from_records(
        [
            {
                "item": "shared_features_before_missingness_filter",
                "value": int(len(aligned[next(iter(aligned))].columns)) if aligned else 0,
            },
            {"item": "features_dropped_for_missingness", "value": int(len(missingness_dropped_features))},
            {"item": "features_after_missingness_filter", "value": int(len(feature_cols))},
            {"item": "zero_only_features_dropped", "value": int(len(zero_only_dropped_features))},
            {"item": "rows_dropped_for_missingness", "value": dropped_rows},
            {"item": "all_zero_rows_dropped", "value": rows_dropped_all_zero},
            {"item": "rows_with_any_zero_dropped", "value": rows_dropped_any_zero},
            {"item": "extreme_values_converted_to_missing", "value": n_extreme},
            {"item": "missing_values_before_imputation", "value": n_missing_before},
            {"item": "missing_values_after_imputation", "value": n_missing_after},
            {"item": "imputation_method", "value": config.imputation_method},
            {"item": "scaling_method", "value": config.scaling_method},
        ]
    )
    if logger is not None:
        logger.info("CLIPn preprocessing summary: %s", summary.to_dict(orient="records"))
    return cleaned, summary


def encode_labels_for_clipn(
    *,
    datasets: Mapping[str, pd.DataFrame],
    config: ClipnAdapterConfig,
) -> tuple[dict[str, np.ndarray], LabelEncoder, pd.DataFrame]:
    """Encode the configured label column globally across datasets."""
    labels = []
    for name, table in datasets.items():
        if config.label_column in table.columns:
            values = table[config.label_column].fillna("unknown").astype(str)
        else:
            values = pd.Series(["unknown"] * table.shape[0], index=table.index)
        labels.extend(values.tolist())
    encoder = LabelEncoder().fit(labels if labels else ["unknown"])
    encoded = {}
    rows = []
    for name, table in datasets.items():
        values = table.get(config.label_column, pd.Series(["unknown"] * table.shape[0])).fillna("unknown").astype(str)
        encoded[name] = encoder.transform(values)
        rows.append(
            {
                "dataset": name,
                "label_column": config.label_column,
                "n_rows": int(table.shape[0]),
                "n_classes_observed": int(values.nunique(dropna=True)),
            }
        )
    mapping = pd.DataFrame(
        {"encoded_label": np.arange(len(encoder.classes_), dtype=int), "label": encoder.classes_}
    )
    return encoded, encoder, pd.concat([pd.DataFrame.from_records(rows), mapping], ignore_index=True, sort=False)


def make_metadata_table(
    *,
    datasets: Mapping[str, pd.DataFrame],
    config: ClipnAdapterConfig,
) -> pd.DataFrame:
    """Return row-level metadata aligned to the original dataset rows."""
    requested = config.metadata_columns or [
        config.id_column,
        config.label_column,
        "Library",
        "Plate_Metadata",
        "Well_Metadata",
        "Metadata_Plate",
        "Metadata_Well",
    ]
    records = []
    for name, table in datasets.items():
        meta = pd.DataFrame({"Dataset": name, "Sample": np.arange(table.shape[0], dtype=int)})
        for column in requested:
            if column in table.columns and column not in meta.columns:
                meta[column] = table[column].to_numpy()
        records.append(meta)
    return pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()


def _resolve_backend_class(config: ClipnAdapterConfig) -> type:
    """Find a CLIPn-compatible model class."""
    module = importlib.import_module(name=config.backend_module)
    if config.model_class is not None:
        return getattr(module, config.model_class)
    if hasattr(module, "CLIPn"):
        return getattr(module, "CLIPn")
    try:
        model_module = importlib.import_module(name=f"{config.backend_module}.model")
        if hasattr(model_module, "CLIPn"):
            return getattr(model_module, "CLIPn")
    except Exception:
        pass
    raise ValueError(
        "Could not find a CLIPn model class. Provide --model_class or a compatible backend."
    )


def _safe_model_eval(model: object) -> None:
    """Put CLIPn wrapper or inner torch module in eval mode when possible."""
    for candidate in [model, getattr(model, "model", None)]:
        if candidate is not None and hasattr(candidate, "eval"):
            try:
                candidate.eval()
            except Exception:
                pass


def _normalise_latent(latent: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalise a latent matrix."""
    values = np.asarray(latent, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return values / norms


def _predict_clipn_model(
    *,
    model: object,
    data_dict: Mapping[int, np.ndarray],
    predict_method: str,
    chunk_rows: int = 100000,
    logger: Optional[logging.Logger] = None,
) -> dict[int, np.ndarray]:
    """Predict latent arrays, using row chunks for large matrices."""
    method = getattr(model, predict_method)
    output: dict[int, list[np.ndarray]] = {key: [] for key in data_dict}
    for key, matrix in data_dict.items():
        n_rows = int(matrix.shape[0])
        if n_rows == 0:
            output[key].append(np.empty((0, 0), dtype=float))
            continue
        for start in range(0, n_rows, max(1, int(chunk_rows))):
            end = min(start + int(chunk_rows), n_rows)
            if logger is not None:
                logger.info("Predicting CLIPn dataset %s rows %d:%d", key, start, end)
            pred = method({key: matrix[start:end, :]})
            values = pred[key] if isinstance(pred, Mapping) and key in pred else pred
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            output[key].append(np.asarray(values))
    return {key: np.concatenate(parts, axis=0) for key, parts in output.items()}


def _build_indexed_arrays(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    labels: Mapping[str, np.ndarray],
    reference_names: Optional[Sequence[str]] = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, str]]:
    """Build integer-keyed CLIPn input dictionaries."""
    names = list(cleaned.keys())
    if reference_names is not None:
        ref_set = set(reference_names)
        names = [name for name in names if name in ref_set]
        if not names:
            raise ValueError("No configured reference_names were present in the datasets.")
    mapping = {index: name for index, name in enumerate(names)}
    data_dict = {index: cleaned[name].to_numpy(dtype=float) for index, name in mapping.items()}
    label_dict = {index: np.asarray(labels[name], dtype=int)[: data_dict[index].shape[0]] for index, name in mapping.items()}
    return data_dict, label_dict, mapping


def _latent_to_table(
    *,
    latent_by_dataset: Mapping[int, np.ndarray],
    dataset_mapping: Mapping[int, str],
    metadata: pd.DataFrame,
    config: ClipnAdapterConfig,
) -> pd.DataFrame:
    """Convert latent dict to a tidy table with metadata."""
    frames = []
    for key, latent in latent_by_dataset.items():
        name = dataset_mapping[key]
        latent_values = _normalise_latent(latent) if config.normalise_latent else np.asarray(latent)
        latent_cols = [f"latent_{idx + 1}" for idx in range(latent_values.shape[1])]
        frame = pd.DataFrame(latent_values, columns=latent_cols)
        frame.insert(0, "Dataset", name)
        frame.insert(1, "Sample", np.arange(frame.shape[0], dtype=int))
        frames.append(frame)
    latent_table = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not latent_table.empty and not metadata.empty:
        latent_table = latent_table.merge(metadata, on=["Dataset", "Sample"], how="left")
    return latent_table


def fit_clipn_backend(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    labels: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """Fit/project a compatible CLIPn backend and return latent table."""
    model_class = _resolve_backend_class(config=config)
    reference_names = config.reference_names if config.mode == "reference_only" else None
    train_x, train_y, train_mapping = _build_indexed_arrays(
        cleaned=cleaned,
        labels=labels,
        reference_names=reference_names,
    )
    model = model_class(train_x, train_y, latent_dim=config.latent_dim)
    fit = getattr(model, config.fit_method)
    loss = fit(train_x, train_y, lr=config.learning_rate, epochs=config.epochs)
    _safe_model_eval(model)

    all_x, _, all_mapping = _build_indexed_arrays(cleaned=cleaned, labels=labels)
    # Best-effort support for reference-only projection with new dataset keys.
    if config.mode == "reference_only" and hasattr(model, "model"):
        try:
            reference_key = next(iter(train_mapping.keys()))
            for new_key in all_mapping:
                if new_key not in getattr(model.model, "encoders", {}):
                    model.model.encoders[new_key] = model.model.encoders[reference_key]
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not extend CLIPn encoders for projection: %s", exc)
    latent = _predict_clipn_model(
        model=model,
        data_dict=all_x,
        predict_method=config.predict_method,
        logger=logger,
    )
    latent_table = _latent_to_table(
        latent_by_dataset=latent,
        dataset_mapping=all_mapping,
        metadata=metadata,
        config=config,
    )
    loss_values = []
    if isinstance(loss, (list, tuple, np.ndarray)):
        loss_values = [float(value) for value in list(loss)]
    elif loss is not None:
        try:
            loss_values = [float(loss)]
        except Exception:
            loss_values = []
    loss_table = pd.DataFrame(
        {"epoch": np.arange(1, len(loss_values) + 1, dtype=int), "loss": loss_values}
    )
    return latent_table, model, loss_table


def fit_pca_fallback(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    config: ClipnAdapterConfig,
) -> tuple[pd.DataFrame, PCA, pd.DataFrame]:
    """Fit a PCA fallback to produce CLIPn-shaped diagnostic output."""
    names = list(cleaned.keys())
    matrix = pd.concat([cleaned[name] for name in names], ignore_index=True, sort=False)
    n_components = min(config.latent_dim, matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=max(1, n_components), random_state=config.random_state)
    values = model.fit_transform(matrix)
    frames = []
    start = 0
    for name in names:
        n_rows = cleaned[name].shape[0]
        frame = pd.DataFrame(values[start:start + n_rows, :])
        frame.insert(0, "Dataset", name)
        frame.insert(1, "Sample", np.arange(n_rows, dtype=int))
        frames.append(frame)
        start += n_rows
    latent = pd.concat(frames, ignore_index=True, sort=False)
    latent.columns = ["Dataset", "Sample", *[f"latent_{i + 1}" for i in range(values.shape[1])]]
    latent = latent.merge(metadata, on=["Dataset", "Sample"], how="left")
    explained = pd.DataFrame(
        {
            "component": [f"latent_{i + 1}" for i in range(values.shape[1])],
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    )
    return latent, model, explained


def calculate_latent_diagnostics(
    *,
    latent_table: pd.DataFrame,
    config: ClipnAdapterConfig,
) -> dict[str, pd.DataFrame]:
    """Calculate CLIPn latent-space diagnostics."""
    latent_cols = [column for column in latent_table.columns if str(column).startswith("latent_")]
    if not latent_cols or latent_table.shape[0] < 2:
        return {"latent_diagnostic_status": pd.DataFrame([{"message": "Insufficient latent data."}])}
    X = latent_table[latent_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variance = X.var(axis=0, ddof=1).reset_index()
    variance.columns = ["latent_dimension", "variance"]
    variance["is_low_variance"] = variance["variance"] < 1e-8

    k = min(max(1, config.n_neighbours), X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=config.distance_metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    rows = []
    id_col = config.id_column if config.id_column in latent_table.columns else None
    label_col = config.label_column if config.label_column in latent_table.columns else None
    for i in range(X.shape[0]):
        rank = 0
        for j, distance in zip(indices[i], distances[i]):
            if j == i:
                continue
            rank += 1
            row = {
                "query_index": int(i),
                "neighbour_index": int(j),
                "rank": int(rank),
                "distance": float(distance),
                "Query_Dataset": latent_table.iloc[i].get("Dataset"),
                "Neighbour_Dataset": latent_table.iloc[j].get("Dataset"),
            }
            if id_col:
                row["Query_id"] = latent_table.iloc[i].get(id_col)
                row["Neighbour_id"] = latent_table.iloc[j].get(id_col)
                row["same_id"] = bool(str(row["Query_id"]) == str(row["Neighbour_id"]))
            if label_col:
                row["Query_label"] = latent_table.iloc[i].get(label_col)
                row["Neighbour_label"] = latent_table.iloc[j].get(label_col)
                row["same_label"] = bool(str(row["Query_label"]) == str(row["Neighbour_label"]))
            rows.append(row)
            if rank >= k:
                break
    neighbours = pd.DataFrame.from_records(rows)
    metrics = []
    if "same_id" in neighbours.columns:
        metrics.append({"metric": "nearest_neighbour_same_id_rate", "value": float(neighbours.loc[neighbours["rank"] == 1, "same_id"].mean())})
    if "same_label" in neighbours.columns:
        metrics.append({"metric": "nearest_neighbour_same_label_rate", "value": float(neighbours.loc[neighbours["rank"] == 1, "same_label"].mean())})
    if "Dataset" in latent_table.columns and latent_table["Dataset"].nunique() > 1:
        same_dataset = neighbours["Query_Dataset"].astype(str) == neighbours["Neighbour_Dataset"].astype(str)
        metrics.append({"metric": "same_dataset_neighbour_rate", "value": float(same_dataset.mean())})
    for col in [config.label_column, "Dataset"]:
        if col in latent_table.columns and latent_table[col].nunique(dropna=True) >= 2:
            labels = latent_table[col].fillna("NA").astype(str)
            if labels.value_counts().min() >= 2:
                try:
                    score = silhouette_score(X, labels, metric=config.distance_metric)
                    metrics.append({"metric": f"silhouette_{col}", "value": float(score)})
                except Exception:
                    pass
    summary = pd.DataFrame.from_records(metrics)
    return {
        "latent_variance": variance,
        "nearest_neighbours": neighbours,
        "latent_diagnostic_summary": summary,
    }


def write_latent_plots(
    *,
    latent_table: pd.DataFrame,
    output_dir: Path,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Write PCA/UMAP-like plots from latent dimensions."""
    latent_cols = [column for column in latent_table.columns if str(column).startswith("latent_")]
    if len(latent_cols) < 2:
        return []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    X = latent_table[latent_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    metadata = latent_table.drop(columns=latent_cols, errors="ignore")
    written: list[Path] = []
    pca_scores, explained = run_pca(features=X, n_components=min(2, X.shape[1]), random_state=config.random_state)
    pca_scores = pca_scores.rename(columns={"PC1": "x", "PC2": "y"})
    if {"x", "y"}.issubset(pca_scores.columns):
        written.extend(
            plot_embedding(
                embedding=pca_scores,
                metadata=metadata,
                x_column="x",
                y_column="y",
                colour_column=config.label_column if config.label_column in metadata.columns else "Dataset",
                output_path_base=plot_dir / "clipn_latent_pca",
                title="CLIPn latent PCA",
                logger=logger,
            )
        )
    plot_pca_variance(
        explained_variance=explained,
        output_path_base=plot_dir / "clipn_latent_pca_variance",
        logger=logger,
    )
    written.extend([plot_dir / "clipn_latent_pca_variance.pdf", plot_dir / "clipn_latent_pca_variance.svg"])
    # UMAP can be expensive on first import because it may trigger numba/JIT
    # compilation.  For very small smoke-test-sized data, a second PCA embedding
    # is more useful and avoids slow optional dependency startup.
    if X.shape[0] >= 10:
        embedding = run_umap_or_pca(features=X, logger=logger, random_state=config.random_state)
        embedding_label = "CLIPn latent UMAP/PCA"
    else:
        embedding, _ = run_pca(features=X, n_components=min(2, X.shape[1]), random_state=config.random_state)
        embedding_label = "CLIPn latent PCA smoke-test embedding"
    x_col, y_col = embedding.columns[:2]
    written.extend(
        plot_embedding(
            embedding=embedding,
            metadata=metadata,
            x_column=x_col,
            y_column=y_col,
            colour_column=config.label_column if config.label_column in metadata.columns else "Dataset",
            output_path_base=plot_dir / "clipn_latent_umap_or_pca",
            title=embedding_label,
            logger=logger,
        )
    )
    try:
        from cpatk.plotting import write_interactive_embedding_html

        html_path = write_interactive_embedding_html(
            embedding=embedding,
            metadata=metadata,
            x_column=x_col,
            y_column=y_col,
            colour_column=config.label_column if config.label_column in metadata.columns else "Dataset",
            output_path=plot_dir / "clipn_latent_interactive.html",
            title="Interactive CLIPn latent embedding",
            logger=logger,
        )
        written.append(html_path)
    except Exception as exc:
        if logger is not None:
            logger.warning("Interactive CLIPn embedding failed: %s", exc)
    return [path for path in written if Path(path).exists()]


def run_clipn_workflow(
    *,
    datasets: Mapping[str, pd.DataFrame],
    output_dir: Path,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
    save_model_path: Optional[Path] = None,
) -> Mapping[str, pd.DataFrame]:
    """Run the full CPATK CLIPn adapter workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_clipn_config(config=config, path=output_dir / "clipn_adapter_config.json")
    metadata = make_metadata_table(datasets=datasets, config=config)
    aligned, feature_summary, feature_report = align_dataset_features(
        datasets=datasets,
        feature_columns=config.feature_columns,
        metadata_columns=config.metadata_columns,
        return_feature_report=True,
    )
    cleaned, preprocessing_summary = clean_impute_and_scale_aligned(
        aligned=aligned,
        metadata=datasets,
        config=config,
        logger=logger,
    )
    labels, label_encoder, label_report = encode_labels_for_clipn(datasets=datasets, config=config)
    status = check_clipn_backend(backend_module=config.backend_module)
    output_tables: dict[str, pd.DataFrame] = {
        "clipn_status": status,
        "clipn_feature_summary": feature_summary,
        "clipn_feature_report": feature_report,
        "clipn_preprocessing_summary": preprocessing_summary,
        "clipn_label_report": label_report,
    }
    latent_table = pd.DataFrame()
    model: object | None = None
    loss_table = pd.DataFrame()
    warnings = []
    if bool(status["available"].iloc[0]):
        try:
            latent_table, model, loss_table = fit_clipn_backend(
                cleaned=cleaned,
                labels=labels,
                metadata=metadata,
                config=config,
                logger=logger,
            )
            run_status = pd.DataFrame(
                [{"backend_run": "success", "message": "CLIPn backend completed."}]
            )
            if save_model_path is not None and model is not None:
                save_model_pickle(model=model, path=save_model_path)
        except Exception as exc:
            if logger is not None:
                logger.exception("CLIPn backend execution failed.")
            warnings.append(f"CLIPn backend execution failed: {exc}")
            run_status = pd.DataFrame(
                [{"backend_run": "failed", "message": str(exc)}]
            )
    else:
        message = str(status["message"].iloc[0])
        warnings.append(f"CLIPn backend unavailable: {message}")
        run_status = pd.DataFrame(
            [{"backend_run": "not_run", "message": message}]
        )
    output_tables["clipn_run_status"] = run_status
    if latent_table.empty and config.allow_pca_fallback:
        latent_table, model, fallback_info = fit_pca_fallback(
            cleaned=cleaned,
            metadata=metadata,
            config=config,
        )
        output_tables["pca_fallback_explained_variance"] = fallback_info
        warnings.append("PCA fallback was used because CLIPn latent output was unavailable.")
    if not latent_table.empty:
        output_tables["clipn_latent"] = latent_table
        if not loss_table.empty:
            output_tables["clipn_training_loss"] = loss_table
        output_tables.update(calculate_latent_diagnostics(latent_table=latent_table, config=config))
        plot_paths = write_latent_plots(
            latent_table=latent_table,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )
    else:
        plot_paths = []
    for name, table in output_tables.items():
        if name == "clipn_latent":
            write_table(data_frame=table, path=output_dir / f"{name}.tsv.gz", logger=logger)
        else:
            write_table(data_frame=table, path=output_dir / f"{name}.tsv", logger=logger)
    write_excel_workbook(tables=output_tables, path=output_dir / "clipn_summary.xlsx", logger=logger)
    make_html_report(
        title="CPATK CLIPn adapter report",
        output_path=output_dir / "clipn_report.html",
        summary_tables=output_tables,
        plot_paths=plot_paths,
        narrative=(
            "This report summarises the optional CLIPn workflow. The adapter first harmonises "
            "features across datasets, cleans non-finite values, handles missing data, scales "
            "features, removes zero-only rows/features for CLIPn compatibility, and then runs a compatible CLIPn backend when available. If the backend "
            "is unavailable, the report records the reason and preserves all preprocessing audits."
        ),
        warnings=warnings,
        methods_text=(
            "CLIPn is treated as an optional integration layer. CPATK freezes the shared feature "
            "intersection before model fitting to prevent accidental feature-order drift between "
            "datasets and requires at least two non-empty datasets. Metadata and encoded labels are excluded from the input feature matrix. "
            "Latent-space diagnostics are descriptive checks of replicate, class and dataset "
            "structure; they should be interpreted alongside the non-AI CPATK workflow."
        ),
    )
    return output_tables


def run_clipn_adapter(
    *,
    datasets: Mapping[str, pd.DataFrame],
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> Mapping[str, pd.DataFrame]:
    """Backward-compatible in-memory CLIPn adapter wrapper."""
    status = check_clipn_backend(backend_module=config.backend_module)
    try:
        aligned, feature_summary, feature_report = align_dataset_features(
            datasets=datasets,
            feature_columns=config.feature_columns,
            metadata_columns=config.metadata_columns,
            return_feature_report=True,
        )
    except Exception as exc:
        return {
            "clipn_status": pd.DataFrame(
                [{"available": False, "backend_module": config.backend_module, "message": str(exc)}]
            )
        }
    if not bool(status["available"].iloc[0]):
        return {
            "clipn_status": status,
            "clipn_feature_summary": feature_summary,
            "clipn_feature_report": feature_report,
        }
    # For backwards compatibility, run a lightweight workflow without writing files by
    # performing the backend fit directly.  Errors are converted to a status table.
    try:
        metadata = make_metadata_table(datasets=datasets, config=config)
        cleaned, preprocessing = clean_impute_and_scale_aligned(
            aligned=aligned,
            metadata=datasets,
            config=config,
            logger=logger,
        )
        labels, _, label_report = encode_labels_for_clipn(datasets=datasets, config=config)
        latent, _, loss_table = fit_clipn_backend(
            cleaned=cleaned,
            labels=labels,
            metadata=metadata,
            config=config,
            logger=logger,
        )
        return {
            "clipn_status": status,
            "clipn_feature_summary": feature_summary,
            "clipn_feature_report": feature_report,
            "clipn_preprocessing_summary": preprocessing,
            "clipn_label_report": label_report,
            "clipn_training_loss": loss_table,
            "clipn_latent": latent,
        }
    except Exception as exc:
        if logger is not None:
            logger.warning("CLIPn adapter failed: %s", exc)
        failure = status.copy()
        failure["message"] = f"Backend import succeeded but adapter execution failed: {exc}"
        return {
            "clipn_status": failure,
            "clipn_feature_summary": feature_summary,
            "clipn_feature_report": feature_report,
        }
