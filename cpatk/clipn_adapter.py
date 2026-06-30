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
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
try:
    from sklearn.decomposition import PCA
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised in broken HPC environments
    PCA = None
    KNNImputer = None
    SimpleImputer = None
    pairwise_distances = None
    silhouette_score = None
    NearestNeighbors = None
    LabelEncoder = None
    RobustScaler = None
    StandardScaler = None
    SKLEARN_IMPORT_ERROR = exc


def require_sklearn_stack(*, purpose: str) -> None:
    """Raise a clear error when the SciPy/scikit-learn stack cannot import."""
    if SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "The SciPy/scikit-learn stack could not be imported for "
            f"{purpose}. This often means the conda environment is using an "
            "older system libstdc++. Put ${CONDA_PREFIX}/lib first in "
            "LD_LIBRARY_PATH or reinstall scipy/scikit-learn from conda-forge."
        ) from SKLEARN_IMPORT_ERROR

from cpatk.contrastive import (
    NativeContrastiveConfig,
    fit_native_contrastive_backend,
    get_native_contrastive_status,
)
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

    backend_module: str = "cpatk_contrastive"
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
    early_stopping: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    early_stopping_chunk_size: int = 10
    validation_fraction: float = 0.0
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
    zero_policy: str = "keep"
    zero_epsilon: float = 1e-8  # Deprecated compatibility field; zeros are not replaced.
    allow_pca_fallback: bool = False
    native_positive_column: Optional[str] = None
    native_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    native_activation: str = "gelu"
    native_normalisation: str = "layernorm"
    native_dropout: float = 0.10
    native_batch_size: int = 256
    native_positives_per_label: int = 2
    native_temperature: float = 0.10
    native_weight_decay: float = 1e-4
    native_eval_batches: int = 4
    native_steps_per_epoch: Optional[int] = None
    native_device: str = "auto"
    native_encode_chunk_size: int = 32768
    run_compound_holdout_validation: bool = False
    compound_holdout_column: Optional[str] = None
    compound_holdout_fraction: float = 0.20
    compound_holdout_repeats: int = 5
    compound_holdout_seed: int = 42
    compound_holdout_min_profiles: int = 4
    n_threads: int = 1


def check_clipn_backend(*, backend_module: str = "clipn") -> pd.DataFrame:
    """Check whether a latent backend module can be imported."""
    if str(backend_module).lower() in {"cpatk_contrastive", "native_contrastive"}:
        return get_native_contrastive_status()
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


def collect_clipn_backend_provenance(
    *,
    backend_module: str,
    backend_run: str = "not_run",
    pca_fallback_used: bool = False,
    loss_table: Optional[pd.DataFrame] = None,
    training_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Collect runtime provenance for a CLIPn adapter run.

    The output is intentionally table-shaped so it can be written directly into
    results folders and HTML reports.  It records enough information to tell
    whether a latent space came from a real CLIPn backend run, a PCA fallback,
    CPU execution or a CUDA-backed run.
    """
    torch_version = "not_imported"
    torch_cuda_available: object = "not_imported"
    torch_cuda_device_count: object = "not_imported"
    torch_cuda_device_names = ""
    try:
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_device_count = int(torch.cuda.device_count())
        names = []
        for index in range(int(torch_cuda_device_count)):
            try:
                names.append(str(torch.cuda.get_device_name(index)))
            except Exception:
                names.append(f"device_{index}_name_unavailable")
        torch_cuda_device_names = ";".join(names)
    except Exception as exc:
        torch_version = f"import_failed: {exc}"
        torch_cuda_available = "unknown"
        torch_cuda_device_count = "unknown"

    n_loss_rows = int(loss_table.shape[0]) if loss_table is not None and not loss_table.empty else 0
    final_loss = float("nan")
    min_loss = float("nan")
    if loss_table is not None and not loss_table.empty:
        loss_column = next(
            (column for column in ["loss", "monitor_loss", "validation_loss", "train_loss"] if column in loss_table.columns),
            None,
        )
        if loss_column is not None:
            losses = pd.to_numeric(loss_table[loss_column], errors="coerce").dropna()
            if not losses.empty:
                final_loss = float(losses.iloc[-1])
                min_loss = float(losses.min())

    training_policy = "unknown"
    stopping_reason = "unknown"
    best_epoch: object = np.nan
    if training_summary is not None and not training_summary.empty:
        row = training_summary.iloc[0]
        training_policy = str(row.get("training_policy", "unknown"))
        stopping_reason = str(row.get("stopping_reason", "unknown"))
        best_epoch = row.get("best_epoch", np.nan)

    return pd.DataFrame.from_records(
        [
            {
                "backend_module": backend_module,
                "backend_run": backend_run,
                "pca_fallback_used": bool(pca_fallback_used),
                "python_version": sys.version.replace("\n", " "),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "torch_version": torch_version,
                "torch_cuda_available": torch_cuda_available,
                "torch_cuda_device_count": torch_cuda_device_count,
                "torch_cuda_device_names": torch_cuda_device_names,
                "training_loss_rows": n_loss_rows,
                "final_training_loss": final_loss,
                "minimum_training_loss": min_loss,
                "training_policy": training_policy,
                "stopping_reason": stopping_reason,
                "best_epoch": best_epoch,
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
    """Remove all-zero rows/features before CLIPn imputation.

    These filters remove profiles or features that contain no signal at all.
    This is an empty-signal QC step, not a CLIPn requirement to remove
    literal zeros. Real zero values are audited later by
    ``apply_clipn_zero_policy`` and are kept by default.
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
    return (
        output,
        [feature for feature in features if feature in output.columns],
        dropped_features,
        rows_dropped_all_zero,
        rows_dropped_any_zero,
    )


def apply_clipn_zero_policy(
    *,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Apply the final CLIPn literal-zero policy after imputation/scaling.

    CLIPn can accept literal zeros, but it cannot accept missing, NaN or
    non-finite values.  This function therefore audits zeros after
    imputation/scaling but leaves them unchanged by default.  Strict row
    dropping and error-on-zero modes remain available for legacy/debugging
    checks, but they are not recommended for general Cell Painting matrices.
    """
    output = table.copy()
    features = [feature for feature in feature_cols if feature in output.columns]
    policy = "drop_rows" if config.drop_rows_with_any_zero else str(config.zero_policy).lower()
    if policy not in {"drop_rows", "keep", "error"}:
        raise ValueError(f"Unsupported CLIPn zero_policy: {config.zero_policy}")
    if not features:
        report = pd.DataFrame.from_records(
            [{"item": "zero_policy", "value": policy}, {"item": "n_features_checked", "value": 0}]
        )
        return output, features, report

    zero_mask = output[features].eq(0.0)
    rows_with_zero_mask = zero_mask.any(axis=1)
    features_with_zero_mask = zero_mask.any(axis=0)
    zeros_before = int(zero_mask.sum().sum())
    rows_with_zero_before = int(rows_with_zero_mask.sum())
    features_with_zero_before = int(features_with_zero_mask.sum())
    rows_dropped = 0
    zeros_replaced = 0

    if zeros_before and policy == "error":
        raise ValueError(
            "CLIPn input contains literal zero values after imputation/scaling. "
            "CLIPn can usually accept zeros, so use --clipn_zero_policy keep unless "
            "you are deliberately auditing zero sensitivity."
        )
    if zeros_before and policy == "drop_rows":
        rows_dropped = rows_with_zero_before
        output = output.loc[~rows_with_zero_mask].copy()
        if output.empty:
            raise ValueError(
                "Strict CLIPn zero filtering removed every sample. "
                "CLIPn can accept literal zeros; rerun with --clipn_zero_policy keep "
                "and without --drop_rows_with_any_zero unless this strict behaviour "
                "was intentional."
            )

    zeros_after = int(output[features].eq(0.0).sum().sum()) if features else 0
    if logger is not None:
        logger.info(
            "CLIPn zero policy applied: policy=%s, zeros_before=%s, "
            "zeros_replaced=%s, rows_dropped=%s, zeros_after=%s",
            policy,
            zeros_before,
            zeros_replaced,
            rows_dropped,
            zeros_after,
        )
    report = pd.DataFrame.from_records(
        [
            {"item": "zero_policy", "value": policy},
            {"item": "n_features_checked", "value": int(len(features))},
            {"item": "literal_zero_values_before_policy", "value": zeros_before},
            {"item": "rows_with_literal_zero_before_policy", "value": rows_with_zero_before},
            {"item": "features_with_literal_zero_before_policy", "value": features_with_zero_before},
            {"item": "rows_dropped_by_zero_policy", "value": rows_dropped},
            {"item": "literal_zero_values_changed_by_policy", "value": zeros_replaced},
            {"item": "literal_zero_values_after_policy", "value": zeros_after},
        ]
    )
    return output, features, report


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
    """Clean non-finite values, impute/scale and prepare CLIPn-safe matrices."""
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
    if table.empty:
        message = (
            "No CLIPn samples remained after missingness/zero filtering. "
            "If --drop_rows_with_any_zero was used, rerun without that strict option; "
            "real Cell Painting matrices often contain some zero-valued measurements "
            "even after valid preprocessing."
        )
        if logger is not None:
            logger.error(message)
        raise ValueError(message)
    empty_datasets = [
        name
        for name in aligned
        if int((table["Dataset"] == name).sum()) == 0
    ]
    if empty_datasets:
        message = (
            "One or more CLIPn datasets have no samples after filtering: "
            f"{', '.join(empty_datasets)}. "
            "Relax missingness/zero filtering or rerun without --drop_rows_with_any_zero."
        )
        if logger is not None:
            logger.error(message)
        raise ValueError(message)
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

    table, feature_cols, zero_policy_report = apply_clipn_zero_policy(
        table=table,
        feature_cols=feature_cols,
        config=config,
        logger=logger,
    )
    if table.empty:
        raise ValueError("No CLIPn samples remained after final zero-policy handling.")

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
            {"item": "rows_with_any_zero_dropped_pre_imputation", "value": rows_dropped_any_zero},
            {"item": "extreme_values_converted_to_missing", "value": n_extreme},
            {"item": "missing_values_before_imputation", "value": n_missing_before},
            {"item": "missing_values_after_imputation", "value": n_missing_after},
            {"item": "imputation_method", "value": config.imputation_method},
            {"item": "scaling_method", "value": config.scaling_method},
        ]
    )
    summary = pd.concat([summary, zero_policy_report], ignore_index=True, sort=False)
    if logger is not None:
        logger.info("CLIPn preprocessing summary: %s", summary.to_dict(orient="records"))
    return cleaned, summary


def encode_labels_for_clipn(
    *,
    datasets: Mapping[str, pd.DataFrame],
    config: ClipnAdapterConfig,
) -> tuple[dict[str, np.ndarray], LabelEncoder, pd.DataFrame]:
    """Encode the configured label column globally across datasets."""
    require_sklearn_stack(purpose="CLIPn label encoding")
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




def _loss_to_float_list(loss: object) -> list[float]:
    """Convert backend loss output to a list of floats when possible."""
    if isinstance(loss, (list, tuple, np.ndarray, pd.Series)):
        values = []
        for value in list(loss):
            try:
                values.append(float(value))
            except Exception:
                continue
        return values
    if loss is None:
        return []
    try:
        return [float(loss)]
    except Exception:
        return []


def _fit_clipn_with_training_policy(
    *,
    model: object,
    fit_method: str,
    train_x: Mapping[int, np.ndarray],
    train_y: Mapping[int, np.ndarray],
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the backend using fixed epochs or conservative loss-plateau stopping.

    Most CLIPn backends expose only a ``fit(X, y, lr, epochs)`` hook and do not
    provide a validation callback. CPATK therefore uses fixed epochs by default.
    If early stopping is enabled, it trains the same model in short chunks and
    stops when the reported training loss has not improved by ``min_delta`` for
    ``patience`` reported loss rows. This is not a validation-loss guarantee;
    the report states that clearly.
    """
    fit = getattr(model, fit_method)
    requested_epochs = int(max(config.epochs, 1))
    records: list[dict] = []
    summary = {
        "training_policy": "fixed_epochs",
        "requested_max_epochs": requested_epochs,
        "early_stopping_enabled": bool(config.early_stopping),
        "early_stopping_patience": int(config.early_stopping_patience),
        "early_stopping_min_delta": float(config.early_stopping_min_delta),
        "early_stopping_chunk_size": int(config.early_stopping_chunk_size),
        "validation_fraction_requested": float(config.validation_fraction),
        "validation_monitor_used": "none_backend_lacks_validation_hook",
        "fit_calls": 0,
        "reported_loss_rows": 0,
        "best_epoch": np.nan,
        "best_loss": np.nan,
        "final_loss": np.nan,
        "stopping_reason": "completed_fixed_epochs",
    }
    if not config.early_stopping:
        loss = fit(train_x, train_y, lr=config.learning_rate, epochs=requested_epochs)
        loss_values = _loss_to_float_list(loss)
        records = [
            {"epoch": int(index), "loss": float(value), "fit_call": 1}
            for index, value in enumerate(loss_values, start=1)
        ]
        summary["fit_calls"] = 1
    else:
        summary["training_policy"] = "chunked_training_loss_early_stopping"
        chunk_size = int(max(config.early_stopping_chunk_size, 1))
        patience = int(max(config.early_stopping_patience, 1))
        min_delta = float(max(config.early_stopping_min_delta, 0.0))
        best_loss = math.inf
        best_epoch: Optional[int] = None
        rows_without_improvement = 0
        fit_call = 0
        reported_epoch = 0
        epochs_requested_so_far = 0
        while epochs_requested_so_far < requested_epochs:
            remaining = requested_epochs - epochs_requested_so_far
            this_chunk = int(min(chunk_size, remaining))
            fit_call += 1
            loss = fit(train_x, train_y, lr=config.learning_rate, epochs=this_chunk)
            epochs_requested_so_far += this_chunk
            loss_values = _loss_to_float_list(loss)
            if not loss_values:
                if logger is not None:
                    logger.warning(
                        "CLIPn backend returned no numeric loss values during fit call %d; continuing until max epochs.",
                        fit_call,
                    )
                continue
            for value in loss_values:
                reported_epoch += 1
                records.append({"epoch": reported_epoch, "loss": float(value), "fit_call": fit_call})
                if np.isfinite(value) and (best_loss - float(value)) > min_delta:
                    best_loss = float(value)
                    best_epoch = reported_epoch
                    rows_without_improvement = 0
                else:
                    rows_without_improvement += 1
            if rows_without_improvement >= patience:
                summary["stopping_reason"] = "early_stopping_training_loss_plateau"
                break
        summary["fit_calls"] = fit_call
        if summary["stopping_reason"] != "early_stopping_training_loss_plateau":
            summary["stopping_reason"] = "completed_max_epochs_before_patience"
        if best_epoch is not None:
            summary["best_epoch"] = int(best_epoch)
            summary["best_loss"] = float(best_loss)
    loss_table = pd.DataFrame.from_records(records, columns=["epoch", "loss", "fit_call"])
    if not loss_table.empty and "loss" in loss_table.columns:
        losses = pd.to_numeric(loss_table["loss"], errors="coerce").dropna()
        summary["reported_loss_rows"] = int(loss_table.shape[0])
        if not losses.empty:
            summary["final_loss"] = float(losses.iloc[-1])
            if not np.isfinite(float(summary["best_loss"])):
                summary["best_loss"] = float(losses.min())
                summary["best_epoch"] = int(losses.idxmin() + 1)
    training_summary = pd.DataFrame.from_records([summary])
    return loss_table, training_summary



def _native_config_from_clipn_config(
    *,
    config: ClipnAdapterConfig,
    heldout_positive_values: Optional[Sequence[str]] = None,
) -> NativeContrastiveConfig:
    """Translate the shared adapter config into native contrastive config."""
    return NativeContrastiveConfig(
        latent_dim=int(config.latent_dim),
        hidden_dims=list(config.native_hidden_dims),
        activation=str(config.native_activation),
        normalisation=str(config.native_normalisation),
        dropout=float(config.native_dropout),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.native_weight_decay),
        epochs=int(config.epochs),
        batch_size=int(config.native_batch_size),
        positives_per_label=int(config.native_positives_per_label),
        temperature=float(config.native_temperature),
        validation_fraction=float(config.validation_fraction),
        early_stopping_patience=int(config.early_stopping_patience),
        early_stopping_min_delta=float(config.early_stopping_min_delta),
        eval_batches=int(config.native_eval_batches),
        steps_per_epoch=(
            None
            if config.native_steps_per_epoch is None or int(config.native_steps_per_epoch) <= 0
            else int(config.native_steps_per_epoch)
        ),
        random_state=int(config.random_state),
        device=str(config.native_device),
        positive_column=str(config.native_positive_column or config.id_column),
        normalise_latent=bool(config.normalise_latent),
        encode_chunk_size=int(config.native_encode_chunk_size),
        n_threads=int(config.n_threads),
        heldout_positive_values=[str(value) for value in (heldout_positive_values or [])],
    )


def fit_cpatk_contrastive_backend(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, object, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit the CPATK-native contrastive backend and return adapter-shaped output."""
    native_config = _native_config_from_clipn_config(config=config)
    result = fit_native_contrastive_backend(
        cleaned=cleaned,
        metadata=metadata,
        config=native_config,
        logger=logger,
    )
    extra_tables = {
        "cpatk_contrastive_positive_label_report": result.positive_label_report,
        "cpatk_contrastive_split_report": result.split_report,
        "cpatk_contrastive_backend_status": result.backend_status,
    }
    return (
        result.latent_table,
        result.model,
        result.training_loss,
        result.training_summary,
        extra_tables,
    )

def fit_clipn_backend(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    labels: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, object, pd.DataFrame, pd.DataFrame]:
    """Fit/project a compatible CLIPn backend and return latent table."""
    model_class = _resolve_backend_class(config=config)
    reference_names = config.reference_names if config.mode == "reference_only" else None
    train_x, train_y, train_mapping = _build_indexed_arrays(
        cleaned=cleaned,
        labels=labels,
        reference_names=reference_names,
    )
    model = model_class(train_x, train_y, latent_dim=config.latent_dim)
    loss_table, training_summary = _fit_clipn_with_training_policy(
        model=model,
        fit_method=config.fit_method,
        train_x=train_x,
        train_y=train_y,
        config=config,
        logger=logger,
    )
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
    return latent_table, model, loss_table, training_summary


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
    require_sklearn_stack(purpose="PCA fallback")
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
    require_sklearn_stack(purpose="latent nearest-neighbour analysis")
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=config.distance_metric,
        n_jobs=max(1, int(config.n_threads)),
    )
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
                    require_sklearn_stack(purpose="latent diagnostics")
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


def diagnose_latent_space_quality(
    *,
    diagnostic_summary: pd.DataFrame,
    config: ClipnAdapterConfig,
    backend_module: str,
) -> pd.DataFrame:
    """Create interpretation warnings for latent-space diagnostics.

    These warnings are deliberately conservative.  They are not used to fail
    a run; they tell users when the latent embedding should be treated as a
    technical visualisation rather than strong biological evidence.
    """
    if diagnostic_summary is None or diagnostic_summary.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "backend_module": backend_module,
                    "severity": "info",
                    "diagnostic": "no_latent_diagnostics",
                    "message": "No latent-space diagnostics were available for interpretation.",
                }
            ]
        )
    values = {
        str(row["metric"]): float(row["value"])
        for _, row in diagnostic_summary.iterrows()
        if "metric" in diagnostic_summary.columns and "value" in diagnostic_summary.columns
        and pd.notna(row.get("value"))
    }
    records = []
    same_id_rate = values.get("nearest_neighbour_same_id_rate")
    if same_id_rate is not None and same_id_rate < 0.25:
        records.append(
            {
                "backend_module": backend_module,
                "severity": "warning",
                "diagnostic": "low_same_id_retrieval",
                "message": (
                    "Nearest-neighbour retrieval of the same compound/id is low. "
                    "Use the latent embedding cautiously and prioritise classical CPATK diagnostics."
                ),
            }
        )
    same_label_rate = values.get("nearest_neighbour_same_label_rate")
    if same_label_rate is not None and same_label_rate < 0.35:
        records.append(
            {
                "backend_module": backend_module,
                "severity": "warning",
                "diagnostic": "low_same_label_retrieval",
                "message": (
                    "Nearest-neighbour retrieval of the same label/class is low. "
                    "The latent space may not support strong class-level interpretation."
                ),
            }
        )
    same_dataset_rate = values.get("same_dataset_neighbour_rate")
    if same_dataset_rate is not None and same_dataset_rate > 0.80:
        records.append(
            {
                "backend_module": backend_module,
                "severity": "warning",
                "diagnostic": "high_same_dataset_neighbour_rate",
                "message": (
                    "Most latent nearest neighbours come from the same dataset/source. "
                    "This suggests residual dataset or batch structure in the embedding."
                ),
            }
        )
    dataset_silhouette = values.get("silhouette_Dataset")
    label_silhouette = values.get(f"silhouette_{config.label_column}")
    if (
        dataset_silhouette is not None
        and label_silhouette is not None
        and dataset_silhouette > label_silhouette
    ):
        records.append(
            {
                "backend_module": backend_module,
                "severity": "warning",
                "diagnostic": "dataset_structure_exceeds_label_structure",
                "message": (
                    "Dataset/source separation is stronger than the configured biological label separation. "
                    "Do not treat the latent space as primarily biological without supporting diagnostics."
                ),
            }
        )
    if not records:
        records.append(
            {
                "backend_module": backend_module,
                "severity": "info",
                "diagnostic": "latent_diagnostics_no_major_warning",
                "message": "No major latent-space warning was triggered by the current simple diagnostic rules.",
            }
        )
    return pd.DataFrame.from_records(records)




def diagnose_feature_alignment_quality(*, feature_summary: pd.DataFrame) -> pd.DataFrame:
    """Warn when shared-feature alignment suggests missing compartments/blocks.

    The native contrastive backend can be useful under ordinary batch effects,
    but the v0.2.31 synthetic validation showed that source-linked missing
    compartments are a serious failure mode. This diagnostic flags cases where
    the shared feature intersection is a small fraction of the feature union or
    where one dataset contributes many fewer candidate features than another.
    """
    if feature_summary is None or feature_summary.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "severity": "info",
                    "diagnostic": "no_feature_alignment_summary",
                    "message": "No feature-alignment summary was available.",
                }
            ]
        )
    records = []
    frame = feature_summary.copy()
    for column in ["n_candidate_features", "n_shared_features", "n_missing_from_union"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if {"n_candidate_features", "n_shared_features", "n_missing_from_union"}.issubset(frame.columns):
        union_estimates = frame["n_candidate_features"] + frame["n_missing_from_union"]
        union_size = float(np.nanmax(union_estimates)) if not union_estimates.empty else float("nan")
        shared_size = float(np.nanmax(frame["n_shared_features"])) if "n_shared_features" in frame.columns else float("nan")
        retained_fraction = shared_size / union_size if np.isfinite(union_size) and union_size > 0 else float("nan")
        max_missing_fraction = float(np.nanmax(frame["n_missing_from_union"] / union_estimates.replace(0, np.nan)))
        if np.isfinite(retained_fraction) and retained_fraction < 0.70:
            records.append(
                {
                    "severity": "warning",
                    "diagnostic": "low_shared_feature_fraction",
                    "value": retained_fraction,
                    "message": (
                        "The shared feature intersection is less than 70% of the apparent feature union. "
                        "This can indicate missing compartments or source-specific feature blocks, which are a known risk for latent contrastive learning."
                    ),
                }
            )
        if np.isfinite(max_missing_fraction) and max_missing_fraction > 0.20:
            records.append(
                {
                    "severity": "warning",
                    "diagnostic": "source_linked_missing_feature_blocks",
                    "value": max_missing_fraction,
                    "message": (
                        "At least one dataset is missing more than 20% of the apparent feature union. "
                        "Check whether compartments/features are source-linked before interpreting the latent space."
                    ),
                }
            )
    if not records:
        records.append(
            {
                "severity": "info",
                "diagnostic": "feature_alignment_no_major_warning",
                "message": "No major shared-feature alignment warning was triggered.",
            }
        )
    return pd.DataFrame.from_records(records)

def make_latent_backend_policy_table(*, config: ClipnAdapterConfig) -> pd.DataFrame:
    """Return a one-row table explaining the latent backend selection policy."""
    backend = str(config.backend_module)
    is_native = backend.lower() in {"cpatk_contrastive", "native_contrastive"}
    return pd.DataFrame.from_records(
        [
            {
                "selected_backend_module": backend,
                "default_backend": "cpatk_contrastive",
                "published_clipn_run_only_if_requested": True,
                "is_cpatk_native_backend": bool(is_native),
                "policy": (
                    "CPATK uses its native supervised contrastive backend by default. "
                    "The published external CLIPn package is run only when --backend_module clipn "
                    "or another external module is explicitly requested."
                ),
            }
        ]
    )


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



def diagnose_training_curve(*, loss_table: pd.DataFrame, backend_module: str) -> pd.DataFrame:
    """Return warning rows for suspicious latent-backend training curves."""
    if loss_table is None or loss_table.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "backend_module": backend_module,
                    "severity": "info",
                    "diagnostic": "no_loss_curve",
                    "message": "No numeric training-loss curve was reported by the latent backend.",
                }
            ]
        )
    loss_columns = [column for column in ["loss", "train_loss", "monitor_loss"] if column in loss_table.columns]
    if not loss_columns:
        return pd.DataFrame.from_records(
            [
                {
                    "backend_module": backend_module,
                    "severity": "info",
                    "diagnostic": "no_standard_loss_column",
                    "message": "Training table was written, but no standard loss column was available for diagnostics.",
                }
            ]
        )
    records = []
    for column in loss_columns:
        values = pd.to_numeric(loss_table[column], errors="coerce").dropna()
        if values.empty:
            continue
        first_value = float(values.iloc[0])
        all_zero = bool(np.allclose(values.to_numpy(dtype=float), 0.0))
        if all_zero:
            records.append(
                {
                    "backend_module": backend_module,
                    "severity": "warning",
                    "diagnostic": f"{column}_all_zero",
                    "message": (
                        f"The reported {column} curve is zero from the first recorded epoch. "
                        "Treat this as a suspicious backend diagnostic rather than evidence of a well-trained latent model."
                    ),
                }
            )
        elif np.isclose(first_value, 0.0):
            records.append(
                {
                    "backend_module": backend_module,
                    "severity": "warning",
                    "diagnostic": f"{column}_starts_at_zero",
                    "message": (
                        f"The reported {column} starts at zero. Check whether the backend is reporting a meaningful loss."
                    ),
                }
            )
    return pd.DataFrame.from_records(records)



def _first_present_column(*, table: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column present in a table."""
    for column in candidates:
        if column in table.columns:
            return str(column)
    return None


def _safe_boolean_rate(values: pd.Series) -> float:
    """Return the mean of a boolean series, or NaN when unavailable."""
    if values is None or values.empty:
        return float("nan")
    return float(values.astype(bool).mean())


def calculate_compound_holdout_embedding_metrics(
    *,
    latent_table: pd.DataFrame,
    holdout_values: Sequence[str],
    holdout_column: str,
    label_column: str,
    distance_metric: str = "cosine",
    threads: int = 1,
    repeat_index: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a latent table from a whole-compound holdout fit.

    Whole-compound holdout asks a stricter question than ordinary row-level
    validation. The held-out compounds are excluded from supervised training,
    then encoded by the trained model. Because those compound labels are absent
    from the training set, same-compound retrieval against the training set is
    impossible. Instead this function reports whether held-out replicates still
    cohere with one another, and whether held-out profiles retrieve similar
    labels/classes from the training set when such labels are available.
    """
    require_sklearn_stack(purpose="compound-holdout latent validation")
    latent_columns = [column for column in latent_table.columns if str(column).startswith("latent_")]
    if not latent_columns:
        return (
            pd.DataFrame.from_records(
                [
                    {
                        "repeat": int(repeat_index),
                        "metric": "compound_holdout_status",
                        "value": np.nan,
                        "message": "No latent columns were available.",
                    }
                ]
            ),
            pd.DataFrame(),
        )
    if holdout_column not in latent_table.columns:
        return (
            pd.DataFrame.from_records(
                [
                    {
                        "repeat": int(repeat_index),
                        "metric": "compound_holdout_status",
                        "value": np.nan,
                        "message": f"Holdout column is missing from latent table: {holdout_column}",
                    }
                ]
            ),
            pd.DataFrame(),
        )
    holdout_set = {str(value) for value in holdout_values}
    group_values = latent_table[holdout_column].fillna("missing").astype(str)
    holdout_mask = group_values.isin(holdout_set).to_numpy(dtype=bool)
    train_mask = ~holdout_mask
    batch_column = _first_present_column(
        table=latent_table,
        candidates=["Metadata_Plate", "Plate_Metadata", "synthetic_batch", "Batch", "Dataset"],
    )
    label_column = label_column if label_column in latent_table.columns else ""
    X = latent_table.loc[:, latent_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    records = [
        {"repeat": int(repeat_index), "metric": "n_holdout_groups", "value": float(len(holdout_set))},
        {"repeat": int(repeat_index), "metric": "n_holdout_rows", "value": float(holdout_mask.sum())},
        {"repeat": int(repeat_index), "metric": "n_training_rows", "value": float(train_mask.sum())},
    ]
    neighbour_rows = []
    if int(holdout_mask.sum()) >= 3:
        holdout_indices = np.where(holdout_mask)[0]
        nn_model = NearestNeighbors(
            n_neighbors=2,
            metric=distance_metric,
            n_jobs=max(1, int(threads)),
        )
        nn_model.fit(X.iloc[holdout_indices, :])
        distances, local_indices = nn_model.kneighbors(X.iloc[holdout_indices, :], return_distance=True)
        rows = []
        for query_pos, query_index in enumerate(holdout_indices):
            neighbour_index = int(holdout_indices[int(local_indices[query_pos, 1])])
            row = {
                "repeat": int(repeat_index),
                "comparison": "heldout_internal",
                "query_index": int(query_index),
                "neighbour_index": neighbour_index,
                "distance": float(distances[query_pos, 1]),
                "Query_holdout_group": latent_table.iloc[query_index].get(holdout_column),
                "Neighbour_holdout_group": latent_table.iloc[neighbour_index].get(holdout_column),
            }
            row["same_holdout_group"] = bool(str(row["Query_holdout_group"]) == str(row["Neighbour_holdout_group"]))
            if label_column:
                row["Query_label"] = latent_table.iloc[query_index].get(label_column)
                row["Neighbour_label"] = latent_table.iloc[neighbour_index].get(label_column)
                row["same_label"] = bool(str(row["Query_label"]) == str(row["Neighbour_label"]))
            if "Dataset" in latent_table.columns:
                row["Query_Dataset"] = latent_table.iloc[query_index].get("Dataset")
                row["Neighbour_Dataset"] = latent_table.iloc[neighbour_index].get("Dataset")
                row["same_dataset"] = bool(str(row["Query_Dataset"]) == str(row["Neighbour_Dataset"]))
            if batch_column:
                row["Query_batch"] = latent_table.iloc[query_index].get(batch_column)
                row["Neighbour_batch"] = latent_table.iloc[neighbour_index].get(batch_column)
                row["same_batch"] = bool(str(row["Query_batch"]) == str(row["Neighbour_batch"]))
            rows.append(row)
        internal = pd.DataFrame.from_records(rows)
        neighbour_rows.append(internal)
        records.extend(
            [
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_internal_top1_same_compound_rate",
                    "value": _safe_boolean_rate(internal.get("same_holdout_group", pd.Series(dtype=bool))),
                },
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_internal_top1_same_label_rate",
                    "value": _safe_boolean_rate(internal.get("same_label", pd.Series(dtype=bool))),
                },
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_internal_top1_same_dataset_rate",
                    "value": _safe_boolean_rate(internal.get("same_dataset", pd.Series(dtype=bool))),
                },
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_internal_top1_same_batch_rate",
                    "value": _safe_boolean_rate(internal.get("same_batch", pd.Series(dtype=bool))),
                },
            ]
        )
    else:
        records.append(
            {
                "repeat": int(repeat_index),
                "metric": "heldout_internal_status",
                "value": np.nan,
                "message": "Fewer than three held-out rows were available for internal nearest-neighbour validation.",
            }
        )
    if int(holdout_mask.sum()) >= 1 and int(train_mask.sum()) >= 2:
        train_indices = np.where(train_mask)[0]
        holdout_indices = np.where(holdout_mask)[0]
        nn_model = NearestNeighbors(
            n_neighbors=1,
            metric=distance_metric,
            n_jobs=max(1, int(threads)),
        )
        nn_model.fit(X.iloc[train_indices, :])
        distances, local_indices = nn_model.kneighbors(X.iloc[holdout_indices, :], return_distance=True)
        rows = []
        for query_pos, query_index in enumerate(holdout_indices):
            neighbour_index = int(train_indices[int(local_indices[query_pos, 0])])
            row = {
                "repeat": int(repeat_index),
                "comparison": "heldout_to_train",
                "query_index": int(query_index),
                "neighbour_index": neighbour_index,
                "distance": float(distances[query_pos, 0]),
                "Query_holdout_group": latent_table.iloc[query_index].get(holdout_column),
                "Neighbour_holdout_group": latent_table.iloc[neighbour_index].get(holdout_column),
                "same_holdout_group": False,
            }
            if label_column:
                row["Query_label"] = latent_table.iloc[query_index].get(label_column)
                row["Neighbour_label"] = latent_table.iloc[neighbour_index].get(label_column)
                row["same_label"] = bool(str(row["Query_label"]) == str(row["Neighbour_label"]))
            if "Dataset" in latent_table.columns:
                row["Query_Dataset"] = latent_table.iloc[query_index].get("Dataset")
                row["Neighbour_Dataset"] = latent_table.iloc[neighbour_index].get("Dataset")
                row["same_dataset"] = bool(str(row["Query_Dataset"]) == str(row["Neighbour_Dataset"]))
            if batch_column:
                row["Query_batch"] = latent_table.iloc[query_index].get(batch_column)
                row["Neighbour_batch"] = latent_table.iloc[neighbour_index].get(batch_column)
                row["same_batch"] = bool(str(row["Query_batch"]) == str(row["Neighbour_batch"]))
            rows.append(row)
        to_train = pd.DataFrame.from_records(rows)
        neighbour_rows.append(to_train)
        records.extend(
            [
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_to_train_top1_same_label_rate",
                    "value": _safe_boolean_rate(to_train.get("same_label", pd.Series(dtype=bool))),
                },
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_to_train_top1_same_dataset_rate",
                    "value": _safe_boolean_rate(to_train.get("same_dataset", pd.Series(dtype=bool))),
                },
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_to_train_top1_same_batch_rate",
                    "value": _safe_boolean_rate(to_train.get("same_batch", pd.Series(dtype=bool))),
                },
                {
                    "repeat": int(repeat_index),
                    "metric": "heldout_to_train_mean_distance",
                    "value": float(to_train["distance"].mean()) if "distance" in to_train.columns else np.nan,
                },
            ]
        )
    neighbours = pd.concat(neighbour_rows, ignore_index=True, sort=False) if neighbour_rows else pd.DataFrame()
    return pd.DataFrame.from_records(records), neighbours


def run_native_compound_holdout_validation(
    *,
    cleaned: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Run repeated whole-compound holdout validation for the native backend."""
    if str(config.backend_module).lower() not in {"cpatk_contrastive", "native_contrastive"}:
        return {
            "compound_holdout_validation_summary": pd.DataFrame.from_records(
                [
                    {
                        "repeat": 0,
                        "metric": "compound_holdout_status",
                        "value": np.nan,
                        "message": "Compound holdout validation is only implemented for cpatk_contrastive.",
                    }
                ]
            )
        }
    holdout_column = str(config.compound_holdout_column or config.native_positive_column or config.id_column)
    if holdout_column not in metadata.columns:
        return {
            "compound_holdout_validation_summary": pd.DataFrame.from_records(
                [
                    {
                        "repeat": 0,
                        "metric": "compound_holdout_status",
                        "value": np.nan,
                        "message": f"Holdout column missing from metadata: {holdout_column}",
                    }
                ]
            )
        }
    values = metadata[holdout_column].fillna("missing").astype(str)
    counts = values.value_counts(dropna=False)
    eligible = counts[counts >= int(max(2, config.compound_holdout_min_profiles))].index.astype(str).tolist()
    group_report = counts.rename_axis("holdout_group").reset_index(name="n_profiles")
    group_report["eligible_for_holdout"] = group_report["holdout_group"].astype(str).isin(set(eligible))
    if len(eligible) < 3:
        return {
            "compound_holdout_validation_group_report": group_report,
            "compound_holdout_validation_summary": pd.DataFrame.from_records(
                [
                    {
                        "repeat": 0,
                        "metric": "compound_holdout_status",
                        "value": np.nan,
                        "message": "Fewer than three eligible repeated compounds/groups were available for whole-group holdout.",
                    }
                ]
            ),
        }
    rng = np.random.default_rng(seed=int(config.compound_holdout_seed))
    fraction = min(max(float(config.compound_holdout_fraction), 0.01), 0.80)
    n_holdout = int(round(len(eligible) * fraction))
    n_holdout = min(max(1, n_holdout), len(eligible) - 2)
    summary_tables = []
    neighbour_tables = []
    training_tables = []
    split_tables = []
    selection_rows = []
    for repeat_index in range(1, int(max(1, config.compound_holdout_repeats)) + 1):
        heldout_values = sorted(rng.choice(np.asarray(eligible, dtype=object), size=n_holdout, replace=False).astype(str).tolist())
        if logger is not None:
            logger.info(
                "Running native compound holdout validation repeat %d/%d with %d held-out groups.",
                repeat_index,
                int(max(1, config.compound_holdout_repeats)),
                len(heldout_values),
            )
        native_config = _native_config_from_clipn_config(
            config=config,
            heldout_positive_values=heldout_values,
        )
        native_config.random_state = int(config.compound_holdout_seed) + int(repeat_index)
        result = fit_native_contrastive_backend(
            cleaned=cleaned,
            metadata=metadata,
            config=native_config,
            logger=logger,
        )
        summary, neighbours = calculate_compound_holdout_embedding_metrics(
            latent_table=result.latent_table,
            holdout_values=heldout_values,
            holdout_column=holdout_column,
            label_column=config.label_column,
            distance_metric=config.distance_metric,
            threads=int(config.n_threads),
            repeat_index=repeat_index,
        )
        summary.insert(1, "holdout_column", holdout_column)
        summary.insert(2, "n_eligible_holdout_groups", int(len(eligible)))
        summary_tables.append(summary)
        if not neighbours.empty:
            neighbour_tables.append(neighbours)
        train = result.training_summary.copy()
        train.insert(0, "repeat", int(repeat_index))
        training_tables.append(train)
        split = result.split_report.copy()
        split.insert(0, "repeat", int(repeat_index))
        split_tables.append(split)
        for value in heldout_values:
            selection_rows.append(
                {
                    "repeat": int(repeat_index),
                    "holdout_column": holdout_column,
                    "heldout_group": value,
                    "n_profiles": int(counts.get(value, 0)),
                }
            )
    outputs = {
        "compound_holdout_validation_group_report": group_report,
        "compound_holdout_validation_selection_report": pd.DataFrame.from_records(selection_rows),
        "compound_holdout_validation_summary": pd.concat(summary_tables, ignore_index=True, sort=False),
    }
    if neighbour_tables:
        outputs["compound_holdout_validation_neighbours"] = pd.concat(neighbour_tables, ignore_index=True, sort=False)
    if training_tables:
        outputs["compound_holdout_validation_training_summary"] = pd.concat(training_tables, ignore_index=True, sort=False)
    if split_tables:
        outputs["compound_holdout_validation_split_report"] = pd.concat(split_tables, ignore_index=True, sort=False)
    return outputs

def run_clipn_workflow(
    *,
    datasets: Mapping[str, pd.DataFrame],
    output_dir: Path,
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
    save_model_path: Optional[Path] = None,
) -> Mapping[str, pd.DataFrame]:
    """Run the full CPATK latent embedding workflow."""
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
    feature_alignment_warnings = diagnose_feature_alignment_quality(feature_summary=feature_summary)
    output_tables: dict[str, pd.DataFrame] = {
        "latent_backend_policy": make_latent_backend_policy_table(config=config),
        "clipn_status": status,
        "clipn_feature_summary": feature_summary,
        "clipn_feature_report": feature_report,
        "clipn_feature_alignment_warnings": feature_alignment_warnings,
        "clipn_preprocessing_summary": preprocessing_summary,
        "clipn_label_report": label_report,
    }
    latent_table = pd.DataFrame()
    model: object | None = None
    loss_table = pd.DataFrame()
    training_summary = pd.DataFrame()
    warnings = feature_alignment_warnings.loc[
        feature_alignment_warnings["severity"].eq("warning"), "message"
    ].astype(str).tolist()
    if bool(status["available"].iloc[0]):
        try:
            if str(config.backend_module).lower() in {"cpatk_contrastive", "native_contrastive"}:
                latent_table, model, loss_table, training_summary, extra_tables = fit_cpatk_contrastive_backend(
                    cleaned=cleaned,
                    metadata=metadata,
                    config=config,
                    logger=logger,
                )
                output_tables.update(extra_tables)
                run_status = pd.DataFrame(
                    [{"backend_run": "success", "message": "CPATK-native contrastive backend completed."}]
                )
            else:
                latent_table, model, loss_table, training_summary = fit_clipn_backend(
                    cleaned=cleaned,
                    labels=labels,
                    metadata=metadata,
                    config=config,
                    logger=logger,
                )
                run_status = pd.DataFrame(
                    [{"backend_run": "success", "message": "External CLIPn backend completed."}]
                )
            if save_model_path is not None and model is not None:
                save_model_pickle(model=model, path=save_model_path)
        except Exception as exc:
            if logger is not None:
                logger.exception("Latent backend execution failed.")
            warnings.append(f"Latent backend execution failed: {exc}")
            run_status = pd.DataFrame(
                [{"backend_run": "failed", "message": str(exc)}]
            )
    else:
        message = str(status["message"].iloc[0])
        warnings.append(f"Latent backend unavailable: {message}")
        run_status = pd.DataFrame(
            [{"backend_run": "not_run", "message": message}]
        )
    output_tables["clipn_run_status"] = run_status
    if training_summary.empty:
        training_summary = pd.DataFrame.from_records([
            {
                "training_policy": "not_run",
                "requested_max_epochs": int(config.epochs),
                "early_stopping_enabled": bool(config.early_stopping),
                "stopping_reason": str(run_status["backend_run"].iloc[0]) if "backend_run" in run_status.columns else "unknown",
            }
        ])
    output_tables["clipn_training_summary"] = training_summary
    training_diagnostics = diagnose_training_curve(loss_table=loss_table, backend_module=config.backend_module)
    if not training_diagnostics.empty:
        output_tables["clipn_training_diagnostics"] = training_diagnostics
        suspicious = training_diagnostics.loc[training_diagnostics["severity"].eq("warning"), "message"].astype(str).tolist()
        warnings.extend(suspicious)
    pca_fallback_used = False
    if latent_table.empty and config.allow_pca_fallback:
        latent_table, model, fallback_info = fit_pca_fallback(
            cleaned=cleaned,
            metadata=metadata,
            config=config,
        )
        pca_fallback_used = True
        output_tables["pca_fallback_explained_variance"] = fallback_info
        warnings.append("PCA fallback was used because CLIPn latent output was unavailable. This is not a CLIPn latent space.")
    output_tables["clipn_backend_provenance"] = collect_clipn_backend_provenance(
        backend_module=config.backend_module,
        backend_run=str(run_status["backend_run"].iloc[0]) if "backend_run" in run_status.columns else "unknown",
        pca_fallback_used=pca_fallback_used,
        loss_table=loss_table,
        training_summary=training_summary,
    )
    if (
        bool(config.run_compound_holdout_validation)
        and str(config.backend_module).lower() in {"cpatk_contrastive", "native_contrastive"}
        and str(run_status["backend_run"].iloc[0]) == "success"
    ):
        try:
            holdout_tables = run_native_compound_holdout_validation(
                cleaned=cleaned,
                metadata=metadata,
                config=config,
                logger=logger,
            )
            output_tables.update(holdout_tables)
            holdout_summary = holdout_tables.get("compound_holdout_validation_summary", pd.DataFrame())
            if not holdout_summary.empty:
                low_internal = holdout_summary.loc[
                    holdout_summary["metric"].eq("heldout_internal_top1_same_compound_rate")
                    & (pd.to_numeric(holdout_summary["value"], errors="coerce") < 0.25)
                ]
                high_leakage = holdout_summary.loc[
                    holdout_summary["metric"].isin(
                        [
                            "heldout_internal_top1_same_dataset_rate",
                            "heldout_to_train_top1_same_dataset_rate",
                        ]
                    )
                    & (pd.to_numeric(holdout_summary["value"], errors="coerce") > 0.80)
                ]
                if not low_internal.empty:
                    warnings.append(
                        "Whole-compound holdout validation found weak held-out replicate cohesion. "
                        "Treat the latent space cautiously for unseen compounds."
                    )
                if not high_leakage.empty:
                    warnings.append(
                        "Whole-compound holdout validation showed high same-dataset/source retrieval. "
                        "This suggests residual batch/source structure in held-out compounds."
                    )
        except Exception as exc:
            if logger is not None:
                logger.exception("Compound holdout validation failed.")
            warnings.append(f"Compound holdout validation failed: {exc}")
            output_tables["compound_holdout_validation_summary"] = pd.DataFrame.from_records(
                [{"repeat": 0, "metric": "compound_holdout_status", "value": np.nan, "message": str(exc)}]
            )
    if not latent_table.empty:
        output_tables["clipn_latent"] = latent_table
        if not loss_table.empty:
            output_tables["clipn_training_loss"] = loss_table
        latent_diagnostics = calculate_latent_diagnostics(latent_table=latent_table, config=config)
        output_tables.update(latent_diagnostics)
        latent_quality = diagnose_latent_space_quality(
            diagnostic_summary=latent_diagnostics.get("latent_diagnostic_summary", pd.DataFrame()),
            config=config,
            backend_module=config.backend_module,
        )
        output_tables["latent_quality_warnings"] = latent_quality
        warnings.extend(
            latent_quality.loc[latent_quality["severity"].eq("warning"), "message"].astype(str).tolist()
        )
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
        title=("CPATK native contrastive report" if str(config.backend_module).lower() in {"cpatk_contrastive", "native_contrastive"} else "CPATK CLIPn adapter report"),
        output_path=output_dir / "clipn_report.html",
        summary_tables=output_tables,
        plot_paths=plot_paths,
        narrative=(
            "This report summarises the optional latent embedding workflow. The adapter first harmonises "
            "features across datasets, cleans non-finite values, handles missing data, scales "
            "features, removes only empty all-zero rows/features as QC, audits literal zeros without changing them, and then runs the selected latent backend. If the backend "
            "is unavailable, the report records the reason and preserves all preprocessing audits. Single-dataset splits are software validation checks, not true multi-dataset integration benchmarks."
        ),
        warnings=warnings,
        methods_text=(
            "Latent embedding is treated as an optional integration layer. CPATK uses its native "
            "supervised contrastive backend by default and runs the published external CLIPn package "
            "only when that backend is explicitly requested. CPATK freezes the shared feature "
            "intersection before model fitting to prevent accidental feature-order drift between "
            "datasets and requires at least two non-empty datasets. Metadata and encoded labels are excluded from the input feature matrix. "
            "Latent-space diagnostics are descriptive checks of replicate, class and dataset "
            "structure; they should be interpreted alongside the non-AI CPATK workflow. CPATK reports backend provenance so that PCA fallback output is not mistaken for true latent-backend biology."
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
    # performing the selected backend fit directly.  Errors are converted to a status table.
    try:
        metadata = make_metadata_table(datasets=datasets, config=config)
        cleaned, preprocessing = clean_impute_and_scale_aligned(
            aligned=aligned,
            metadata=datasets,
            config=config,
            logger=logger,
        )
        labels, _, label_report = encode_labels_for_clipn(datasets=datasets, config=config)
        if str(config.backend_module).lower() in {"cpatk_contrastive", "native_contrastive"}:
            latent, _, loss_table, training_summary, extra_tables = fit_cpatk_contrastive_backend(
                cleaned=cleaned,
                metadata=metadata,
                config=config,
                logger=logger,
            )
        else:
            latent, _, loss_table, training_summary = fit_clipn_backend(
                cleaned=cleaned,
                labels=labels,
                metadata=metadata,
                config=config,
                logger=logger,
            )
            extra_tables = {}
        result = {
            "clipn_status": status,
            "clipn_feature_summary": feature_summary,
            "clipn_feature_report": feature_report,
            "clipn_preprocessing_summary": preprocessing,
            "clipn_label_report": label_report,
            "clipn_training_loss": loss_table,
            "clipn_training_summary": training_summary,
            "clipn_latent": latent,
        }
        result.update(extra_tables)
        return result
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
