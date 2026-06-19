"""Optional CLIPn adapter for CPATK.

The adapter is intentionally defensive. CLIPn installations and local wrappers
vary across projects, so CPATK exposes a small compatibility layer that can:

* check whether CLIPn or a project-specific backend is importable;
* align named datasets on shared features;
* call common backend method names when they exist;
* load and save adapter configuration/provenance;
* preserve metadata and return tidy latent tables.

If CLIPn is not installed, classical CPATK workflows remain fully usable.
"""

from __future__ import annotations

import importlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


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
    metadata_columns: Optional[list[str]] = None
    feature_columns: Optional[list[str]] = None


def check_clipn_backend(*, backend_module: str = "clipn") -> pd.DataFrame:
    """Check whether a CLIPn backend module can be imported.

    Parameters
    ----------
    backend_module:
        Backend module name.

    Returns
    -------
    pandas.DataFrame
        Backend status table.
    """
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


def align_dataset_features(
    *,
    datasets: Mapping[str, pd.DataFrame],
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Align multiple datasets to a shared numeric feature set.

    Parameters
    ----------
    datasets:
        Named data frames.
    feature_columns:
        Optional candidate features. If omitted, numeric shared columns are used.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], pandas.DataFrame]
        Aligned feature matrices and feature summary.
    """
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if feature_columns is None:
        shared = None
        for data_frame in datasets.values():
            numeric = set(data_frame.select_dtypes(include="number").columns.tolist())
            shared = numeric if shared is None else shared & numeric
        features = sorted(shared or [])
    else:
        requested = list(feature_columns)
        features = [
            column for column in requested
            if all(column in data_frame.columns for data_frame in datasets.values())
        ]
    if not features:
        raise ValueError("No shared numeric feature columns were found across datasets.")
    aligned = {
        name: data_frame.loc[:, features].apply(pd.to_numeric, errors="coerce").copy()
        for name, data_frame in datasets.items()
    }
    summary = pd.DataFrame.from_records(
        [
            {
                "dataset": name,
                "n_rows": int(data_frame.shape[0]),
                "n_original_columns": int(datasets[name].shape[1]),
                "n_shared_features": int(len(features)),
            }
            for name, data_frame in aligned.items()
        ]
    )
    return aligned, summary


def save_clipn_config(*, config: ClipnAdapterConfig, path: Path) -> Path:
    """Save CLIPn adapter configuration as JSON.

    Parameters
    ----------
    config:
        Adapter configuration.
    path:
        Output JSON path.

    Returns
    -------
    pathlib.Path
        Written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data=json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def load_clipn_config(*, path: Path) -> ClipnAdapterConfig:
    """Load CLIPn adapter configuration from JSON.

    Parameters
    ----------
    path:
        Input JSON path.

    Returns
    -------
    ClipnAdapterConfig
        Configuration object.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClipnAdapterConfig(**data)


def save_model_pickle(*, model: object, path: Path) -> Path:
    """Save a Python model object using pickle.

    Parameters
    ----------
    model:
        Model object.
    path:
        Output pickle path.

    Returns
    -------
    pathlib.Path
        Written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="wb") as handle:
        pickle.dump(obj=model, file=handle)
    return path


def load_model_pickle(*, path: Path) -> object:
    """Load a Python model object using pickle.

    Parameters
    ----------
    path:
        Pickle path.

    Returns
    -------
    object
        Loaded model.
    """
    with Path(path).open(mode="rb") as handle:
        return pickle.load(file=handle)


def _get_backend_model(*, config: ClipnAdapterConfig) -> object:
    """Instantiate a configured backend model when possible."""
    module = importlib.import_module(name=config.backend_module)
    if config.model_class is None:
        if hasattr(module, "CLIPn"):
            return getattr(module, "CLIPn")()
        if hasattr(module, "Clipn"):
            return getattr(module, "Clipn")()
        raise ValueError(
            "No model_class was provided and common CLIPn class names were not found."
        )
    model_class = getattr(module, config.model_class)
    return model_class()


def run_clipn_adapter(
    *,
    datasets: Mapping[str, pd.DataFrame],
    config: ClipnAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> Mapping[str, pd.DataFrame]:
    """Run a best-effort CLIPn backend adapter.

    Parameters
    ----------
    datasets:
        Named input data frames.
    config:
        Adapter configuration.
    logger:
        Optional logger.

    Returns
    -------
    mapping[str, pandas.DataFrame]
        Status, feature summary and latent outputs when available.
    """
    status = check_clipn_backend(backend_module=config.backend_module)
    aligned, feature_summary = align_dataset_features(
        datasets=datasets,
        feature_columns=config.feature_columns,
    )
    if not bool(status["available"].iloc[0]):
        return {"clipn_status": status, "clipn_feature_summary": feature_summary}
    try:
        model = _get_backend_model(config=config)
        fit_method = getattr(model, config.fit_method)
        fit_method(aligned)
        method_name = config.transform_method or config.predict_method
        transform_method = getattr(model, method_name)
        latent_tables = []
        for dataset_name, feature_table in aligned.items():
            latent = transform_method(feature_table)
            latent_frame = pd.DataFrame(latent)
            latent_frame.insert(0, "Dataset", dataset_name)
            latent_frame.insert(1, "row_index", feature_table.index.to_numpy())
            latent_tables.append(latent_frame)
        latent_table = pd.concat(latent_tables, ignore_index=True)
        return {
            "clipn_status": status,
            "clipn_feature_summary": feature_summary,
            "clipn_latent": latent_table,
        }
    except Exception as exc:
        if logger is not None:
            logger.warning("CLIPn adapter failed: %s", exc)
        failure = status.copy()
        failure["available"] = bool(status["available"].iloc[0])
        failure["message"] = f"Backend import succeeded but adapter execution failed: {exc}"
        return {"clipn_status": failure, "clipn_feature_summary": feature_summary}
