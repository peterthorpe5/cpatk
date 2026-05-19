"""Preprocessing workflows for generic Cell Painting data."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from cpatk.features import split_metadata_and_features
from cpatk.qc import (
    calculate_feature_qc,
    calculate_sample_qc,
    flag_samples_by_qc,
    select_features_by_qc,
)


def impute_features(
    *,
    features: pd.DataFrame,
    method: str = "median",
    n_neighbors: int = 5,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Impute missing values in a feature matrix.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    method:
        Imputation method: ``median``, ``mean``, ``zero`` or ``knn``.
    n_neighbors:
        Number of neighbours for KNN imputation.
    logger:
        Optional logger.

    Returns
    -------
    pandas.DataFrame
        Imputed feature matrix.
    """
    method = method.lower()
    if logger is not None:
        logger.info("Imputing features using method=%s", method)

    if method in {"median", "mean"}:
        imputer = SimpleImputer(strategy=method)
    elif method == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    values = imputer.fit_transform(X=features)
    return pd.DataFrame(data=values, columns=features.columns, index=features.index)


def scale_features(
    *,
    features: pd.DataFrame,
    method: str = "robust",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Scale a feature matrix.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    method:
        Scaling method: ``robust``, ``standard``, ``minmax`` or ``none``.
    logger:
        Optional logger.

    Returns
    -------
    pandas.DataFrame
        Scaled feature matrix.
    """
    method = method.lower()
    if logger is not None:
        logger.info("Scaling features using method=%s", method)

    if method == "none":
        return features.copy()
    if method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    values = scaler.fit_transform(X=features)
    return pd.DataFrame(data=values, columns=features.columns, index=features.index)


def remove_correlated_features(
    *,
    features: pd.DataFrame,
    max_absolute_correlation: float = 0.95,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove one feature from each highly correlated pair.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    max_absolute_correlation:
        Maximum allowed absolute pairwise correlation.
    logger:
        Optional logger.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Filtered feature matrix and table of removed features.
    """
    if features.shape[1] <= 1:
        return features.copy(), pd.DataFrame(columns=["removed_feature", "correlated_with", "correlation"])

    correlation = features.corr(method="pearson").abs()
    upper_mask = np.triu(np.ones(shape=correlation.shape), k=1).astype(bool)
    upper = correlation.where(cond=upper_mask)
    removed_records = []
    to_remove = set()
    for column in upper.columns:
        if column in to_remove:
            continue
        high = upper.index[upper[column] > max_absolute_correlation].tolist()
        for other in high:
            if column not in to_remove:
                to_remove.add(column)
                removed_records.append(
                    {
                        "removed_feature": column,
                        "correlated_with": other,
                        "correlation": float(upper.loc[other, column]),
                    }
                )
                break
    if logger is not None:
        logger.info("Removed %s highly correlated features", len(to_remove))
    retained = [column for column in features.columns if column not in to_remove]
    return features.loc[:, retained].copy(), pd.DataFrame.from_records(removed_records)


def aggregate_profiles(
    *,
    data_frame: pd.DataFrame,
    group_columns: Sequence[str],
    feature_columns: Sequence[str],
    statistic: str = "median",
) -> pd.DataFrame:
    """Aggregate object-level measurements to profile-level summaries.

    Parameters
    ----------
    data_frame:
        Object-level or single-cell table.
    group_columns:
        Columns defining profiles, for example plate/well/treatment.
    feature_columns:
        Numeric feature columns to aggregate.
    statistic:
        Summary statistic: ``median`` or ``mean``.

    Returns
    -------
    pandas.DataFrame
        Aggregated profile table.
    """
    valid_groups = [column for column in group_columns if column in data_frame.columns]
    valid_features = [column for column in feature_columns if column in data_frame.columns]
    if not valid_groups:
        raise ValueError("At least one valid group column is required.")
    if not valid_features:
        raise ValueError("At least one valid feature column is required.")
    if statistic == "median":
        aggregated = data_frame.groupby(valid_groups, dropna=False)[valid_features].median().reset_index()
    elif statistic == "mean":
        aggregated = data_frame.groupby(valid_groups, dropna=False)[valid_features].mean().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation statistic: {statistic}")
    counts = data_frame.groupby(valid_groups, dropna=False).size().reset_index(name="n_objects")
    return counts.merge(right=aggregated, on=valid_groups, how="left")


def preprocess_profiles(
    *,
    data_frame: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    additional_metadata_columns: Optional[Sequence[str]] = None,
    max_feature_missing_fraction: float = 0.2,
    max_sample_missing_fraction: float = 0.5,
    min_feature_variance: float = 1e-12,
    min_unique_values: int = 2,
    remove_correlated: bool = True,
    max_absolute_correlation: float = 0.95,
    imputation_method: str = "median",
    scaling_method: str = "robust",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """Run a generic Cell Painting preprocessing workflow.

    Parameters
    ----------
    data_frame:
        Input profile or object-level table.
    metadata_columns:
        Optional explicit metadata columns.
    feature_columns:
        Optional explicit feature columns.
    additional_metadata_columns:
        Additional metadata columns used during automatic inference.
    max_feature_missing_fraction:
        Maximum feature missingness.
    max_sample_missing_fraction:
        Maximum sample missingness.
    min_feature_variance:
        Minimum feature variance.
    min_unique_values:
        Minimum number of unique values per feature.
    remove_correlated:
        Whether to drop highly correlated features.
    max_absolute_correlation:
        Correlation threshold for feature filtering.
    imputation_method:
        Feature imputation method.
    scaling_method:
        Feature scaling method.
    logger:
        Optional logger.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Preprocessed table and QC/report tables.
    """
    if logger is not None:
        logger.info("Starting CPATK preprocessing workflow")
    metadata, features, metadata_names, feature_names = split_metadata_and_features(
        data_frame=data_frame,
        metadata_columns=metadata_columns,
        feature_columns=feature_columns,
        additional_metadata_columns=additional_metadata_columns,
    )
    if logger is not None:
        logger.info("Detected %s metadata columns", len(metadata_names))
        logger.info("Detected %s feature columns", len(feature_names))

    feature_qc = calculate_feature_qc(features=features)
    selected_features, feature_qc = select_features_by_qc(
        feature_qc=feature_qc,
        max_missing_fraction=max_feature_missing_fraction,
        min_variance=min_feature_variance,
        min_unique_values=min_unique_values,
    )
    sample_qc = calculate_sample_qc(features=features.loc[:, selected_features], metadata=metadata)
    sample_qc = flag_samples_by_qc(
        sample_qc=sample_qc,
        max_missing_fraction=max_sample_missing_fraction,
    )
    passed_rows = sample_qc["sample_qc_pass"].to_numpy(dtype=bool)
    selected_metadata = metadata.loc[passed_rows, :].reset_index(drop=True)
    selected_feature_matrix = features.loc[passed_rows, selected_features].reset_index(drop=True)

    imputed = impute_features(
        features=selected_feature_matrix,
        method=imputation_method,
        logger=logger,
    )
    scaled = scale_features(features=imputed, method=scaling_method, logger=logger)

    if remove_correlated:
        scaled, correlation_report = remove_correlated_features(
            features=scaled,
            max_absolute_correlation=max_absolute_correlation,
            logger=logger,
        )
    else:
        correlation_report = pd.DataFrame(columns=["removed_feature", "correlated_with", "correlation"])

    preprocessed = pd.concat([selected_metadata.reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
    retained_features = pd.DataFrame({"feature": scaled.columns.tolist()})
    summary = pd.DataFrame.from_records(
        [
            {"item": "n_rows_input", "value": int(data_frame.shape[0])},
            {"item": "n_rows_passing_qc", "value": int(preprocessed.shape[0])},
            {"item": "n_features_input", "value": int(len(feature_names))},
            {"item": "n_features_after_qc", "value": int(len(selected_features))},
            {"item": "n_features_after_correlation_filter", "value": int(len(scaled.columns))},
            {"item": "imputation_method", "value": imputation_method},
            {"item": "scaling_method", "value": scaling_method},
        ]
    )
    return {
        "preprocessed": preprocessed,
        "feature_qc": feature_qc,
        "sample_qc": sample_qc,
        "correlation_filter_report": correlation_report,
        "retained_features": retained_features,
        "preprocessing_summary": summary,
    }
