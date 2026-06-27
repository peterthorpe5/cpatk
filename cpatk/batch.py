"""Batch-effect, domain-shift and unwanted-structure diagnostics."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cpatk.embedding import run_pca


def calculate_batch_centroid_distances(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_column: str,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Calculate distances between batch centroids.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    metadata:
        Metadata aligned to features.
    batch_column:
        Batch/domain label column.
    metric:
        Distance metric used by ``scipy.spatial.distance.cdist``.

    Returns
    -------
    pandas.DataFrame
        Long-format centroid distance table.
    """
    if batch_column not in metadata.columns:
        raise ValueError(f"Batch column is missing: {batch_column}")
    labels = metadata[batch_column].astype(str).reset_index(drop=True)
    table = features.reset_index(drop=True).copy()
    table["__batch__"] = labels
    centroids = table.groupby("__batch__", dropna=False).median()
    distances = cdist(XA=centroids, XB=centroids, metric=metric)
    records = []
    batch_labels = centroids.index.astype(str).tolist()
    for first_index, first_label in enumerate(batch_labels):
        for second_index, second_label in enumerate(batch_labels):
            records.append(
                {
                    "batch_1": first_label,
                    "batch_2": second_label,
                    "distance": float(distances[first_index, second_index]),
                    "metric": metric,
                }
            )
    return pd.DataFrame.from_records(records)


def calculate_metadata_association_with_pcs(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    columns_to_test: Sequence[str],
    n_components: int = 5,
) -> pd.DataFrame:
    """Quantify one-variable-at-a-time association of metadata with PCs.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    metadata:
        Metadata aligned to features.
    columns_to_test:
        Metadata columns to test descriptively.
    n_components:
        Number of principal components.

    Returns
    -------
    pandas.DataFrame
        Association table using eta-squared style variance ratios.
    """
    n_components = max(1, min(n_components, features.shape[0], features.shape[1]))
    scores, explained = run_pca(features=features, n_components=n_components)
    records = []
    aligned_metadata = metadata.reset_index(drop=True)
    for metadata_column in columns_to_test:
        if metadata_column not in aligned_metadata.columns:
            continue
        labels = aligned_metadata[metadata_column].astype(str)
        for pc_column in scores.columns:
            values = scores[pc_column].to_numpy(dtype=float)
            grand_mean = np.nanmean(values)
            total_ss = np.nansum((values - grand_mean) ** 2)
            between_ss = 0.0
            for _, index_values in labels.groupby(labels, dropna=False).groups.items():
                group_values = values[list(index_values)]
                between_ss += len(group_values) * (np.nanmean(group_values) - grand_mean) ** 2
            eta_squared = between_ss / total_ss if total_ss > 0 else np.nan
            records.append(
                {
                    "metadata_column": metadata_column,
                    "component": pc_column,
                    "eta_squared": float(eta_squared),
                    "explained_variance_ratio": float(
                        explained.loc[explained["component"] == pc_column, "explained_variance_ratio"].iloc[0]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def cross_validated_batch_prediction(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_column: str,
    n_splits: int = 5,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Estimate whether batch/domain labels are predictable from profiles.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    metadata:
        Metadata aligned to features.
    batch_column:
        Batch/domain label column.
    n_splits:
        Maximum cross-validation folds.
    random_state:
        Random seed.
    logger:
        Optional logger.

    Returns
    -------
    pandas.DataFrame
        Predictability summary.
    """
    if batch_column not in metadata.columns:
        raise ValueError(f"Batch column is missing: {batch_column}")
    labels = metadata[batch_column].astype(str).reset_index(drop=True)
    counts = labels.value_counts()
    if counts.shape[0] < 2 or counts.min() < 2:
        return pd.DataFrame.from_records(
            [
                {
                    "batch_column": batch_column,
                    "status": "not_tested",
                    "reason": "At least two classes with at least two profiles each are required.",
                }
            ]
        )
    folds = min(n_splits, int(counts.min()))
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    predicted = cross_val_predict(estimator=model, X=features, y=labels, cv=cv)
    accuracy = accuracy_score(y_true=labels, y_pred=predicted)
    balanced = balanced_accuracy_score(y_true=labels, y_pred=predicted)
    if logger is not None:
        logger.info("Batch prediction balanced accuracy for %s: %.3f", batch_column, balanced)
    return pd.DataFrame.from_records(
        [
            {
                "batch_column": batch_column,
                "status": "tested",
                "n_profiles": int(features.shape[0]),
                "n_classes": int(counts.shape[0]),
                "n_splits": int(folds),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced),
                "interpretation": (
                    "High cross-validated predictability suggests profiles contain strong batch/domain structure. "
                    "This is diagnostic rather than proof of a technical artefact."
                ),
            }
        ]
    )
