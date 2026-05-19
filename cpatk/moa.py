"""Mechanism-of-action classification helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


def calculate_class_centroids(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    min_class_size: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate class centroids for known mechanisms of action.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    labels:
        Class labels aligned to features.
    min_class_size:
        Minimum number of profiles required per class.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Centroid matrix and class-size summary.
    """
    table = features.copy()
    table["__label__"] = labels.astype(str).to_numpy()
    counts = table.groupby("__label__", dropna=False).size().reset_index(name="n_profiles")
    valid_labels = counts.loc[counts["n_profiles"] >= min_class_size, "__label__"].tolist()
    centroids = table.loc[table["__label__"].isin(valid_labels)].groupby("__label__").median()
    centroids.index.name = "class_label"
    return centroids, counts.rename(columns={"__label__": "class_label"})


def score_profiles_against_centroids(
    *,
    query_features: pd.DataFrame,
    centroids: pd.DataFrame,
    metric: str = "cosine",
    top_n: int = 5,
) -> pd.DataFrame:
    """Score query profiles against class centroids.

    Parameters
    ----------
    query_features:
        Query feature matrix.
    centroids:
        Class centroid matrix.
    metric:
        Pairwise distance metric.
    top_n:
        Number of best classes to return per profile.

    Returns
    -------
    pandas.DataFrame
        Long-format centroid scoring table.
    """
    shared_columns = [column for column in query_features.columns if column in centroids.columns]
    if not shared_columns:
        raise ValueError("No shared feature columns between query features and centroids.")
    distances = pairwise_distances(
        X=query_features.loc[:, shared_columns],
        Y=centroids.loc[:, shared_columns],
        metric=metric,
    )
    records = []
    labels = centroids.index.astype(str).tolist()
    for row_index, query_index in enumerate(query_features.index):
        order = np.argsort(distances[row_index, :])
        for rank, class_position in enumerate(order[:top_n], start=1):
            records.append(
                {
                    "query_index": query_index,
                    "rank": rank,
                    "predicted_class": labels[class_position],
                    "distance": float(distances[row_index, class_position]),
                    "similarity": float(1 / (1 + distances[row_index, class_position])),
                }
            )
    return pd.DataFrame.from_records(records)


def classify_by_knn(
    *,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    query_features: pd.DataFrame,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> pd.DataFrame:
    """Classify query profiles using K-nearest neighbours.

    Parameters
    ----------
    train_features:
        Training feature matrix.
    train_labels:
        Known labels aligned to training features.
    query_features:
        Query feature matrix.
    n_neighbors:
        Number of neighbours.
    weights:
        KNN weighting strategy.

    Returns
    -------
    pandas.DataFrame
        Predicted labels and maximum probability.
    """
    shared_columns = [column for column in train_features.columns if column in query_features.columns]
    if not shared_columns:
        raise ValueError("No shared feature columns between training and query features.")
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X=train_features.loc[:, shared_columns], y=train_labels.astype(str))
    predicted = model.predict(X=query_features.loc[:, shared_columns])
    probabilities = model.predict_proba(X=query_features.loc[:, shared_columns])
    return pd.DataFrame(
        {
            "query_index": query_features.index,
            "predicted_class": predicted,
            "max_probability": probabilities.max(axis=1),
        }
    )


def summarise_moa_predictions(
    *,
    predictions: pd.DataFrame,
    group_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Summarise predicted mechanisms of action.

    Parameters
    ----------
    predictions:
        Prediction table containing ``predicted_class``.
    group_columns:
        Optional columns to group by.

    Returns
    -------
    pandas.DataFrame
        Prediction count summary.
    """
    if "predicted_class" not in predictions.columns:
        raise ValueError("Prediction table must contain predicted_class.")
    group_columns = [column for column in (group_columns or []) if column in predictions.columns]
    grouped = group_columns + ["predicted_class"]
    counts = predictions.groupby(grouped, dropna=False).size().reset_index(name="n_profiles")
    totals = counts.groupby(group_columns, dropna=False)["n_profiles"].transform("sum") if group_columns else counts["n_profiles"].sum()
    counts["fraction"] = counts["n_profiles"] / totals
    return counts
