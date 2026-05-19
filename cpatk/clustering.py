"""Clustering helpers for Cell Painting profile tables."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(
    *,
    features: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run K-means clustering.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_clusters:
        Number of clusters.
    random_state:
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Cluster assignment table.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X=features)
    return pd.DataFrame({"row_index": features.index, "cluster": labels})


def run_agglomerative(
    *,
    features: pd.DataFrame,
    n_clusters: int = 8,
    linkage: str = "ward",
) -> pd.DataFrame:
    """Run agglomerative clustering.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    n_clusters:
        Number of clusters.
    linkage:
        Linkage strategy.

    Returns
    -------
    pandas.DataFrame
        Cluster assignment table.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X=features)
    return pd.DataFrame({"row_index": features.index, "cluster": labels})


def run_dbscan(
    *,
    features: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Run DBSCAN clustering.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    eps:
        DBSCAN radius parameter.
    min_samples:
        Minimum samples per dense region.
    metric:
        Distance metric.

    Returns
    -------
    pandas.DataFrame
        Cluster assignment table.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X=features)
    return pd.DataFrame({"row_index": features.index, "cluster": labels})


def calculate_silhouette_summary(
    *,
    features: pd.DataFrame,
    clusters: pd.Series,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Calculate a silhouette summary for clustering.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    clusters:
        Cluster labels.
    metric:
        Distance metric.

    Returns
    -------
    pandas.DataFrame
        Single-row summary table.
    """
    labels = pd.Series(clusters).to_numpy()
    valid_labels = set(labels)
    if len(valid_labels) < 2 or len(valid_labels) >= features.shape[0]:
        score = np.nan
    else:
        score = float(silhouette_score(X=features, labels=labels, metric=metric))
    return pd.DataFrame.from_records(
        [
            {
                "n_clusters_observed": int(len(valid_labels)),
                "n_profiles": int(features.shape[0]),
                "silhouette_score": score,
            }
        ]
    )


def summarise_clusters(
    *,
    metadata: pd.DataFrame,
    clusters: pd.Series,
    group_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Summarise cluster composition.

    Parameters
    ----------
    metadata:
        Metadata table aligned to cluster labels.
    clusters:
        Cluster labels.
    group_columns:
        Metadata columns to tabulate within clusters.

    Returns
    -------
    pandas.DataFrame
        Cluster composition summary.
    """
    table = metadata.copy()
    table["cluster"] = list(clusters)
    group_columns = [column for column in (group_columns or []) if column in table.columns]
    if not group_columns:
        return table.groupby("cluster", dropna=False).size().reset_index(name="n_profiles")
    return table.groupby(["cluster", *group_columns], dropna=False).size().reset_index(name="n_profiles")
