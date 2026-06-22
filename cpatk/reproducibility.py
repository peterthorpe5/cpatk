"""Replicate, neighbour and cluster-stability diagnostics for CPATK.

These helpers are intended to make unsupervised Cell Painting analyses more
honest. A UMAP or cluster plot can look convincing even when the structure is
unstable. CPATK therefore combines replicate correlations, neighbour stability,
consensus clustering, bootstrap ARI and a feature-permutation null model.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors




def _kmeans_fit_predict(
    *,
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    n_init: int = 1,
) -> np.ndarray:
    """Run K-means with a one-thread native-library guard when available.

    Repeated small K-means calls in test and CI environments can interact badly
    with optional BLAS/OpenMP thread pools. Limiting native threads keeps the
    workflow deterministic and prevents full unittest discovery from hanging in
    constrained environments.
    """
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        with threadpool_limits(limits=1):
            return KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                algorithm="lloyd",
            ).fit_predict(X=features)
    except Exception:
        return KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            algorithm="lloyd",
        ).fit_predict(X=features)


def _validate_non_empty_numeric_features(*, features: pd.DataFrame) -> pd.DataFrame:
    """Return a finite numeric feature matrix or raise an informative error."""
    if features.empty:
        raise ValueError("Feature matrix is empty.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Feature matrix contains missing/non-numeric values. Run CPATK preprocessing first.")
    if numeric.shape[0] < 2 or numeric.shape[1] < 1:
        raise ValueError("At least two profiles and one feature are required.")
    return numeric


def _validate_cluster_parameters(*, features: pd.DataFrame, n_clusters: int) -> None:
    """Validate cluster count for profile-level clustering."""
    if n_clusters < 2 or n_clusters >= features.shape[0]:
        raise ValueError("n_clusters must be at least 2 and smaller than n_profiles.")


def _validate_iterations(*, value: int, name: str) -> None:
    """Validate a positive iteration count."""
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1.")


def calculate_replicate_correlations(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    replicate_group_columns: Sequence[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Calculate pairwise replicate profile correlations within groups."""
    valid_groups = [column for column in replicate_group_columns if column in metadata.columns]
    if not valid_groups:
        raise ValueError("At least one valid replicate-group column is required.")
    aligned_metadata = metadata.reset_index(drop=True)
    aligned_features = features.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    records = []
    for group_key, index_values in aligned_metadata.groupby(valid_groups, dropna=False).groups.items():
        indices = list(index_values)
        if len(indices) < 2:
            continue
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        for first_index, second_index in combinations(indices, 2):
            first = aligned_features.iloc[first_index, :]
            second = aligned_features.iloc[second_index, :]
            valid = first.notna() & second.notna()
            correlation = first.loc[valid].corr(other=second.loc[valid], method=method) if valid.sum() > 1 else np.nan
            record = {
                "replicate_group": "|".join(str(item) for item in key_tuple),
                "row_index_1": int(first_index),
                "row_index_2": int(second_index),
                "correlation": float(correlation) if pd.notna(correlation) else np.nan,
                "n_features_compared": int(valid.sum()),
                "method": method,
            }
            for column, value in zip(valid_groups, key_tuple):
                record[column] = value
            records.append(record)
    return pd.DataFrame.from_records(records)


def summarise_replicate_correlations(
    *,
    replicate_correlations: pd.DataFrame,
    group_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Summarise replicate correlations by group."""
    if replicate_correlations.empty:
        return pd.DataFrame(columns=["n_pairs", "median_correlation", "mean_correlation", "sd_correlation"])
    group_columns = [column for column in (group_columns or []) if column in replicate_correlations.columns]
    if not group_columns:
        values = pd.to_numeric(replicate_correlations["correlation"], errors="coerce")
        return pd.DataFrame.from_records(
            [
                {
                    "n_pairs": int(values.notna().sum()),
                    "median_correlation": float(values.median(skipna=True)),
                    "mean_correlation": float(values.mean(skipna=True)),
                    "sd_correlation": float(values.std(skipna=True)),
                }
            ]
        )
    return (
        replicate_correlations.groupby(group_columns, dropna=False)["correlation"]
        .agg(n_pairs="count", median_correlation="median", mean_correlation="mean", sd_correlation="std")
        .reset_index()
    )


def calculate_neighbour_sets(
    *,
    features: pd.DataFrame,
    n_neighbours: int = 10,
    metric: str = "cosine",
) -> list[set[int]]:
    """Calculate nearest-neighbour index sets for each profile."""
    numeric = _validate_non_empty_numeric_features(features=features)
    if numeric.shape[0] <= 1:
        return [set() for _ in range(numeric.shape[0])]
    n_neighbours = max(1, int(n_neighbours))
    n_fit = min(n_neighbours + 1, numeric.shape[0])
    model = NearestNeighbors(n_neighbors=n_fit, metric=metric)
    model.fit(X=numeric)
    indices = model.kneighbors(X=numeric, return_distance=False)
    neighbour_sets = []
    for row_index, row_indices in enumerate(indices):
        values = [int(item) for item in row_indices if int(item) != row_index]
        neighbour_sets.append(set(values[:n_neighbours]))
    return neighbour_sets


def bootstrap_neighbour_stability(
    *,
    features: pd.DataFrame,
    n_neighbours: int = 10,
    n_bootstraps: int = 50,
    feature_fraction: float = 0.8,
    metric: str = "cosine",
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Estimate nearest-neighbour stability under feature subsampling."""
    numeric = _validate_non_empty_numeric_features(features=features)
    _validate_iterations(value=n_bootstraps, name="n_bootstraps")
    if not 0 < feature_fraction <= 1:
        raise ValueError("feature_fraction must be in the interval (0, 1].")
    rng = np.random.default_rng(seed=random_state)
    base_sets = calculate_neighbour_sets(features=numeric, n_neighbours=n_neighbours, metric=metric)
    n_features = numeric.shape[1]
    n_selected = max(1, int(round(n_features * feature_fraction)))
    scores = [[] for _ in range(numeric.shape[0])]
    for iteration in range(n_bootstraps):
        sampled = rng.choice(n_features, size=n_selected, replace=False)
        subset = numeric.iloc[:, sampled]
        boot_sets = calculate_neighbour_sets(features=subset, n_neighbours=n_neighbours, metric=metric)
        for row_index, base_set in enumerate(base_sets):
            comparison_set = boot_sets[row_index]
            union = base_set | comparison_set
            jaccard = 1.0 if not union else len(base_set & comparison_set) / len(union)
            scores[row_index].append(float(jaccard))
        if logger is not None and (iteration + 1) % max(1, n_bootstraps // 5) == 0:
            logger.info("Completed neighbour-stability feature subsample %s/%s", iteration + 1, n_bootstraps)
    records = []
    for row_index, row_scores in enumerate(scores):
        records.append(
            {
                "row_index": numeric.index[row_index],
                "mean_neighbour_jaccard": float(np.mean(row_scores)),
                "median_neighbour_jaccard": float(np.median(row_scores)),
                "sd_neighbour_jaccard": float(np.std(row_scores, ddof=1)) if len(row_scores) > 1 else 0.0,
                "n_bootstraps": int(n_bootstraps),
                "feature_fraction": float(feature_fraction),
                "metric": metric,
            }
        )
    return pd.DataFrame.from_records(records)


def consensus_clustering(
    *,
    features: pd.DataFrame,
    n_clusters: int = 8,
    n_bootstraps: int = 50,
    sample_fraction: float = 0.8,
    random_state: int = 42,
    n_init: int = 1,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate a consensus co-clustering matrix by sample subsampling."""
    numeric = _validate_non_empty_numeric_features(features=features)
    _validate_cluster_parameters(features=numeric, n_clusters=n_clusters)
    _validate_iterations(value=n_bootstraps, name="n_bootstraps")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    rng = np.random.default_rng(seed=random_state)
    n_profiles = numeric.shape[0]
    co_cluster = np.zeros((n_profiles, n_profiles), dtype=float)
    co_observed = np.zeros((n_profiles, n_profiles), dtype=float)
    sample_size = max(n_clusters + 1, int(round(n_profiles * sample_fraction)))
    sample_size = min(sample_size, n_profiles)
    for iteration in range(n_bootstraps):
        sampled = np.sort(rng.choice(n_profiles, size=sample_size, replace=False))
        subset = numeric.iloc[sampled, :]
        labels = _kmeans_fit_predict(
            features=subset,
            n_clusters=n_clusters,
            random_state=random_state + iteration,
            n_init=n_init,
        )
        same_cluster = labels[:, None] == labels[None, :]
        co_observed[np.ix_(sampled, sampled)] += 1.0
        co_cluster[np.ix_(sampled, sampled)] += same_cluster.astype(float)
        if logger is not None and (iteration + 1) % max(1, n_bootstraps // 5) == 0:
            logger.info("Completed consensus bootstrap %s/%s", iteration + 1, n_bootstraps)
    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.divide(co_cluster, co_observed, out=np.zeros_like(co_cluster), where=co_observed > 0)
    matrix = pd.DataFrame(data=consensus, index=numeric.index, columns=numeric.index)
    upper = consensus[np.triu_indices(n_profiles, k=1)]
    summary = pd.DataFrame.from_records(
        [
            {
                "n_profiles": int(n_profiles),
                "n_clusters": int(n_clusters),
                "n_bootstraps": int(n_bootstraps),
                "sample_fraction": float(sample_fraction),
                "mean_pair_consensus": float(np.mean(upper)) if upper.size else np.nan,
                "sd_pair_consensus": float(np.std(upper, ddof=1)) if upper.size > 1 else np.nan,
                "mean_observation_count_per_pair": float(np.mean(co_observed[np.triu_indices(n_profiles, k=1)])) if n_profiles > 1 else np.nan,
            }
        ]
    )
    return matrix, summary


def permutation_test_cluster_structure_detailed(
    *,
    features: pd.DataFrame,
    n_clusters: int = 8,
    n_permutations: int = 100,
    metric: str = "euclidean",
    random_state: int = 42,
    n_init: int = 1,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test cluster separation against a feature-wise permutation null.

    The permutation null independently shuffles each feature across profiles,
    preserving each feature's marginal distribution while breaking coordinated
    multivariate morphology. The test does not prove the biological truth of a
    cluster count, but it asks whether the observed K-means separation is larger
    than expected after destroying profile-level feature covariance.
    """
    numeric = _validate_non_empty_numeric_features(features=features)
    _validate_cluster_parameters(features=numeric, n_clusters=n_clusters)
    _validate_iterations(value=n_permutations, name="n_permutations")
    rng = np.random.default_rng(seed=random_state)
    labels = _kmeans_fit_predict(
        features=numeric,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    observed = float(silhouette_score(X=numeric, labels=labels, metric=metric))
    null_records = []
    values = numeric.to_numpy(copy=True)
    for iteration in range(n_permutations):
        permuted = values.copy()
        for column_index in range(permuted.shape[1]):
            rng.shuffle(permuted[:, column_index])
        permuted_frame = pd.DataFrame(data=permuted, columns=numeric.columns)
        permuted_labels = _kmeans_fit_predict(
            features=permuted_frame,
            n_clusters=n_clusters,
            random_state=random_state + iteration + 1,
            n_init=n_init,
        )
        null_score = float(silhouette_score(X=permuted_frame, labels=permuted_labels, metric=metric))
        null_records.append(
            {
                "permutation_index": int(iteration),
                "n_clusters": int(n_clusters),
                "null_silhouette": null_score,
                "observed_silhouette": observed,
                "metric": metric,
            }
        )
        if logger is not None and (iteration + 1) % max(1, n_permutations // 5) == 0:
            logger.info("Completed cluster permutation %s/%s", iteration + 1, n_permutations)
    null_table = pd.DataFrame.from_records(null_records)
    null_array = null_table["null_silhouette"].to_numpy(dtype=float)
    p_value = (1.0 + np.sum(null_array >= observed)) / (n_permutations + 1.0)
    summary = pd.DataFrame.from_records(
        [
            {
                "n_clusters": int(n_clusters),
                "n_permutations": int(n_permutations),
                "observed_silhouette": observed,
                "null_mean_silhouette": float(np.mean(null_array)),
                "null_median_silhouette": float(np.median(null_array)),
                "null_sd_silhouette": float(np.std(null_array, ddof=1)) if n_permutations > 1 else np.nan,
                "empirical_p_value": float(p_value),
                "metric": metric,
                "interpretation_caution": (
                    "A low p-value supports multivariate structure relative to shuffled features, "
                    "but does not by itself identify the biologically correct number of clusters."
                ),
            }
        ]
    )
    return summary, null_table


def permutation_test_cluster_structure(
    *,
    features: pd.DataFrame,
    n_clusters: int = 8,
    n_permutations: int = 100,
    metric: str = "euclidean",
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Return a one-row permutation-test summary for backward compatibility."""
    summary, _ = permutation_test_cluster_structure_detailed(
        features=features,
        n_clusters=n_clusters,
        n_permutations=n_permutations,
        metric=metric,
        random_state=random_state,
        logger=logger,
    )
    return summary


def bootstrap_cluster_stability(
    *,
    features: pd.DataFrame,
    n_clusters: int = 8,
    n_bootstraps: int = 50,
    sample_fraction: float = 0.8,
    random_state: int = 42,
    n_init: int = 1,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Estimate clustering stability using adjusted Rand index on overlaps."""
    numeric = _validate_non_empty_numeric_features(features=features)
    _validate_cluster_parameters(features=numeric, n_clusters=n_clusters)
    _validate_iterations(value=n_bootstraps, name="n_bootstraps")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    rng = np.random.default_rng(seed=random_state)
    n_profiles = numeric.shape[0]
    full_labels = _kmeans_fit_predict(
        features=numeric,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    sample_size = max(n_clusters + 1, int(round(n_profiles * sample_fraction)))
    sample_size = min(sample_size, n_profiles)
    records = []
    for iteration in range(n_bootstraps):
        sampled = np.sort(rng.choice(n_profiles, size=sample_size, replace=False))
        labels = _kmeans_fit_predict(
            features=numeric.iloc[sampled, :],
            n_clusters=n_clusters,
            random_state=random_state + iteration + 1,
            n_init=n_init,
        )
        records.append(
            {
                "bootstrap_index": int(iteration),
                "n_clusters": int(n_clusters),
                "adjusted_rand_index": float(adjusted_rand_score(labels_true=full_labels[sampled], labels_pred=labels)),
                "sample_size": int(sample_size),
            }
        )
        if logger is not None and (iteration + 1) % max(1, n_bootstraps // 5) == 0:
            logger.info("Completed cluster-stability bootstrap %s/%s", iteration + 1, n_bootstraps)
    details = pd.DataFrame.from_records(records)
    values = details["adjusted_rand_index"].to_numpy(dtype=float)
    return pd.DataFrame.from_records(
        [
            {
                "n_clusters": int(n_clusters),
                "n_bootstraps": int(n_bootstraps),
                "sample_fraction": float(sample_fraction),
                "mean_adjusted_rand_index": float(np.mean(values)),
                "median_adjusted_rand_index": float(np.median(values)),
                "sd_adjusted_rand_index": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min_adjusted_rand_index": float(np.min(values)),
                "max_adjusted_rand_index": float(np.max(values)),
            }
        ]
    )


def evaluate_kmeans_k_range(
    *,
    features: pd.DataFrame,
    k_values: Sequence[int],
    n_bootstraps: int = 30,
    n_permutations: int = 50,
    sample_fraction: float = 0.8,
    metric: str = "euclidean",
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Evaluate a range of K-means cluster counts with several diagnostics."""
    numeric = _validate_non_empty_numeric_features(features=features)
    records = []
    for k in k_values:
        k = int(k)
        if k < 2 or k >= numeric.shape[0]:
            records.append({"n_clusters": k, "status": "skipped_invalid_k"})
            continue
        labels = _kmeans_fit_predict(
            features=numeric,
            n_clusters=k,
            random_state=random_state,
            n_init=1,
        )
        silhouette = float(silhouette_score(X=numeric, labels=labels, metric=metric))
        stability = bootstrap_cluster_stability(
            features=numeric,
            n_clusters=k,
            n_bootstraps=n_bootstraps,
            sample_fraction=sample_fraction,
            random_state=random_state + k,
            logger=logger,
        )
        permutation = permutation_test_cluster_structure(
            features=numeric,
            n_clusters=k,
            n_permutations=n_permutations,
            metric=metric,
            random_state=random_state + 1000 + k,
            logger=logger,
        )
        records.append(
            {
                "n_clusters": k,
                "status": "ok",
                "silhouette_score": silhouette,
                "mean_bootstrap_ari": float(stability["mean_adjusted_rand_index"].iloc[0]),
                "median_bootstrap_ari": float(stability["median_adjusted_rand_index"].iloc[0]),
                "permutation_empirical_p_value": float(permutation["empirical_p_value"].iloc[0]),
                "null_mean_silhouette": float(permutation["null_mean_silhouette"].iloc[0]),
                "n_bootstraps": int(n_bootstraps),
                "n_permutations": int(n_permutations),
                "metric": metric,
            }
        )
    return pd.DataFrame.from_records(records)
