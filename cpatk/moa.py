"""Mechanism-of-action classification and enrichment helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def calculate_class_centroids(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    min_class_size: int = 2,
    statistic: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate class centroids for known mechanisms of action."""
    if features.shape[0] != len(labels):
        raise ValueError("features and labels must contain the same number of rows.")
    statistic = statistic.lower()
    table = features.copy()
    table["__label__"] = labels.astype(str).to_numpy()
    counts = table.groupby("__label__", dropna=False).size().reset_index(name="n_profiles")
    valid_labels = counts.loc[counts["n_profiles"] >= min_class_size, "__label__"].tolist()
    filtered = table.loc[table["__label__"].isin(valid_labels)]
    if statistic == "median":
        centroids = filtered.groupby("__label__").median(numeric_only=True)
    elif statistic == "mean":
        centroids = filtered.groupby("__label__").mean(numeric_only=True)
    else:
        raise ValueError(f"Unsupported centroid statistic: {statistic}")
    centroids.index.name = "class_label"
    counts = counts.rename(columns={"__label__": "class_label"})
    counts["used_for_centroid"] = counts["class_label"].isin(valid_labels)
    counts["min_class_size"] = int(min_class_size)
    return centroids, counts


def _similarity_from_distance(*, distance: np.ndarray, metric: str) -> np.ndarray:
    """Convert distances to a bounded larger-is-better similarity score."""
    if metric in {"cosine", "correlation"}:
        return 1.0 - distance
    return 1.0 / (1.0 + distance)


def _softmax(values: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """Numerically stable softmax for centroid confidence scores."""
    temperature = max(float(temperature), 1e-12)
    shifted = values / temperature
    shifted = shifted - np.nanmax(shifted)
    exp_values = np.exp(shifted)
    denominator = np.nansum(exp_values)
    if not np.isfinite(denominator) or denominator <= 0:
        return np.repeat(1.0 / len(values), len(values))
    return exp_values / denominator


def score_profiles_against_centroids(
    *,
    query_features: pd.DataFrame,
    centroids: pd.DataFrame,
    metric: str = "cosine",
    top_n: int = 5,
    confidence_temperature: float = 0.1,
) -> pd.DataFrame:
    """Score query profiles against class centroids.

    The returned long table contains rank, distance, similarity, an approximate
    softmax confidence and the margin from the next-best centroid.  The scores
    are useful for triage but should be validated using replicate consistency
    and cross-validation when known MOA labels are available.
    """
    shared_columns = [column for column in query_features.columns if column in centroids.columns]
    if not shared_columns:
        raise ValueError("No shared feature columns between query features and centroids.")
    distances = pairwise_distances(
        X=query_features.loc[:, shared_columns],
        Y=centroids.loc[:, shared_columns],
        metric=metric,
    )
    similarities = _similarity_from_distance(distance=distances, metric=metric)
    records = []
    labels = centroids.index.astype(str).tolist()
    top_n = max(1, min(int(top_n), len(labels)))
    for row_index, query_index in enumerate(query_features.index):
        order = np.argsort(distances[row_index, :])
        probs = _softmax(similarities[row_index, :], temperature=confidence_temperature)
        best = order[0]
        second = order[1] if len(order) > 1 else order[0]
        margin = similarities[row_index, best] - similarities[row_index, second]
        for rank, class_position in enumerate(order[:top_n], start=1):
            records.append(
                {
                    "query_index": query_index,
                    "rank": rank,
                    "predicted_class": labels[class_position],
                    "distance": float(distances[row_index, class_position]),
                    "similarity": float(similarities[row_index, class_position]),
                    "softmax_confidence": float(probs[class_position]),
                    "top1_similarity_margin": float(margin),
                    "metric": metric,
                    "n_shared_features": int(len(shared_columns)),
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
    return_neighbour_table: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Classify query profiles using K-nearest neighbours."""
    shared_columns = [column for column in train_features.columns if column in query_features.columns]
    if not shared_columns:
        raise ValueError("No shared feature columns between training and query features.")
    clean_labels = train_labels.astype(str).reset_index(drop=True)
    n_neighbors = min(max(1, int(n_neighbors)), train_features.shape[0])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X=train_features.loc[:, shared_columns], y=clean_labels)
    query_x = query_features.loc[:, shared_columns]
    predicted = model.predict(X=query_x)
    probabilities = model.predict_proba(X=query_x)
    result = pd.DataFrame(
        {
            "query_index": query_features.index,
            "predicted_class": predicted,
            "max_probability": probabilities.max(axis=1),
            "n_neighbors": int(n_neighbors),
            "weights": weights,
            "n_shared_features": int(len(shared_columns)),
        }
    )
    for class_index, class_name in enumerate(model.classes_):
        result[f"probability_{class_name}"] = probabilities[:, class_index]
    if not return_neighbour_table:
        return result

    distances, indices = model.kneighbors(X=query_x, n_neighbors=n_neighbors, return_distance=True)
    neighbour_records = []
    train_index_values = list(train_features.index)
    for query_row, query_index in enumerate(query_features.index):
        for rank in range(n_neighbors):
            train_position = int(indices[query_row, rank])
            neighbour_records.append(
                {
                    "query_index": query_index,
                    "rank": rank + 1,
                    "train_index": train_index_values[train_position],
                    "neighbour_class": clean_labels.iloc[train_position],
                    "distance": float(distances[query_row, rank]),
                }
            )
    return result, pd.DataFrame.from_records(neighbour_records)


def calculate_nearest_neighbour_moa_enrichment(
    *,
    neighbour_table: pd.DataFrame,
    class_labels: pd.Series,
    class_column: str = "neighbour_class",
    query_column: str = "query_index",
) -> pd.DataFrame:
    """Calculate hypergeometric enrichment of MOA classes among neighbours."""
    if class_column not in neighbour_table.columns:
        raise ValueError(f"Neighbour table is missing class column: {class_column}")
    background_counts = class_labels.astype(str).value_counts()
    total_background = int(background_counts.sum())
    records = []
    for query_id, query_table in neighbour_table.groupby(query_column, dropna=False):
        neighbour_classes = query_table[class_column].astype(str)
        n_neighbours = int(neighbour_classes.shape[0])
        observed_counts = neighbour_classes.value_counts()
        for moa_class, observed in observed_counts.items():
            background_class_count = int(background_counts.get(moa_class, 0))
            p_value = hypergeom.sf(observed - 1, total_background, background_class_count, n_neighbours)
            records.append(
                {
                    "query_index": query_id,
                    "moa_class": moa_class,
                    "n_neighbours": n_neighbours,
                    "observed_neighbours_in_class": int(observed),
                    "background_profiles_in_class": background_class_count,
                    "background_profiles_total": total_background,
                    "fraction_neighbours_in_class": float(observed / n_neighbours) if n_neighbours else np.nan,
                    "hypergeom_p_value": float(p_value),
                }
            )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    table = table.sort_values(["query_index", "hypergeom_p_value", "moa_class"])
    return table


def leave_one_out_centroid_validation(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    min_class_size: int = 2,
    metric: str = "cosine",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate centroid MOA classification using leave-one-out profiles."""
    clean_labels = labels.astype(str).reset_index(drop=True)
    predictions = []
    for row_index in range(features.shape[0]):
        train_mask = np.ones(features.shape[0], dtype=bool)
        train_mask[row_index] = False
        centroids, _ = calculate_class_centroids(
            features=features.loc[train_mask, :].reset_index(drop=True),
            labels=clean_labels.loc[train_mask].reset_index(drop=True),
            min_class_size=min_class_size,
        )
        if centroids.empty:
            continue
        scores = score_profiles_against_centroids(
            query_features=features.iloc[[row_index], :],
            centroids=centroids,
            metric=metric,
            top_n=1,
        )
        top = scores.iloc[0].to_dict()
        top["true_class"] = clean_labels.iloc[row_index]
        top["correct"] = str(top["predicted_class"]) == clean_labels.iloc[row_index]
        predictions.append(top)
    prediction_table = pd.DataFrame.from_records(predictions)
    if prediction_table.empty:
        summary = pd.DataFrame.from_records([{"status": "not_tested", "reason": "No leave-one-out predictions could be made."}])
    else:
        summary = pd.DataFrame.from_records(
            [
                {
                    "status": "tested",
                    "n_profiles": int(prediction_table.shape[0]),
                    "accuracy": float(prediction_table["correct"].mean()),
                    "metric": metric,
                    "min_class_size": int(min_class_size),
                }
            ]
        )
    return prediction_table, summary


def summarise_moa_predictions(
    *,
    predictions: pd.DataFrame,
    group_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Summarise predicted mechanisms of action."""
    if "predicted_class" not in predictions.columns:
        raise ValueError("Prediction table must contain predicted_class.")
    group_columns = [column for column in (group_columns or []) if column in predictions.columns]
    grouped = group_columns + ["predicted_class"]
    counts = predictions.groupby(grouped, dropna=False).size().reset_index(name="n_profiles")
    if group_columns:
        totals = counts.groupby(group_columns, dropna=False)["n_profiles"].transform("sum")
    else:
        totals = counts["n_profiles"].sum()
    counts["fraction"] = counts["n_profiles"] / totals
    return counts


def calculate_moa_separability(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    metric: str = "cosine",
    n_permutations: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare within-MOA and between-MOA distances with a permutation null.

    A useful MOA classifier should not only produce labels; known profiles from
    the same MOA should also tend to be closer to each other than to profiles
    from other MOAs. This diagnostic estimates the observed within-minus-between
    distance separation and compares it with shuffled MOA labels.
    """
    if features.shape[0] != len(labels):
        raise ValueError("features and labels must contain the same number of rows.")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be at least 1.")
    clean_features = features.apply(pd.to_numeric, errors="coerce")
    if clean_features.isna().any().any():
        raise ValueError("MOA separability requires preprocessed features without missing values.")
    clean_labels = labels.astype(str).reset_index(drop=True)
    distances = pairwise_distances(X=clean_features, metric=metric)
    same = clean_labels.to_numpy()[:, None] == clean_labels.to_numpy()[None, :]
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    within = distances[upper & same]
    between = distances[upper & ~same]
    observed = float(np.nanmedian(between) - np.nanmedian(within)) if within.size and between.size else np.nan
    rng = np.random.default_rng(seed=random_state)
    null_records = []
    label_values = clean_labels.to_numpy(copy=True)
    for iteration in range(n_permutations):
        shuffled = label_values.copy()
        rng.shuffle(shuffled)
        same_null = shuffled[:, None] == shuffled[None, :]
        null_within = distances[upper & same_null]
        null_between = distances[upper & ~same_null]
        null_score = float(np.nanmedian(null_between) - np.nanmedian(null_within)) if null_within.size and null_between.size else np.nan
        null_records.append({"permutation_index": int(iteration), "null_separation": null_score})
    null_table = pd.DataFrame.from_records(null_records)
    null_values = pd.to_numeric(null_table["null_separation"], errors="coerce").dropna().to_numpy()
    empirical_p = float((1.0 + np.sum(null_values >= observed)) / (len(null_values) + 1.0)) if len(null_values) else np.nan
    summary = pd.DataFrame.from_records(
        [
            {
                "metric": metric,
                "n_profiles": int(clean_features.shape[0]),
                "n_classes": int(clean_labels.nunique()),
                "median_within_moa_distance": float(np.nanmedian(within)) if within.size else np.nan,
                "median_between_moa_distance": float(np.nanmedian(between)) if between.size else np.nan,
                "observed_between_minus_within": observed,
                "null_mean_between_minus_within": float(np.nanmean(null_values)) if len(null_values) else np.nan,
                "empirical_p_value": empirical_p,
                "n_permutations": int(n_permutations),
            }
        ]
    )
    return summary, null_table


def summarise_prediction_confidence(
    *,
    predictions: pd.DataFrame,
    probability_column: str = "max_probability",
    margin_column: str = "top1_similarity_margin",
) -> pd.DataFrame:
    """Summarise confidence columns from MOA prediction tables."""
    records = []
    if probability_column in predictions.columns:
        values = pd.to_numeric(predictions[probability_column], errors="coerce")
        records.append(
            {
                "confidence_metric": probability_column,
                "n_values": int(values.notna().sum()),
                "median": float(values.median(skipna=True)),
                "mean": float(values.mean(skipna=True)),
                "min": float(values.min(skipna=True)),
                "max": float(values.max(skipna=True)),
            }
        )
    if margin_column in predictions.columns:
        values = pd.to_numeric(predictions[margin_column], errors="coerce")
        records.append(
            {
                "confidence_metric": margin_column,
                "n_values": int(values.notna().sum()),
                "median": float(values.median(skipna=True)),
                "mean": float(values.mean(skipna=True)),
                "min": float(values.min(skipna=True)),
                "max": float(values.max(skipna=True)),
            }
        )
    return pd.DataFrame.from_records(records)
