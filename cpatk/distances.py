"""Distance and nearest-neighbour calculations for Cell Painting profiles."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def calculate_pairwise_distance_matrix(
    *,
    features: pd.DataFrame,
    metric: str = "cosine",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Calculate a pairwise distance matrix.

    Parameters
    ----------
    features:
        Numeric feature matrix.
    metric:
        Distance metric accepted by scikit-learn.
    n_jobs:
        Number of jobs for scikit-learn pairwise distance calculation.

    Returns
    -------
    pandas.DataFrame
        Square distance matrix.
    """
    distances = pairwise_distances(X=features, metric=metric, n_jobs=max(1, int(n_jobs)))
    return pd.DataFrame(data=distances, index=features.index, columns=features.index)


def calculate_nearest_neighbours(
    *,
    distance_matrix: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    id_column: Optional[str] = None,
    n_neighbours: int = 10,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Create a nearest-neighbour table from a distance matrix.

    Parameters
    ----------
    distance_matrix:
        Square distance matrix.
    metadata:
        Optional metadata table aligned to matrix rows.
    id_column:
        Optional metadata column used as profile labels.
    n_neighbours:
        Number of neighbours per profile.
    exclude_self:
        Whether to exclude self-distances.

    Returns
    -------
    pandas.DataFrame
        Long-format nearest-neighbour table.
    """
    records = []
    labels = list(distance_matrix.index)
    if metadata is not None and id_column in metadata.columns:
        profile_ids = metadata[id_column].astype(str).tolist()
    else:
        profile_ids = [str(label) for label in labels]

    values = distance_matrix.to_numpy(copy=True)
    for row_index, profile_id in enumerate(profile_ids):
        row = values[row_index, :].copy()
        order = np.argsort(row)
        rank = 0
        for neighbour_index in order:
            if exclude_self and neighbour_index == row_index:
                continue
            rank += 1
            records.append(
                {
                    "query_index": labels[row_index],
                    "query_id": profile_id,
                    "neighbour_index": labels[neighbour_index],
                    "neighbour_id": profile_ids[neighbour_index],
                    "rank": rank,
                    "distance": float(row[neighbour_index]),
                }
            )
            if rank >= n_neighbours:
                break
    return pd.DataFrame.from_records(records)


def summarise_neighbour_classes(
    *,
    neighbours: pd.DataFrame,
    metadata: pd.DataFrame,
    query_key: str = "query_index",
    neighbour_key: str = "neighbour_index",
    class_column: str = "moa",
) -> pd.DataFrame:
    """Summarise class labels among nearest neighbours.

    Parameters
    ----------
    neighbours:
        Nearest-neighbour table.
    metadata:
        Metadata table indexed consistently with neighbour indices.
    query_key:
        Query index column in the neighbour table.
    neighbour_key:
        Neighbour index column in the neighbour table.
    class_column:
        Metadata class column.

    Returns
    -------
    pandas.DataFrame
        Counts and fractions of neighbour classes per query.
    """
    if class_column not in metadata.columns:
        raise ValueError(f"Missing class column in metadata: {class_column}")
    class_lookup = metadata[class_column].to_dict()
    annotated = neighbours.copy()
    annotated["neighbour_class"] = annotated[neighbour_key].map(class_lookup)
    counts = (
        annotated.groupby([query_key, "neighbour_class"], dropna=False)
        .size()
        .reset_index(name="n_neighbours")
    )
    totals = counts.groupby(query_key)["n_neighbours"].transform("sum")
    counts["fraction_neighbours"] = counts["n_neighbours"] / totals
    return counts
