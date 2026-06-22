"""Nearest-neighbour comparison and plotting utilities for CPATK."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpatk.io import read_table, write_excel_workbook, write_table
from cpatk.plotting import save_current_figure
from cpatk.reporting import make_html_report

QUERY_CANDIDATES = ("query_id", "QueryID", "cpd_id", "Query", "source", "compound")
NEIGHBOUR_CANDIDATES = (
    "neighbour_id",
    "NeighbourID",
    "neighbor_id",
    "Neighbour",
    "Neighbor",
    "target",
)
RANK_CANDIDATES = ("rank", "Rank", "neighbour_rank", "nn_rank", "k")
DISTANCE_CANDIDATES = ("distance", "Distance", "cosine_distance", "euclidean_distance")


def _try_gaussian_kde_density(*, x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """Return KDE point density when SciPy is available, otherwise None.

    Some HPC environments expose an older system ``libstdc++`` before the
    conda environment library path. In that case importing ``scipy.stats`` can
    fail even when SciPy is installed. KDE colouring is optional, so CPATK
    should still import and plot a plain scatter in that situation.
    """
    try:
        from scipy.stats import gaussian_kde  # type: ignore

        return gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    except Exception:
        return None


def detect_column(*, data_frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column present in a table."""
    for candidate in candidates:
        if candidate in data_frame.columns:
            return candidate
    return None


def wide_neighbour_columns(*, data_frame: pd.DataFrame, prefix: str = "NN") -> list[str]:
    """Return columns named like NN1, NN2 and so on."""
    columns = [column for column in data_frame.columns if str(column).startswith(prefix)]

    def sort_key(column: str) -> tuple[int, str]:
        suffix = str(column)[len(prefix):]
        try:
            return (int(suffix), "")
        except ValueError:
            return (10**9, str(column))

    return sorted(columns, key=sort_key)


def load_neighbour_table(
    *,
    path: Path,
    nn_prefix: str = "NN",
) -> tuple[str, dict[str, list[str]], dict[str, Optional[list[float]]], int]:
    """Load a long or wide nearest-neighbour table.

    Returns
    -------
    tuple
        Run name, ordered neighbour lists, aligned distance lists and median k.
    """
    df = read_table(path=path)
    run_name = Path(path).stem.replace("_nearest_neighbours", "")
    wide_cols = wide_neighbour_columns(data_frame=df, prefix=nn_prefix)
    lists: dict[str, list[str]] = {}
    dists: dict[str, Optional[list[float]]] = {}
    if wide_cols:
        id_candidates = [column for column in df.columns if column not in wide_cols]
        if not id_candidates:
            raise ValueError(f"Could not detect query ID column in wide NN table: {path}")
        id_col = id_candidates[0]
        for _, row in df.iterrows():
            query = str(row[id_col])
            neighbours = [str(row[col]) for col in wide_cols if col in row and pd.notna(row[col])]
            lists[query] = [n for n in neighbours if n != query]
            dists[query] = None
    else:
        query_col = detect_column(data_frame=df, candidates=QUERY_CANDIDATES)
        neighbour_col = detect_column(data_frame=df, candidates=NEIGHBOUR_CANDIDATES)
        rank_col = detect_column(data_frame=df, candidates=RANK_CANDIDATES)
        distance_col = detect_column(data_frame=df, candidates=DISTANCE_CANDIDATES)
        if query_col is None or neighbour_col is None:
            raise ValueError(f"Could not detect query/neighbour columns in {path}: {list(df.columns)}")
        work = df.copy()
        work[query_col] = work[query_col].astype(str)
        work[neighbour_col] = work[neighbour_col].astype(str)
        if rank_col is not None:
            work = work.sort_values([query_col, rank_col], ascending=[True, True])
        elif distance_col is not None:
            work = work.sort_values([query_col, distance_col], ascending=[True, True])
        for query, block in work.groupby(query_col, sort=False):
            block = block.loc[block[neighbour_col] != str(query)]
            lists[str(query)] = block[neighbour_col].astype(str).tolist()
            if distance_col is not None:
                dists[str(query)] = pd.to_numeric(block[distance_col], errors="coerce").tolist()
            else:
                dists[str(query)] = None
    lengths = [len(value) for value in lists.values()]
    median_k = int(np.median(lengths)) if lengths else 0
    return run_name, lists, dists, median_k


def select_set_with_ties(
    *,
    ordered_ids: Sequence[str],
    ordered_distances: Optional[Sequence[float]],
    depth: int,
    include_ties: bool = False,
) -> set[str]:
    """Select top-k neighbours with optional tie expansion by distance."""
    if depth <= 0:
        return set()
    depth = min(depth, len(ordered_ids))
    selected = list(ordered_ids[:depth])
    if include_ties and ordered_distances is not None and depth > 0:
        kth_distance = ordered_distances[depth - 1]
        selected = [item for item, distance in zip(ordered_ids, ordered_distances) if distance <= kth_distance]
    return set(map(str, selected))


def jaccard(*, first: set[str], second: set[str]) -> float:
    """Return Jaccard similarity between two sets."""
    if not first and not second:
        return 0.0
    union = first | second
    if not union:
        return 0.0
    return float(len(first & second) / len(union))


def rank_biased_overlap(
    *,
    first: Sequence[str],
    second: Sequence[str],
    p: float = 0.9,
    depth: Optional[int] = None,
) -> float:
    """Calculate finite-depth rank-biased overlap."""
    if not first or not second:
        return 0.0
    depth = min(depth or min(len(first), len(second)), len(first), len(second))
    if depth <= 0:
        return 0.0
    seen_first: set[str] = set()
    seen_second: set[str] = set()
    score = 0.0
    for idx in range(depth):
        seen_first.add(str(first[idx]))
        seen_second.add(str(second[idx]))
        overlap = len(seen_first & seen_second) / float(idx + 1)
        score += overlap * (p ** idx)
    return float((1.0 - p) * score)


def evaluate_neighbour_overlap(
    *,
    baseline_path: Path,
    run_paths: Sequence[Path],
    k: str = "auto",
    include_ties_at_k: bool = False,
    with_rbo: bool = True,
    rbo_p: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate neighbour-list overlap against a baseline run."""
    base_name, base_lists, base_dists, base_k = load_neighbour_table(path=baseline_path)
    fixed_k: Optional[int]
    if str(k).lower() == "auto":
        fixed_k = None
    else:
        fixed_k = int(k)
    item_rows = []
    summary_rows = []
    for run_path in run_paths:
        run_name, run_lists, run_dists, run_k = load_neighbour_table(path=run_path)
        shared_queries = sorted(set(base_lists) & set(run_lists))
        depth = min(base_k, run_k) if fixed_k is None else fixed_k
        values = []
        rbo_values = []
        for query in shared_queries:
            a_set = select_set_with_ties(
                ordered_ids=base_lists[query],
                ordered_distances=base_dists.get(query),
                depth=depth,
                include_ties=include_ties_at_k,
            )
            b_set = select_set_with_ties(
                ordered_ids=run_lists[query],
                ordered_distances=run_dists.get(query),
                depth=depth,
                include_ties=include_ties_at_k,
            )
            jac = jaccard(first=a_set, second=b_set)
            values.append(jac)
            rbo = rank_biased_overlap(
                first=base_lists[query],
                second=run_lists[query],
                p=rbo_p,
                depth=depth,
            ) if with_rbo else np.nan
            rbo_values.append(rbo)
            item_rows.append(
                {
                    "query_id": query,
                    "baseline": base_name,
                    "run": run_name,
                    "k": depth,
                    "jaccard": jac,
                    "rbo": rbo,
                    "baseline_set_size": len(a_set),
                    "run_set_size": len(b_set),
                }
            )
        values_arr = np.asarray(values, dtype=float)
        rbo_arr = np.asarray(rbo_values, dtype=float)
        summary_rows.append(
            {
                "baseline": base_name,
                "run": run_name,
                "k": depth,
                "n_queries": len(shared_queries),
                "mean_jaccard": float(np.nanmean(values_arr)) if values_arr.size else np.nan,
                "median_jaccard": float(np.nanmedian(values_arr)) if values_arr.size else np.nan,
                "q25_jaccard": float(np.nanquantile(values_arr, 0.25)) if values_arr.size else np.nan,
                "q75_jaccard": float(np.nanquantile(values_arr, 0.75)) if values_arr.size else np.nan,
                "mean_rbo": float(np.nanmean(rbo_arr)) if rbo_arr.size else np.nan,
                "median_rbo": float(np.nanmedian(rbo_arr)) if rbo_arr.size else np.nan,
            }
        )
    return pd.DataFrame(item_rows), pd.DataFrame(summary_rows)


def normalise_neighbour_long_table(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a neighbour table with query_id, neighbour_id and distance columns."""
    query_col = detect_column(data_frame=data_frame, candidates=QUERY_CANDIDATES)
    neighbour_col = detect_column(data_frame=data_frame, candidates=NEIGHBOUR_CANDIDATES)
    distance_col = detect_column(data_frame=data_frame, candidates=DISTANCE_CANDIDATES)
    if query_col is None or neighbour_col is None or distance_col is None:
        raise ValueError("Nearest-neighbour table must include query, neighbour and distance columns.")
    out = data_frame[[query_col, neighbour_col, distance_col]].copy()
    out.columns = ["query_id", "neighbour_id", "distance"]
    out["query_id"] = out["query_id"].astype(str)
    out["neighbour_id"] = out["neighbour_id"].astype(str)
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    return out.dropna(subset=["distance"])


def plot_top_neighbours(
    *,
    neighbours: pd.DataFrame,
    compound_id: str,
    output_path_base: Path,
    top_n: int = 10,
    use_similarity: bool = True,
) -> tuple[pd.DataFrame, list[Path]]:
    """Plot top-N neighbours for a single compound."""
    table = normalise_neighbour_long_table(data_frame=neighbours)
    subset = table.loc[table["query_id"] == str(compound_id)].copy()
    if subset.empty:
        empty = pd.DataFrame(columns=table.columns)
        return empty, []
    subset = subset.sort_values("distance", ascending=True).head(top_n)
    if use_similarity:
        subset["similarity"] = 1.0 - subset["distance"]
        metric = "similarity"
        xlabel = "1 - distance"
    else:
        metric = "distance"
        xlabel = "Distance"
    plt.figure(figsize=(9, max(4, 0.35 * subset.shape[0] + 2)))
    plt.barh(subset["neighbour_id"].astype(str), subset[metric].astype(float))
    plt.xlabel(xlabel)
    plt.ylabel("Neighbour")
    plt.title(f"Top {top_n} neighbours for {compound_id}")
    plt.gca().invert_yaxis()
    written = save_current_figure(output_path_base=output_path_base)
    return subset, written


def shared_neighbour_table(
    *,
    neighbours: pd.DataFrame,
    first_compound: str,
    second_compound: str,
) -> pd.DataFrame:
    """Create an aligned table of neighbours shared by two compounds."""
    table = normalise_neighbour_long_table(data_frame=neighbours)
    table = table.sort_values("distance").drop_duplicates(["query_id", "neighbour_id"], keep="first")
    first = table.loc[table["query_id"] == str(first_compound)].set_index("neighbour_id")
    second = table.loc[table["query_id"] == str(second_compound)].set_index("neighbour_id")
    common = sorted(set(first.index) & set(second.index))
    rows = []
    for neighbour in common:
        d1 = float(first.loc[neighbour, "distance"])
        d2 = float(second.loc[neighbour, "distance"])
        rows.append(
            {
                "neighbour_id": neighbour,
                "first_distance": d1,
                "second_distance": d2,
                "first_similarity": 1.0 - d1,
                "second_similarity": 1.0 - d2,
            }
        )
    return pd.DataFrame(rows)


def plot_shared_neighbours(
    *,
    shared: pd.DataFrame,
    first_compound: str,
    second_compound: str,
    output_path_base: Path,
) -> list[Path]:
    """Plot similarity of shared neighbours for two compounds."""
    if shared.empty:
        return []
    x = shared["first_similarity"].to_numpy(dtype=float)
    y = shared["second_similarity"].to_numpy(dtype=float)
    plt.figure(figsize=(6, 6))
    if x.size >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        density = _try_gaussian_kde_density(x=x, y=y)
        if density is not None:
            order = np.argsort(density)
            plt.scatter(x[order], y[order], c=density[order], s=48)
            plt.colorbar(label="Density")
        else:
            plt.scatter(x, y, s=48)
    else:
        plt.scatter(x, y, s=48)
    plt.xlabel(f"{first_compound}: 1 - distance")
    plt.ylabel(f"{second_compound}: 1 - distance")
    plt.title("Shared-neighbour similarity")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    return save_current_figure(output_path_base=output_path_base)


def write_interactive_shared_neighbours(
    *,
    shared: pd.DataFrame,
    first_compound: str,
    second_compound: str,
    output_path: Path,
) -> Optional[Path]:
    """Write an interactive shared-neighbour scatter plot if Plotly is available."""
    if shared.empty:
        return None
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return None
    fig = px.scatter(
        shared,
        x="first_similarity",
        y="second_similarity",
        hover_name="neighbour_id",
        title=f"Shared neighbours: {first_compound} vs {second_compound}",
        template="simple_white",
    )
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path


def run_neighbour_workflow(
    *,
    output_dir: Path,
    input_neighbours: Optional[Path] = None,
    compounds: Optional[Sequence[str]] = None,
    baseline_neighbours: Optional[Path] = None,
    run_neighbours: Optional[Sequence[Path]] = None,
    top_n: int = 10,
    k: str = "auto",
    include_ties_at_k: bool = False,
    with_rbo: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run nearest-neighbour plotting and/or overlap evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    plot_paths: list[Path] = []
    if input_neighbours is not None:
        neighbours = read_table(path=input_neighbours)
        tables["nearest_neighbours"] = normalise_neighbour_long_table(data_frame=neighbours).head(100)
        for compound in compounds or []:
            top_table, written = plot_top_neighbours(
                neighbours=neighbours,
                compound_id=compound,
                output_path_base=plots_dir / f"top_neighbours_{compound}",
                top_n=top_n,
            )
            if not top_table.empty:
                write_table(data_frame=top_table, path=output_dir / f"top_neighbours_{compound}.tsv")
                tables[f"top_neighbours_{compound}"] = top_table
                plot_paths.extend(written)
        if compounds is not None and len(compounds) >= 2:
            first, second = compounds[0], compounds[1]
            shared = shared_neighbour_table(neighbours=neighbours, first_compound=first, second_compound=second)
            write_table(data_frame=shared, path=output_dir / f"shared_neighbours_{first}_vs_{second}.tsv")
            tables["shared_neighbours"] = shared
            plot_paths.extend(
                plot_shared_neighbours(
                    shared=shared,
                    first_compound=first,
                    second_compound=second,
                    output_path_base=plots_dir / f"shared_neighbours_{first}_vs_{second}",
                )
            )
            html_path = write_interactive_shared_neighbours(
                shared=shared,
                first_compound=first,
                second_compound=second,
                output_path=plots_dir / f"shared_neighbours_{first}_vs_{second}.html",
            )
            if html_path is not None:
                plot_paths.append(html_path)
    if baseline_neighbours is not None and run_neighbours:
        item, summary = evaluate_neighbour_overlap(
            baseline_path=baseline_neighbours,
            run_paths=run_neighbours,
            k=k,
            include_ties_at_k=include_ties_at_k,
            with_rbo=with_rbo,
        )
        write_table(data_frame=item, path=output_dir / "neighbour_overlap_per_query.tsv")
        write_table(data_frame=summary, path=output_dir / "neighbour_overlap_summary.tsv")
        tables["neighbour_overlap_summary"] = summary
        tables["neighbour_overlap_per_query"] = item
    if tables:
        write_excel_workbook(tables=tables, path=output_dir / "neighbour_analysis_summary.xlsx")
    make_html_report(
        title="CPATK nearest-neighbour analysis report",
        output_path=output_dir / "neighbour_analysis_report.html",
        summary_tables=tables,
        plot_paths=plot_paths,
        narrative=(
            "Nearest-neighbour plots and stability comparisons were generated. "
            "Jaccard overlap summarises shared top-k neighbours; rank-biased "
            "overlap additionally rewards agreement near the top of ranked lists."
        ),
    )
    return tables
