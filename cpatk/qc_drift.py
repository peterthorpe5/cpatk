"""Acquisition-drift QC for object-level CellProfiler outputs.

The functions here generalise the user's older per-compartment drift scripts.
They operate on CellProfiler object tables before profile aggregation so that
Cell, Nuclei, Cytoplasm and other compartments can be inspected separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpatk.io import (
    is_ignored_sidecar_path,
    read_table,
    write_excel_workbook,
    write_table,
)
from cpatk.plotting import save_current_figure
from cpatk.reporting import make_html_report

COMPARTMENT_NAMES = (
    "Nuclei",
    "Cytoplasm",
    "Cell",
    "Mitochondria",
    "Acrosome",
    "Image",
)

FEATURE_PREFIXES = (
    "Intensity_",
    "AreaShape_",
    "Texture_",
    "Granularity_",
    "RadialDistribution_",
    "Neighbors_",
    "Correlation_",
)

BLOCKED_PREFIXES = (
    "FileName_",
    "PathName_",
    "URL_",
    "MD5Digest_",
    "ExecutionTime_",
    "Group_",
    "Parent_",
    "Children_",
    "Metadata_",
)

BLOCKED_EXACT = {
    "ImageNumber",
    "ObjectNumber",
    "TableNumber",
    "Number_Object_Number",
    "Plate",
    "Well",
    "Plate_Metadata",
    "Well_Metadata",
    "cpd_id",
    "cpd_type",
    "Library",
    "Dataset",
    "Sample",
}



def safe_spearmanr(*, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return Spearman correlation while avoiding eager SciPy imports.

    Parameters
    ----------
    x:
        First numeric vector.
    y:
        Second numeric vector.

    Returns
    -------
    tuple[float, float]
        Spearman rho and p-value. If SciPy cannot be imported, rho is computed
        from ranked values and the p-value is returned as NaN.

    Notes
    -----
    Some HPC environments can have a broken SciPy binary stack even when basic
    NumPy and pandas work. Drift QC should still be importable so that CPATK can
    report the environment problem at the point where a SciPy p-value is needed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = x[mask]
    y = y[mask]
    try:
        from scipy.stats import spearmanr  # type: ignore

        rho, p_value = spearmanr(x, y)
        return float(rho), float(p_value)
    except Exception:
        x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        if np.nanstd(x_rank) == 0.0 or np.nanstd(y_rank) == 0.0:
            return np.nan, np.nan
        rho = np.corrcoef(x_rank, y_rank)[0, 1]
        return float(rho), np.nan

def infer_compartment_from_name(*, filename: str) -> Optional[str]:
    """Infer compartment name from a CellProfiler output filename."""
    lower = filename.lower()
    if "nuclei" in lower:
        return "Nuclei"
    if "cytoplasm" in lower:
        return "Cytoplasm"
    if "mitochondria" in lower:
        return "Mitochondria"
    if "acrosome" in lower:
        return "Acrosome"
    if "cell" in lower and "image" not in lower:
        return "Cell"
    if "image" in lower:
        return "Image"
    return None


def list_compartment_files(
    *,
    input_dir: Path,
    include_globs: Optional[Sequence[str]] = None,
) -> dict[str, list[Path]]:
    """List likely CellProfiler files grouped by compartment."""
    input_dir = Path(input_dir)
    patterns = list(include_globs or ["*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz"])
    mapping: dict[str, list[Path]] = {}
    for pattern in patterns:
        for path in sorted(input_dir.rglob(pattern)):
            if is_ignored_sidecar_path(path=path):
                continue
            compartment = infer_compartment_from_name(filename=path.name)
            if compartment is None or compartment == "Image":
                continue
            mapping.setdefault(compartment, []).append(path)
    return mapping


def select_drift_features(
    *,
    data_frame: pd.DataFrame,
    max_features: int = 200,
    explicit_features: Optional[Sequence[str]] = None,
) -> list[str]:
    """Select numeric object-level features for drift QC."""
    if explicit_features:
        missing = [feature for feature in explicit_features if feature not in data_frame.columns]
        if missing:
            raise KeyError(f"Requested drift features are missing: {missing}")
        return list(explicit_features)
    features: list[str] = []
    for column in data_frame.select_dtypes(include=[np.number]).columns:
        name = str(column)
        if name in BLOCKED_EXACT:
            continue
        if any(name.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            continue
        if any(name.startswith(prefix) for prefix in FEATURE_PREFIXES):
            features.append(name)
    return features[:max_features]


def benjamini_hochberg(*, p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not finite_mask.any():
        return q_values
    p = p_values[finite_mask]
    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1)
    q_sorted = p[order] * p.size / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(q_sorted, 1.0)
    q_values[finite_mask] = q
    return q_values


def cliffs_delta(*, early: np.ndarray, late: np.ndarray) -> float:
    """Compute Cliff's delta for early versus late values."""
    early = np.asarray(early, dtype=float)
    late = np.asarray(late, dtype=float)
    early = early[np.isfinite(early)]
    late = late[np.isfinite(late)]
    if early.size == 0 or late.size == 0:
        return np.nan
    # Vectorised in chunks to avoid excessive memory for huge tables.
    more = 0
    less = 0
    for value in early:
        more += int(np.sum(value > late))
        less += int(np.sum(value < late))
    return float((more - less) / (early.size * late.size))


def per_image_median_series(
    *,
    data_frame: pd.DataFrame,
    image_col: str,
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted per-image feature medians."""
    sub = data_frame[[image_col, feature]].copy()
    sub[image_col] = pd.to_numeric(sub[image_col], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return np.array([]), np.array([])
    med = sub.groupby(image_col, observed=False)[feature].median().dropna()
    x = med.index.to_numpy(dtype=float)
    y = med.to_numpy(dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def rolling_median(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return centred rolling median values."""
    if values.size == 0 or window <= 1:
        return values
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).median().to_numpy()


def compute_drift_statistics(
    *,
    data_frame: pd.DataFrame,
    image_col: str = "ImageNumber",
    feature_columns: Sequence[str],
    early_fraction: float = 0.2,
    min_points: int = 50,
) -> pd.DataFrame:
    """Compute acquisition-drift statistics for selected features.

    Statistics include Spearman correlation against acquisition order using
    per-image medians, early/late median shift, Cliff's delta and a robust slope
    fitted to per-image medians.
    """
    if image_col not in data_frame.columns:
        raise KeyError(f"Missing acquisition-order column: {image_col}")
    image_values = pd.to_numeric(data_frame[image_col], errors="coerce")
    q_low = image_values.quantile(early_fraction)
    q_high = image_values.quantile(1.0 - early_fraction)
    rows = []
    for feature in feature_columns:
        if feature not in data_frame.columns:
            continue
        values = pd.to_numeric(data_frame[feature], errors="coerce")
        mask = image_values.notna() & values.notna()
        n_points = int(mask.sum())
        if n_points < min_points:
            continue
        x_img, y_img = per_image_median_series(
            data_frame=data_frame.loc[mask, [image_col, feature]],
            image_col=image_col,
            feature=feature,
        )
        if x_img.size < 4 or np.nanstd(y_img) == 0.0:
            rho = np.nan
            p_value = np.nan
            slope = np.nan
        else:
            rho, p_value = safe_spearmanr(x=x_img, y=y_img)
            x0 = x_img - np.nanmedian(x_img)
            try:
                slope = float(np.polyfit(x0, y_img, deg=1)[0])
            except Exception:
                slope = np.nan
        early = values.loc[mask & (image_values <= q_low)].to_numpy(dtype=float)
        late = values.loc[mask & (image_values >= q_high)].to_numpy(dtype=float)
        early_median = float(np.nanmedian(early)) if early.size else np.nan
        late_median = float(np.nanmedian(late)) if late.size else np.nan
        rows.append(
            {
                "feature": feature,
                "n_objects": n_points,
                "n_images": int(x_img.size),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p_value) if np.isfinite(p_value) else np.nan,
                "slope_per_image": slope,
                "early_median": early_median,
                "late_median": late_median,
                "late_minus_early_median": late_median - early_median
                if np.isfinite(early_median) and np.isfinite(late_median)
                else np.nan,
                "cliffs_delta_early_vs_late": cliffs_delta(early=early, late=late),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["spearman_q"] = benjamini_hochberg(p_values=out["spearman_p"].to_numpy())
    out["drift_flag"] = (out["spearman_q"] <= 0.01) & (out["spearman_rho"].abs() >= 0.10)
    return out.sort_values(["drift_flag", "spearman_q", "spearman_rho"], ascending=[False, True, False])


def per_image_summary(
    *,
    data_frame: pd.DataFrame,
    image_col: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarise selected features per image."""
    cols = [image_col] + list(feature_columns)
    sub = data_frame[[col for col in cols if col in data_frame.columns]].copy()
    grouped = sub.groupby(image_col, observed=False)
    med = grouped[list(feature_columns)].median(numeric_only=True).add_suffix("__median")
    q1 = grouped[list(feature_columns)].quantile(0.25).add_suffix("__q1")
    q3 = grouped[list(feature_columns)].quantile(0.75).add_suffix("__q3")
    counts = grouped.size().rename("object_count")
    return med.join([q1, q3, counts]).reset_index()


def plot_feature_drift(
    *,
    data_frame: pd.DataFrame,
    image_col: str,
    feature: str,
    stats_row: Optional[pd.Series],
    output_path_base: Path,
    max_points: int = 200_000,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """Plot feature intensity/morphology against acquisition order."""
    sub = data_frame[[image_col, feature]].dropna().copy()
    sub[image_col] = pd.to_numeric(sub[image_col], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return []
    if sub.shape[0] > max_points:
        sub = sub.sample(n=max_points, random_state=0)
    sub = sub.sort_values(image_col)
    x = sub[image_col].to_numpy(dtype=float)
    y = sub[feature].to_numpy(dtype=float)
    y_roll = rolling_median(values=y, window=max(3, min(301, max(3, y.size // 50))))
    plt.figure(figsize=(8, 4.8))
    plt.hexbin(x, y, gridsize=70, mincnt=1)
    plt.plot(x, y_roll, linewidth=1.5, label="rolling median")
    if stats_row is not None:
        rho = stats_row.get("spearman_rho", np.nan)
        q_val = stats_row.get("spearman_q", np.nan)
        delta = stats_row.get("late_minus_early_median", np.nan)
        text = f"rho={rho:.3g}; q={q_val:.3g}; late-early={delta:.3g}"
        plt.text(0.01, 0.98, text, transform=plt.gca().transAxes, ha="left", va="top")
    plt.xlabel(image_col)
    plt.ylabel(feature)
    plt.title(f"Acquisition drift: {feature}")
    plt.legend(loc="best")
    return save_current_figure(output_path_base=output_path_base, logger=logger)


def run_drift_qc(
    *,
    input_dir: Path,
    output_dir: Path,
    image_col: str = "ImageNumber",
    include_globs: Optional[Sequence[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    max_features: int = 200,
    plot_top_n: int = 8,
    min_points: int = 50,
    logger: Optional[logging.Logger] = None,
) -> dict[str, pd.DataFrame]:
    """Run per-compartment acquisition-drift QC.

    Parameters
    ----------
    input_dir:
        Folder containing CellProfiler object tables.
    output_dir:
        Output folder.
    image_col:
        Acquisition-order column, usually ``ImageNumber``.
    include_globs:
        File patterns to scan.
    feature_columns:
        Optional exact feature list.
    max_features:
        Maximum automatically selected features per compartment.
    plot_top_n:
        Number of top drifting features to plot per compartment.
    min_points:
        Minimum non-missing points required per feature.
    logger:
        Optional logger.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Summary tables keyed by name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files_by_compartment = list_compartment_files(input_dir=Path(input_dir), include_globs=include_globs)
    inventory_rows = []
    summary_tables: dict[str, pd.DataFrame] = {}
    plot_paths: list[Path] = []

    for compartment, files in sorted(files_by_compartment.items()):
        comp_dir = output_dir / compartment
        comp_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for path in files:
            try:
                frame = read_table(path=path, logger=logger)
            except Exception as exc:
                if logger is not None:
                    logger.warning("Could not read %s: %s", path, exc)
                continue
            frame["__source_file"] = path.name
            frames.append(frame)
            inventory_rows.append({"compartment": compartment, "file": str(path), "rows": frame.shape[0], "columns": frame.shape[1]})
        if not frames:
            continue
        df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        if image_col not in df.columns:
            if logger is not None:
                logger.warning("Skipping %s: missing %s", compartment, image_col)
            continue
        features = select_drift_features(
            data_frame=df,
            max_features=max_features,
            explicit_features=feature_columns,
        )
        stats = compute_drift_statistics(
            data_frame=df,
            image_col=image_col,
            feature_columns=features,
            min_points=min_points,
        )
        write_table(data_frame=stats, path=comp_dir / "drift_statistics.tsv", logger=logger)
        summary_tables[f"{compartment}_drift_statistics"] = stats
        if not stats.empty:
            top_features = stats.head(plot_top_n)["feature"].tolist()
            img_summary = per_image_summary(data_frame=df, image_col=image_col, feature_columns=top_features)
            write_table(data_frame=img_summary, path=comp_dir / "per_image_summary.tsv", logger=logger)
            for feature in top_features:
                row = stats.loc[stats["feature"] == feature].iloc[0]
                written = plot_feature_drift(
                    data_frame=df,
                    image_col=image_col,
                    feature=feature,
                    stats_row=row,
                    output_path_base=comp_dir / "plots" / f"drift_{feature[:80].replace('/', '_')}",
                    logger=logger,
                )
                plot_paths.extend(written)

    inventory = pd.DataFrame(inventory_rows)
    write_table(data_frame=inventory, path=output_dir / "drift_input_inventory.tsv", logger=logger)
    if summary_tables:
        write_excel_workbook(tables=summary_tables, path=output_dir / "drift_qc_summary.xlsx", logger=logger)
    make_html_report(
        title="CPATK acquisition-drift QC report",
        output_path=output_dir / "drift_qc_report.html",
        summary_tables={"input_inventory": inventory, **summary_tables},
        plot_paths=plot_paths[:60],
        narrative=(
            "Per-compartment acquisition-drift QC was run before profile aggregation. "
            "Features are tested against ImageNumber using per-image medians, "
            "early/late shifts and robust effect-size summaries."
        ),
        warnings=[] if summary_tables else ["No compartment drift statistics were produced."],
    )
    return {"input_inventory": inventory, **summary_tables}
