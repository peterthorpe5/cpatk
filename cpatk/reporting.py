"""HTML report generation for CPATK workflows."""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from cpatk.io import data_frame_to_html_table


def _make_anchor(text: str) -> str:
    """Create a safe HTML anchor ID."""
    return "".join(character.lower() if character.isalnum() else "-" for character in text).strip("-")


def _make_table_block(*, name: str, table: pd.DataFrame, max_rows: int = 50) -> str:
    """Render an HTML section for a table."""
    escaped = html.escape(name)
    anchor = _make_anchor(name)
    row_note = ""
    if table.shape[0] > max_rows:
        row_note = f"<p class='note'>Showing first {max_rows} of {table.shape[0]} rows.</p>"
    return (
        f"<section id='{anchor}' class='table-section'>"
        f"<details open><summary><strong>{escaped}</strong> "
        f"<span class='pill'>{table.shape[0]} rows × {table.shape[1]} columns</span></summary>"
        f"{row_note}{data_frame_to_html_table(data_frame=table, max_rows=max_rows)}</details></section>"
    )


def _copy_asset_if_requested(path: Path, assets_dir: Optional[Path]) -> tuple[Path, str]:
    """Copy an asset and return both copied path and report-relative link.

    Parameters
    ----------
    path:
        Source asset path.
    assets_dir:
        Optional report-assets directory.

    Returns
    -------
    tuple[pathlib.Path, str]
        The path to display/read and the href relative to the report file.
    """
    if assets_dir is None or not path.exists():
        return path, path.name
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(src=path, dst=target)
    return target, f"{assets_dir.name}/{target.name}"


def _make_plot_block(*, path: Path, assets_dir: Optional[Path] = None) -> str:
    """Render an HTML section for a plot path."""
    path = Path(path)
    display_path, href = _copy_asset_if_requested(path=path, assets_dir=assets_dir)
    title = html.escape(path.stem.replace("_", " "))
    href_escaped = html.escape(href)
    if path.suffix.lower() == ".svg" and path.exists():
        svg_text = path.read_text(encoding="utf-8", errors="replace")
        return f"<section class='plot'><h3>{title}</h3><div class='svg-plot'>{svg_text}</div></section>"
    if path.suffix.lower() in {".html", ".htm"}:
        return f"<section class='plot'><h3>{title}</h3><p><a href='{href_escaped}'>Open interactive HTML output</a></p></section>"
    return f"<section class='plot'><h3>{title}</h3><p><a href='{href_escaped}'>{html.escape(display_path.name)}</a></p></section>"


def _summary_cards(summary_tables: Mapping[str, pd.DataFrame]) -> str:
    """Create compact cards from recognised summary tables."""
    cards = []
    for name, table in summary_tables.items():
        if table.empty:
            continue
        if {"item", "value"}.issubset(table.columns):
            for _, row in table.head(10).iterrows():
                cards.append(
                    f"<div class='card'><div class='card-label'>{html.escape(str(row['item']))}</div>"
                    f"<div class='card-value'>{html.escape(str(row['value']))}</div></div>"
                )
            break
    if not cards:
        return ""
    return "<section><h2>Key summary values</h2><div class='card-grid'>" + "".join(cards) + "</div></section>"


def make_html_report(
    *,
    title: str,
    output_path: Path,
    summary_tables: Optional[Mapping[str, pd.DataFrame]] = None,
    plot_paths: Optional[Sequence[Path]] = None,
    narrative: Optional[str] = None,
    methods_text: Optional[str] = None,
    warnings: Optional[Sequence[str]] = None,
    max_table_rows: int = 50,
    copy_assets: bool = True,
) -> Path:
    """Create an HTML analysis report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_tables = summary_tables or {}
    plot_paths = plot_paths or []
    assets_dir = output_path.parent / "report_assets" if copy_assets else None
    warning_blocks = "".join(
        f"<div class='notice'>{html.escape(str(warning))}</div>" for warning in (warnings or [])
    )
    table_toc = "".join(
        f"<li><a href='#{_make_anchor(name)}'>{html.escape(name)}</a></li>" for name in summary_tables
    )
    table_blocks = "".join(
        _make_table_block(name=name, table=table, max_rows=max_table_rows)
        for name, table in summary_tables.items()
    )
    plot_blocks = "".join(_make_plot_block(path=Path(path), assets_dir=assets_dir) for path in plot_paths)
    methods_block = ""
    if methods_text:
        paragraphs = "".join(
            f"<p>{html.escape(paragraph.strip())}</p>"
            for paragraph in methods_text.split("\n\n")
            if paragraph.strip()
        )
        methods_block = f"<section><h2>Methods explained</h2>{paragraphs}</section>"
    cards = _summary_cards(summary_tables)
    document = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.55; margin: 0; background: #f7f7f7; color: #222; }}
header {{ background: linear-gradient(135deg, #263238, #455a64); color: white; padding: 2rem 3rem; }}
main {{ max-width: 1280px; margin: auto; background: white; padding: 2rem 3rem; }}
h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; margin-top: 0.8rem; }}
th, td {{ border: 1px solid #d8d8d8; padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }}
th {{ background: #eceff1; position: sticky; top: 0; }}
summary {{ cursor: pointer; margin: 0.8rem 0; }}
.plot, .table-section {{ margin: 1rem 0 2rem 0; padding: 1rem; border: 1px solid #e0e0e0; background: #fafafa; border-radius: 0.5rem; }}
.svg-plot svg {{ max-width: 100%; height: auto; background: white; }}
.notice {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 1rem; margin: 1rem 0; }}
.note {{ color: #555; }}
.pill {{ background: #eceff1; border-radius: 999px; padding: 0.15rem 0.55rem; font-size: 0.78rem; color: #455a64; }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.8rem; margin: 1rem 0; }}
.card {{ border: 1px solid #d8d8d8; border-radius: 0.5rem; padding: 0.8rem; background: #fafafa; }}
.card-label {{ color: #555; font-size: 0.82rem; }}
.card-value {{ font-size: 1.2rem; font-weight: 650; margin-top: 0.2rem; word-break: break-word; }}
.toc {{ background: #f5f5f5; padding: 1rem; border-radius: 0.5rem; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>Generated by CPATK.</p></header>
<main>
<section><h2>Executive summary</h2><div class='notice'>{html.escape(narrative or 'CPATK report generated successfully.')}</div>{warning_blocks}</section>
{cards}
{methods_block}
<section class='toc'><h2>Table index</h2><ul>{table_toc}</ul></section>
<section><h2>Summary tables</h2>{table_blocks}</section>
<section><h2>Plots and interactive outputs</h2>{plot_blocks}</section>
</main>
</body>
</html>
"""
    output_path.write_text(data=document, encoding="utf-8")
    return output_path


def default_methods_text() -> str:
    """Return default report text explaining CPATK analysis concepts."""
    return (
        "CPATK separates metadata from numeric Cell Painting features using an auditable column-role table. "
        "By default, obvious provenance and QC columns such as image identifiers, file names, checksums, execution times, "
        "object counts and image dimensions are not treated as biological morphology features unless explicitly requested.\n\n"
        "Preprocessing applies feature-level and profile-level QC before imputation. After all CellProfiler tables have been merged, rows whose observed retained feature values are all zero are removed by default, because these usually represent failed/empty profiles rather than valid morphology. Median imputation is the default because "
        "it is robust and avoids borrowing structure across treatments.  Group-median, mean, zero and KNN imputation are available, "
        "but KNN imputation should be used cautiously because it can smooth real batch or perturbation structure.\n\n"
        "Optional control/reference normalisation can centre profiles against reference wells such as DMSO within each plate or batch. "
        "This is often useful in Cell Painting, but it is disabled by default because it depends on correctly annotated controls and an appropriate design.\n\n"
        "Classical analysis uses PCA, optional UMAP and t-SNE, pairwise distances, nearest neighbours and clustering. "
        "These methods provide a non-AI analysis route and are also important QC before any AI/CLIPn workflow.\n\n"
        "Replicate reproducibility is assessed by within-group profile correlations. Neighbour stability and cluster stability are assessed by bootstrap resampling. "
        "Cluster permutation testing compares the observed silhouette score with scores obtained after feature-wise permutation, which helps determine whether clustering is stronger than expected from marginal feature distributions alone.\n\n"
        "MOA classification can be performed using centroid similarity, KNN and supervised machine-learning models such as random forests, extra trees, gradient boosting, logistic regression and calibrated linear SVMs. "
        "Prediction confidence should be interpreted alongside class size, replicate consistency, cross-validation performance and possible batch effects.\n\n"
        "Feature attribution is provided by permutation importance and, when available, SHAP. Permutation importance asks how much held-out classifier performance drops when a feature is shuffled. "
        "SHAP provides model-specific local/global attribution, but it is still a model explanation rather than proof of biological causality."
    )
