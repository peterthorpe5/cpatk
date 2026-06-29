"""HTML report generation for CPATK workflows."""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from cpatk.io import data_frame_to_html_table
from cpatk.method_guidance import ml_nn_method_guide_html

PLOT_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".pdf", ".html", ".htm"}
DEFAULT_MAX_INLINE_SVG_BYTES = 500_000


def _make_anchor(text: str) -> str:
    """Create a safe HTML anchor ID."""
    return "".join(character.lower() if character.isalnum() else "-" for character in text).strip("-")


def _normalise_relative_path(*, path: Path, base_dir: Path) -> str:
    """Return a readable relative path where possible."""
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _table_help_text(*, name: str, table: pd.DataFrame) -> str:
    """Return short plain-language guidance for recognised report tables."""
    lowered = name.lower()
    if "metadata" in lowered and "validation" in lowered:
        return "Checks that assay plate/well metadata are present, unique and not confused with source-transfer plate/well fields."
    if "profile build" in lowered:
        return "Summarises how raw CellProfiler image/object tables were merged into one image/profile-level table."
    if "preprocessing summary" in lowered:
        return "Records how many profiles/features were retained, how missing values were imputed, and how scaling/normalisation was applied."
    if "matrix validation" in lowered:
        return "Confirms the final numeric matrix is finite and safe for downstream PCA, neighbours, MOA and optional AI workflows."
    if "control qc" in lowered or "reference" in lowered:
        return "Checks whether the configured control/reference wells are present and usable before reference normalisation."
    if "replicate" in lowered:
        return "Shows whether replicate profiles for the same compound are more similar to each other after preprocessing."
    if "batch" in lowered or "pc association" in lowered:
        return "Reports how strongly metadata such as plate or compound explains the leading PCs; high plate association may indicate batch structure."
    if "nearest" in lowered or "neighbour" in lowered:
        return "Lists the closest profiles or compounds in feature space; useful for checking whether similar treatments group together."
    if "clipn" in lowered and "preprocessing" in lowered:
        return "Audits CLIPn-specific matrix preparation, especially missing, NaN and non-finite value handling before latent modelling. Literal zeros are audited but kept by default."
    if "clipn" in lowered and "status" in lowered:
        return "Reports whether the CLIPn backend ran, failed, or fell back to PCA diagnostic output."
    if "pseudo-anchor" in lowered or "moa" in lowered:
        return "Summarises pseudo-anchor or MOA-style labels. Treat these as hypotheses unless supported by controls and replicate consistency."
    return "Preview of the linked output table. Use the full TSV/Parquet file for downstream analysis."


def _status_badge(*, name: str, table: pd.DataFrame) -> str:
    """Return a small status badge inferred from common validation columns."""
    if table.empty:
        return "<span class='badge muted'>empty</span>"
    status_cols = [column for column in table.columns if str(column).lower() in {"status", "backend_run"}]
    text_values = []
    for column in status_cols:
        text_values.extend(table[column].dropna().astype(str).str.lower().tolist())
    if any(value in {"failed", "error", "not_run"} or "fail" in value for value in text_values):
        return "<span class='badge warn'>review</span>"
    if any(value in {"ok", "success", "tested"} for value in text_values):
        return "<span class='badge ok'>ok</span>"
    return "<span class='badge muted'>table</span>"


def _make_table_block(
    *,
    name: str,
    table: pd.DataFrame,
    table_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_rows: int = 50,
) -> str:
    """Render an HTML section for a table."""
    escaped = html.escape(name)
    anchor = _make_anchor(name)
    row_note = ""
    if table.shape[0] > max_rows:
        row_note = f"<p class='note'>Showing first {max_rows} of {table.shape[0]} rows.</p>"
    link_block = ""
    if table_path is not None:
        href = _normalise_relative_path(path=Path(table_path), base_dir=output_dir or Path.cwd())
        link_block = f"<p class='note'><a href='{html.escape(href)}'>Open full table</a></p>"
    help_text = html.escape(_table_help_text(name=name, table=table))
    badge = _status_badge(name=name, table=table)
    return (
        f"<section id='{anchor}' class='table-section'>"
        f"<details open><summary><strong>{escaped}</strong> "
        f"<span class='pill'>{table.shape[0]} rows × {table.shape[1]} columns</span> {badge}</summary>"
        f"<p class='help'>{help_text}</p>"
        f"{row_note}{link_block}{data_frame_to_html_table(data_frame=table, max_rows=max_rows)}</details></section>"
    )


def _copy_asset_if_requested(path: Path, assets_dir: Optional[Path]) -> tuple[Path, str]:
    """Copy an asset and return both copied path and report-relative link."""
    if assets_dir is None or not path.exists():
        return path, path.name
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(src=path, dst=target)
    return target, f"{assets_dir.name}/{target.name}"


def _make_plot_block(
    *,
    path: Path,
    assets_dir: Optional[Path] = None,
    max_inline_svg_bytes: int = DEFAULT_MAX_INLINE_SVG_BYTES,
) -> str:
    """Render an HTML section for a plot path."""
    path = Path(path)
    display_path, href = _copy_asset_if_requested(path=path, assets_dir=assets_dir)
    title = html.escape(path.stem.replace("_", " "))
    context = html.escape(path.parent.name.replace("_", " "))
    href_escaped = html.escape(href)
    if path.suffix.lower() == ".svg" and path.exists() and path.stat().st_size <= max_inline_svg_bytes:
        svg_text = path.read_text(encoding="utf-8", errors="replace")
        return (
            f"<section class='plot'><h3>{title}</h3><p class='note'>From: {context}</p>"
            f"<p class='note'><a href='{href_escaped}'>Open SVG file</a></p>"
            f"<div class='svg-plot'>{svg_text}</div></section>"
        )
    if path.suffix.lower() == ".svg" and path.exists():
        return (
            f"<section class='plot'><h3>{title}</h3><p class='note'>From: {context}</p>"
            f"<p class='note'>Large SVG not embedded to keep the report readable.</p>"
            f"<p><a href='{href_escaped}'>Open SVG file</a></p></section>"
        )
    if path.suffix.lower() in {".html", ".htm"}:
        return f"<section class='plot'><h3>{title}</h3><p class='note'>From: {context}</p><p><a href='{href_escaped}'>Open interactive HTML output</a></p></section>"
    return f"<section class='plot'><h3>{title}</h3><p class='note'>From: {context}</p><p><a href='{href_escaped}'>{html.escape(display_path.name)}</a></p></section>"


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


def _result_map(
    *,
    summary_tables: Mapping[str, pd.DataFrame],
    plot_paths: Sequence[Path],
    table_paths: Optional[Mapping[str, Path]],
    output_dir: Path,
) -> str:
    """Create a plain-language map of the report contents."""
    rows = []
    for name, table in summary_tables.items():
        href = f"#{_make_anchor(name)}"
        source = ""
        if table_paths and name in table_paths:
            source = _normalise_relative_path(path=Path(table_paths[name]), base_dir=output_dir)
        rows.append(
            {
                "section": name,
                "type": "table",
                "size": f"{table.shape[0]} rows × {table.shape[1]} columns",
                "where": source,
                "link": href,
            }
        )
    for path in plot_paths:
        rows.append(
            {
                "section": Path(path).stem.replace("_", " "),
                "type": "plot/output",
                "size": Path(path).suffix.lower().lstrip(".") or "file",
                "where": _normalise_relative_path(path=Path(path), base_dir=output_dir),
                "link": "#plots-and-interactive-outputs",
            }
        )
    if not rows:
        return ""
    data_frame = pd.DataFrame.from_records(rows[:200])
    note = ""
    if len(rows) > 200:
        note = f"<p class='note'>Showing first 200 of {len(rows)} mapped outputs.</p>"
    return (
        "<section><h2>Result map</h2>"
        "<p>This section lists the main tables and plots included in this report, with links back to the full files where available.</p>"
        f"{note}{data_frame_to_html_table(data_frame=data_frame, max_rows=200)}</section>"
    )


def discover_plot_paths(
    *,
    root_dir: Path,
    output_html: Optional[Path] = None,
    max_plots: int = 200,
) -> list[Path]:
    """Discover plot and interactive-output files below a results directory."""
    root_dir = Path(root_dir)
    output_html = Path(output_html) if output_html is not None else None
    if not root_dir.exists():
        return []
    candidates: list[Path] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._") or path.name.startswith("."):
            continue
        if output_html is not None and path.resolve() == output_html.resolve():
            continue
        if "report_assets" in path.parts:
            continue
        if path.suffix.lower() in PLOT_SUFFIXES:
            candidates.append(path)
        if len(candidates) >= max_plots:
            break
    return candidates


def _quick_reading_guide() -> str:
    """Return a short report-reading guide."""
    items = [
        "Start with metadata validation and profile-build summary before trusting any biology.",
        "Use preprocessing summaries to check how much data was removed or imputed.",
        "Compare replicate consistency and batch association before choosing a normalisation strategy.",
        "Use PCA/UMAP/neighbour plots as the first biological sanity check.",
        "Treat ML, SHAP, CLIPn and MOA outputs as interpretation layers, not proof on their own.",
    ]
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<section class='guide'><h2>How to read this report</h2><ul>{bullets}</ul></section>"


def make_html_report(
    *,
    title: str,
    output_path: Path,
    summary_tables: Optional[Mapping[str, pd.DataFrame]] = None,
    plot_paths: Optional[Sequence[Path]] = None,
    table_paths: Optional[Mapping[str, Path]] = None,
    narrative: Optional[str] = None,
    methods_text: Optional[str] = None,
    warnings: Optional[Sequence[str]] = None,
    max_table_rows: int = 50,
    copy_assets: bool = True,
    max_inline_svg_bytes: int = DEFAULT_MAX_INLINE_SVG_BYTES,
) -> Path:
    """Create an HTML analysis report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent
    summary_tables = summary_tables or {}
    plot_paths = list(plot_paths or [])
    table_paths = table_paths or {}
    assets_dir = output_path.parent / "report_assets" if copy_assets else None
    warning_blocks = "".join(
        f"<div class='notice'>{html.escape(str(warning))}</div>" for warning in (warnings or [])
    )
    table_toc = "".join(
        f"<li><a href='#{_make_anchor(name)}'>{html.escape(name)}</a></li>" for name in summary_tables
    )
    table_blocks = "".join(
        _make_table_block(
            name=name,
            table=table,
            table_path=table_paths.get(name),
            output_dir=output_dir,
            max_rows=max_table_rows,
        )
        for name, table in summary_tables.items()
    )
    plot_blocks = "".join(
        _make_plot_block(
            path=Path(path),
            assets_dir=assets_dir,
            max_inline_svg_bytes=max_inline_svg_bytes,
        )
        for path in plot_paths
    )
    if not plot_blocks:
        plot_blocks = "<p class='note'>No plots were supplied or discovered for this report.</p>"
    methods_block = ""
    if methods_text:
        paragraphs = "".join(
            f"<p>{html.escape(paragraph.strip())}</p>"
            for paragraph in methods_text.split("\n\n")
            if paragraph.strip()
        )
        methods_block = f"<section><h2>Methods explained</h2>{paragraphs}</section>"
    cards = _summary_cards(summary_tables)
    result_map = _result_map(
        summary_tables=summary_tables,
        plot_paths=plot_paths,
        table_paths=table_paths,
        output_dir=output_dir,
    )
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
.toc, .guide {{ background: #f5f5f5; padding: 1rem; border-radius: 0.5rem; }}
.help {{ color: #444; margin: 0.4rem 0 0.7rem 0; }}
.badge {{ border-radius: 999px; padding: 0.12rem 0.45rem; font-size: 0.72rem; margin-left: 0.35rem; }}
.badge.ok {{ background: #e8f5e9; color: #1b5e20; }}
.badge.warn {{ background: #fff3e0; color: #e65100; }}
.badge.muted {{ background: #eeeeee; color: #555; }}
.plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1rem; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>Generated by CPATK.</p></header>
<main>
<section><h2>Summary</h2><div class='notice'>{html.escape(narrative or 'CPATK report generated successfully.')}</div>{warning_blocks}</section>
{cards}
{_quick_reading_guide()}
{ml_nn_method_guide_html()}
{result_map}
{methods_block}
<section class='toc'><h2>Table index</h2><ul>{table_toc}</ul></section>
<section><h2>Summary tables</h2>{table_blocks}</section>
<section id='plots-and-interactive-outputs'><h2>Plots and interactive outputs</h2><div class='plot-grid'>{plot_blocks}</div></section>
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
        "The report is a map into the results folder, not a replacement for the full outputs. Small previews are shown in the HTML. "
        "The full TSV or Parquet files should be used for downstream analysis and archiving.\n\n"
        "Preprocessing applies feature-level and profile-level QC before imputation. After all CellProfiler tables have been merged, rows whose observed retained feature values are all zero are removed by default, because these usually represent failed/empty profiles rather than valid morphology. For CLIPn, missing/NA/non-finite values are the problem; real zero values are audited but kept by default. Median imputation is the default because "
        "it is robust and avoids borrowing structure across treatments. Group-median, mean, zero and KNN imputation are available, "
        "but KNN imputation should be used cautiously because it can smooth real batch or perturbation structure.\n\n"
        "Optional control/reference normalisation can centre profiles against reference wells such as DMSO within each plate or batch. "
        "This is often useful in Cell Painting, but it is disabled by default because it depends on correctly annotated controls and an appropriate design.\n\n"
        "Classical analysis uses PCA, optional UMAP and t-SNE, pairwise distances, nearest neighbours and clustering. "
        "These methods provide a non-AI analysis route and are also important QC before any AI/CLIPn workflow. MOA analysis can be run on the classical preprocessed feature space and, optionally, on a CLIPn latent space; the latter is labelled separately because it answers a different question.\n\n"
        "Replicate reproducibility is assessed by within-group profile correlations. Neighbour stability and cluster stability are assessed by bootstrap resampling. "
        "Cluster permutation testing compares the observed silhouette score with scores obtained after feature-wise permutation, which helps determine whether clustering is stronger than expected from marginal feature distributions alone.\n\n"
        "MOA classification can be performed using centroid similarity, KNN and supervised machine-learning models such as random forests, extra trees, gradient boosting, logistic regression and calibrated linear SVMs. "
        "Prediction confidence should be interpreted alongside class size, replicate consistency, cross-validation performance and possible batch effects.\n\n"
        "Feature attribution is provided by permutation importance and, when available, SHAP. Permutation importance asks how much held-out classifier performance drops when a feature is shuffled. "
        "SHAP provides model-specific local/global attribution, but it is still a model explanation rather than proof of biological causality."
    )
