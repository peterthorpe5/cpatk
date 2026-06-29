"""Reusable method guidance text for CPATK reports.

The functions in this module read the package's method-guide resource files and
return both table and HTML representations.  The content is deliberately plain
language because it is intended for end users reading analysis reports, not only
for developers.
"""

from __future__ import annotations

import html
from importlib import resources
from pathlib import Path
from typing import Optional

import pandas as pd

RESOURCE_PACKAGE = "cpatk.resources"
GUIDE_TSV = "ml_nn_method_guide.tsv"
GUIDE_MARKDOWN = "ML_NN_METHOD_GUIDE.md"


_REQUIRED_COLUMNS = [
    "method",
    "method_group",
    "what_it_does",
    "best_use_case",
    "what_results_mean",
    "when_not_to_use",
    "main_caveats",
]


def load_ml_nn_method_guide() -> pd.DataFrame:
    """Load the bundled ML/NN method guide as a data frame.

    Returns
    -------
    pandas.DataFrame
        One row per method with plain-language interpretation guidance.

    Raises
    ------
    ValueError
        If the bundled guide is missing required columns.
    """
    with resources.files(RESOURCE_PACKAGE).joinpath(GUIDE_TSV).open(
        "r", encoding="utf-8"
    ) as handle:
        guide = pd.read_csv(handle, sep="\t")
    missing = [column for column in _REQUIRED_COLUMNS if column not in guide.columns]
    if missing:
        raise ValueError(f"Bundled CPATK method guide is missing columns: {missing}")
    return guide


def read_ml_nn_method_guide_markdown() -> str:
    """Return the bundled ML/NN method guide markdown text."""
    with resources.files(RESOURCE_PACKAGE).joinpath(GUIDE_MARKDOWN).open(
        "r", encoding="utf-8"
    ) as handle:
        return handle.read()


def export_ml_nn_method_guide(*, output_dir: Path) -> tuple[Path, Path]:
    """Write the bundled method guide to a results directory.

    Parameters
    ----------
    output_dir:
        Directory where the guide copies should be written.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the written TSV and Markdown files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guide = load_ml_nn_method_guide()
    tsv_path = output_dir / GUIDE_TSV
    md_path = output_dir / GUIDE_MARKDOWN
    guide.to_csv(tsv_path, sep="\t", index=False)
    md_path.write_text(read_ml_nn_method_guide_markdown(), encoding="utf-8")
    return tsv_path, md_path


def ml_nn_method_guide_html(*, max_methods: Optional[int] = None) -> str:
    """Render the bundled ML/NN method guide as an HTML report section."""
    guide = load_ml_nn_method_guide()
    if max_methods is not None:
        guide = guide.head(max_methods)
    rows = []
    for _, row in guide.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['method']))}</td>"
            f"<td>{html.escape(str(row['method_group']))}</td>"
            f"<td>{html.escape(str(row['what_it_does']))}</td>"
            f"<td>{html.escape(str(row['best_use_case']))}</td>"
            f"<td>{html.escape(str(row['what_results_mean']))}</td>"
            f"<td>{html.escape(str(row['when_not_to_use']))}</td>"
            f"<td>{html.escape(str(row['main_caveats']))}</td>"
            "</tr>"
        )
    return (
        "<section class='guide'><h2>ML and nearest-neighbour method guide</h2>"
        "<p>This guide is bundled with CPATK and is included in reports to help users decide which "
        "analysis layer is appropriate for their data. It is deliberately cautious: no single ML, "
        "nearest-neighbour or latent-space result should override metadata, replicate QC and batch QC.</p>"
        "<table><thead><tr>"
        "<th>Method</th><th>Group</th><th>What it does</th><th>Best use case</th>"
        "<th>What the results mean</th><th>When not to use</th><th>Main caveats</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )
