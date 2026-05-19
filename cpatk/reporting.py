"""HTML report generation for CPATK workflows."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from cpatk.io import data_frame_to_html_table


def make_html_report(
    *,
    title: str,
    output_path: Path,
    summary_tables: Optional[Mapping[str, pd.DataFrame]] = None,
    plot_paths: Optional[Sequence[Path]] = None,
    narrative: Optional[str] = None,
) -> Path:
    """Create a compact HTML analysis report.

    Parameters
    ----------
    title:
        Report title.
    output_path:
        Output HTML path.
    summary_tables:
        Optional named tables to include.
    plot_paths:
        Optional plots to embed or link.
    narrative:
        Optional narrative text for the executive summary.

    Returns
    -------
    pathlib.Path
        Written HTML path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_blocks = []
    for name, table in (summary_tables or {}).items():
        table_blocks.append(
            f"<section><h2>{html.escape(name)}</h2>"
            f"{data_frame_to_html_table(data_frame=table, max_rows=50)}</section>"
        )
    plot_blocks = []
    for path in plot_paths or []:
        path = Path(path)
        relative = path.name
        if path.suffix.lower() == ".svg" and path.exists():
            svg_text = path.read_text(encoding="utf-8", errors="replace")
            plot_blocks.append(
                f"<section><h2>{html.escape(path.stem)}</h2>"
                f"<div class='plot'>{svg_text}</div></section>"
            )
        else:
            plot_blocks.append(
                f"<section><h2>{html.escape(path.stem)}</h2>"
                f"<p><a href='{html.escape(relative)}'>{html.escape(path.name)}</a></p></section>"
            )
    document = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.55; margin: 0; background: #f7f7f7; color: #222; }}
header {{ background: #263238; color: white; padding: 2rem 3rem; }}
main {{ max-width: 1200px; margin: auto; background: white; padding: 2rem 3rem; }}
h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border: 1px solid #d8d8d8; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #eceff1; }}
.plot svg {{ max-width: 100%; height: auto; }}
.notice {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 1rem; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1></header>
<main>
<section><h2>Executive summary</h2><div class='notice'>{html.escape(narrative or 'CPATK report generated successfully.')}</div></section>
{''.join(table_blocks)}
{''.join(plot_blocks)}
</main>
</body>
</html>
"""
    output_path.write_text(data=document, encoding="utf-8")
    return output_path
