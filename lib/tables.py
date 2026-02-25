"""Markdown table generation helpers.

Provides a generic ``md_table()`` function that constructs a Markdown table
from headers, rows, and optional column alignments. Extracted from
sci-trends/report/tables.py where it was the only tool with a dedicated
table-building abstraction; all other tools build tables inline.

Example::

    from lib.tables import md_table

    print(md_table(
        headers=["City", "Pop (M)", "Growth"],
        rows=[["Tokyo", "13.9", "+0.3%"], ["Delhi", "11.0", "+2.1%"]],
        alignments=["l", "r", "r"],
    ))

    # | City | Pop (M) | Growth |
    # | --- | ---: | ---: |
    # | Tokyo | 13.9 | +0.3% |
    # | Delhi | 11.0 | +2.1% |
"""

from __future__ import annotations

from typing import Sequence


def md_table(
    headers: Sequence[str],
    rows: Sequence[Sequence],
    alignments: Sequence[str] | None = None,
) -> str:
    """Build a Markdown table from headers and rows.

    Args:
        headers: Column header strings.
        rows: List of rows, where each row is a sequence of cell values.
            Non-string values are converted via ``str()``.
        alignments: Optional list of alignment codes, one per column:
            ``'l'`` for left (default), ``'r'`` for right, ``'c'`` for center.

    Returns:
        A Markdown-formatted table string, or ``""`` if *rows* is empty.
    """
    if not rows:
        return ""

    n_cols = len(headers)

    # Convert all cells to strings
    str_rows: list[list[str]] = []
    for row in rows:
        str_rows.append([str(c) for c in row])

    # Header row
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")

    # Separator with alignment markers
    if alignments is None:
        alignments = ["l"] * n_cols

    seps: list[str] = []
    for a in alignments:
        if a == "r":
            seps.append("---:")
        elif a == "c":
            seps.append(":---:")
        else:
            seps.append("---")
    lines.append("| " + " | ".join(seps) + " |")

    # Data rows
    for row in str_rows:
        # Pad short rows with empty cells
        while len(row) < n_cols:
            row.append("")
        # Truncate to header count
        lines.append("| " + " | ".join(row[:n_cols]) + " |")

    return "\n".join(lines)
