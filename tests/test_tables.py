"""Tests for lib.tables — Markdown table generation.

RED phase: these tests must FAIL before the module exists.
"""

import pytest

from lib.tables import md_table


class TestMdTableBasic:
    """Basic table generation."""

    def test_simple_two_column(self):
        result = md_table(["Name", "Value"], [["Alice", "10"], ["Bob", "20"]])
        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows
        assert lines[0] == "| Name | Value |"
        assert "---" in lines[1]
        assert lines[2] == "| Alice | 10 |"
        assert lines[3] == "| Bob | 20 |"

    def test_single_row(self):
        result = md_table(["X"], [["1"]])
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "| X |"
        assert lines[2] == "| 1 |"

    def test_empty_rows_returns_empty(self):
        result = md_table(["A", "B"], [])
        assert result == ""

    def test_numeric_cells_converted_to_string(self):
        result = md_table(["Count"], [[42]])
        assert "| 42 |" in result

    def test_none_cell_converted_to_string(self):
        result = md_table(["Val"], [[None]])
        assert "| None |" in result

    def test_three_columns(self):
        result = md_table(
            ["A", "B", "C"],
            [["1", "2", "3"], ["4", "5", "6"]],
        )
        lines = result.split("\n")
        assert lines[0] == "| A | B | C |"
        assert lines[2] == "| 1 | 2 | 3 |"
        assert lines[3] == "| 4 | 5 | 6 |"


class TestMdTableAlignment:
    """Column alignment support."""

    def test_left_alignment_default(self):
        result = md_table(["Col"], [["x"]])
        sep_line = result.split("\n")[1]
        assert "---" in sep_line
        # Left-aligned: no colon on right
        assert ":---:" not in sep_line
        assert "---:" not in sep_line

    def test_right_alignment(self):
        result = md_table(["Num"], [["42"]], alignments=["r"])
        sep_line = result.split("\n")[1]
        assert "---:" in sep_line

    def test_center_alignment(self):
        result = md_table(["Mid"], [["x"]], alignments=["c"])
        sep_line = result.split("\n")[1]
        assert ":---:" in sep_line

    def test_mixed_alignment(self):
        result = md_table(
            ["Name", "Count", "Status"],
            [["A", "1", "OK"]],
            alignments=["l", "r", "c"],
        )
        sep_line = result.split("\n")[1]
        parts = [p.strip() for p in sep_line.split("|") if p.strip()]
        assert parts[0] == "---"       # left
        assert parts[1] == "---:"      # right
        assert parts[2] == ":---:"     # center


class TestMdTableEdgeCases:
    """Edge cases and robustness."""

    def test_row_shorter_than_headers_padded(self):
        result = md_table(["A", "B", "C"], [["1"]])
        data_line = result.split("\n")[2]
        # Split by '|' and take inner cells (skip leading/trailing empties from split)
        parts = [p.strip() for p in data_line.split("|")[1:-1]]
        assert len(parts) == 3
        assert parts[0] == "1"
        assert parts[1] == ""
        assert parts[2] == ""

    def test_row_longer_than_headers_truncated(self):
        result = md_table(["A"], [["1", "2", "3"]])
        data_line = result.split("\n")[2]
        parts = [p.strip() for p in data_line.split("|") if p.strip()]
        assert len(parts) == 1
        assert parts[0] == "1"

    def test_pipe_character_in_cell(self):
        """Cells with pipe characters should still produce valid table."""
        result = md_table(["Data"], [["a|b"]])
        # The function doesn't escape pipes — this documents current behavior
        assert "a|b" in result

    def test_many_rows(self):
        rows = [[str(i), str(i * 2)] for i in range(100)]
        result = md_table(["X", "Y"], rows)
        lines = result.split("\n")
        assert len(lines) == 102  # header + separator + 100 data rows

    def test_multiline_result_is_valid_markdown(self):
        result = md_table(["H1", "H2"], [["a", "b"]])
        lines = result.split("\n")
        for line in lines:
            assert line.startswith("|")
            assert line.endswith("|")
