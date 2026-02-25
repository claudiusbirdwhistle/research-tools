"""Number formatting helpers for research reports.

Provides consistent formatting for numbers, p-values, significance
levels, percentages, and comma-separated integers across all research
tools. Uses em-dash ('\u2014') as the standard sentinel for missing or
invalid values (None, NaN, inf).

Extracted from duplicate implementations in ocean-warming, climate-trends,
seismicity, attention-gap, sci-trends, uk-grid-decarb, exoplanet-census,
and solar-cycles.

Usage::

    from lib.formatting import fmt, sign, p_str, stars, fmt_pct, fmt_num

    fmt(1.23456)        # '1.235'
    sign(-0.05, 2)      # '-0.05'
    sign(0.12, 2)       # '+0.12'
    p_str(0.0001)       # '<0.001'
    stars(0.003)        # '**'
    fmt_pct(0.85)       # '85.0%'
    fmt_num(12345)      # '12,345'
"""

import math
from typing import Optional, Union

# Sentinel for missing/invalid values
_DASH = "\u2014"

Numeric = Optional[Union[int, float]]


def _is_missing(x: Numeric) -> bool:
    """Check if a value is None, NaN, or infinite."""
    if x is None:
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    return False


def fmt(x: Numeric, decimals: int = 3, comma: bool = False) -> str:
    """Format a number with fixed decimal places.

    Args:
        x: The number to format. Returns em-dash for None/NaN/inf.
        decimals: Number of decimal places (default 3).
        comma: If True, add thousands separators (default False).

    Returns:
        Formatted string, or '\u2014' if the value is missing.

    Examples:
        >>> fmt(1.23456)
        '1.235'
        >>> fmt(None)
        '\u2014'
        >>> fmt(42, 1)
        '42.0'
        >>> fmt(12345.6, 1, comma=True)
        '12,345.6'
    """
    if _is_missing(x):
        return _DASH
    sep = "," if comma else ""
    return f"{x:{sep}.{decimals}f}"


def sign(x: Numeric, decimals: int = 3) -> str:
    """Format a number with an explicit +/\u2212 sign prefix.

    Args:
        x: The number to format. Returns em-dash for None/NaN/inf.
        decimals: Number of decimal places (default 3).

    Returns:
        Formatted string with '+' or '-' prefix, or '\u2014' if missing.

    Examples:
        >>> sign(0.123, 3)
        '+0.123'
        >>> sign(-0.456, 3)
        '-0.456'
        >>> sign(0.0, 2)
        '+0.00'
    """
    if _is_missing(x):
        return _DASH
    return f"+{x:.{decimals}f}" if x >= 0 else f"{x:.{decimals}f}"


def p_str(p: Numeric) -> str:
    """Format a p-value for display.

    Uses tiered precision:
    - p < 0.001 \u2192 '<0.001'
    - 0.001 \u2264 p < 0.01 \u2192 3 decimal places
    - p \u2265 0.01 \u2192 2 decimal places

    Args:
        p: The p-value to format. Returns em-dash for None.

    Returns:
        Formatted p-value string.

    Examples:
        >>> p_str(0.0001)
        '<0.001'
        >>> p_str(0.005)
        '0.005'
        >>> p_str(0.05)
        '0.05'
    """
    if p is None:
        return _DASH
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def stars(p: Numeric) -> str:
    """Return significance stars for a p-value.

    Thresholds (conventional):
    - p < 0.001 \u2192 '***'
    - p < 0.01  \u2192 '**'
    - p < 0.05  \u2192 '*'
    - p \u2265 0.05  \u2192 ''

    Args:
        p: The p-value. Returns '' for None.

    Returns:
        Significance stars string.

    Examples:
        >>> stars(0.0001)
        '***'
        >>> stars(0.005)
        '**'
        >>> stars(0.1)
        ''
    """
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_pct(x: Numeric, decimals: int = 1) -> str:
    """Format a fraction (0\u20131 scale) as a percentage string.

    Args:
        x: The fraction to format (e.g. 0.85 \u2192 '85.0%').
            Returns em-dash for None.
        decimals: Number of decimal places (default 1).

    Returns:
        Percentage string with '%' suffix, or '\u2014' if missing.

    Examples:
        >>> fmt_pct(0.5)
        '50.0%'
        >>> fmt_pct(0.0312, 2)
        '3.12%'
    """
    if x is None:
        return _DASH
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x: Numeric) -> str:
    """Format a number with comma separators.

    Integers get no decimal places; floats get 1 decimal place.

    Args:
        x: The number to format. Returns em-dash for None.

    Returns:
        Comma-separated number string, or '\u2014' if missing.

    Examples:
        >>> fmt_num(1234567)
        '1,234,567'
        >>> fmt_num(1234.5)
        '1,234.5'
    """
    if x is None:
        return _DASH
    if isinstance(x, float):
        return f"{x:,.1f}"
    return f"{x:,}"
