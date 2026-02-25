"""Ocean warming report generation package.

Provides generate_report() to produce a comprehensive Markdown report
from ocean warming analysis results (trends, acceleration, ENSO, and
ocean-atmosphere comparison).
"""

from .generator import generate_report, generate_summary_json

__all__ = ["generate_report", "generate_summary_json"]
