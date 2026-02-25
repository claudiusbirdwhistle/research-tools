"""OpenAlex API client for scientific publication trend analysis."""

from .client import OpenAlexClient
from lib.cache import ResponseCache
from .models import Field, Topic, CountryStats, WorkSummary, GroupResult

__all__ = [
    "OpenAlexClient",
    "ResponseCache",
    "Field",
    "Topic",
    "CountryStats",
    "WorkSummary",
    "GroupResult",
]
