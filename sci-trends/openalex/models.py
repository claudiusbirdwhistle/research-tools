"""Data models for OpenAlex entities."""

from dataclasses import dataclass, field


@dataclass
class GroupResult:
    """A single result from a group_by aggregation query."""
    key: str
    key_display_name: str
    count: int


@dataclass
class Field:
    """An OpenAlex field of study (top-level domain)."""
    id: str
    display_name: str
    works_count: int = 0
    description: str = ""
    subfields: list[str] = field(default_factory=list)


@dataclass
class Topic:
    """An OpenAlex research topic."""
    id: str
    display_name: str
    works_count: int = 0
    field_id: str = ""
    field_name: str = ""
    subfield_id: str = ""
    subfield_name: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class CountryStats:
    """Publication statistics for a country."""
    country_code: str
    country_name: str
    works_count: int = 0
    share: float = 0.0


@dataclass
class WorkSummary:
    """Summary of a single published work."""
    id: str
    title: str
    publication_year: int
    cited_by_count: int = 0
    doi: str = ""
    primary_topic: str = ""
    source_name: str = ""
    authors: list[str] = field(default_factory=list)
