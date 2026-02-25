"""Field-level publication trend analysis (2015-2024).

Collects publication counts for all OpenAlex fields across years,
computes CAGR, absolute growth, and pre/post-COVID structural shifts.

Usage:
    from analysis.field_trends import collect_field_trends, compute_metrics
    data = collect_field_trends(client)
    metrics = compute_metrics(data)
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openalex import OpenAlexClient

logger = logging.getLogger(__name__)

YEAR_START = 2015
YEAR_END = 2024
CAGR_5Y_START = 2019
CAGR_10Y_START = 2015
COVID_SPLIT_YEAR = 2020  # pre: 2015-2019, post: 2020-2024

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class FieldYearData:
    """Publication counts per year for a field."""
    field_id: str
    field_name: str
    year_counts: dict[int, int] = field(default_factory=dict)  # {year: count}


@dataclass
class FieldMetrics:
    """Computed growth metrics for a field."""
    field_id: str
    field_name: str
    year_counts: dict[int, int]
    total_2024: int = 0
    total_2019: int = 0
    total_2015: int = 0
    cagr_5y: float | None = None   # 2019->2024
    cagr_10y: float | None = None  # 2015->2024
    abs_growth_5y: int = 0         # 2024 - 2019
    abs_growth_10y: int = 0        # 2024 - 2015
    pre_covid_cagr: float | None = None   # 2015->2019
    post_covid_cagr: float | None = None  # 2020->2024
    acceleration: float | None = None     # post_covid_cagr - pre_covid_cagr
    share_2024: float = 0.0        # fraction of global output in 2024


def _cagr(start_val: int, end_val: int, years: int) -> float | None:
    """Compute Compound Annual Growth Rate. Returns None if inputs invalid."""
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return None
    return (end_val / start_val) ** (1 / years) - 1


def collect_field_trends(client: OpenAlexClient) -> tuple[list[FieldYearData], dict[int, int]]:
    """Collect year-by-year publication counts for each field and globally.

    Returns:
        (field_data_list, global_year_counts)
    """
    # Step 1: Get list of all fields
    logger.info("Fetching field list...")
    fields = client.get_fields()
    logger.info("Found %d fields", len(fields))

    # Step 2: For each field, get publication counts by year (2015-2024)
    field_data = []
    for i, f in enumerate(fields):
        # Extract the short field id (e.g., "fields/17" from the full URL)
        # Full URL looks like "https://openalex.org/fields/17"
        # API filter needs just "fields/17" (not the full URL)
        fid = f.id
        if fid.startswith("https://"):
            # Extract "fields/17" from "https://openalex.org/fields/17"
            parts = fid.split("openalex.org/")
            fid = parts[-1] if len(parts) > 1 else fid

        logger.info("[%d/%d] Fetching year data for %s (%s)...",
                     i + 1, len(fields), f.display_name, fid)

        groups = client.get_grouped(
            "/works",
            filters={"primary_topic.field.id": fid},
            group_by="publication_year",
        )

        year_counts = {}
        for g in groups:
            try:
                year = int(g.key)
            except (ValueError, TypeError):
                continue
            if YEAR_START <= year <= YEAR_END:
                year_counts[year] = g.count

        fd = FieldYearData(
            field_id=fid,
            field_name=f.display_name,
            year_counts=year_counts,
        )
        field_data.append(fd)

    # Step 3: Get global publication counts by year
    logger.info("Fetching global year counts...")
    global_groups = client.get_grouped("/works", group_by="publication_year")
    global_year_counts = {}
    for g in global_groups:
        try:
            year = int(g.key)
        except (ValueError, TypeError):
            continue
        if YEAR_START <= year <= YEAR_END:
            global_year_counts[year] = g.count

    logger.info("Data collection complete. %d fields, %d years of global data.",
                 len(field_data), len(global_year_counts))
    return field_data, global_year_counts


def compute_metrics(
    field_data: list[FieldYearData],
    global_year_counts: dict[int, int],
) -> list[FieldMetrics]:
    """Compute growth metrics for each field."""
    global_2024 = global_year_counts.get(2024, 1)  # avoid div/0

    metrics = []
    for fd in field_data:
        yc = fd.year_counts
        t2024 = yc.get(2024, 0)
        t2019 = yc.get(2019, 0)
        t2015 = yc.get(2015, 0)
        t2020 = yc.get(2020, 0)

        m = FieldMetrics(
            field_id=fd.field_id,
            field_name=fd.field_name,
            year_counts=dict(sorted(yc.items())),
            total_2024=t2024,
            total_2019=t2019,
            total_2015=t2015,
            cagr_5y=_cagr(t2019, t2024, 5),
            cagr_10y=_cagr(t2015, t2024, 9),  # 2015->2024 = 9 years
            abs_growth_5y=t2024 - t2019,
            abs_growth_10y=t2024 - t2015,
            pre_covid_cagr=_cagr(t2015, t2019, 4),   # 2015->2019 = 4 years
            post_covid_cagr=_cagr(t2020, t2024, 4),   # 2020->2024 = 4 years
            share_2024=t2024 / global_2024 if global_2024 > 0 else 0,
        )

        # Acceleration = post_covid_cagr - pre_covid_cagr
        if m.pre_covid_cagr is not None and m.post_covid_cagr is not None:
            m.acceleration = m.post_covid_cagr - m.pre_covid_cagr

        metrics.append(m)

    # Sort by 5-year CAGR descending
    metrics.sort(key=lambda m: m.cagr_5y if m.cagr_5y is not None else -999, reverse=True)
    return metrics


def save_results(
    metrics: list[FieldMetrics],
    global_year_counts: dict[int, int],
    output_path: Path | None = None,
):
    """Save analysis results to JSON."""
    if output_path is None:
        output_path = DATA_DIR / "field_trends.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "year_range": [YEAR_START, YEAR_END],
        "global_year_counts": {str(k): v for k, v in sorted(global_year_counts.items())},
        "fields": [],
    }

    for m in metrics:
        field_entry = {
            "field_id": m.field_id,
            "field_name": m.field_name,
            "year_counts": {str(k): v for k, v in sorted(m.year_counts.items())},
            "total_2024": m.total_2024,
            "total_2019": m.total_2019,
            "total_2015": m.total_2015,
            "cagr_5y": round(m.cagr_5y, 6) if m.cagr_5y is not None else None,
            "cagr_10y": round(m.cagr_10y, 6) if m.cagr_10y is not None else None,
            "abs_growth_5y": m.abs_growth_5y,
            "abs_growth_10y": m.abs_growth_10y,
            "pre_covid_cagr": round(m.pre_covid_cagr, 6) if m.pre_covid_cagr is not None else None,
            "post_covid_cagr": round(m.post_covid_cagr, 6) if m.post_covid_cagr is not None else None,
            "acceleration": round(m.acceleration, 6) if m.acceleration is not None else None,
            "share_2024": round(m.share_2024, 6),
        }
        data["fields"].append(field_entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved field trends to %s", output_path)
    return output_path


def format_summary_table(metrics: list[FieldMetrics]) -> str:
    """Format a summary table of field metrics as Markdown."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = ["Rank", "Field", "2024 Works", "5yr CAGR", "10yr CAGR",
               "Pre-COVID", "Post-COVID", "Accel.", "Share"]
    rows = []
    for i, m in enumerate(metrics, 1):
        rows.append([
            i,
            m.field_name,
            f"{m.total_2024:,}",
            f"{m.cagr_5y*100:.1f}%" if m.cagr_5y is not None else "N/A",
            f"{m.cagr_10y*100:.1f}%" if m.cagr_10y is not None else "N/A",
            f"{m.pre_covid_cagr*100:.1f}%" if m.pre_covid_cagr is not None else "N/A",
            f"{m.post_covid_cagr*100:.1f}%" if m.post_covid_cagr is not None else "N/A",
            f"{m.acceleration*100:+.1f}pp" if m.acceleration is not None else "N/A",
            f"{m.share_2024*100:.1f}%",
        ])

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    else:
        # Fallback: simple pipe table
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def run(client: OpenAlexClient | None = None) -> list[FieldMetrics]:
    """Main entry point: collect data, compute metrics, save results."""
    close_client = False
    if client is None:
        from openalex.cache import ResponseCache
        cache = ResponseCache()
        client = OpenAlexClient(cache=cache)
        close_client = True

    try:
        field_data, global_year_counts = collect_field_trends(client)
        metrics = compute_metrics(field_data, global_year_counts)
        save_results(metrics, global_year_counts)

        # Print summary
        print("\n" + format_summary_table(metrics))
        print(f"\nGlobal totals (2024): {global_year_counts.get(2024, 'N/A'):,}")
        print(f"Global totals (2015): {global_year_counts.get(2015, 'N/A'):,}")
        g_cagr = _cagr(global_year_counts.get(2015, 0), global_year_counts.get(2024, 0), 9)
        if g_cagr:
            print(f"Global 10yr CAGR: {g_cagr*100:.1f}%")

        stats = client.stats()
        print(f"\nAPI stats: {stats['requests_made']} requests, {stats['cache_hits']} cache hits")

        return metrics
    finally:
        if close_client:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
