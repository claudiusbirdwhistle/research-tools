"""Geographic distribution analysis of global research output (2015-2024).

Analyzes how research publication volume is distributed across countries,
how shares are shifting over time, and what fields each country specializes in.

Method:
1. For each year 2015-2024: query group_by=authorships.countries (10 calls)
2. Build per-country time series of publication counts
3. Compute shares, rank changes, CAGR for top countries
4. For top-10 countries: break down by field to identify specializations (10 calls)

Usage:
    from analysis.geography import run
    results = run()
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
DATA_DIR = Path(__file__).parent.parent / "data"

# Top N countries for detailed analysis
TOP_N_RANKING = 20    # for ranking table
TOP_N_FIELDS = 10     # for field specialization breakdown


@dataclass
class CountryTimeSeries:
    """Publication counts per year for a country."""
    country_code: str
    country_name: str
    year_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class CountryMetrics:
    """Computed metrics for a country's research output."""
    country_code: str
    country_name: str
    year_counts: dict[int, int]
    total_2024: int = 0
    total_2015: int = 0
    total_2019: int = 0
    share_2024: float = 0.0
    share_2015: float = 0.0
    share_change: float = 0.0   # share_2024 - share_2015
    rank_2024: int = 0
    rank_2015: int = 0
    rank_change: int = 0        # rank_2015 - rank_2024 (positive = improved)
    cagr_5y: float | None = None    # 2019->2024
    cagr_10y: float | None = None   # 2015->2024


@dataclass
class CountryFieldProfile:
    """Field breakdown for a country's 2024 output."""
    country_code: str
    country_name: str
    field_counts: dict[str, int] = field(default_factory=dict)    # {field_name: count}
    field_shares: dict[str, float] = field(default_factory=dict)  # {field_name: share}
    top_fields: list[tuple[str, float]] = field(default_factory=list)  # [(field_name, share), ...]
    specialization_index: dict[str, float] = field(default_factory=dict)  # {field: RCA}


def _cagr(start_val: int, end_val: int, years: int) -> float | None:
    """Compound Annual Growth Rate. Returns None if inputs invalid."""
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return None
    return (end_val / start_val) ** (1 / years) - 1


def _shorten_id(full_id: str) -> str:
    """Convert full OpenAlex URL to short ID.

    'https://openalex.org/fields/17' -> 'fields/17'
    """
    if full_id.startswith("https://"):
        parts = full_id.split("openalex.org/")
        return parts[-1] if len(parts) > 1 else full_id
    return full_id


def collect_country_data(client: OpenAlexClient) -> list[CountryTimeSeries]:
    """Collect per-country publication counts for each year 2015-2024.

    Uses group_by=authorships.countries which returns up to 200 countries.
    This captures all significant research nations.

    Returns:
        List of CountryTimeSeries, one per country seen in any year.
    """
    # Collect per-year country data
    all_country_data: dict[str, CountryTimeSeries] = {}

    for year in range(YEAR_START, YEAR_END + 1):
        logger.info("Fetching country data for %d...", year)
        groups = client.get_grouped(
            "/works",
            filters={"publication_year": str(year)},
            group_by="authorships.countries",
        )

        for g in groups:
            # Key is a full URL: https://openalex.org/countries/CN
            raw_key = g.key or ""
            if raw_key.startswith("https://"):
                code = raw_key.split("/")[-1].upper()
            else:
                code = raw_key.upper()
            if not code or len(code) != 2:
                # Skip invalid country codes
                continue

            if code not in all_country_data:
                all_country_data[code] = CountryTimeSeries(
                    country_code=code,
                    country_name=g.key_display_name or code,
                    year_counts={},
                )

            all_country_data[code].year_counts[year] = g.count

    countries = list(all_country_data.values())
    logger.info("Collected data for %d countries across %d years",
                len(countries), YEAR_END - YEAR_START + 1)
    return countries


def compute_country_metrics(
    countries: list[CountryTimeSeries],
    global_year_counts: dict[int, int] | None = None,
) -> list[CountryMetrics]:
    """Compute growth and share metrics for each country.

    Args:
        countries: Raw per-country time series data
        global_year_counts: Optional pre-computed global totals. If None,
            will sum from country data (note: country totals can exceed
            global works because multi-country collaborations are counted
            for each contributing country).
    """
    # Compute totals per year from country data if not provided
    if global_year_counts is None:
        # Use the sum of country counts as a proxy
        # Note: this overcounts due to multi-country papers
        global_year_counts = {}
        for year in range(YEAR_START, YEAR_END + 1):
            total = sum(c.year_counts.get(year, 0) for c in countries)
            global_year_counts[year] = total

    global_2024 = global_year_counts.get(2024, 1)
    global_2015 = global_year_counts.get(2015, 1)

    # Build metrics for each country
    metrics = []
    for c in countries:
        yc = c.year_counts
        t2024 = yc.get(2024, 0)
        t2015 = yc.get(2015, 0)
        t2019 = yc.get(2019, 0)

        m = CountryMetrics(
            country_code=c.country_code,
            country_name=c.country_name,
            year_counts=dict(sorted(yc.items())),
            total_2024=t2024,
            total_2015=t2015,
            total_2019=t2019,
            share_2024=t2024 / global_2024 if global_2024 > 0 else 0,
            share_2015=t2015 / global_2015 if global_2015 > 0 else 0,
            cagr_5y=_cagr(t2019, t2024, 5),
            cagr_10y=_cagr(t2015, t2024, 9),  # 2015->2024 = 9 years
        )
        m.share_change = m.share_2024 - m.share_2015
        metrics.append(m)

    # Compute ranks for 2024 and 2015
    by_2024 = sorted(metrics, key=lambda m: m.total_2024, reverse=True)
    for i, m in enumerate(by_2024, 1):
        m.rank_2024 = i

    by_2015 = sorted(metrics, key=lambda m: m.total_2015, reverse=True)
    for i, m in enumerate(by_2015, 1):
        m.rank_2015 = i

    for m in metrics:
        m.rank_change = m.rank_2015 - m.rank_2024  # positive = moved up

    # Sort final result by 2024 rank
    metrics.sort(key=lambda m: m.rank_2024)
    return metrics


def collect_field_profiles(
    client: OpenAlexClient,
    top_countries: list[CountryMetrics],
    global_field_shares: dict[str, float] | None = None,
) -> list[CountryFieldProfile]:
    """Collect field breakdown for top countries.

    For each country, queries group_by=primary_topic.field.id filtered
    by country and year 2024 to get the distribution across fields.

    Also computes Revealed Comparative Advantage (RCA) as a specialization
    index: RCA = (country's field share) / (world's field share).
    RCA > 1 means the country is specialized in that field.

    Args:
        client: OpenAlex API client
        top_countries: List of countries to profile
        global_field_shares: {field_name: share_of_global_2024_output}.
            If None, will be computed from a global query.
    """
    # Get global field distribution if not provided
    if global_field_shares is None:
        logger.info("Fetching global field distribution for 2024...")
        global_groups = client.get_grouped(
            "/works",
            filters={"publication_year": "2024"},
            group_by="primary_topic.field.id",
        )
        total = sum(g.count for g in global_groups)
        global_field_shares = {}
        for g in global_groups:
            global_field_shares[g.key_display_name] = g.count / total if total > 0 else 0

    profiles = []
    for i, cm in enumerate(top_countries):
        logger.info("[%d/%d] Fetching field profile for %s (%s)...",
                    i + 1, len(top_countries), cm.country_name, cm.country_code)

        groups = client.get_grouped(
            "/works",
            filters={
                "authorships.countries": cm.country_code,
                "publication_year": "2024",
            },
            group_by="primary_topic.field.id",
        )

        total_country = sum(g.count for g in groups)
        field_counts = {}
        field_shares = {}
        for g in groups:
            field_counts[g.key_display_name] = g.count
            field_shares[g.key_display_name] = g.count / total_country if total_country > 0 else 0

        # Top fields by share
        top_fields = sorted(field_shares.items(), key=lambda x: x[1], reverse=True)[:5]

        # Revealed Comparative Advantage
        rca = {}
        for fname, fshare in field_shares.items():
            global_share = global_field_shares.get(fname, 0)
            if global_share > 0:
                rca[fname] = fshare / global_share
            else:
                rca[fname] = 0.0

        profiles.append(CountryFieldProfile(
            country_code=cm.country_code,
            country_name=cm.country_name,
            field_counts=field_counts,
            field_shares=field_shares,
            top_fields=top_fields,
            specialization_index=rca,
        ))

    return profiles


def find_rising_countries(metrics: list[CountryMetrics], min_works_2024: int = 10000) -> list[CountryMetrics]:
    """Identify countries with fastest growth rates (among significant producers).

    Filters to countries with at least min_works_2024 publications in 2024,
    then sorts by 5-year CAGR.
    """
    significant = [m for m in metrics
                   if m.total_2024 >= min_works_2024 and m.cagr_5y is not None]
    return sorted(significant, key=lambda m: m.cagr_5y, reverse=True)


def save_results(
    metrics: list[CountryMetrics],
    profiles: list[CountryFieldProfile],
    global_year_counts: dict[int, int],
    output_path: Path | None = None,
):
    """Save geographic analysis results to JSON."""
    if output_path is None:
        output_path = DATA_DIR / "geography.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "year_range": [YEAR_START, YEAR_END],
        "global_year_counts": {str(k): v for k, v in sorted(global_year_counts.items())},
        "total_countries": len(metrics),
        "top_20_rankings": [],
        "field_profiles": [],
        "rising_countries": [],
    }

    # Top 20 by 2024 rank
    for m in metrics[:TOP_N_RANKING]:
        entry = {
            "country_code": m.country_code,
            "country_name": m.country_name,
            "rank_2024": m.rank_2024,
            "rank_2015": m.rank_2015,
            "rank_change": m.rank_change,
            "total_2024": m.total_2024,
            "total_2015": m.total_2015,
            "share_2024": round(m.share_2024, 6),
            "share_2015": round(m.share_2015, 6),
            "share_change_pp": round(m.share_change * 100, 2),
            "cagr_5y": round(m.cagr_5y, 6) if m.cagr_5y is not None else None,
            "cagr_10y": round(m.cagr_10y, 6) if m.cagr_10y is not None else None,
            "year_counts": {str(k): v for k, v in sorted(m.year_counts.items())},
        }
        data["top_20_rankings"].append(entry)

    # Field profiles for top countries
    for p in profiles:
        top_specializations = sorted(
            p.specialization_index.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        entry = {
            "country_code": p.country_code,
            "country_name": p.country_name,
            "top_fields": [
                {"field": f, "share": round(s * 100, 1)}
                for f, s in p.top_fields
            ],
            "top_specializations": [
                {"field": f, "rca": round(rca, 2)}
                for f, rca in top_specializations
            ],
            "all_field_shares": {
                f: round(s * 100, 2)
                for f, s in sorted(p.field_shares.items(), key=lambda x: x[1], reverse=True)
            },
        }
        data["field_profiles"].append(entry)

    # Rising countries (fastest CAGR among significant producers)
    rising = find_rising_countries(metrics)[:15]
    for m in rising:
        entry = {
            "country_code": m.country_code,
            "country_name": m.country_name,
            "total_2024": m.total_2024,
            "total_2015": m.total_2015,
            "cagr_5y": round(m.cagr_5y, 6) if m.cagr_5y is not None else None,
            "cagr_10y": round(m.cagr_10y, 6) if m.cagr_10y is not None else None,
            "rank_2024": m.rank_2024,
            "rank_change": m.rank_change,
        }
        data["rising_countries"].append(entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved geography analysis to %s", output_path)
    return output_path


def format_ranking_table(metrics: list[CountryMetrics], top_n: int = 20) -> str:
    """Format top countries as a Markdown table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = ["Rank", "Country", "2024 Works", "2015 Works", "Rank Chg",
               "Share '24", "Share Chg", "5yr CAGR"]
    rows = []
    for m in metrics[:top_n]:
        rank_chg = ""
        if m.rank_change > 0:
            rank_chg = f"+{m.rank_change}"
        elif m.rank_change < 0:
            rank_chg = str(m.rank_change)
        else:
            rank_chg = "="

        rows.append([
            m.rank_2024,
            f"{m.country_name} ({m.country_code})",
            f"{m.total_2024:,}",
            f"{m.total_2015:,}",
            rank_chg,
            f"{m.share_2024 * 100:.1f}%",
            f"{m.share_change * 100:+.1f}pp",
            f"{m.cagr_5y * 100:.1f}%" if m.cagr_5y is not None else "N/A",
        ])

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    else:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def format_specialization_table(profiles: list[CountryFieldProfile]) -> str:
    """Format field specialization profiles as a Markdown table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = ["Country", "Top Field (Share)", "#2 Field", "#3 Field",
               "Top Specialization (RCA)"]
    rows = []
    for p in profiles:
        top3 = p.top_fields[:3]
        top_rca = sorted(
            p.specialization_index.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:1]

        cols = [f"{p.country_name} ({p.country_code})"]
        for i in range(3):
            if i < len(top3):
                fname, share = top3[i]
                # Abbreviate long field names
                fname_short = fname[:20] + "..." if len(fname) > 23 else fname
                cols.append(f"{fname_short} ({share * 100:.0f}%)")
            else:
                cols.append("—")

        if top_rca:
            rca_field, rca_val = top_rca[0]
            rca_short = rca_field[:18] + "..." if len(rca_field) > 21 else rca_field
            cols.append(f"{rca_short} ({rca_val:.1f}x)")
        else:
            cols.append("—")

        rows.append(cols)

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    else:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def format_rising_table(metrics: list[CountryMetrics], top_n: int = 15) -> str:
    """Format fastest-growing research nations as a Markdown table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    rising = find_rising_countries(metrics)[:top_n]

    headers = ["Rank", "Country", "5yr CAGR", "10yr CAGR",
               "2024 Works", "2015 Works", "Growth"]
    rows = []
    for i, m in enumerate(rising, 1):
        growth_factor = m.total_2024 / m.total_2015 if m.total_2015 > 0 else float('inf')
        rows.append([
            i,
            f"{m.country_name} ({m.country_code})",
            f"{m.cagr_5y * 100:.1f}%" if m.cagr_5y is not None else "N/A",
            f"{m.cagr_10y * 100:.1f}%" if m.cagr_10y is not None else "N/A",
            f"{m.total_2024:,}",
            f"{m.total_2015:,}",
            f"{growth_factor:.1f}x",
        ])

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    else:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def run(client: OpenAlexClient | None = None) -> dict:
    """Main entry point: collect data, compute metrics, save results.

    Returns dict with 'metrics', 'profiles', 'global_year_counts' keys.
    """
    close_client = False
    if client is None:
        from openalex.cache import ResponseCache
        cache = ResponseCache()
        client = OpenAlexClient(cache=cache)
        close_client = True

    try:
        # Step 1: Collect per-country per-year data
        countries = collect_country_data(client)

        # Step 2: Get global year counts for share computation
        # Use actual global counts (not sum of countries, which overcounts due to collabs)
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

        # Step 3: Compute metrics
        metrics = compute_country_metrics(countries, global_year_counts)

        # Step 4: Field profiles for top countries
        top_for_fields = metrics[:TOP_N_FIELDS]
        profiles = collect_field_profiles(client, top_for_fields)

        # Step 5: Save results
        save_results(metrics, profiles, global_year_counts)

        # Print summaries
        print("\n=== TOP 20 RESEARCH NATIONS (2024) ===\n")
        print(format_ranking_table(metrics))

        print("\n\n=== FIELD SPECIALIZATIONS (TOP 10 COUNTRIES) ===\n")
        print(format_specialization_table(profiles))

        print("\n\n=== FASTEST-GROWING RESEARCH NATIONS ===\n")
        print(format_rising_table(metrics))

        # Key stats
        if metrics:
            top = metrics[0]
            print(f"\n--- Key Stats ---")
            print(f"Total countries tracked: {len(metrics)}")
            print(f"#1 by output (2024): {top.country_name} ({top.total_2024:,} works)")
            print(f"Global 2024 total: {global_year_counts.get(2024, 0):,}")
            print(f"Global 2015 total: {global_year_counts.get(2015, 0):,}")

            # Country attribution coverage
            country_sum_2024 = sum(m.total_2024 for m in metrics)
            coverage = country_sum_2024 / global_year_counts.get(2024, 1)
            if coverage > 1.0:
                print(f"Country sum / global ratio: {coverage:.2f}x "
                      f"(>1 due to multi-country collaborations)")
            else:
                print(f"Country attribution coverage: {coverage:.0%} "
                      f"({100 - coverage * 100:.0f}% of works lack country data)")

        stats = client.stats()
        print(f"\nAPI stats: {stats['requests_made']} requests, "
              f"{stats['cache_hits']} cache hits")

        return {
            "metrics": metrics,
            "profiles": profiles,
            "global_year_counts": global_year_counts,
        }

    finally:
        if close_client:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run()
