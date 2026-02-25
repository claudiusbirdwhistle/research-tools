"""Synthesis: combine all analyses into summary statistics and hypothesis table."""
import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

def run():
    geo = json.loads((DATA / "analysis" / "geographic.json").read_text())
    tax = json.loads((DATA / "analysis" / "taxonomic.json").read_text())
    temp = json.loads((DATA / "analysis" / "temporal.json").read_text())
    accum = json.loads((DATA / "analysis" / "accumulation.json").read_text())

    # === Hypothesis summary ===
    hypotheses = [
        {
            "id": "H1",
            "name": "Tropical Paradox",
            "statement": "Biodiversity-rich countries have LOW observation density",
            "test": "Spearman correlation (density vs species richness)",
            "rho": geo["h1_tropical_paradox"]["spearman_rho"],
            "p_value": geo["h1_tropical_paradox"]["p_value"],
            "n": geo["h1_tropical_paradox"]["n_countries"],
            "supported": geo["h1_tropical_paradox"]["supported"],
            "interpretation": geo["h1_tropical_paradox"]["interpretation"],
        },
        {
            "id": "H2",
            "name": "Geographic Inequality",
            "statement": "Citizen science INCREASED geographic inequality",
            "test": "OLS + Mann-Kendall on Gini coefficient (2000-2024)",
            "slope": temp["h2_geographic_inequality"].get("ols_slope_per_year"),
            "p_value": temp["h2_geographic_inequality"].get("ols_p_value"),
            "gini_change": f"{temp['h2_geographic_inequality'].get('gini_2000', '?')} → {temp['h2_geographic_inequality'].get('gini_2024', '?')}",
            "supported": temp["h2_geographic_inequality"]["supported"],
            "interpretation": temp["h2_geographic_inequality"].get("interpretation", ""),
        },
        {
            "id": "H3",
            "name": "Bird Bias Worsening",
            "statement": "Bird proportion is increasing over time",
            "test": "OLS + Mann-Kendall on bird share (2000-2024)",
            "slope": tax["h3_bird_bias"].get("ols_slope_pct_per_year"),
            "p_value": tax["h3_bird_bias"].get("ols_p_value"),
            "change": f"{tax['h3_bird_bias'].get('bird_share_2000', '?')}% → {tax['h3_bird_bias'].get('bird_share_2024', '?')}%",
            "supported": tax["h3_bird_bias"]["supported"],
            "interpretation": f"Bird share changed by {tax['h3_bird_bias'].get('change', '?')} percentage points",
        },
        {
            "id": "H4",
            "name": "Differential Saturation",
            "statement": "Bird species saturating, insects still growing",
            "test": "Growth rate comparison across 3 countries (2005-2024)",
            "n_countries": accum["h4_differential_saturation"]["n_countries"],
            "supported": accum["h4_differential_saturation"]["supported"],
            "interpretation": accum["h4_differential_saturation"]["interpretation"],
        },
        {
            "id": "H5",
            "name": "GBIF Membership Effect",
            "statement": "GBIF members have higher per-capita observation rates",
            "test": "Mann-Whitney U (member vs non-member per-capita rates)",
            "p_value": geo["h5_membership_effect"]["p_value"],
            "ratio": geo["h5_membership_effect"]["ratio"],
            "supported": geo["h5_membership_effect"]["supported"],
            "interpretation": f"Member median: {geo['h5_membership_effect']['median_member_per_capita']}, Non-member: {geo['h5_membership_effect']['median_nonmember_per_capita']} ({geo['h5_membership_effect']['ratio']}x ratio)",
        },
    ]

    supported_count = sum(1 for h in hypotheses if h["supported"])

    # === Key findings ===
    findings = [
        f"GBIF contains {geo['summary']['total_records']:,} occurrence records across {geo['summary']['n_countries']} countries",
        f"Top 1 country ({geo['summary']['top1_country']}) holds {100*geo['summary']['top1_share']:.1f}% of all records; top 10 hold {100*geo['summary']['top10_share']:.1f}%",
        f"Observation volume grew {temp['growth'].get('fold_increase', '?')}x since 2000 (doubling every {temp['growth'].get('doubling_time_years', '?')} years)",
        f"Citizen science (HUMAN_OBSERVATION) accounts for {temp['citizen_science'][-1]['pct'] if temp['citizen_science'] else '?'}% of all records in 2024",
        f"Animalia dominates at {tax['kingdom_proportions'][0]['pct']}% of records, with Aves alone at ~{tax['bird_share_timeseries'][-1]['bird_share'] if tax['bird_share_timeseries'] else '?'}%",
        f"{supported_count}/5 hypotheses supported by data",
    ]

    # === Citizen science by country ===
    # Load country basis data
    cs_by_country = []
    master = json.loads((DATA / "raw" / "master_results.json").read_text())
    for cc in master["top30_countries"][:20]:
        try:
            basis = json.loads((DATA / "raw" / f"country_{cc}_basis.json").read_text())
            ho = next((n for name, n in basis if name == "HUMAN_OBSERVATION"), 0)
            total = sum(n for _, n in basis)
            if total > 0:
                cs_by_country.append({"country": cc, "human_obs_pct": round(100 * ho / total, 1), "total": total})
        except:
            pass

    result = {
        "hypotheses": hypotheses,
        "supported_count": supported_count,
        "total_hypotheses": 5,
        "key_findings": findings,
        "citizen_science_by_country": sorted(cs_by_country, key=lambda x: x["human_obs_pct"], reverse=True),
    }

    out = DATA / "analysis" / "synthesis.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Synthesis saved to {out}")
    print(f"  {supported_count}/5 hypotheses supported")
    for h in hypotheses:
        status = "SUPPORTED" if h["supported"] else "NOT SUPPORTED"
        print(f"  {h['id']}: {status} — {h['name']}")
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
