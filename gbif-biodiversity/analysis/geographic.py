"""Geographic bias analysis: observation density, H1 (tropical paradox), H5 (membership effect)."""
import json, math
from pathlib import Path
from scipy import stats
import numpy as np

DATA = Path(__file__).parent.parent / "data"

def load(name):
    return json.loads((DATA / "raw" / f"{name}.json").read_text())

def run():
    from data.country_metadata import COUNTRIES
    from data.species_richness import SPECIES_RICHNESS

    # Load data
    country_counts = load("global_country")  # [(code, count), ...]
    gbif_members = load("gbif_members")
    taxa_animalia = load("taxa_animalia_country")
    taxa_plantae = load("taxa_plantae_country")
    taxa_insecta = load("taxa_insecta_country")
    taxa_fungi = load("taxa_fungi_country")

    count_map = {c: n for c, n in country_counts}

    # === Per-country metrics ===
    country_metrics = []
    for cc, meta in COUNTRIES.items():
        count = count_map.get(cc, 0)
        if count == 0:
            continue
        area = meta["area_km2"]
        pop = meta["pop"]
        density = count / area if area > 0 else 0
        per_capita = count / pop if pop > 0 else 0
        country_metrics.append({
            "code": cc,
            "name": meta["name"],
            "continent": meta["continent"],
            "count": count,
            "area_km2": area,
            "population": pop,
            "gdp_pc": meta["gdp_pc"],
            "density_per_km2": round(density, 2),
            "per_capita": round(per_capita, 4),
            "is_gbif_member": cc in gbif_members,
        })

    country_metrics.sort(key=lambda x: x["count"], reverse=True)

    # === Continental breakdown ===
    continent_stats = {}
    for cm in country_metrics:
        c = cm["continent"]
        if c not in continent_stats:
            continent_stats[c] = {"count": 0, "area": 0, "pop": 0, "n_countries": 0}
        continent_stats[c]["count"] += cm["count"]
        continent_stats[c]["area"] += cm["area_km2"]
        continent_stats[c]["pop"] += cm["population"]
        continent_stats[c]["n_countries"] += 1

    continental = []
    for c, s in sorted(continent_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        continental.append({
            "continent": c,
            "total_records": s["count"],
            "n_countries": s["n_countries"],
            "density_per_km2": round(s["count"] / s["area"], 2) if s["area"] > 0 else 0,
            "per_capita": round(s["count"] / s["pop"], 2) if s["pop"] > 0 else 0,
        })

    # === Top/bottom rankings ===
    by_density = sorted([c for c in country_metrics if c["area_km2"] > 1000], key=lambda x: x["density_per_km2"], reverse=True)
    top20_density = by_density[:20]
    bottom20_density = by_density[-20:]

    # === H1: Tropical Paradox ===
    # Spearman correlation between observation density and species richness
    h1_pairs = []
    for cc, sr in SPECIES_RICHNESS.items():
        cm = next((c for c in country_metrics if c["code"] == cc), None)
        if cm and cm["density_per_km2"] > 0:
            h1_pairs.append((cm["density_per_km2"], sr["species"]))

    h1_rho, h1_p = stats.spearmanr([p[0] for p in h1_pairs], [p[1] for p in h1_pairs])
    h1_n = len(h1_pairs)

    # Log-transform for additional insight
    h1_log_rho, h1_log_p = stats.spearmanr(
        [math.log10(p[0]) for p in h1_pairs],
        [math.log10(p[1]) for p in h1_pairs]
    )

    h1_result = {
        "hypothesis": "Countries with highest biodiversity have disproportionately LOW observation density",
        "n_countries": h1_n,
        "spearman_rho": round(h1_rho, 4),
        "p_value": round(h1_p, 6),
        "log_spearman_rho": round(h1_log_rho, 4),
        "log_p_value": round(h1_log_p, 6),
        "supported": bool(h1_rho < 0 and h1_p < 0.05),
        "interpretation": (
            f"Spearman rho = {h1_rho:.3f} (p = {h1_p:.4f}). "
            + ("Significant negative correlation — biodiversity-rich countries are systematically under-sampled." if (h1_rho < 0 and h1_p < 0.05)
               else "No significant negative correlation — H1 not supported." if h1_p >= 0.05
               else "Positive correlation — opposite of prediction.")
        ),
        "detail": [(cc, COUNTRIES.get(cc, {}).get("name", cc),
                     next((c["density_per_km2"] for c in country_metrics if c["code"] == cc), 0),
                     sr["species"])
                    for cc, sr in SPECIES_RICHNESS.items()
                    if any(c["code"] == cc for c in country_metrics)]
    }

    # === H5: GBIF Membership Effect ===
    member_rates = [c["per_capita"] for c in country_metrics if c["is_gbif_member"] and c["per_capita"] > 0]
    nonmember_rates = [c["per_capita"] for c in country_metrics if not c["is_gbif_member"] and c["per_capita"] > 0]

    if member_rates and nonmember_rates:
        h5_u, h5_p = stats.mannwhitneyu(member_rates, nonmember_rates, alternative="greater")
        h5_median_member = float(np.median(member_rates))
        h5_median_nonmember = float(np.median(nonmember_rates))

        # GDP-controlled: split into high/low GDP
        high_gdp_members = [c["per_capita"] for c in country_metrics if c["is_gbif_member"] and c["gdp_pc"] > 15000 and c["per_capita"] > 0]
        high_gdp_nonmembers = [c["per_capita"] for c in country_metrics if not c["is_gbif_member"] and c["gdp_pc"] > 15000 and c["per_capita"] > 0]
        low_gdp_members = [c["per_capita"] for c in country_metrics if c["is_gbif_member"] and c["gdp_pc"] <= 15000 and c["per_capita"] > 0]
        low_gdp_nonmembers = [c["per_capita"] for c in country_metrics if not c["is_gbif_member"] and c["gdp_pc"] <= 15000 and c["per_capita"] > 0]

        h5_high_gdp = None
        if high_gdp_members and high_gdp_nonmembers:
            u2, p2 = stats.mannwhitneyu(high_gdp_members, high_gdp_nonmembers, alternative="greater")
            h5_high_gdp = {"u": float(u2), "p": round(float(p2), 6), "n_member": len(high_gdp_members), "n_nonmember": len(high_gdp_nonmembers)}

        h5_low_gdp = None
        if low_gdp_members and low_gdp_nonmembers:
            u3, p3 = stats.mannwhitneyu(low_gdp_members, low_gdp_nonmembers, alternative="greater")
            h5_low_gdp = {"u": float(u3), "p": round(float(p3), 6), "n_member": len(low_gdp_members), "n_nonmember": len(low_gdp_nonmembers)}
    else:
        h5_u, h5_p = 0, 1
        h5_median_member, h5_median_nonmember = 0, 0
        h5_high_gdp, h5_low_gdp = None, None

    h5_result = {
        "hypothesis": "GBIF member countries have higher per-capita observation rates",
        "n_members": len(member_rates),
        "n_nonmembers": len(nonmember_rates),
        "median_member_per_capita": round(h5_median_member, 4),
        "median_nonmember_per_capita": round(h5_median_nonmember, 4),
        "ratio": round(h5_median_member / h5_median_nonmember, 1) if h5_median_nonmember > 0 else float('inf'),
        "mann_whitney_u": float(h5_u),
        "p_value": round(float(h5_p), 6),
        "supported": bool(float(h5_p) < 0.05),
        "gdp_controlled_high": h5_high_gdp,
        "gdp_controlled_low": h5_low_gdp,
    }

    # === Taxonomic coverage by country ===
    anim_map = {c: n for c, n in taxa_animalia}
    plant_map = {c: n for c, n in taxa_plantae}
    insect_map = {c: n for c, n in taxa_insecta}
    fungi_map = {c: n for c, n in taxa_fungi}

    # Compute bird% for top countries
    for cm in country_metrics[:50]:
        cc = cm["code"]
        total = count_map.get(cc, 0)
        cm["animalia_count"] = anim_map.get(cc, 0)
        cm["plantae_count"] = plant_map.get(cc, 0)
        cm["insecta_count"] = insect_map.get(cc, 0)
        cm["fungi_count"] = fungi_map.get(cc, 0)

    # === Summary stats ===
    total_records = sum(n for _, n in country_counts)
    top10_share = sum(count_map.get(c, 0) for c in [cm["code"] for cm in country_metrics[:10]]) / total_records
    top1_share = country_metrics[0]["count"] / total_records if country_metrics else 0

    summary = {
        "total_records": total_records,
        "n_countries": len(country_counts),
        "n_countries_with_metadata": len(country_metrics),
        "top1_country": country_metrics[0]["code"] if country_metrics else "",
        "top1_share": round(top1_share, 4),
        "top10_share": round(top10_share, 4),
        "max_density_per_km2": top20_density[0]["density_per_km2"] if top20_density else 0,
        "min_density_per_km2": bottom20_density[-1]["density_per_km2"] if bottom20_density else 0,
        "density_range_ratio": round(top20_density[0]["density_per_km2"] / bottom20_density[-1]["density_per_km2"], 0) if bottom20_density and bottom20_density[-1]["density_per_km2"] > 0 else 0,
    }

    result = {
        "summary": summary,
        "continental": continental,
        "country_metrics": country_metrics[:60],  # top 60 for report
        "top20_density": top20_density,
        "bottom20_density": bottom20_density,
        "h1_tropical_paradox": h1_result,
        "h5_membership_effect": h5_result,
    }

    out = DATA / "analysis" / "geographic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"Geographic analysis saved to {out}")
    print(f"  Total records: {total_records:,}")
    print(f"  Countries: {len(country_counts)}")
    print(f"  H1 (tropical paradox): rho={h1_rho:.3f}, p={h1_p:.4f}, supported={h1_rho < 0 and h1_p < 0.05}")
    print(f"  H5 (membership): p={float(h5_p):.4f}, member median={h5_median_member:.4f}, non-member={h5_median_nonmember:.4f}")
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run()
