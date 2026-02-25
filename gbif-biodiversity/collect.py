#!/usr/bin/env python3
"""GBIF data collection: runs all ~121 faceted queries and saves results."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gbif.client import GBIFClient

OUT = Path(__file__).parent / "data" / "raw"
CACHE_DB = Path(__file__).parent / "data" / "cache.db"
OUT.mkdir(parents=True, exist_ok=True)

def save(name, data):
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))
    print(f"  Saved {name} ({len(data)} items)")

def collect_all():
    t0 = time.time()
    results = {}

    with GBIFClient(cache_path=CACHE_DB) as client:
        # === A. Global Overview (5 queries) ===
        print("=== A. Global Overview ===")
        r = client.facet_query("country", 300)
        save("global_country", r)
        results["global_country"] = r

        r = client.facet_query("year", 200)
        save("global_year", r)
        results["global_year"] = r

        r = client.facet_query("month", 12)
        save("global_month", r)

        r = client.facet_query("basisOfRecord", 20)
        save("global_basis", r)
        results["global_basis"] = r

        r = client.facet_query("kingdomKey", 10)
        save("global_kingdom", r)
        results["global_kingdom"] = r

        # === B. Kingdom x Year (7 queries) ===
        print("=== B. Kingdom x Year ===")
        kingdoms = {1: "Animalia", 6: "Plantae", 5: "Fungi", 3: "Bacteria", 4: "Chromista", 2: "Archaea", 7: "Protozoa"}
        kingdom_year = {}
        for k, name in kingdoms.items():
            r = client.facet_query("year", 60, {"kingdomKey": k})
            kingdom_year[str(k)] = r
            save(f"kingdom_{k}_year", r)
        results["kingdom_year"] = kingdom_year

        # === C. Key Class x Year (6 queries) ===
        print("=== C. Class x Year ===")
        classes = {212: "Aves", 216: "Insecta", 359: "Mammalia", 131: "Amphibia", 220: "Magnoliopsida"}
        class_year = {}
        for c, name in classes.items():
            r = client.facet_query("year", 60, {"classKey": c})
            class_year[str(c)] = r
            save(f"class_{c}_year", r)
        # Fungi kingdom-level (already got in B, just reference it)
        class_year["5_fungi"] = kingdom_year["5"]
        results["class_year"] = class_year

        # === D. BasisOfRecord x Year (6 queries) ===
        print("=== D. BasisOfRecord x Year ===")
        bases = ["HUMAN_OBSERVATION", "PRESERVED_SPECIMEN", "MATERIAL_SAMPLE", "MACHINE_OBSERVATION", "OCCURRENCE", "OBSERVATION"]
        basis_year = {}
        for b in bases:
            r = client.facet_query("year", 60, {"basisOfRecord": b})
            basis_year[b] = r
            save(f"basis_{b}_year", r)
        results["basis_year"] = basis_year

        # === E. Country x Year for top-30 (30 queries) ===
        print("=== E. Country x Year (top 30) ===")
        # Get top-30 countries by count
        top30 = sorted(results["global_country"], key=lambda x: x[1], reverse=True)[:30]
        country_codes = [c[0] for c in top30]
        country_year = {}
        for i, cc in enumerate(country_codes):
            r = client.facet_query("year", 60, {"country": cc})
            country_year[cc] = r
            save(f"country_{cc}_year", r)
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/30 countries done")
        results["country_year"] = country_year
        results["top30_countries"] = country_codes

        # === F. Kingdom x Country cross-tab (4 queries) ===
        print("=== F. Kingdom/Class x Country ===")
        taxa_country = {}
        for key, filters, label in [
            ("animalia", {"kingdomKey": 1}, "Animalia"),
            ("plantae", {"kingdomKey": 6}, "Plantae"),
            ("fungi", {"kingdomKey": 5}, "Fungi"),
            ("insecta", {"classKey": 216}, "Insecta"),
        ]:
            r = client.facet_query("country", 300, filters)
            taxa_country[key] = r
            save(f"taxa_{key}_country", r)
        results["taxa_country"] = taxa_country

        # === G. Country x BasisOfRecord for top-30 (30 queries) ===
        print("=== G. Country x BasisOfRecord (top 30) ===")
        country_basis = {}
        for i, cc in enumerate(country_codes):
            r = client.facet_query("basisOfRecord", 10, {"country": cc})
            country_basis[cc] = r
            save(f"country_{cc}_basis", r)
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/30 countries done")
        results["country_basis"] = country_basis

        # === H. Species Accumulation (30 queries) ===
        print("=== H. Species Accumulation ===")
        accum_countries = ["US", "GB", "AU"]
        accum_classes = {212: "Aves", 216: "Insecta"}
        accum_years = [2005, 2010, 2015, 2020, 2024]
        species_accum = {}
        qcount = 0
        for cc in accum_countries:
            species_accum[cc] = {}
            for ckey, cname in accum_classes.items():
                species_accum[cc][str(ckey)] = {}
                for yr in accum_years:
                    r = client.facet_query("speciesKey", 100000, {"country": cc, "classKey": ckey, "year": yr})
                    species_accum[cc][str(ckey)][str(yr)] = len(r)  # distinct species count
                    save(f"accum_{cc}_{ckey}_{yr}", {"species_count": len(r), "country": cc, "class": cname, "year": yr})
                    qcount += 1
                    if qcount % 5 == 0:
                        print(f"  ... {qcount}/30 accumulation queries done")
        results["species_accum"] = species_accum

        # === I. GBIF Membership (1 query) ===
        print("=== I. GBIF Nodes ===")
        nodes = client.get_nodes(300)
        country_nodes = [n for n in nodes if n.get("type") == "COUNTRY"]
        member_countries = []
        for n in country_nodes:
            cc = n.get("country", "")
            if cc:
                member_countries.append(cc)
        save("gbif_members", member_countries)
        results["gbif_members"] = member_countries

        # === J. Class breakdown for Animalia and Plantae ===
        print("=== J. Class breakdowns ===")
        for kkey, kname in [(1, "Animalia"), (6, "Plantae")]:
            r = client.facet_query("classKey", 50, {"kingdomKey": kkey})
            save(f"kingdom_{kkey}_classes", r)

    elapsed = time.time() - t0
    print(f"\n=== Collection complete: {elapsed:.1f}s ===")

    # Save master results file
    save("master_results", {
        "top30_countries": country_codes,
        "gbif_members": member_countries,
        "species_accum": species_accum,
        "collection_time_s": round(elapsed, 1),
    })

    return results

if __name__ == "__main__":
    collect_all()
