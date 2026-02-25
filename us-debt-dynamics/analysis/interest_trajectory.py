"""Analysis 3: Interest Expense Trajectory.

Tracks interest expense growth, compares to defense spending,
computes ratios (interest/GDP, interest/revenue, interest/outlays),
and projects forward.
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmark_data import GDP, NET_INTEREST, DEFENSE, TOTAL_OUTLAYS, TOTAL_REVENUE
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_processed(name):
    with open(os.path.join(DATA_DIR, 'processed', f'{name}.json')) as f:
        return json.load(f)

def run():
    debt = load_processed('debt_monthly')
    blended_path = os.path.join(DATA_DIR, 'analysis', 'blended_rate.json')
    with open(blended_path) as f:
        blended = json.load(f)

    years = sorted(GDP.keys())

    # === Ratio Time Series ===
    ratios = {}
    for yr in years:
        interest = NET_INTEREST.get(yr)
        gdp = GDP.get(yr)
        defense = DEFENSE.get(yr)
        outlays = TOTAL_OUTLAYS.get(yr)
        revenue = TOTAL_REVENUE.get(yr)

        if not all([interest, gdp, defense, outlays, revenue]):
            continue

        # Get end-of-FY debt (use September of that year or nearest)
        fy_end = f"{yr}-09"
        debt_val = None
        for month_offset in [0, -1, 1, -2, 2]:  # Try nearby months
            check = f"{yr}-{9 + month_offset:02d}" if 1 <= 9 + month_offset <= 12 else None
            if check and check in debt:
                debt_val = debt[check].get('total')
                if debt_val:
                    break
        if not debt_val:
            # Try December
            dec = f"{yr}-12"
            if dec in debt:
                debt_val = debt[dec].get('total')

        ratios[yr] = {
            'net_interest_bil': interest,
            'gdp_bil': gdp,
            'defense_bil': defense,
            'total_outlays_bil': outlays,
            'total_revenue_bil': revenue,
            'debt_bil': round(debt_val / 1e9, 1) if debt_val else None,
            'interest_to_gdp_pct': round(interest / gdp * 100, 2),
            'interest_to_revenue_pct': round(interest / revenue * 100, 2),
            'interest_to_outlays_pct': round(interest / outlays * 100, 2),
            'interest_to_defense_ratio': round(interest / defense, 3),
            'deficit_bil': outlays - revenue,
            'interest_as_share_of_deficit_pct': round(interest / (outlays - revenue) * 100, 1) if outlays > revenue else None,
        }

    # === Interest vs Defense Crossover ===
    crossover = None
    for yr in years:
        if NET_INTEREST.get(yr) and DEFENSE.get(yr):
            if NET_INTEREST[yr] > DEFENSE[yr]:
                crossover = {
                    'year': yr,
                    'interest_bil': NET_INTEREST[yr],
                    'defense_bil': DEFENSE[yr],
                    'margin_bil': NET_INTEREST[yr] - DEFENSE[yr],
                }
                break

    # === Growth Rates ===
    growth = {}
    for start, end, label in [
        (2001, 2025, 'full_period'),
        (2001, 2007, 'pre_crisis'),
        (2008, 2015, 'crisis_and_zirp'),
        (2016, 2019, 'normalization'),
        (2019, 2025, 'post_covid'),
        (2022, 2025, 'recent_surge'),
    ]:
        if start in NET_INTEREST and end in NET_INTEREST:
            n_years = end - start
            if n_years > 0 and NET_INTEREST[start] > 0:
                cagr = (NET_INTEREST[end] / NET_INTEREST[start]) ** (1 / n_years) - 1
                growth[label] = {
                    'start_year': start,
                    'end_year': end,
                    'start_bil': NET_INTEREST[start],
                    'end_bil': NET_INTEREST[end],
                    'total_growth_pct': round((NET_INTEREST[end] / NET_INTEREST[start] - 1) * 100, 1),
                    'cagr_pct': round(cagr * 100, 2),
                }

    # === Forward Projection ===
    # Simple model: if effective rate converges to current market rate and debt grows at recent pace
    projection = {}

    # Get current effective rate and debt growth rate
    weighted_rates = blended.get('weighted_blended_rate', {})
    rate_dates = sorted(weighted_rates.keys())

    current_rate = weighted_rates[rate_dates[-1]] if rate_dates else 3.0

    # Debt growth rate from last 5 years
    debt_vals = []
    for yr in range(2020, 2026):
        dec = f"{yr}-12"
        if dec in debt and debt[dec].get('total'):
            debt_vals.append((yr, debt[dec]['total']))
    if len(debt_vals) >= 2:
        debt_cagr = (debt_vals[-1][1] / debt_vals[0][1]) ** (1 / (debt_vals[-1][0] - debt_vals[0][0])) - 1
    else:
        debt_cagr = 0.05  # fallback 5%

    # Latest debt level
    latest_debt_dates = sorted(debt.keys())
    latest_debt = debt[latest_debt_dates[-1]]['total'] if latest_debt_dates else 38e12

    # Market rate target — where the blended rate is heading
    # Use the latest avg rate for Total Marketable as proxy
    market_rate = current_rate  # Use current blended as floor

    # Blended rate convergence: assume it rises by ~0.15pp/year toward market rate
    rate_convergence_per_year = 0.15

    for proj_year in range(2026, 2036):
        years_out = proj_year - 2025
        proj_debt = latest_debt * (1 + debt_cagr) ** years_out
        proj_rate = min(current_rate + rate_convergence_per_year * years_out, 4.5)  # Cap at 4.5%
        proj_interest = proj_debt * proj_rate / 100

        # GDP projection (assume 4% nominal growth)
        proj_gdp = GDP[2025] * 1e9 * (1.04 ** years_out)

        projection[str(proj_year)] = {
            'projected_debt_tril': round(proj_debt / 1e12, 2),
            'projected_effective_rate_pct': round(proj_rate, 2),
            'projected_interest_bil': round(proj_interest / 1e9, 0),
            'projected_interest_to_gdp_pct': round(proj_interest / proj_gdp * 100, 2),
        }

    # === Summary ===
    summary = {
        'crossover': crossover,
        'growth': growth,
        'current_state': ratios.get(2025) or ratios.get(2024),
        'projection_assumptions': {
            'debt_cagr_pct': round(debt_cagr * 100, 2),
            'current_effective_rate': current_rate,
            'rate_convergence_per_year': rate_convergence_per_year,
            'gdp_nominal_growth_pct': 4.0,
        },
        'key_findings': [],
    }

    # Key findings
    if crossover:
        summary['key_findings'].append(
            f"Interest costs first exceeded defense spending in FY{crossover['year']}: "
            f"${crossover['interest_bil']}B vs ${crossover['defense_bil']}B (margin: ${crossover['margin_bil']}B)"
        )

    if 'post_covid' in growth:
        g = growth['post_covid']
        summary['key_findings'].append(
            f"Interest costs grew {g['total_growth_pct']}% in {g['end_year'] - g['start_year']} years "
            f"(FY{g['start_year']} ${g['start_bil']}B → FY{g['end_year']} ${g['end_bil']}B)"
        )

    yr_2025 = ratios.get(2025)
    if yr_2025:
        summary['key_findings'].append(
            f"FY2025: Interest = {yr_2025['interest_to_gdp_pct']}% of GDP, "
            f"{yr_2025['interest_to_revenue_pct']}% of revenue, "
            f"{yr_2025['interest_to_outlays_pct']}% of spending"
        )

    results = {
        'ratios': ratios,
        'crossover': crossover,
        'growth': growth,
        'projection': projection,
        'summary': summary,
    }

    # Save
    out_path = os.path.join(DATA_DIR, 'analysis', 'interest_trajectory.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Interest trajectory analysis saved to {out_path}")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    if crossover:
        print(f"  Defense crossover: FY{crossover['year']} (${crossover['margin_bil']}B margin)")
    for yr_key in ['2030', '2035']:
        if yr_key in projection:
            p = projection[yr_key]
            print(f"  Projection {yr_key}: ${p['projected_interest_bil']:.0f}B interest, "
                  f"{p['projected_interest_to_gdp_pct']}% of GDP")

    return results

if __name__ == '__main__':
    run()
