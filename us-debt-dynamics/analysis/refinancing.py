"""Analysis 2: Refinancing Wall Model.

Models how ZIRP-era debt (2009-2021) reprices at current rates as it matures.
Estimates additional annual interest costs from the mechanical repricing.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_processed(name):
    with open(os.path.join(DATA_DIR, 'processed', f'{name}.json')) as f:
        return json.load(f)

def run():
    mspd = load_processed('mspd')
    rates = load_processed('interest_rates')

    # === Model Parameters ===
    # Average maturity assumptions (years) — standard estimates
    MATURITY = {
        'Bills': 0.5,    # T-Bills: 4-52 weeks, avg ~6 months
        'Notes': 5.0,    # T-Notes: 2-10 years, avg ~5 years
        'Bonds': 20.0,   # T-Bonds: 20-30 years, avg ~20 years
        'TIPS': 10.0,    # TIPS: 5-30 years, avg ~10 years
        'FRNs': 0.0,     # Already floating — no repricing needed
    }

    # Get latest MSPD composition
    mspd_dates = sorted(mspd.keys())
    latest_date = mspd_dates[-1]
    latest_mspd = mspd[latest_date]

    # Get ZIRP-era average rates and current rates
    # ZIRP era: 2009-2021 (Fed Funds at 0-0.25%)
    TYPES = ['Bills', 'Notes', 'Bonds', 'TIPS']

    zirp_rates = {}
    current_rates = {}
    for t in TYPES:
        # Average rate during ZIRP (2009-2021)
        zirp_vals = []
        for date, r in rates.items():
            if '2009' <= date[:4] <= '2021' and t in r:
                zirp_vals.append(r[t])
        zirp_rates[t] = sum(zirp_vals) / len(zirp_vals) if zirp_vals else None

        # Current rate (latest available)
        rate_dates = sorted(rates.keys())
        for d in reversed(rate_dates):
            if t in rates[d]:
                current_rates[t] = rates[d][t]
                break

    # === Composition History ===
    # Track how composition has shifted over time
    composition_history = {}
    for date in sorted(mspd.keys()):
        m = mspd[date]
        total = sum(
            (m.get(t, {}).get('total_mil', 0) or 0)
            for t in TYPES + ['FRNs']
        )
        if total > 0:
            composition_history[date] = {
                t: round((m.get(t, {}).get('total_mil', 0) or 0) / total * 100, 2)
                for t in TYPES + ['FRNs']
            }

    # === Refinancing Model ===
    # For each security type, estimate when ZIRP-era issuance fully reprices

    refinancing_schedule = {}
    total_additional_cost = 0

    for t in TYPES:
        outstanding_mil = latest_mspd.get(t, {}).get('total_mil', 0) or 0
        outstanding_bil = outstanding_mil / 1000
        maturity = MATURITY[t]
        zirp_rate = zirp_rates.get(t)
        curr_rate = current_rates.get(t)

        if zirp_rate is None or curr_rate is None or maturity == 0:
            continue

        rate_spread = curr_rate - zirp_rate

        # Model: assume linear rollover of ZIRP-era debt
        # Bills: already repriced (maturity < 1yr, all post-ZIRP)
        # Notes: ~60% issued during ZIRP (2009-2021 = 13 of ~25 years of data)
        # Bonds: ~30% issued during ZIRP (13 of 40+ years)
        # TIPS: ~50% issued during ZIRP (13 of ~25 years)

        # Estimate ZIRP-era share based on issuance window vs maturity
        # If maturity >> ZIRP window, less is still outstanding
        zirp_window = 13  # years (2009-2021)
        years_since_zirp_end = 4  # 2022-2025

        if t == 'Bills':
            zirp_share = 0  # Already fully repriced
        elif maturity <= years_since_zirp_end:
            zirp_share = 0  # All matured already
        else:
            # Fraction of ZIRP-era debt still outstanding
            # Debt issued uniformly during ZIRP, maturity = M years
            # After Y years post-ZIRP, fraction remaining = max(0, 1 - Y/M) of ZIRP issuance
            # But only ZIRP issuance = zirp_window/M of total (if M >= zirp_window)
            if maturity >= zirp_window:
                zirp_issuance_share = zirp_window / maturity
                still_outstanding = max(0, 1 - years_since_zirp_end / maturity)
                # But some of the ZIRP issuance has already matured
                # Remaining ZIRP debt = integrate over issuance dates
                remaining = 0
                for yr_back in range(years_since_zirp_end, years_since_zirp_end + zirp_window):
                    if yr_back < maturity:
                        remaining += 1 / maturity  # uniform issuance model
                zirp_share = remaining
            else:
                # Multiple cycles within ZIRP window, mostly repriced already
                remaining = max(0, (maturity - years_since_zirp_end) / maturity)
                zirp_share = remaining

        zirp_outstanding_bil = outstanding_bil * zirp_share

        # Additional annual cost from repricing this ZIRP debt at current rates
        additional_cost_bil = zirp_outstanding_bil * rate_spread / 100

        # Remaining repricing timeline
        if t == 'Bills':
            years_to_full_reprice = 0
        else:
            years_to_full_reprice = max(0, maturity - years_since_zirp_end)

        refinancing_schedule[t] = {
            'outstanding_bil': round(outstanding_bil, 1),
            'maturity_years': maturity,
            'zirp_avg_rate': round(zirp_rate, 3),
            'current_rate': round(curr_rate, 3),
            'rate_spread_pp': round(rate_spread, 3),
            'zirp_share_pct': round(zirp_share * 100, 1),
            'zirp_outstanding_bil': round(zirp_outstanding_bil, 1),
            'additional_annual_cost_bil': round(additional_cost_bil, 1),
            'years_to_full_reprice': round(years_to_full_reprice, 1),
        }
        total_additional_cost += additional_cost_bil

    # === Sensitivity Analysis ===
    # What if rates are ±100bp from current?
    sensitivity = {}
    for scenario_name, rate_delta in [('rates_down_100bp', -1.0), ('rates_unchanged', 0), ('rates_up_100bp', 1.0)]:
        scenario_cost = 0
        for t in TYPES:
            entry = refinancing_schedule.get(t)
            if not entry or entry['zirp_outstanding_bil'] == 0:
                continue
            adjusted_spread = entry['rate_spread_pp'] + rate_delta
            scenario_cost += entry['zirp_outstanding_bil'] * adjusted_spread / 100
        sensitivity[scenario_name] = round(scenario_cost, 1)

    # === Forward Projection ===
    # Model year-by-year repricing for 2025-2035
    projection = {}
    for proj_year in range(2025, 2036):
        years_post_zirp = proj_year - 2021
        year_cost = 0
        year_detail = {}
        for t in TYPES:
            outstanding_bil = (latest_mspd.get(t, {}).get('total_mil', 0) or 0) / 1000
            maturity = MATURITY[t]
            zirp_rate = zirp_rates.get(t)
            curr_rate = current_rates.get(t)
            if not zirp_rate or not curr_rate or maturity == 0:
                continue

            # Remaining ZIRP share at this future date
            zirp_window = 13
            if maturity <= years_post_zirp:
                remaining_zirp = 0
            else:
                remaining = 0
                for yr_back in range(years_post_zirp, years_post_zirp + zirp_window):
                    if yr_back < maturity:
                        remaining += 1 / maturity
                remaining_zirp = remaining

            zirp_bil = outstanding_bil * remaining_zirp
            spread = curr_rate - zirp_rate
            cost = zirp_bil * spread / 100
            year_cost += cost
            year_detail[t] = {
                'zirp_remaining_pct': round(remaining_zirp * 100, 1),
                'cost_bil': round(cost, 1)
            }

        # Also compute the non-ZIRP debt cost (at current rates)
        total_marketable_bil = sum(
            (latest_mspd.get(t, {}).get('total_mil', 0) or 0) / 1000
            for t in TYPES + ['FRNs']
        )
        # Estimate total interest if ALL debt were at current weighted rate
        latest_rates = rates[sorted(rates.keys())[-1]]
        if 'Total_Marketable' in latest_rates:
            full_cost_bil = total_marketable_bil * latest_rates['Total_Marketable'] / 100
        else:
            full_cost_bil = None

        projection[str(proj_year)] = {
            'additional_zirp_repricing_cost_bil': round(year_cost, 1),
            'detail': year_detail,
            'full_rate_cost_estimate_bil': round(full_cost_bil, 1) if full_cost_bil else None,
        }

    results = {
        'latest_date': latest_date,
        'refinancing_schedule': refinancing_schedule,
        'total_additional_annual_cost_bil': round(total_additional_cost, 1),
        'sensitivity': sensitivity,
        'projection': projection,
        'composition_history': composition_history,
        'model_assumptions': {
            'zirp_era': '2009-2021',
            'years_since_zirp_end': years_since_zirp_end,
            'maturity_assumptions': MATURITY,
            'issuance_model': 'uniform within ZIRP window',
            'debt_growth': 'static (current outstanding, no growth modeled)',
        },
    }

    # Save
    out_path = os.path.join(DATA_DIR, 'analysis', 'refinancing.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Refinancing wall analysis saved to {out_path}")
    print(f"  Total additional annual cost from ZIRP repricing: ${total_additional_cost:.1f}B")
    for t, entry in refinancing_schedule.items():
        print(f"  {t}: ${entry['outstanding_bil']:.0f}B outstanding, "
              f"{entry['zirp_share_pct']:.0f}% ZIRP-era, "
              f"spread {entry['rate_spread_pp']:.2f}pp, "
              f"additional cost ${entry['additional_annual_cost_bil']:.1f}B/yr")
    print(f"  Sensitivity: {sensitivity}")

    return results

if __name__ == '__main__':
    run()
