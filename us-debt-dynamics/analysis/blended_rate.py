"""Analysis 1: Effective Blended Interest Rate Decomposition.

Two methods:
- Method A (weighted): avg_rate[type] × outstanding[type] / total_outstanding
- Method B (direct): annualized interest_expense / average total debt
"""
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_processed(name):
    with open(os.path.join(DATA_DIR, 'processed', f'{name}.json')) as f:
        return json.load(f)

def run():
    rates = load_processed('interest_rates')
    mspd = load_processed('mspd')
    expense = load_processed('interest_expense')
    debt = load_processed('debt_monthly')

    results = {
        'weighted_blended_rate': {},
        'direct_effective_rate': {},
        'by_security_type': {},
        'composition_share': {},
        'summary': {},
    }

    # ===== METHOD A: Weighted blended rate =====
    # For each month where both rates and MSPD exist, compute weighted average
    TYPES = ['Bills', 'Notes', 'Bonds', 'TIPS', 'FRNs']

    for date in sorted(rates.keys()):
        if date not in mspd:
            continue
        r = rates[date]
        m = mspd[date]

        total_outstanding = 0
        weighted_sum = 0
        type_contributions = {}

        for t in TYPES:
            rate_val = r.get(t)
            mspd_val = m.get(t)
            if rate_val is None or mspd_val is None:
                continue
            outstanding = mspd_val.get('total_mil', 0) or 0
            if outstanding <= 0:
                continue
            total_outstanding += outstanding
            contribution = rate_val * outstanding
            weighted_sum += contribution
            type_contributions[t] = {
                'rate': rate_val,
                'outstanding_mil': outstanding,
                'contribution': contribution,
            }

        if total_outstanding > 0:
            blended = weighted_sum / total_outstanding
            results['weighted_blended_rate'][date] = round(blended, 4)

            # Store per-type data
            for t, c in type_contributions.items():
                if t not in results['by_security_type']:
                    results['by_security_type'][t] = {}
                results['by_security_type'][t][date] = {
                    'rate': c['rate'],
                    'share': round(c['outstanding_mil'] / total_outstanding * 100, 2),
                    'weighted_contribution': round(c['rate'] * c['outstanding_mil'] / total_outstanding, 4),
                }

            # Composition shares
            results['composition_share'][date] = {
                t: round(c['outstanding_mil'] / total_outstanding * 100, 2)
                for t, c in type_contributions.items()
            }

    # ===== METHOD B: Direct effective rate =====
    # Annual: sum monthly interest expense / avg monthly debt × 100
    # Only available from 2010+ (interest_expense start date)
    annual_expense = {}
    annual_debt = {}

    for date, exp in expense.items():
        year = date[:4]
        # Use fiscal year (Oct-Sep): FY2024 = Oct 2023 - Sep 2024
        month = int(date[5:7])
        fy = int(year) if month >= 10 else int(year)
        # Actually, for simplicity compute by calendar year first
        if year not in annual_expense:
            annual_expense[year] = 0
        annual_expense[year] += exp['total_monthly']

    for date, d in debt.items():
        year = date[:4]
        if year not in annual_debt:
            annual_debt[year] = []
        if d.get('total'):
            annual_debt[year].append(d['total'])

    for year in sorted(annual_expense.keys()):
        if year in annual_debt and annual_debt[year]:
            avg_debt = sum(annual_debt[year]) / len(annual_debt[year])
            if avg_debt > 0:
                eff_rate = (annual_expense[year] / avg_debt) * 100
                results['direct_effective_rate'][year] = round(eff_rate, 4)

    # ===== SUMMARY =====
    wbr_dates = sorted(results['weighted_blended_rate'].keys())
    if wbr_dates:
        # Key periods
        def avg_rate_period(start_y, end_y):
            vals = [v for d, v in results['weighted_blended_rate'].items()
                    if start_y <= d[:4] <= end_y]
            return round(sum(vals) / len(vals), 3) if vals else None

        results['summary'] = {
            'method_a_months': len(wbr_dates),
            'method_b_years': len(results['direct_effective_rate']),
            'first_date': wbr_dates[0],
            'last_date': wbr_dates[-1],
            'current_rate': results['weighted_blended_rate'][wbr_dates[-1]],
            'peak_rate': max(results['weighted_blended_rate'].values()),
            'peak_date': max(results['weighted_blended_rate'], key=results['weighted_blended_rate'].get),
            'trough_rate': min(results['weighted_blended_rate'].values()),
            'trough_date': min(results['weighted_blended_rate'], key=results['weighted_blended_rate'].get),
            'period_averages': {
                'pre_crisis_2001_2007': avg_rate_period('2001', '2007'),
                'crisis_response_2008_2009': avg_rate_period('2008', '2009'),
                'zirp_era_2010_2015': avg_rate_period('2010', '2015'),
                'normalization_2016_2019': avg_rate_period('2016', '2019'),
                'covid_zirp_2020_2021': avg_rate_period('2020', '2021'),
                'tightening_2022_2024': avg_rate_period('2022', '2024'),
                'current_2025_2026': avg_rate_period('2025', '2026'),
            },
        }

        # Rate by type for latest month
        latest = wbr_dates[-1]
        results['summary']['latest_rates_by_type'] = {}
        for t in TYPES:
            if t in results['by_security_type'] and latest in results['by_security_type'][t]:
                entry = results['by_security_type'][t][latest]
                results['summary']['latest_rates_by_type'][t] = {
                    'rate': entry['rate'],
                    'share': entry['share'],
                }

    # Save
    os.makedirs(os.path.join(DATA_DIR, 'analysis'), exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'analysis', 'blended_rate.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Blended rate analysis saved to {out_path}")
    print(f"  Weighted rate time series: {len(results['weighted_blended_rate'])} months")
    print(f"  Direct effective rate: {len(results['direct_effective_rate'])} years")
    if results['summary']:
        s = results['summary']
        print(f"  Current blended rate: {s['current_rate']}%")
        print(f"  Peak: {s['peak_rate']}% ({s['peak_date']})")
        print(f"  Trough: {s['trough_rate']}% ({s['trough_date']})")
        for period, avg in s.get('period_averages', {}).items():
            if avg:
                print(f"  {period}: {avg}%")

    return results

if __name__ == '__main__':
    run()
