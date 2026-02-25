"""Process raw Treasury API data into analysis-ready time series."""
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def _load_raw(name):
    with open(os.path.join(DATA_DIR, 'raw', f'{name}.json')) as f:
        return json.load(f)

def _safe_float(v):
    if v is None or v == '' or v == 'null':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def process_debt_to_penny():
    """Convert daily debt records to monthly averages."""
    data = _load_raw('debt_to_penny')
    monthly = defaultdict(lambda: {'total': [], 'public': [], 'intragov': []})

    for r in data:
        date = r['record_date'][:7]  # YYYY-MM
        total = _safe_float(r.get('tot_pub_debt_out_amt'))
        public = _safe_float(r.get('debt_held_public_amt'))
        intragov = _safe_float(r.get('intragov_hold_amt'))
        if total:
            monthly[date]['total'].append(total)
        if public:
            monthly[date]['public'].append(public)
        if intragov:
            monthly[date]['intragov'].append(intragov)

    result = {}
    for date in sorted(monthly.keys()):
        m = monthly[date]
        result[date] = {
            'total': sum(m['total']) / len(m['total']) if m['total'] else None,
            'public': sum(m['public']) / len(m['public']) if m['public'] else None,
            'intragov': sum(m['intragov']) / len(m['intragov']) if m['intragov'] else None,
        }
    return result

def process_avg_interest_rates():
    """Extract monthly interest rates by security type."""
    data = _load_raw('avg_interest_rates')

    # Map security descriptions to categories
    SECURITY_MAP = {
        'Treasury Bills': 'Bills',
        'Treasury Notes': 'Notes',
        'Treasury Bonds': 'Bonds',
        'Treasury Inflation-Protected Securities (TIPS)': 'TIPS',
        'Treasury Floating Rate Note (FRN)': 'FRNs',
        'Federal Financing Bank': 'FFB',
        'Total Marketable': 'Total_Marketable',
        'Total Non-marketable': 'Total_Nonmarketable',
        'Total Interest-bearing Debt': 'Total',
    }

    by_date = defaultdict(dict)
    for r in data:
        date = r['record_date'][:7]
        desc = r.get('security_desc', '')
        rate = _safe_float(r.get('avg_interest_rate_amt'))
        cat = SECURITY_MAP.get(desc)
        if cat and rate is not None:
            by_date[date][cat] = rate

    return dict(sorted(by_date.items()))

def process_interest_expense():
    """Aggregate monthly interest expense by category."""
    data = _load_raw('interest_expense')

    by_date = defaultdict(lambda: {'total': 0, 'by_category': defaultdict(float), 'fytd': 0})
    for r in data:
        date = r['record_date'][:7]
        catg = r.get('expense_catg_desc', 'Unknown')
        month_amt = _safe_float(r.get('month_expense_amt')) or 0
        fytd_amt = _safe_float(r.get('fytd_expense_amt')) or 0
        by_date[date]['total'] += month_amt
        by_date[date]['by_category'][catg] += month_amt
        # Track the max FYTD (last record of the month should be the total)
        by_date[date]['fytd'] = max(by_date[date]['fytd'], fytd_amt)

    result = {}
    for date in sorted(by_date.keys()):
        d = by_date[date]
        result[date] = {
            'total_monthly': d['total'],
            'by_category': dict(d['by_category']),
            'fytd': d['fytd'],
        }
    return result

def process_mspd():
    """Extract monthly outstanding debt by security class."""
    data = _load_raw('mspd')

    # Key security classes to track
    CLASS_MAP = {
        'Bills': 'Bills',
        'Notes': 'Notes',
        'Bonds': 'Bonds',
        'Treasury Inflation-Protected Securities': 'TIPS',
        'Floating Rate Notes': 'FRNs',
        'Federal Financing Bank Securities': 'FFB',
        'Government Account Series': 'GAS',
        'State and Local Government Series': 'SLGS',
        'Savings Bonds': 'Savings',
        'Other': 'Other',
    }

    by_date = defaultdict(dict)
    for r in data:
        date = r['record_date'][:7]
        sec_class = r.get('security_class_desc', '')
        sec_type = r.get('security_type_desc', '')
        total_mil = _safe_float(r.get('total_mil_amt'))
        public_mil = _safe_float(r.get('debt_held_public_mil_amt'))

        cat = CLASS_MAP.get(sec_class)
        if cat and total_mil is not None:
            by_date[date][cat] = {
                'total_mil': total_mil,
                'public_mil': public_mil,
            }

        # Also capture the summary rows
        if sec_type == 'Total Public Debt Outstanding' and sec_class == '':
            by_date[date]['_total'] = {
                'total_mil': total_mil,
                'public_mil': public_mil,
            }

    return dict(sorted(by_date.items()))

def process_debt_outstanding():
    """Process 236-year historical debt series."""
    data = _load_raw('debt_outstanding')
    result = {}
    for r in data:
        year = r['record_date'][:4]
        # debt_outstanding has different field names - check what's available
        amt = None
        for field in ['debt_outstanding_amt', 'tot_pub_debt_out_amt', 'total_debt']:
            if field in r and r[field]:
                amt = _safe_float(r[field])
                if amt:
                    break
        if amt is None:
            # Try any numeric-looking field
            for k, v in r.items():
                if k != 'record_date' and v:
                    amt = _safe_float(v)
                    if amt:
                        break
        if amt:
            result[year] = amt
    return dict(sorted(result.items()))

def process_all():
    """Process all raw data and save processed files."""
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)

    print("Processing debt_to_penny...")
    debt_monthly = process_debt_to_penny()
    with open(os.path.join(DATA_DIR, 'processed', 'debt_monthly.json'), 'w') as f:
        json.dump(debt_monthly, f, indent=2)
    print(f"  → {len(debt_monthly)} months")

    print("Processing avg_interest_rates...")
    rates = process_avg_interest_rates()
    with open(os.path.join(DATA_DIR, 'processed', 'interest_rates.json'), 'w') as f:
        json.dump(rates, f, indent=2)
    print(f"  → {len(rates)} months")

    print("Processing interest_expense...")
    expense = process_interest_expense()
    with open(os.path.join(DATA_DIR, 'processed', 'interest_expense.json'), 'w') as f:
        json.dump(expense, f, indent=2)
    print(f"  → {len(expense)} months")

    print("Processing mspd...")
    mspd = process_mspd()
    with open(os.path.join(DATA_DIR, 'processed', 'mspd.json'), 'w') as f:
        json.dump(mspd, f, indent=2)
    print(f"  → {len(mspd)} months")

    print("Processing debt_outstanding...")
    historical = process_debt_outstanding()
    with open(os.path.join(DATA_DIR, 'processed', 'debt_historical.json'), 'w') as f:
        json.dump(historical, f, indent=2)
    print(f"  → {len(historical)} years")

    return {
        'debt_monthly': debt_monthly,
        'interest_rates': rates,
        'interest_expense': expense,
        'mspd': mspd,
        'debt_historical': historical,
    }

if __name__ == '__main__':
    process_all()
