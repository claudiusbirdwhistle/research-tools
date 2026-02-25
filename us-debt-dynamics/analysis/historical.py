"""Analysis 4: 236-Year Historical Regime Analysis (1790-2025).

Identifies structural regimes in US federal debt:
- Pre-Civil War baseline
- War spikes (1812, Civil War, WWI, WWII)
- Post-war paydown cycles
- The 1981 structural shift
- Modern structural growth
"""
import json
import os
import math
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_processed(name):
    with open(os.path.join(DATA_DIR, 'processed', f'{name}.json')) as f:
        return json.load(f)

def run():
    historical = load_processed('debt_historical')

    years = sorted(int(y) for y in historical.keys())
    debts = [historical[str(y)] for y in years]

    # Convert to billions for readability
    debts_bil = []
    for d in debts:
        if d > 1e12:
            debts_bil.append(d / 1e9)  # Already in dollars, convert to billions
        elif d > 1e9:
            debts_bil.append(d / 1e9)
        elif d > 1e6:
            debts_bil.append(d / 1e6)  # Might be in thousands
        else:
            debts_bil.append(d / 1e6)  # Early years in raw dollars, convert to millions

    # Actually, let's check the scale of the data
    # 1790 should be ~$75M, 2025 should be ~$36T
    first_val = debts[0]
    last_val = debts[-1]
    print(f"Data range: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"First value (1790): {first_val:,.0f}")
    print(f"Last value ({years[-1]}): {last_val:,.0f}")

    # Determine scale factor
    # If 1790 value is ~75M and stored as 75,463,476.52, it's in dollars
    # If stored as 75.46, it's in millions
    if first_val > 1e6:
        # Values are in raw dollars
        scale = 'dollars'
        to_billions = lambda x: x / 1e9
        to_millions = lambda x: x / 1e6
    elif first_val > 1e3:
        # Values might be in thousands
        scale = 'thousands'
        to_billions = lambda x: x / 1e6
        to_millions = lambda x: x / 1e3
    else:
        # Values in millions
        scale = 'millions'
        to_billions = lambda x: x / 1e3
        to_millions = lambda x: x

    print(f"Scale detected: {scale}")

    # === Key Historical Milestones ===
    milestones = []
    milestone_years = {
        1790: "Post-Revolution baseline",
        1812: "War of 1812",
        1835: "Jackson pays off national debt",
        1866: "Civil War peak",
        1893: "Post-Civil War paydown low",
        1919: "WWI peak",
        1930: "Post-WWI paydown low",
        1946: "WWII peak",
        1974: "Post-WWII paydown low (as % GDP)",
        1981: "Reagan structural shift begins",
        2001: "Clinton surplus era ends",
        2008: "Great Recession begins",
        2019: "Pre-COVID",
        2020: "COVID spike",
        2025: "Current",
    }

    for yr, desc in milestone_years.items():
        if str(yr) in historical:
            val = historical[str(yr)]
            milestones.append({
                'year': yr,
                'description': desc,
                'debt_billions': round(to_billions(val), 3),
                'debt_millions': round(to_millions(val), 1),
            })

    # === Doubling Times by Era ===
    doubling_times = []

    def compute_doubling_time(start_yr, end_yr, label):
        s = str(start_yr)
        e = str(end_yr)
        if s not in historical or e not in historical:
            return None
        start_val = historical[s]
        end_val = historical[e]
        if start_val <= 0 or end_val <= start_val:
            return None
        n_years = end_yr - start_yr
        growth_rate = (end_val / start_val) ** (1 / n_years) - 1
        if growth_rate <= 0:
            return None
        doubling_time = math.log(2) / math.log(1 + growth_rate)
        return {
            'era': label,
            'start_year': start_yr,
            'end_year': end_yr,
            'start_debt_bil': round(to_billions(start_val), 3),
            'end_debt_bil': round(to_billions(end_val), 3),
            'cagr_pct': round(growth_rate * 100, 2),
            'doubling_time_years': round(doubling_time, 1),
            'total_growth_factor': round(end_val / start_val, 2),
        }

    eras = [
        (1790, 1860, "Pre-Civil War"),
        (1860, 1865, "Civil War"),
        (1866, 1893, "Post-Civil War paydown"),
        (1893, 1916, "Progressive Era"),
        (1916, 1919, "WWI"),
        (1919, 1930, "Post-WWI paydown"),
        (1930, 1940, "Great Depression"),
        (1940, 1946, "WWII"),
        (1946, 1981, "Post-WWII era"),
        (1981, 2001, "Reagan-Clinton era"),
        (2001, 2008, "Bush era"),
        (2008, 2019, "Post-crisis expansion"),
        (2019, 2025, "COVID and aftermath"),
    ]

    for start, end, label in eras:
        dt = compute_doubling_time(start, end, label)
        if dt:
            doubling_times.append(dt)

    # === Post-War Paydown Analysis ===
    # Key question: after each major war, did debt decline?
    paydown_analysis = []

    war_peaks = [
        (1866, "Civil War", 1893),    # Peak → trough
        (1919, "WWI", 1930),
        (1946, "WWII", 1974),         # Debt stopped declining in absolute terms ~1950s but fell as % GDP until 1974
    ]

    for peak_yr, war, trough_yr in war_peaks:
        s_peak = str(peak_yr)
        s_trough = str(trough_yr)
        if s_peak in historical and s_trough in historical:
            peak_val = historical[s_peak]
            trough_val = historical[s_trough]
            pct_change = (trough_val / peak_val - 1) * 100
            paydown_analysis.append({
                'war': war,
                'peak_year': peak_yr,
                'trough_year': trough_yr,
                'peak_debt_bil': round(to_billions(peak_val), 3),
                'trough_debt_bil': round(to_billions(trough_val), 3),
                'absolute_change_pct': round(pct_change, 1),
                'years_to_trough': trough_yr - peak_yr,
                'debt_decreased': trough_val < peak_val,
            })

    # === Structural Break Detection ===
    # Simple approach: segmented regression on log(debt)
    # Look for years where the growth rate permanently shifted

    log_debts = []
    valid_years = []
    for yr in years:
        val = historical[str(yr)]
        if val > 0:
            log_debts.append(math.log(val))
            valid_years.append(yr)

    log_debts = np.array(log_debts)
    valid_years = np.array(valid_years)

    # Test candidate breakpoints
    candidate_breaks = [1835, 1860, 1866, 1916, 1919, 1930, 1940, 1946, 1981, 2001, 2008]
    breakpoint_results = []

    for bp in candidate_breaks:
        if bp not in valid_years:
            # Find nearest
            idx = np.argmin(np.abs(valid_years - bp))
            bp = valid_years[idx]

        bp_idx = np.where(valid_years == bp)[0]
        if len(bp_idx) == 0:
            continue
        bp_idx = bp_idx[0]

        # Compute growth rates before and after breakpoint
        window = 15  # years

        pre_start = max(0, bp_idx - window)
        pre_end = bp_idx
        post_start = bp_idx
        post_end = min(len(valid_years), bp_idx + window)

        if pre_end - pre_start < 5 or post_end - post_start < 5:
            continue

        # OLS on log(debt) vs year
        pre_years = valid_years[pre_start:pre_end]
        pre_log = log_debts[pre_start:pre_end]
        post_years = valid_years[post_start:post_end]
        post_log = log_debts[post_start:post_end]

        pre_slope = np.polyfit(pre_years, pre_log, 1)[0]
        post_slope = np.polyfit(post_years, post_log, 1)[0]

        # Convert log-slopes to annual growth rates
        pre_growth = (math.exp(pre_slope) - 1) * 100
        post_growth = (math.exp(post_slope) - 1) * 100

        breakpoint_results.append({
            'year': int(bp),
            'pre_growth_rate_pct': round(pre_growth, 2),
            'post_growth_rate_pct': round(post_growth, 2),
            'growth_change_pp': round(post_growth - pre_growth, 2),
            'direction': 'acceleration' if post_growth > pre_growth else 'deceleration',
        })

    # === The Key Regime Shift: Pre-1981 vs Post-1981 ===
    # Before 1981: debt was episodic (war spike → peacetime paydown)
    # After 1981: debt is structural (persistent growth regardless of conditions)

    # Post-war: find years where debt declined year-over-year
    decline_years = []
    for i in range(1, len(valid_years)):
        if historical[str(valid_years[i])] < historical[str(valid_years[i-1])]:
            decline_years.append(int(valid_years[i]))

    last_decline_year = max(decline_years) if decline_years else None

    regime_analysis = {
        'episodic_era': {
            'period': '1790-1981',
            'description': 'Debt rises during wars/crises, then declines during peacetime',
            'decline_years_count': len([y for y in decline_years if y <= 1981]),
            'total_years': len([y for y in valid_years if y <= 1981]),
        },
        'structural_era': {
            'period': '1981-2025',
            'description': 'Debt grows persistently regardless of war, peace, or economic conditions',
            'decline_years_count': len([y for y in decline_years if y > 1981]),
            'total_years': len([y for y in valid_years if y > 1981]),
        },
        'last_year_debt_declined': last_decline_year,
        'years_since_last_decline': years[-1] - last_decline_year if last_decline_year else None,
    }

    # === Full Growth Rate Time Series ===
    growth_series = {}
    for i in range(1, len(valid_years)):
        yr = int(valid_years[i])
        prev_yr = int(valid_years[i-1])
        curr = historical[str(yr)]
        prev = historical[str(prev_yr)]
        if prev > 0:
            growth = (curr / prev - 1) * 100
            growth_series[yr] = round(growth, 2)

    results = {
        'data_range': {'first_year': int(years[0]), 'last_year': int(years[-1]), 'n_years': len(years)},
        'milestones': milestones,
        'doubling_times': doubling_times,
        'paydown_analysis': paydown_analysis,
        'breakpoints': breakpoint_results,
        'regime_analysis': regime_analysis,
        'growth_series': growth_series,
        'decline_years': decline_years,
    }

    # Save
    out_path = os.path.join(DATA_DIR, 'analysis', 'historical.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Historical analysis saved to {out_path}")
    print(f"  Data range: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Milestones: {len(milestones)}")
    print(f"  Doubling times by era:")
    for dt in doubling_times:
        print(f"    {dt['era']}: {dt['doubling_time_years']:.1f} years (CAGR {dt['cagr_pct']}%)")
    print(f"  Post-war paydown analysis:")
    for pa in paydown_analysis:
        direction = "↓ decreased" if pa['debt_decreased'] else "↑ increased"
        print(f"    {pa['war']}: {direction} {pa['absolute_change_pct']:.1f}% over {pa['years_to_trough']} years")
    print(f"  Regime shift:")
    print(f"    Last year debt declined: {regime_analysis['last_year_debt_declined']}")
    print(f"    Years since: {regime_analysis['years_since_last_decline']}")
    print(f"    Episodic era decline years: {regime_analysis['episodic_era']['decline_years_count']}/{regime_analysis['episodic_era']['total_years']}")
    print(f"    Structural era decline years: {regime_analysis['structural_era']['decline_years_count']}/{regime_analysis['structural_era']['total_years']}")

    return results

if __name__ == '__main__':
    run()
