"""
ENSO-River Flow Composition Analysis
Tests whether El Niño / La Niña events systematically affect streamflow
in 10 major US rivers. Composes ocean-warming (ENSO events) + river-flow data.
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime

# River metadata (station_id -> info)
RIVER_INFO = {
    '09380000': {'river': 'Colorado', 'basin': 'Colorado/West', 'regime': 'snowmelt'},
    '07022000': {'river': 'Mississippi', 'basin': 'Mississippi/Central', 'regime': 'mixed'},
    '14105700': {'river': 'Columbia', 'basin': 'Columbia/PNW', 'regime': 'snowmelt'},
    '01646500': {'river': 'Potomac', 'basin': 'Atlantic/East', 'regime': 'rain'},
    '11377100': {'river': 'Sacramento', 'basin': 'California', 'regime': 'mixed'},
    '06893000': {'river': 'Missouri', 'basin': 'Missouri/Plains', 'regime': 'mixed'},
    '03294500': {'river': 'Ohio', 'basin': 'Ohio/Appalachian', 'regime': 'rain'},
    '08313000': {'river': 'Rio Grande', 'basin': 'Rio Grande/SW', 'regime': 'snowmelt'},
    '06191500': {'river': 'Yellowstone', 'basin': 'Yellowstone/N.Rockies', 'regime': 'snowmelt'},
    '01570500': {'river': 'Susquehanna', 'basin': 'Chesapeake/NE', 'regime': 'rain'},
}


def load_nino34():
    """Load Niño 3.4 monthly SST, compute anomalies."""
    with open('/tools/ocean-warming/data/processed/nino34.json') as f:
        raw = json.load(f)

    # Build monthly time series
    months = {}
    for key, val in raw.items():
        months[key] = val['mean_sst']

    # Compute climatology (1971-2000 base period)
    clim = defaultdict(list)
    for key, sst in months.items():
        yr, mo = key.split('-')
        if 1971 <= int(yr) <= 2000:
            clim[int(mo)].append(sst)
    climatology = {mo: np.mean(vals) for mo, vals in clim.items()}

    # Compute anomalies
    anomalies = {}
    for key, sst in months.items():
        yr, mo = key.split('-')
        anomalies[key] = sst - climatology[int(mo)]

    return months, anomalies, climatology


def load_enso_events():
    """Load ENSO event catalog from ocean-warming analysis."""
    with open('/tools/ocean-warming/data/analysis/enso.json') as f:
        data = json.load(f)

    ed = data['event_detection']
    el_nino = ed['el_nino_events']
    la_nina = ed['la_nina_events']
    return el_nino, la_nina


def load_river_monthly(station_id):
    """Load river daily data, aggregate to monthly mean flow."""
    with open(f'/tools/river-flow/data/raw/{station_id}.json') as f:
        data = json.load(f)

    info = data
    monthly = defaultdict(list)
    for rec in data['records']:
        if rec['flow_cfs'] is None:
            continue
        date = rec['date']  # YYYY-MM-DD
        ym = date[:7]  # YYYY-MM
        monthly[ym].append(rec['flow_cfs'])

    # Average each month
    monthly_mean = {}
    for ym, flows in monthly.items():
        if len(flows) >= 15:  # require >=15 days
            monthly_mean[ym] = np.mean(flows)

    return monthly_mean, info['site_name']


def compute_flow_anomalies(monthly_mean):
    """Compute monthly flow anomalies (z-scores) relative to calendar month climatology."""
    # Build climatology
    clim = defaultdict(list)
    for ym, flow in monthly_mean.items():
        mo = int(ym.split('-')[1])
        clim[mo].append(flow)

    climatology = {}
    std_dev = {}
    for mo, vals in clim.items():
        climatology[mo] = np.mean(vals)
        std_dev[mo] = np.std(vals) if len(vals) > 1 else 1.0

    # Z-scores
    anomalies = {}
    for ym, flow in monthly_mean.items():
        mo = int(ym.split('-')[1])
        if std_dev[mo] > 0:
            anomalies[ym] = (flow - climatology[mo]) / std_dev[mo]
        else:
            anomalies[ym] = 0.0

    return anomalies, climatology, std_dev


def month_offset(base_ym, offset):
    """Add offset months to YYYY-MM string."""
    yr, mo = int(base_ym[:4]), int(base_ym[5:7])
    mo += offset
    while mo > 12:
        yr += 1
        mo -= 12
    while mo < 1:
        yr -= 1
        mo += 12
    return f'{yr:04d}-{mo:02d}'


def superposed_epoch_analysis(events, flow_anomalies, window=(-6, 18)):
    """
    Superposed epoch analysis: composite flow anomaly aligned to event peak.
    Returns composite mean, std, n_events, and per-offset arrays.
    """
    lo, hi = window
    offsets = list(range(lo, hi + 1))
    composites = {off: [] for off in offsets}

    for event in events:
        peak = event['peak_date']  # YYYY-MM
        for off in offsets:
            ym = month_offset(peak, off)
            if ym in flow_anomalies:
                composites[off].append(flow_anomalies[ym])

    result = {}
    for off in offsets:
        vals = composites[off]
        if len(vals) >= 5:
            result[off] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'n': len(vals),
                'se': float(np.std(vals) / np.sqrt(len(vals))),
            }

    return result


def bootstrap_significance(events, flow_anomalies, n_boot=5000, window=(-6, 18)):
    """Bootstrap significance test: shuffle event years, recompute composites."""
    lo, hi = window
    offsets = list(range(lo, hi + 1))

    # Get observed composite
    observed = superposed_epoch_analysis(events, flow_anomalies, window)

    # Get all available years in flow data
    all_years = sorted(set(int(ym[:4]) for ym in flow_anomalies.keys()))

    # Bootstrap: randomly assign events to different years
    rng = np.random.default_rng(42)
    boot_means = {off: [] for off in offsets if off in observed}

    for _ in range(n_boot):
        # Randomly shift each event's peak year
        fake_events = []
        for event in events:
            peak = event['peak_date']
            orig_yr = int(peak[:4])
            new_yr = rng.choice(all_years)
            new_peak = f'{new_yr:04d}{peak[4:]}'
            fake_events.append({'peak_date': new_peak})

        fake_composite = superposed_epoch_analysis(fake_events, flow_anomalies, window)
        for off in boot_means:
            if off in fake_composite:
                boot_means[off].append(fake_composite[off]['mean'])

    # Compute p-values
    p_values = {}
    for off in boot_means:
        if off in observed and len(boot_means[off]) > 100:
            obs_val = observed[off]['mean']
            boot_arr = np.array(boot_means[off])
            # Two-sided p-value
            p = np.mean(np.abs(boot_arr) >= np.abs(obs_val))
            p_values[off] = float(p)

    return p_values


def monthly_correlation(nino34_anomalies, flow_anomalies):
    """Compute monthly correlation between Niño 3.4 and flow anomalies."""
    from scipy import stats

    common = sorted(set(nino34_anomalies.keys()) & set(flow_anomalies.keys()))
    if len(common) < 24:
        return None

    x = np.array([nino34_anomalies[ym] for ym in common])
    y = np.array([flow_anomalies[ym] for ym in common])

    # Overall correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # Seasonal correlation
    seasonal = {}
    for season, months in [('DJF', [12, 1, 2]), ('MAM', [3, 4, 5]),
                           ('JJA', [6, 7, 8]), ('SON', [9, 10, 11])]:
        sx, sy = [], []
        for ym in common:
            mo = int(ym.split('-')[1])
            if mo in months:
                sx.append(nino34_anomalies[ym])
                sy.append(flow_anomalies[ym])
        if len(sx) >= 20:
            sr, sp = stats.pearsonr(sx, sy)
            seasonal[season] = {'r': float(sr), 'p': float(sp), 'n': len(sx)}

    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_months': len(common),
        'period': f'{common[0]} to {common[-1]}',
        'seasonal': seasonal,
    }


def lag_correlation(nino34_anomalies, flow_anomalies, max_lag=12):
    """Cross-correlation at lags 0 to max_lag months (Niño leads flow)."""
    from scipy import stats

    common_base = sorted(set(nino34_anomalies.keys()) & set(flow_anomalies.keys()))
    if len(common_base) < 24:
        return None

    results = []
    for lag in range(0, max_lag + 1):
        pairs = []
        for ym in common_base:
            lagged_ym = month_offset(ym, lag)
            if lagged_ym in flow_anomalies:
                pairs.append((nino34_anomalies[ym], flow_anomalies[lagged_ym]))

        if len(pairs) >= 20:
            x, y = zip(*pairs)
            r, p = stats.pearsonr(x, y)
            results.append({
                'lag_months': lag,
                'r': float(r),
                'p': float(p),
                'n': len(pairs),
            })

    return results


def asymmetry_test(el_nino_composite, la_nina_composite):
    """Test if El Niño and La Niña produce asymmetric flow responses."""
    # Compare mean absolute response during event (months 0-6)
    el_response = []
    la_response = []
    for off in range(0, 7):
        if off in el_nino_composite:
            el_response.append(abs(el_nino_composite[off]['mean']))
        if off in la_nina_composite:
            la_response.append(abs(la_nina_composite[off]['mean']))

    if el_response and la_response:
        el_mean = np.mean(el_response)
        la_mean = np.mean(la_response)
        ratio = el_mean / la_mean if la_mean > 0 else float('inf')
        return {
            'el_nino_mean_abs_response': float(el_mean),
            'la_nina_mean_abs_response': float(la_mean),
            'asymmetry_ratio': float(ratio),
            'stronger': 'El Niño' if el_mean > la_mean else 'La Niña',
        }
    return None


def run_analysis():
    """Run full ENSO-river flow analysis."""
    print("Loading ENSO data...")
    nino34_sst, nino34_anom, _ = load_nino34()
    el_nino_events, la_nina_events = load_enso_events()

    print(f"  {len(el_nino_events)} El Niño events, {len(la_nina_events)} La Niña events")
    print(f"  Niño 3.4: {len(nino34_anom)} months")

    results = {'rivers': {}, 'metadata': {
        'n_el_nino': len(el_nino_events),
        'n_la_nina': len(la_nina_events),
        'nino34_months': len(nino34_anom),
        'analysis_date': datetime.utcnow().isoformat() + 'Z',
        'window': [-6, 18],
        'bootstrap_n': 5000,
    }}

    for station_id, info in RIVER_INFO.items():
        river = info['river']
        print(f"\nAnalyzing {river} ({station_id})...")

        # Load and compute anomalies
        monthly_mean, site_name = load_river_monthly(station_id)
        flow_anom, flow_clim, flow_std = compute_flow_anomalies(monthly_mean)

        print(f"  {len(monthly_mean)} months, {len(flow_anom)} anomalies")

        # Superposed epoch analysis
        print("  SEA: El Niño...")
        el_composite = superposed_epoch_analysis(el_nino_events, flow_anom)
        print("  SEA: La Niña...")
        la_composite = superposed_epoch_analysis(la_nina_events, flow_anom)

        # Bootstrap significance (El Niño only — expensive)
        print("  Bootstrap significance (5000 iterations)...")
        el_pvals = bootstrap_significance(el_nino_events, flow_anom, n_boot=5000)
        la_pvals = bootstrap_significance(la_nina_events, flow_anom, n_boot=5000)

        # Monthly correlation
        print("  Correlation analysis...")
        corr = monthly_correlation(nino34_anom, flow_anom)

        # Lag correlation
        lag_corr = lag_correlation(nino34_anom, flow_anom, max_lag=12)

        # Asymmetry test
        asym = asymmetry_test(el_composite, la_composite)

        # Find peak response
        el_peak_off = max(el_composite.keys(), key=lambda k: abs(el_composite[k]['mean'])) if el_composite else None
        la_peak_off = max(la_composite.keys(), key=lambda k: abs(la_composite[k]['mean'])) if la_composite else None

        # Best lag
        best_lag = None
        if lag_corr:
            best_lag = max(lag_corr, key=lambda x: abs(x['r']))

        # Count significant months
        el_sig = sum(1 for off, p in el_pvals.items() if p < 0.05 and 0 <= off <= 12)
        la_sig = sum(1 for off, p in la_pvals.items() if p < 0.05 and 0 <= off <= 12)

        river_result = {
            'station_id': station_id,
            'river': river,
            'basin': info['basin'],
            'regime': info['regime'],
            'n_months': len(monthly_mean),
            'el_nino_composite': {str(k): v for k, v in el_composite.items()},
            'la_nina_composite': {str(k): v for k, v in la_composite.items()},
            'el_nino_pvalues': {str(k): v for k, v in el_pvals.items()},
            'la_nina_pvalues': {str(k): v for k, v in la_pvals.items()},
            'correlation': corr,
            'lag_correlation': lag_corr,
            'asymmetry': asym,
            'el_nino_peak': {
                'offset': el_peak_off,
                'anomaly': el_composite[el_peak_off]['mean'] if el_peak_off is not None else None,
                'p_value': el_pvals.get(el_peak_off),
            } if el_peak_off is not None else None,
            'la_nina_peak': {
                'offset': la_peak_off,
                'anomaly': la_composite[la_peak_off]['mean'] if la_peak_off is not None else None,
                'p_value': la_pvals.get(la_peak_off),
            } if la_peak_off is not None else None,
            'el_significant_months': el_sig,
            'la_significant_months': la_sig,
            'best_lag': best_lag,
        }

        results['rivers'][river] = river_result

        # Print summary
        if corr:
            print(f"  Overall r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f})")
        if best_lag:
            print(f"  Best lag: {best_lag['lag_months']}mo, r={best_lag['r']:.3f} (p={best_lag['p']:.4f})")
        print(f"  El Niño significant months (0-12): {el_sig}")
        print(f"  La Niña significant months (0-12): {la_sig}")

    # Compute rankings
    rankings = []
    for river, data in results['rivers'].items():
        sensitivity = 0
        if data['correlation']:
            sensitivity = abs(data['correlation']['pearson_r'])
        rankings.append({
            'river': river,
            'abs_correlation': float(sensitivity),
            'el_sig_months': data['el_significant_months'],
            'la_sig_months': data['la_significant_months'],
            'total_sig_months': data['el_significant_months'] + data['la_significant_months'],
            'regime': data['regime'],
            'basin': data['basin'],
        })

    rankings.sort(key=lambda x: x['total_sig_months'], reverse=True)
    results['rankings'] = rankings

    # Save results
    output_path = '/tools/enso-river/data/analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    run_analysis()
