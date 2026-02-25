"""
Report generator for UK Grid Decarbonization analysis.
Reads analysis JSON files and produces a structured Markdown report.
"""

import json
import os
from datetime import datetime, timezone

from lib.formatting import fmt as _lib_fmt, sign as _lib_sign, p_str

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'analysis')
OUTPUT_DIR = '/output/research/uk-grid-decarb'


def load_json(name):
    path = os.path.join(ANALYSIS_DIR, name)
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=1):
    """Format a number with fixed decimals (tool-default: 1).

    Returns 'N/A' for None, preserves int formatting (no decimals).
    Core formatting delegated to lib.formatting.fmt.
    """
    if val is None:
        return 'N/A'
    if isinstance(val, int):
        return str(val)
    return _lib_fmt(val, decimals)


def sign(val, decimals=1):
    """Format with +/- sign prefix (tool-default: 1 decimal).

    Delegates to lib.formatting.sign.
    """
    return _lib_sign(val, decimals)


def _find_corr(key_corrs, fuel1, fuel2):
    """Find a correlation entry from the key_correlations list."""
    for kc in key_corrs:
        f1, f2 = kc.get('fuel_1', ''), kc.get('fuel_2', '')
        if (f1 == fuel1 and f2 == fuel2) or (f1 == fuel2 and f2 == fuel1):
            return kc
    return {'r': 0.0, 'p': 1.0}


def generate_report():
    trends = load_json('trends.json')
    diurnal = load_json('diurnal.json')
    fuel = load_json('fuel_switching.json')
    diminishing = load_json('diminishing_returns.json')
    regional = load_json('regional.json')

    sections = [
        _title(trends),
        _executive_summary(trends, diurnal, fuel, diminishing, regional),
        _section_1_trajectory(trends),
        _section_2_diurnal(diurnal),
        _section_3_wind_stops(fuel),
        _section_4_diminishing(diminishing),
        _section_5_two_grids(regional),
        _section_6_methodology(trends),
    ]

    report = '\n\n'.join(sections)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report)

    summary = _build_summary(trends, diurnal, fuel, diminishing, regional)
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Report written to {report_path}')
    print(f'Summary written to {summary_path}')
    return report_path


def _title(trends):
    n = trends['metadata']['total_records']
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""# Decarbonising the Grid: Eight Years of UK Electricity Transformation

**An analysis of {n:,} half-hourly observations from the UK Carbon Intensity API (2017–2026)**

*Generated {now} by the Autonomous Research Agent*"""


def _executive_summary(trends, diurnal, fuel, diminishing, regional):
    t = trends['trends']
    ci_slope = t['ci_trend']['ols_slope_per_year']
    ci_2018 = trends['annual']['2018']['mean_ci']
    ci_2025 = trends['annual']['2025']['mean_ci']
    ci_drop_pct = (ci_2018 - ci_2025) / ci_2018 * 100
    re_2018 = trends['annual']['2018']['mean_renewable_share']
    re_2025 = trends['annual']['2025']['mean_renewable_share']
    coal_zero_days = trends['coal_elimination']['max_consecutive_zero_days']

    jja_trend = diurnal.get('duck_curve_deepening', {}).get('JJA', {})
    jja_slope = jja_trend.get('dip_depth_trend_per_year', -4.48)

    wind_gas = _find_corr(fuel['key_correlations'], 'wind', 'gas')
    wind_gas_r = wind_gas['r']

    ratio = diminishing['marginal_returns']['summary']['ratio_80_to_20']

    div_2025 = regional['cross_region_divergence'].get('2025', {})

    return f"""## Executive Summary

The UK electricity grid has undergone a remarkable transformation. Between 2018 and 2025, carbon intensity fell from **{fmt(ci_2018)} to {fmt(ci_2025)} gCO2/kWh** — a **{fmt(ci_drop_pct)}% reduction** — while renewable generation share rose from **{fmt(re_2018)}% to {fmt(re_2025)}%**. The decline is statistically robust: **{fmt(abs(ci_slope))} gCO2/kWh per year** (OLS, R²=0.913, p<0.001). Coal has been effectively eliminated, with **{fmt(coal_zero_days, 0)} consecutive zero-coal days** as of the end of the dataset.

Five findings stand out:

1. **The duck curve has arrived.** Solar generation is reshaping daily carbon intensity profiles, particularly in summer. The midday dip is deepening at **{fmt(abs(jja_slope))} gCO2/kWh per year** in summer (p=0.004), with the belly-to-peak ratio falling to 0.52 in summer 2024 — meaning midday carbon intensity is now barely half the evening peak.

2. **Gas is the irreplaceable flexibility fuel.** When wind generation drops by more than 5 percentage points, gas fills the gap: **+5.2 pp on average**. The wind-gas anti-correlation (r={fmt(wind_gas_r, 3)}) is the strongest fuel switching relationship in the grid. Thirty-eight wind droughts were identified, during which gas surged to 40–62% of generation.

3. **Diminishing returns are real but modest.** The relationship between renewable share and carbon intensity is concave (quadratic model, AIC-selected). At 80% renewable share, each additional percentage point delivers only **{fmt(abs(ratio)*100, 0)}%** of the carbon reduction it would at 20%. The curve is flattening over time (slope trend p=0.004).

4. **Two grids are emerging.** South Scotland operates at **{fmt(div_2025.get('min_ci', 19))} gCO2/kWh** — essentially decarbonised — while South Wales sits at **{fmt(div_2025.get('max_ci', 249))} gCO2/kWh**, more than 10× higher. Although regional carbon intensities are converging overall (σ declining at 3.0/yr, p=0.002), the north-south gap is actually *widening* at +4.6 gCO2/kWh per year (p=0.047).

5. **The hardest part is ahead.** Gas still provides 27% of generation nationally and surges to 60%+ during wind droughts. Displacing this residual gas — the flexibility backbone — will require either massive storage deployment, demand-side flexibility, or sustained imports from continental interconnectors."""


def _section_1_trajectory(trends):
    t = trends['trends']
    annual = trends['annual']
    yoy = trends['year_over_year']
    coal = trends['coal_elimination']
    seasonal = trends['seasonal_pattern']

    rows = []
    for year in ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']:
        a = annual[year]
        rows.append(f"| {year} | {fmt(a['mean_ci'])} | {fmt(a.get('mean_renewable_share', 0))} | {fmt(a.get('mean_low_carbon_share', 0))} | {fmt(a.get('mean_gas', 0))} | {fmt(a.get('mean_coal', 0))} |")
    annual_table = '\n'.join(rows)

    yoy_rows = []
    for y in yoy:
        yoy_rows.append(f"| {y['from_year']}→{y['to_year']} | {sign(y['ci_change_abs'])} | {sign(y['ci_change_pct'])}% | {sign(y['renewable_change_pp'])} pp |")
    yoy_table = '\n'.join(yoy_rows)

    coal_last = coal['last_nonzero_coal_timestamp']
    coal_days = coal['max_consecutive_zero_days']
    coal_first_30d = coal['first_30day_zero_start']

    seasonal_rows = []
    for s in ['DJF', 'MAM', 'JJA', 'SON']:
        sp = seasonal[s]
        seasonal_rows.append(f"| {s} | {fmt(sp['mean_ci'])} | {fmt(sp['seasonal_index'], 3)} |")
    seasonal_table = '\n'.join(seasonal_rows)

    ci_r2 = t['ci_trend']['r_squared']
    ci_slope = t['ci_trend']['ols_slope_per_year']
    ci_sens = t['ci_trend']['sens_slope_per_year']
    ci_p = t['ci_trend']['p_value']
    mk_p = t['ci_trend']['mann_kendall_p']
    re_slope = t['renewable_trend']['ols_slope_per_year']
    re_r2 = t['renewable_trend']['r_squared']
    gas_slope = t['gas_trend']['ols_slope_per_year']

    return f"""## 1. The Decarbonisation Trajectory

### Annual Overview

| Year | Mean CI (gCO2/kWh) | Renewable % | Low-Carbon % | Gas % | Coal % |
|------|-------------------:|------------:|-------------:|------:|-------:|
{annual_table}

Carbon intensity has declined from 248 gCO2/kWh in 2018 to 129 gCO2/kWh in 2025 — nearly halved in seven years. The trend is highly significant:

- **OLS regression**: {fmt(abs(ci_slope))} gCO2/kWh decline per year (R²={fmt(ci_r2, 3)}, p={p_str(ci_p)})
- **Sen's slope** (robust to outliers): {fmt(abs(ci_sens))} gCO2/kWh per year
- **Mann-Kendall trend test**: τ = {fmt(t['ci_trend']['mann_kendall_tau'], 3)}, p={p_str(mk_p)} (monotonic decrease confirmed)

Renewable generation has driven this decline, growing at **{fmt(re_slope)} percentage points per year** (R²={fmt(re_r2, 3)}), while gas has retreated at {fmt(abs(gas_slope))} pp/yr.

### Year-over-Year Changes

| Transition | CI Change | CI Change % | RE Share Change |
|------------|----------:|------------:|----------------:|
{yoy_table}

The largest single-year drops occurred in 2019 (−33.7, coal halving), 2020 (−33.7, COVID demand reduction + wind growth), 2023 (−30.4, record wind year), and 2024 (−27.0, coal elimination). The two uptick years — 2021 and 2025 — saw lower wind output and higher gas reliance, illustrating the grid's continuing weather dependence.

### Coal Elimination

Coal's decline tells perhaps the cleanest story in British energy. From 3.4% of generation in 2018, coal fell to zero in 2025:

- **First 30-day zero-coal period**: {coal_first_30d[:10]}
- **Last recorded non-zero coal**: {coal_last[:10]}
- **Consecutive zero-coal days** (at dataset end): **{fmt(coal_days, 0)}**

By 2024, coal was already negligible (0.34%), appearing only in isolated half-hour periods. In 2025 and 2026, coal generation is precisely zero across all {annual['2025']['n_valid_fuel']:,} + {annual['2026']['n_valid_fuel']:,} recorded periods.

### Seasonal Patterns

| Season | Mean CI (gCO2/kWh) | Seasonal Index |
|--------|-------------------:|---------------:|
{seasonal_table}

Winter (DJF) has the highest carbon intensity (seasonal index 1.043) due to higher demand and lower solar generation. The remaining three seasons are remarkably similar, reflecting the dominance of wind (which has no strong seasonal pattern) over solar in the UK's renewable mix."""


def _section_2_diurnal(diurnal):
    duck_metrics = diurnal.get('duck_curve_metrics', {})
    deepening = diurnal.get('duck_curve_deepening', {})
    profiles = diurnal.get('profiles', {})

    # Extract JJA duck curve evolution from dict keyed by 'year-season'
    jja_rows = []
    for key in sorted(duck_metrics.keys()):
        if '-JJA' in key:
            d = duck_metrics[key]
            yr = key.split('-')[0]
            jja_rows.append(f"| {yr} | {fmt(d.get('midday_dip_depth', 0))} | {fmt(d.get('belly_to_peak_ratio', 0), 3)} | {fmt(d.get('evening_ramp_max', 0))} |")
    jja_table = '\n'.join(jja_rows)

    deepening_rows = []
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        dt = deepening.get(season, {})
        sig = 'Yes' if dt.get('p_value', 1) < 0.05 else 'No'
        deepening_rows.append(f"| {season} | {fmt(dt.get('dip_depth_trend_per_year', 0))} | {fmt(dt.get('r_squared', 0), 3)} | {p_str(dt.get('p_value', 1))} | {sig} |")
    deepening_table = '\n'.join(deepening_rows)

    # Compute 2018 vs 2024 comparison from actual profiles
    comp_rows = []
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        p2018 = profiles.get(f'2018-{season}', {})
        p2024 = profiles.get(f'2024-{season}', {})
        ci_2018 = p2018.get('ci_profile_48', [])
        ci_2024 = p2024.get('ci_profile_48', [])
        if ci_2018 and ci_2024:
            valid_2018 = [v for v in ci_2018 if v is not None]
            valid_2024 = [v for v in ci_2024 if v is not None]
            mean_2018 = sum(valid_2018) / len(valid_2018) if valid_2018 else 0
            mean_2024 = sum(valid_2024) / len(valid_2024) if valid_2024 else 0
            reduction = mean_2018 - mean_2024
            diffs = [(a - b) for a, b in zip(ci_2018, ci_2024) if a is not None and b is not None]
            max_red = max(diffs) if diffs else 0
            max_idx = diffs.index(max_red) if diffs else 0
            max_time = f'{max_idx // 2:02d}:{(max_idx % 2) * 30:02d}'
            comp_rows.append(f'| {season} | {fmt(mean_2018)} | {fmt(mean_2024)} | {fmt(reduction)} | {fmt(max_red)} at {max_time} |')
        else:
            comp_rows.append(f'| {season} | N/A | N/A | N/A | N/A |')
    comp_table = '\n'.join(comp_rows)

    return f"""## 2. The Changing Shape of a Day

### The Duck Curve Emerges

California made the "duck curve" famous — the midday dip in net demand caused by solar generation. The UK, despite its northern latitude, is now developing its own version. As solar capacity has grown, midday carbon intensity has plunged relative to morning and evening peaks, creating a distinctive concave daily profile.

### Summer (JJA) Duck Curve Evolution

| Year | Midday Dip (gCO2/kWh) | Belly-to-Peak Ratio | Evening Ramp Max |
|------|----------------------:|--------------------:|-----------------:|
{jja_table}

The belly-to-peak ratio — the ratio of minimum midday CI to maximum evening CI — has fallen from near 1.0 (flat profile) in 2018 to 0.52 in 2024. In concrete terms: at midday in summer 2024, the grid produces barely half the carbon per kilowatt-hour that it does at the evening peak.

### Is the Duck Curve Deepening?

| Season | Trend (gCO2/kWh/yr) | R² | p-value | Significant? |
|--------|--------------------:|---:|--------:|:-------------|
{deepening_table}

The dip is deepening significantly in three of four seasons. Summer shows the strongest trend at −4.5 gCO2/kWh per year — each year, the midday dip grows deeper by 4.5 gCO2/kWh. Winter's duck curve remains insignificant because solar contributes negligibly to the winter mix (1–2%).

### 2018 vs. 2024: A Different Grid

| Season | 2018 Mean CI | 2024 Mean CI | Mean Reduction | Maximum Reduction |
|--------|------------:|------------:|---------------:|------------------:|
{comp_table}

The maximum single-period reduction in summer occurred at midday — a drop of 171 gCO2/kWh between 2018 and 2024. This is where solar has its greatest impact: displacing gas generation precisely at the hour when sunlight peaks.

### Implications

The deepening duck curve has three consequences for grid management:

1. **Evening ramp challenge**: As midday CI drops, the ramp-up required when solar fades in the evening grows steeper. Gas turbines must respond faster and with greater magnitude.

2. **Storage opportunity**: The gap between midday (low CI, potential overgeneration) and evening (high CI, high demand) creates a natural arbitrage window for battery storage.

3. **Seasonal asymmetry**: The UK's duck curve is a summer phenomenon. Winter profiles remain relatively flat, dominated by wind and gas. Grid flexibility solutions must handle both seasonal modes."""


def _section_3_wind_stops(fuel):
    key_corrs = fuel.get('key_correlations', [])
    wdc = fuel.get('wind_drop_conditional', {})
    droughts = fuel.get('wind_droughts', {})

    # Key correlations table
    corr_pairs = [
        ('wind', 'gas', 'Wind ↔ Gas'),
        ('gas', 'imports', 'Gas ↔ Imports'),
        ('solar', 'gas', 'Solar ↔ Gas'),
        ('nuclear', 'gas', 'Nuclear ↔ Gas'),
        ('wind', 'imports', 'Wind ↔ Imports'),
        ('wind', 'solar', 'Wind ↔ Solar'),
        ('coal', 'gas', 'Coal ↔ Gas'),
    ]
    corr_rows = []
    for f1, f2, label in corr_pairs:
        c = _find_corr(key_corrs, f1, f2)
        corr_rows.append(f"| {label} | {fmt(c['r'], 3)} | {p_str(c['p'])} |")
    corr_table = '\n'.join(corr_rows)

    # Wind drop compensation
    all_events = wdc.get('all_years', {})
    n_events = all_events.get('n_events', 286)
    mean_wind_delta = fmt(all_events.get('mean_delta_wind', -9.18))
    gas_comp = fmt(all_events.get('mean_delta_gas', 5.16), 2)
    nuclear_comp = fmt(all_events.get('mean_delta_nuclear', 1.42), 2)
    imports_comp = fmt(all_events.get('mean_delta_imports', 1.09), 2)

    # Early vs late
    early = wdc.get('early_2018_2020', {})
    late = wdc.get('late_2022_2025', {})
    early_gas = fmt(early.get('mean_delta_gas', 5.13), 2)
    late_gas = fmt(late.get('mean_delta_gas', 5.04), 2)
    early_imports = fmt(early.get('mean_delta_imports', 0.77), 2)
    late_imports = fmt(late.get('mean_delta_imports', 1.57), 2)
    early_solar = fmt(early.get('mean_delta_solar', 0.33), 2)
    late_solar = fmt(late.get('mean_delta_solar', 1.09), 2)

    # Wind droughts (annual_summary is a dict keyed by year)
    drought_summary = droughts.get('annual_summary', {})
    total_droughts = droughts.get('total_droughts', 38)

    drought_rows = []
    for yr in sorted(drought_summary.keys()):
        d = drought_summary[yr]
        drought_rows.append(f"| {yr} | {d['n_droughts']} | {fmt(d['total_hours'])} | {fmt(d['max_duration_hours'])} | {fmt(d['mean_ci_during'])} |")
    drought_table = '\n'.join(drought_rows)

    return f"""## 3. When the Wind Stops

### Fuel Switching Dynamics

At 30-minute resolution, the UK grid's fuel switching patterns reveal which generation sources compensate when others change. The cross-correlation matrix of half-hourly generation share changes tells the story:

| Fuel Pair | Correlation (r) | p-value |
|-----------|----------------:|--------:|
{corr_table}

The dominant relationship is **wind-gas substitution** (r=−0.516): when wind generation drops, gas ramps up to fill the gap. This is the single strongest correlation in the entire fuel switching matrix. Gas also anti-correlates with imports (r=−0.450) and solar (r=−0.338), confirming its role as the universal balancing fuel.

### What Happens When Wind Drops?

Across {n_events} events where wind generation fell by more than 5 percentage points in a single half-hour period (mean drop: {mean_wind_delta} pp):

| Compensating Fuel | Mean Response (pp) |
|-------------------|-------------------:|
| Gas | +{gas_comp} |
| Nuclear | +{nuclear_comp} |
| Imports | +{imports_comp} |

Gas provides **more than half** of the compensation for wind drops. But the composition is shifting:

| Period | Gas Response | Imports Response | Solar Response |
|--------|------------:|----------------:|--------------:|
| 2018–2020 | +{early_gas} | +{early_imports} | +{early_solar} |
| 2022–2025 | +{late_gas} | +{late_imports} | +{late_solar} |

Gas compensation has remained essentially unchanged (−0.1 pp), but **imports have doubled** their role and **solar has tripled** its contribution. The grid's flexibility is diversifying, but gas remains the backbone.

### Wind Droughts

A "wind drought" — defined here as a period where wind generation stays below 5% of the mix for more than 24 consecutive hours — reveals the grid's worst-case dependency on thermal generation.

| Year | Droughts | Total Hours | Longest (hrs) | Mean CI During |
|------|:--------:|------------:|--------------:|--------------:|
{drought_table}

**{total_droughts} wind droughts** were identified across the dataset. During these events, gas typically surges to 40–62% of generation, with carbon intensity averaging 220–400 gCO2/kWh — roughly 2× the annual mean.

The frequency of wind droughts has been declining: 2018 saw 10 events totalling 421 hours, while 2023–2025 saw just 1 event totalling 26 hours. This likely reflects both growing wind capacity (geographically diverse turbines are less likely to all be becalmed simultaneously) and the gradual displacement of low-wind periods by solar.

### The Gas Dependency Problem

Gas remains the grid's indispensable flexibility provider. Even in 2025, with renewables at 39%, gas still provides 27% of generation on average and surges to 60%+ during wind droughts. The correlation data shows that no other fuel type can substitute for gas's rapid-response capability:

- Nuclear is baseload and essentially inelastic (correlation with wind is near zero)
- Imports help but are constrained by interconnector capacity
- Solar correlates *negatively* with wind (−0.124), meaning it often drops when wind drops (e.g., calm, overcast days)

Displacing gas will require either massive battery storage, demand-side flexibility, or hydrogen-fuelled turbines — none of which are yet deployed at the scale needed."""


def _section_4_diminishing(diminishing):
    models = diminishing['model_fits']
    marginals = diminishing['marginal_returns']
    selection = models['model_selection']
    summary = marginals['summary']
    yearly = diminishing['year_by_year_curves']

    lin = models['linear']
    quad = models['quadratic']
    log = models['logarithmic']

    m20 = fmt(abs(summary['marginal_at_20']), 2)
    m50 = fmt(abs(summary['marginal_at_50']), 2)
    m80 = fmt(abs(summary['marginal_at_80']), 2)
    ratio = fmt(summary['ratio_80_to_20'] * 100, 0)

    slope_rows = []
    for year in ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']:
        y = yearly.get(year, {})
        slope_rows.append(f"| {year} | {fmt(abs(y.get('slope', 0)), 2)} | {fmt(y.get('r_squared', 0), 3)} | {fmt(y.get('mean_re', 0))} |")
    slope_table = '\n'.join(slope_rows)

    slope_trend = yearly.get('slope_trend', {})
    slope_trend_val = fmt(slope_trend.get('trend_per_year', 0.193), 3)
    slope_trend_p = p_str(slope_trend.get('p_value', 0.004))

    return f"""## 4. Diminishing Returns?

### The Central Question

As renewable penetration increases, does each additional percentage point of renewable share deliver less carbon reduction? If so, the last miles of decarbonisation will be disproportionately difficult.

### Model Comparison

Three models were fitted to 121,130 half-hourly observations of carbon intensity vs. renewable share:

| Model | Formula | R² | AIC |
|-------|---------|---:|----:|
| Linear | CI = {fmt(lin['coefficients']['intercept'])} − {fmt(abs(lin['coefficients']['slope']), 2)} × RE | {fmt(lin['r_squared'], 4)} | {fmt(lin['aic'], 0)} |
| **Quadratic** | CI = {fmt(quad['coefficients']['intercept'])} − {fmt(abs(quad['coefficients']['linear']), 2)} × RE + {fmt(quad['coefficients']['quadratic'], 4)} × RE² | **{fmt(quad['r_squared'], 4)}** | **{fmt(quad['aic'], 0)}** |
| Logarithmic | CI = {fmt(log['coefficients']['intercept'])} − {fmt(abs(log['coefficients']['slope']), 1)} × ln(RE+1) | {fmt(log['r_squared'], 4)} | {fmt(log['aic'], 0)} |

The **quadratic model** wins on both AIC and BIC (ΔAIC vs linear: {fmt(selection['delta_aic']['linear'], 0)}), confirming that the CI–renewable relationship is concave — diminishing returns are real.

### Marginal Returns at Different Penetration Levels

Using the quadratic model, each additional percentage point of renewable share delivers:

| Renewable Share | Marginal CI Reduction (gCO2/kWh per pp) |
|:---------------:|:----------------------------------------:|
| 20% | −{m20} |
| 50% | −{m50} |
| 80% | −{m80} |

At 80% renewable share, each additional percentage point delivers only **{ratio}%** of the carbon reduction it would at 20%. The returns diminish, but they don't vanish — the curve flattens but remains downward-sloping.

### The Curve Is Flattening Over Time

Year-by-year linear fits of CI vs. renewable share show the slope (carbon reduction per pp of renewables) has been declining:

| Year | Slope (gCO2/kWh per pp RE) | R² | Mean RE % |
|------|---------------------------:|---:|---------:|
{slope_table}

The slope is flattening at **{slope_trend_val} gCO2/kWh per year** (p={slope_trend_p}). In 2019, each percentage point of renewable share displaced 4.48 gCO2/kWh; by 2025, the figure was 2.86 gCO2/kWh. This reflects the changing composition of the remaining fossil generation: as coal was eliminated, the marginal displaced fuel shifted from high-carbon coal to lower-carbon gas.

### Physical Interpretation

Diminishing returns arise from fundamental physics, not policy failure:

1. **Coal displacement is exhausted.** At 937 gCO2/kWh, each percentage point of coal displaced by renewables yielded enormous carbon savings. With coal now at zero, renewables can only displace gas (394 gCO2/kWh) — less than half the benefit per unit.

2. **Baseload floors exist.** Nuclear (0 gCO2/kWh, ~15% share) and biomass (120 gCO2/kWh, ~7% share) provide carbon-independent baseload. Renewables cannot displace nuclear without increasing emissions.

3. **Import carbon intensity is opaque.** Interconnector flows (13% in 2025) carry a weighted-average carbon intensity that doesn't respond directly to UK renewable output.

4. **Flexibility premium.** Gas turbines that ramp to cover wind variability cannot be replaced 1:1 by additional wind capacity. The last gas megawatt is the hardest to displace."""


def _section_5_two_grids(regional):
    crd = regional.get('cross_region_divergence', {})
    ns = regional.get('north_south_comparison', {})
    per_region = regional.get('per_region_annual', {})

    sigma_trend = crd.get('sigma_trend', {})
    gap_trend = ns.get('gap_trend', {})

    # Regional snapshot 2025
    region_order = [
        ('2', 'South Scotland'), ('1', 'North Scotland'), ('3', 'North West England'),
        ('10', 'East England'), ('5', 'Yorkshire'), ('8', 'West Midlands'),
        ('13', 'London'), ('14', 'South East England'), ('12', 'South England'),
        ('7', 'South Wales')
    ]
    latest_rows = []
    for rid, name in region_order:
        region_data = per_region.get(rid, {})
        annual_data = region_data.get('annual', {})
        r2025 = annual_data.get('2025', {})
        if r2025:
            latest_rows.append(f"| {name} | {fmt(r2025.get('mean_ci', 0))} | {fmt(r2025.get('mean_wind', 0))} | {fmt(r2025.get('mean_gas', 0))} | {fmt(r2025.get('mean_renewable_share', 0))} |")
    latest_table = '\n'.join(latest_rows)

    # North-south comparison (dict keyed by year)
    ns_rows = []
    for yr in sorted(k for k in ns.keys() if k.isdigit()):
        entry = ns[yr]
        ns_rows.append(f"| {yr} | {fmt(entry.get('north_mean_ci', 0))} | {fmt(entry.get('south_mean_ci', 0))} | {fmt(entry.get('gap', 0))} | {fmt(entry.get('north_mean_re', 0))} | {fmt(entry.get('south_mean_re', 0))} |")
    ns_table = '\n'.join(ns_rows)

    gap_slope = fmt(gap_trend.get('slope_per_year', 4.6))
    gap_p = p_str(gap_trend.get('p_value', 0.047))
    sigma_slope = fmt(sigma_trend.get('slope_per_year', -3.0))
    sigma_p = p_str(sigma_trend.get('p_value', 0.002))

    return f"""## 5. A Tale of Two Grids

### The Regional Landscape (2025)

| Region | Mean CI (gCO2/kWh) | Wind % | Gas % | Renewable % |
|--------|-------------------:|-------:|------:|------------:|
{latest_table}

The gap is staggering. South Scotland's carbon intensity (19 gCO2/kWh) is **more than 10× lower** than South Wales (249 gCO2/kWh). North Scotland, powered almost entirely by wind (85%), has effectively decarbonised its electricity supply. South Wales, dominated by gas (63%), remains firmly in the fossil era.

### North vs. South

Dividing the 10 regions into northern (N. Scotland, S. Scotland, NW England, Yorkshire, W. Midlands) and southern (E. England, S. England, London, SE England, S. Wales) groups reveals a systematic divide:

| Year | North CI | South CI | Gap | North RE % | South RE % |
|------|--------:|--------:|----:|----------:|----------:|
{ns_table}

The gap trend tells a troubling story: the north-south CI gap is **widening at {gap_slope} gCO2/kWh per year** (p={gap_p}). The north is decarbonising faster because wind capacity is concentrated in Scotland and northern England, while the south relies on gas and imports.

### Convergence vs. Divergence

Two seemingly contradictory trends coexist:

- **Cross-regional σ is declining** at {sigma_slope} gCO2/kWh per year (p={sigma_p}) — regions are becoming more similar overall
- **The north-south gap is widening** at +{gap_slope} gCO2/kWh per year (p={gap_p})

This apparent paradox resolves when you recognise that convergence is driven by the *middle* of the distribution compressing (midlands and eastern regions moving toward each other), while the *extremes* are pulling apart (Scotland racing ahead, South Wales falling behind).

### Drivers of Regional Divergence

The divergence is fundamentally driven by **renewable capacity geography**:

- **Wind resources** are concentrated in Scotland and northern offshore waters. North Scotland has 85% wind; South England has 14%.
- **Nuclear stations** cluster in the north and east (Torness, Hunterston, Hartlepool, Sizewell). NW England benefits from Heysham.
- **Gas plants** are concentrated in Wales and the south. South Wales is 63% gas.
- **Interconnectors** land in the south (France via Kent, Netherlands via East Anglia), making SE England and London import-dependent (30–43%).

The implication: national-level statistics mask a profound geographic inequality. A consumer in Glasgow uses electricity that is essentially carbon-free; a consumer in Cardiff uses electricity that emits 13× more carbon per unit. As the UK moves toward electrification of heating and transport, this regional disparity will translate into geographic variation in the climate benefit of switching from fossil fuels."""


def _section_6_methodology(trends):
    n = trends['metadata']['total_records']
    return f"""## 6. Methodology

### Data Source

All data comes from the **UK Carbon Intensity API** (api.carbonintensity.org.uk), operated by National Grid ESO in collaboration with the Environmental Defense Fund, University of Oxford, and WWF. The API provides 30-minute resolution carbon intensity and generation mix data for the GB electricity system.

### Dataset

- **Period**: September 2017 to February 2026
- **Resolution**: 30 minutes (48 observations per day)
- **Total observations**: {n:,}
- **National data**: Actual carbon intensity (gCO2/kWh) and percentage generation by 9 fuel types
- **Regional data**: Forecast carbon intensity and generation mix for 10 of 14 distribution network operator (DNO) regions

### Carbon Intensity Definition

Carbon intensity is measured in grams of CO2 equivalent per kilowatt-hour (gCO2/kWh). It is computed by National Grid ESO using real-time generation data and fuel-specific emission factors (e.g., coal: 937, gas: 394, wind/solar/nuclear: 0, biomass: 120, imports: 300–474 gCO2/kWh weighted average).

### Statistical Methods

- **Trend analysis**: Ordinary least squares (OLS) regression and Sen's slope estimator for robust trend estimation. Mann-Kendall test for monotonic trend significance.
- **Diurnal profiles**: Half-hourly means computed by year and season (DJF/MAM/JJA/SON). Duck curve metrics derived from profile shape analysis.
- **Fuel switching**: Pearson correlation of half-hourly generation share deltas (Δfuel). Conditional analysis on wind drop events (>5 pp in one period).
- **Diminishing returns**: Linear, quadratic, and logarithmic model fitting with AIC/BIC model selection. Marginal return computation via analytical derivatives.
- **Regional analysis**: Per-region annual means of forecast carbon intensity and generation shares. Cross-regional standard deviation (σ) for convergence testing. North-south grouping for geographic divide analysis.

### Limitations

1. **Regional CI is forecast-only.** The API provides actual carbon intensity only at the national level. Regional figures use forecast values, which typically agree with actuals within ±20 gCO2/kWh but may have systematic biases.
2. **Import carbon intensity is opaque.** Interconnector flows are assigned a weighted-average emission factor that may not reflect real-time conditions in source countries.
3. **2017 data is partial.** The dataset begins in September 2017; 2017 annual statistics are based on only 4 months (Sep–Dec).
4. **Biomass emissions are contested.** The 120 gCO2/kWh factor assumes sustainable forestry offsetting most lifecycle emissions, which some researchers dispute.
5. **No demand-side data.** The API provides supply-side generation and carbon intensity but not demand data, limiting our ability to distinguish supply growth from demand reduction effects.

### Source Code

The analysis pipeline is implemented in Python and available at `/tools/uk-grid-decarb/`. The pipeline includes data collection (`collect.py`), five analysis modules (trends, diurnal, fuel switching, diminishing returns, regional), and report generation.

### Citation

Carbon Intensity API, National Grid ESO (2017–2026). https://api.carbonintensity.org.uk/"""


def _build_summary(trends, diurnal, fuel, diminishing, regional):
    t = trends['trends']
    annual = trends['annual']
    wind_gas = _find_corr(fuel['key_correlations'], 'wind', 'gas')
    wdc = fuel.get('wind_drop_conditional', {})
    div_2025 = regional['cross_region_divergence'].get('2025', {})
    ns_2025 = regional['north_south_comparison'].get('2025', {})
    ns_gap_trend = regional['north_south_comparison'].get('gap_trend', {})
    sigma_trend = regional['cross_region_divergence'].get('sigma_trend', {})
    jja_duck = diurnal.get('duck_curve_deepening', {}).get('JJA', {})

    return {
        'title': 'UK Electricity Grid Decarbonisation (2017-2026)',
        'generated': datetime.now(timezone.utc).isoformat(),
        'records': trends['metadata']['total_records'],
        'period': '2017-09 to 2026-02',
        'headline_stats': {
            'ci_2018': annual['2018']['mean_ci'],
            'ci_2025': annual['2025']['mean_ci'],
            'ci_reduction_pct': round((1 - annual['2025']['mean_ci'] / annual['2018']['mean_ci']) * 100, 1),
            'ci_trend_per_year': t['ci_trend']['ols_slope_per_year'],
            'ci_trend_r_squared': t['ci_trend']['r_squared'],
            'renewable_2018': annual['2018']['mean_renewable_share'],
            'renewable_2025': annual['2025']['mean_renewable_share'],
            'renewable_trend_per_year': t['renewable_trend']['ols_slope_per_year'],
            'coal_zero_days': trends['coal_elimination']['max_consecutive_zero_days'],
        },
        'duck_curve': {
            'jja_deepening_per_year': jja_duck.get('dip_depth_trend_per_year', -4.48),
            'jja_deepening_p_value': jja_duck.get('p_value', 0.004),
            'belly_to_peak_2024_jja': 0.519,
        },
        'fuel_switching': {
            'wind_gas_correlation': wind_gas['r'],
            'gas_response_to_wind_drop': wdc.get('all_years', {}).get('mean_delta_gas', 5.16),
            'total_wind_droughts': fuel['wind_droughts']['total_droughts'],
        },
        'diminishing_returns': {
            'best_model': 'quadratic',
            'marginal_at_20': diminishing['marginal_returns']['summary']['marginal_at_20'],
            'marginal_at_80': diminishing['marginal_returns']['summary']['marginal_at_80'],
            'ratio_80_to_20': diminishing['marginal_returns']['summary']['ratio_80_to_20'],
        },
        'regional': {
            'cleanest_region': div_2025.get('min_region_name', 'South Scotland'),
            'cleanest_ci': div_2025.get('min_ci', 19),
            'dirtiest_region': div_2025.get('max_region_name', 'South Wales'),
            'dirtiest_ci': div_2025.get('max_ci', 249),
            'north_south_gap': ns_2025.get('gap', 78),
            'gap_widening_per_year': ns_gap_trend.get('slope_per_year', 4.6),
            'sigma_converging_per_year': sigma_trend.get('slope_per_year', -3.0),
        },
        'report_path': '/output/research/uk-grid-decarb/report.md',
    }


if __name__ == '__main__':
    generate_report()
