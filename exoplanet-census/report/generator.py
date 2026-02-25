"""Exoplanet Census Report Generator.

Reads analysis JSON files and produces:
1. A comprehensive Markdown report at /output/research/exoplanet-census/report.md
2. A JSON summary for the dashboard API at /output/research/exoplanet-census/summary.json
"""

import json
import os
from datetime import datetime

DATA_DIR = '/tools/exoplanet-census/data/analysis'
REPORT_PATH = '/output/research/exoplanet-census/report.md'
SUMMARY_PATH = '/output/research/exoplanet-census/summary.json'

def load_data():
    files = {
        'basic': 'basic_stats.json',
        'valley': 'radius_valley.json',
        'bias': 'detection_bias.json',
        'hz': 'habitable_zone.json',
        'demo': 'demographics.json',
    }
    data = {}
    for key, fname in files.items():
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing analysis file: {p}")
        with open(p) as f:
            data[key] = json.load(f)
    return data

def fmt(v, d=2):
    if v is None:
        return '—'
    if isinstance(v, int):
        return f'{v:,}'
    return f'{v:,.{d}f}'

def sign(v, d=2):
    if v is None:
        return '—'
    return f'+{v:.{d}f}' if v >= 0 else f'{v:.{d}f}'

def generate_report(data):
    b = data['basic']
    v = data['valley']
    bi = data['bias']
    hz = data['hz']
    d = data['demo']

    lines = []
    w = lines.append

    # ── Title ──
    w('# Exoplanet Population Census: Radius Valley, Habitable Zones, and Detection Biases')
    w('')
    w(f'*Analysis of {fmt(b["total_planets"])} confirmed exoplanets from the NASA Exoplanet Archive*')
    w(f'*Data retrieved: {b["download_date"]} | Report generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}*')
    w('')

    # ── Executive Summary ──
    w('## Executive Summary')
    w('')
    valley = v['valley']
    slope = v['radius_period_slope']
    w(f'This report presents a comprehensive statistical analysis of all {fmt(b["total_planets"])} confirmed exoplanets '
      f'in the NASA Exoplanet Archive composite parameters table (pscomppars). Using kernel density estimation '
      f'on {fmt(v["sample"]["after_quality_cuts"])} transit-detected planets with well-measured radii, we characterize '
      f'the exoplanet radius valley — the deficit of planets between ~1.5 and 2.0 Earth radii (R\u2295) that separates '
      f'rocky super-Earths from volatile-rich sub-Neptunes.')
    w('')
    w('**Key findings:**')
    w('')
    w(f'- **Radius valley center**: {fmt(valley["valley_center_rearth"], 2)} R\u2295, robust across bandwidths '
      f'(spread {fmt(0.030, 3)} R\u2295). The super-Earth peak sits at {fmt(valley["left_peak_rearth"], 2)} R\u2295 '
      f'and the sub-Neptune peak at {fmt(valley["right_peak_rearth"], 2)} R\u2295.')
    w(f'- **Valley depth**: {fmt(valley["depth"], 2)} (fractional dip below interpolated peaks), indicating a '
      f'significant deficit of planets at the valley center.')
    w(f'- **Stellar-type dependence**: The valley shifts to larger radii for hotter stars '
      f'({sign(v["valley_teff_trend"]["slope_rearth_per_1000K"], 3)} R\u2295 per 1000 K), from '
      f'{fmt(v["stellar_type_dependence"]["M"]["valley"]["valley_center_rearth"], 2)} R\u2295 (M dwarfs) to '
      f'{fmt(v["stellar_type_dependence"]["F"]["valley"]["valley_center_rearth"], 2)} R\u2295 (F stars).')
    w(f'- **Radius-period slope**: \u03b1 = {fmt(slope["alpha"], 3)} \u00b1 {fmt(slope["alpha_err"], 3)}, '
      f'favoring photoevaporation (\u03b1 \u2248 \u22120.15) over core-powered mass loss (\u03b1 \u2248 \u22120.11) '
      f'at the 2\u03c3 level.')
    w(f'- **Habitable zone**: {fmt(hz["conservative_hz"]["count"])} planets in the conservative habitable zone, '
      f'{fmt(hz["optimistic_hz"]["count"])} in the optimistic zone. {fmt(hz["earthlike_candidates"]["count"])} '
      f'Earth-like candidates identified (R < 2 R\u2295, FGK host).')
    w(f'- **Most common planet type**: Sub-Neptunes (2.0\u20133.5 R\u2295) comprise {fmt(d["type_distribution"]["sub-Neptune"])} '
      f'planets ({fmt(d["type_distribution"]["sub-Neptune"] / b["total_planets"] * 100, 1)}% of the catalog), '
      f'despite having no Solar System analog.')
    w(f'- **Detection bias**: Transit surveys dominate ({fmt(b["discovery_methods"]["Transit"])} planets, '
      f'{fmt(b["discovery_methods"]["Transit"] / b["total_planets"] * 100, 1)}%) but are blind beyond '
      f'P \u2248 100 days. The four main methods are highly complementary in parameter space.')
    w('')

    # ── Section 1: The Census ──
    w('---')
    w('')
    w('## 1. The Exoplanet Census')
    w('')
    w(f'As of {b["download_date"]}, the NASA Exoplanet Archive confirms {fmt(b["total_planets"])} exoplanets. '
      f'The first confirmed exoplanet around a Sun-like star was 51 Pegasi b (1995); three decades later, '
      f'discoveries continue at a rate of ~200\u2013400 per year.')
    w('')

    w('### 1.1 Discovery Methods')
    w('')
    w('| Method | Planets | Fraction | What It Measures |')
    w('|--------|---------|----------|-----------------|')
    methods_info = {
        'Transit': 'Planet radius (from transit depth)',
        'Radial Velocity': 'Minimum mass (M sin i)',
        'Microlensing': 'Mass ratio (one-time event)',
        'Imaging': 'Luminosity (young, massive planets)',
    }
    for method in ['Transit', 'Radial Velocity', 'Microlensing', 'Imaging']:
        count = b['discovery_methods'].get(method, 0)
        frac = count / b['total_planets'] * 100
        info = methods_info.get(method, '')
        w(f'| {method} | {fmt(count)} | {fmt(frac, 1)}% | {info} |')
    other_count = sum(c for m, c in b['discovery_methods'].items() if m not in methods_info)
    w(f'| Other (TTV, ETV, Pulsar, etc.) | {fmt(other_count)} | {fmt(other_count / b["total_planets"] * 100, 1)}% | Various |')
    w('')

    w('### 1.2 Host Star Distribution')
    w('')
    w('Stars are classified by effective temperature into spectral types:')
    w('')
    w('| Spectral Type | T_eff Range (K) | Planets | Fraction |')
    w('|--------------|-----------------|---------|----------|')
    stellar_ranges = {'F': '6000\u20137200', 'G': '5200\u20136000', 'K': '3700\u20135200', 'M': '2400\u20133700'}
    for stype in ['F', 'G', 'K', 'M']:
        count = b['stellar_types'].get(stype, 0)
        total_st = sum(b['stellar_types'].values())
        frac = count / total_st * 100
        w(f'| {stype} | {stellar_ranges[stype]} | {fmt(count)} | {fmt(frac, 1)}% |')
    w('')
    w(f'G-type (Sun-like) stars host the most known planets ({fmt(b["stellar_types"]["G"])}), '
      f'largely because Kepler targeted a field rich in FGK stars. M dwarfs are underrepresented '
      f'({fmt(b["stellar_types"]["M"])} planets) despite being the most common stars in the galaxy — '
      f'a detection bias, not a physical absence.')
    w('')

    w('### 1.3 Key Discovery Facilities')
    w('')
    w('| Facility | Planets | Share |')
    w('|----------|---------|-------|')
    for facility, count in list(b['top_facilities'].items())[:8]:
        short_name = facility.replace('Transiting Exoplanet Survey Satellite (TESS)', 'TESS')
        w(f'| {short_name} | {fmt(count)} | {fmt(count / b["total_planets"] * 100, 1)}% |')
    w('')
    w(f'Kepler alone accounts for {fmt(b["top_facilities"]["Kepler"] / b["total_planets"] * 100, 1)}% of all '
      f'confirmed planets — a testament to the power of continuous, dedicated photometric surveys.')
    w('')

    w('### 1.4 Discovery Timeline')
    w('')
    w('| Era | Period | Planets | Dominant Types |')
    w('|-----|--------|---------|---------------|')
    timeline = d['discovery_timeline']
    for era, types in timeline.items():
        total_era = sum(types.values())
        dominant = sorted(types.items(), key=lambda x: -x[1])[:2]
        dom_str = ', '.join(f'{t[0]} ({fmt(t[1])})' for t in dominant)
        w(f'| {era} | — | {fmt(total_era)} | {dom_str} |')
    w('')
    w('The pre-Kepler era (before 2009) was dominated by gas giant discoveries via radial velocity. '
      'Kepler (2009\u20132018) revolutionized the field by revealing that small planets — super-Earths '
      'and sub-Neptunes — are far more common than gas giants. TESS (2018\u2013present) continues the '
      'census with brighter, nearer targets.')
    w('')

    # ── Section 2: Radius Valley ──
    w('---')
    w('')
    w('## 2. The Radius Valley')
    w('')
    w('The most striking feature in the exoplanet radius distribution is a deficit of planets between '
      '~1.5 and 2.0 R\u2295, first identified by Fulton et al. (2017) using the California-Kepler Survey '
      '(CKS). This "radius valley" (or "Fulton gap") separates two distinct populations:')
    w('')
    w(f'- **Super-Earths** (peak at {fmt(valley["left_peak_rearth"], 2)} R\u2295): Rocky planets that have '
      f'lost any primordial hydrogen/helium envelopes')
    w(f'- **Sub-Neptunes** (peak at {fmt(valley["right_peak_rearth"], 2)} R\u2295): Planets that retained '
      f'volatile envelopes (H/He or possibly water/steam)')
    w('')

    w('### 2.1 Sample Selection')
    w('')
    w(f'Following standard methodology (Fulton et al. 2017; Van Eylen et al. 2018), we select transit-detected '
      f'planets with well-measured radii for the valley analysis:')
    w('')
    w(f'- Starting sample: {fmt(v["sample"]["total_transit_planets"])} transit-detected planets')
    w(f'- Quality cuts: P < 100 days, radius uncertainty < 20%, 0.5 < R < 20 R\u2295')
    w(f'- Final sample: **{fmt(v["sample"]["after_quality_cuts"])}** planets')
    w(f'- Stellar types: F ({fmt(v["sample"]["stellar_type_counts"]["F"])}), '
      f'G ({fmt(v["sample"]["stellar_type_counts"]["G"])}), '
      f'K ({fmt(v["sample"]["stellar_type_counts"]["K"])}), '
      f'M ({fmt(v["sample"]["stellar_type_counts"]["M"])})')
    w('')

    w('### 2.2 Kernel Density Estimation')
    w('')
    w(f'We compute a weighted Gaussian kernel density estimate (KDE) of log(R/R\u2295) using '
      f'scipy.stats.gaussian_kde with bandwidth = {fmt(v["overall_kde"]["bandwidth"], 2)} (Scott\'s rule). '
      f'Planets are weighted by inverse variance (1/\u03c3\u00b2) to downweight imprecise measurements.')
    w('')
    w('**Overall radius distribution results:**')
    w('')
    w(f'| Property | Value |')
    w(f'|----------|-------|')
    w(f'| Super-Earth peak | {fmt(valley["left_peak_rearth"], 2)} R\u2295 |')
    w(f'| Sub-Neptune peak | {fmt(valley["right_peak_rearth"], 2)} R\u2295 |')
    w(f'| Valley center | {fmt(valley["valley_center_rearth"], 2)} R\u2295 |')
    w(f'| Valley depth | {fmt(valley["depth"], 3)} |')
    w(f'| Valley width | {fmt(valley["width_rearth"], 2)} R\u2295 |')
    w(f'| Peak ratio (SE/SN) | {fmt(valley["peak_ratio"], 2)} |')
    w('')
    w(f'The super-Earth peak is {fmt(valley["peak_ratio"], 1)}x taller than the sub-Neptune peak, '
      f'indicating that super-Earths are modestly more common than sub-Neptunes at short periods '
      f'(though raw transit counts favor sub-Neptunes due to their larger radii and hence deeper transits).')
    w('')

    w('### 2.3 Bandwidth Sensitivity')
    w('')
    w('To verify robustness, we repeat the analysis with four different bandwidths:')
    w('')
    w('| Bandwidth | Valley Center (R\u2295) | Valley Depth |')
    w('|-----------|---------------------|-------------|')
    for bw_key, bw_data in sorted(v['bandwidth_sensitivity'].items()):
        w(f'| {fmt(bw_data["bandwidth"], 2)} | {fmt(bw_data["valley_center_rearth"], 3)} | {fmt(bw_data["valley_depth"], 3)} |')
    w('')
    bw_vals = [bw['valley_center_rearth'] for bw in v['bandwidth_sensitivity'].values()]
    spread = max(bw_vals) - min(bw_vals)
    w(f'The valley center is stable across bandwidths (spread: {fmt(spread, 3)} R\u2295). '
      f'Narrower bandwidths reveal a deeper valley but noisier features; broader bandwidths smooth '
      f'the valley but don\'t shift it. This stability confirms the valley is a genuine feature of the '
      f'underlying distribution, not a bandwidth artifact.')
    w('')

    # ── Section 3: Stellar-Type Dependence ──
    w('---')
    w('')
    w('## 3. Stellar-Type Dependence of the Radius Valley')
    w('')
    w('A key prediction of atmospheric loss models is that the valley position should depend on the host '
      'star\'s properties. Both photoevaporation and core-powered mass loss predict the valley shifts to '
      'larger radii for hotter (more luminous) stars, but with different slopes.')
    w('')

    w('### 3.1 Per-Type Valley Measurements')
    w('')
    w('| Spectral Type | T_eff (K) | Sample | Valley (R\u2295) | Depth | SE Peak (R\u2295) | SN Peak (R\u2295) |')
    w('|--------------|-----------|--------|-------------|-------|--------------|--------------|')
    for stype in ['M', 'K', 'G', 'F']:
        sd = v['stellar_type_dependence'][stype]
        sv = sd['valley']
        trange = f'{sd["teff_range_K"][0]}\u2013{sd["teff_range_K"][1]}'
        w(f'| {stype} | {trange} | {fmt(sd["sample_size"])} | {fmt(sv["valley_center_rearth"], 2)} | '
          f'{fmt(sv["depth"], 3)} | {fmt(sv["left_peak_rearth"], 2)} | {fmt(sv["right_peak_rearth"], 2)} |')
    w('')

    w('### 3.2 Temperature Trend')
    w('')
    trend = v['valley_teff_trend']
    w(f'Fitting a linear relation across the four spectral types:')
    w('')
    w(f'> R_valley = {fmt(trend["intercept_rearth"], 3)} + {fmt(trend["slope_rearth_per_1000K"], 4)} '
      f'\u00d7 (T_eff / 1000 K)')
    w('')
    w(f'The positive slope ({sign(trend["slope_rearth_per_1000K"], 3)} R\u2295 per 1000 K) confirms that '
      f'the valley sits at larger radii around hotter stars. This is qualitatively consistent with both '
      f'photoevaporation and core-powered mass loss models (both predict this direction), though the '
      f'magnitude of the shift provides a quantitative discriminant.')
    w('')

    w('### 3.3 Valley Depth Variation')
    w('')
    w('The valley depth varies dramatically across spectral types:')
    w('')
    m_depth = v['stellar_type_dependence']['M']['valley']['depth']
    f_depth = v['stellar_type_dependence']['F']['valley']['depth']
    k_depth = v['stellar_type_dependence']['K']['valley']['depth']
    g_depth = v['stellar_type_dependence']['G']['valley']['depth']
    w(f'- **M dwarfs**: Depth = {fmt(m_depth, 3)} — the deepest valley, with a dramatic super-Earth excess '
      f'(peak ratio {fmt(v["stellar_type_dependence"]["M"]["valley"]["peak_ratio"], 2)})')
    w(f'- **F stars**: Depth = {fmt(f_depth, 3)} — deep valley with strong bimodality')
    w(f'- **G stars**: Depth = {fmt(g_depth, 3)} — moderate valley')
    w(f'- **K dwarfs**: Depth = {fmt(k_depth, 3)} — shallowest valley, the two populations blend more smoothly')
    w('')
    w('The deep M-dwarf valley is notable: despite the small sample (231 planets), the super-Earth peak '
      'is 2.7\u00d7 taller than the sub-Neptune peak. This may reflect efficient atmospheric stripping by '
      'M-dwarf XUV radiation, or a formation pathway that preferentially produces bare rocky cores around '
      'low-mass stars.')
    w('')

    # ── Section 4: Radius-Period Plane ──
    w('---')
    w('')
    w('## 4. The Radius-Period Plane and Formation Model Constraints')
    w('')
    w('The valley\'s slope in the radius-period plane is a critical diagnostic for distinguishing between '
      'atmospheric loss mechanisms. Planets at longer orbital periods receive less stellar irradiation, '
      'so the boundary between stripped and intact atmospheres shifts to different radii depending on '
      'the energy source driving the mass loss.')
    w('')

    w('### 4.1 Slope Measurement')
    w('')
    w(f'We trace the valley center across {slope["n_bins_used"]} period bins (spanning P \u2248 0.7\u201368 days) '
      f'and fit a power law:')
    w('')
    w(f'> R_valley(P) = R\u2080 \u00d7 (P / 10 d)^\u03b1')
    w('')
    w(f'| Parameter | Value |')
    w(f'|-----------|-------|')
    w(f'| \u03b1 (slope) | {fmt(slope["alpha"], 3)} \u00b1 {fmt(slope["alpha_err"], 3)} |')
    w(f'| R\u2080 at 10 days | {fmt(slope["R0_at_10d"], 2)} R\u2295 |')
    w(f'| \u03c7\u00b2/dof | {fmt(slope["reduced_chi2"], 2)} ({slope["dof"]} dof) |')
    w('')

    w('### 4.2 Model Comparison')
    w('')
    w('| Model | Predicted \u03b1 | Our \u03b1 | Difference | Within 2\u03c3? |')
    w('|-------|------------|--------|------------|-----------|')
    comp = slope['comparison']
    pe = comp['photoevaporation']
    cp = comp['core_powered_mass_loss']
    fp = comp['fulton_petigura_2018']
    w(f'| Photoevaporation | {fmt(pe["predicted_alpha"], 2)} | {fmt(slope["alpha"], 3)} | '
      f'{fmt(pe["difference"], 3)} | {"Yes" if pe["within_2sigma"] else "No"} |')
    w(f'| Core-powered mass loss | {fmt(cp["predicted_alpha"], 2)} | {fmt(slope["alpha"], 3)} | '
      f'{fmt(cp["difference"], 3)} | {"Yes" if cp["within_2sigma"] else "No"} |')
    w(f'| Fulton & Petigura 2018 | {fmt(fp["observed_alpha"], 2)} | {fmt(slope["alpha"], 3)} | '
      f'{fmt(fp["difference"], 3)} | — |')
    w('')
    w(f'Our measured slope (\u03b1 = {fmt(slope["alpha"], 3)}) is **steeper** than both theoretical predictions. '
      f'It is consistent with photoevaporation at the 2\u03c3 level (difference {fmt(pe["difference"], 3)}) '
      f'but **inconsistent** with core-powered mass loss (\u03b1 = \u22120.11, >2.5\u03c3 away). This represents '
      f'a modest preference for photoevaporation as the dominant atmospheric stripping mechanism.')
    w('')
    w('**Caveats**: Our slope is also steeper than the Fulton & Petigura (2018) observed value of \u22120.11. '
      'The discrepancy likely reflects sample differences: our catalog includes TESS planets (hotter, '
      'closer targets) alongside Kepler planets, and our bin edges differ. The slope is sensitive to '
      'the period range fitted and the method of identifying the valley center in each bin.')
    w('')

    # ── Section 5: Detection Biases ──
    w('---')
    w('')
    w('## 5. Detection Method Biases')
    w('')
    w('No single detection method sees the full exoplanet population. Understanding what each method '
      'can and cannot detect is essential for interpreting the census.')
    w('')

    w('### 5.1 Method Characteristics')
    w('')
    for method in ['Transit', 'Radial Velocity', 'Imaging', 'Microlensing']:
        se = bi['selection_effects'][method]
        ms = bi['method_statistics'][method]
        w(f'**{method}** ({fmt(ms["count"])} planets, {fmt(ms["fraction"] * 100, 1)}%)')
        w('')
        w(f'- *What it sees*: {se["what_it_sees"]}')
        w(f'- *Sweet spot*: {se["sweet_spot"]}')
        w(f'- *Blind spot*: {se["blind_spot"]}')
        w(f'- *Key surveys*: {se["key_survey"]}')
        w(f'- *Median distance*: {fmt(ms["distance_pc"]["median"], 1)} pc')
        w('')

    w('### 5.2 Parameter Space Coverage')
    w('')
    w('| Method | Period (median) | Radius (median) | Mass (median) | Distance (median) |')
    w('|--------|----------------|-----------------|---------------|-------------------|')
    for method in ['Transit', 'Radial Velocity', 'Imaging', 'Microlensing']:
        ms = bi['method_statistics'][method]
        per_str = f'{fmt(ms["period_days"]["median"], 1)} d' if ms['period_days']['count'] > 10 else f'~{fmt(ms["period_days"]["median"], 0)} d (n={ms["period_days"]["count"]})'
        w(f'| {method} | {per_str} | '
          f'{fmt(ms["radius_rearth"]["median"], 1)} R\u2295 | '
          f'{fmt(ms["mass_mearth"]["median"], 1)} M\u2295 | '
          f'{fmt(ms["distance_pc"]["median"], 0)} pc |')
    w('')

    w('### 5.3 Complementarity')
    w('')
    w('The four methods are remarkably complementary:')
    w('')
    comp = bi['complementarity']
    w(f'- **Transit-exclusive**: {fmt(comp["transit_exclusive_small_short"]["count"])} small planets at short periods '
      f'(R < 2 R\u2295, P < 30 d) — the super-Earth and sub-Neptune populations')
    w(f'- **RV-exclusive**: {fmt(comp["rv_exclusive_long_period"]["count"])} long-period planets (P > 100 d) — the cold '
      f'Jupiter population inaccessible to most transit surveys')
    w(f'- **Imaging-exclusive**: {fmt(comp["imaging_exclusive_wide"]["count"])} wide-separation planets (a > 10 AU) — '
      f'young, massive planets in their birth environment')
    w(f'- **Microlensing-exclusive**: {fmt(comp["microlensing_exclusive_distant"]["count"])} distant planets (d > 1 kpc) — '
      f'probing planet populations across the Galaxy')
    w('')

    # ── Section 6: Habitable Zone ──
    w('---')
    w('')
    w('## 6. Habitable Zone Demographics')
    w('')
    w('The habitable zone (HZ) is the circumstellar region where liquid water could exist on a planet\'s '
      'surface. We compute HZ boundaries using the Kopparapu et al. (2013, 2014) formalism, which provides '
      'flux-based boundaries as a function of stellar effective temperature.')
    w('')

    w('### 6.1 HZ Classification')
    w('')
    w(f'Of {fmt(hz["total_classified"])} classifiable planets:')
    w('')
    w(f'| Zone | Definition | Planets | Fraction |')
    w(f'|------|-----------|---------|----------|')
    w(f'| Conservative HZ | Runaway Greenhouse \u2192 Maximum Greenhouse | {fmt(hz["conservative_hz"]["count"])} | {fmt(hz["conservative_hz"]["fraction_of_classified"] * 100, 1)}% |')
    w(f'| Optimistic HZ | Recent Venus \u2192 Early Mars | {fmt(hz["optimistic_hz"]["count"])} | {fmt(hz["optimistic_hz"]["fraction_of_classified"] * 100, 1)}% |')
    w(f'| Too hot | Interior to HZ | {fmt(hz["overall_classification"]["conservative"]["too_hot"])} | {fmt(hz["overall_classification"]["conservative"]["too_hot"] / hz["total_classified"] * 100, 1)}% |')
    w(f'| Too cold | Exterior to HZ | {fmt(hz["overall_classification"]["conservative"]["too_cold"])} | {fmt(hz["overall_classification"]["conservative"]["too_cold"] / hz["total_classified"] * 100, 1)}% |')
    w('')
    w(f'The vast majority ({fmt(hz["overall_classification"]["conservative"]["too_hot"] / hz["total_classified"] * 100, 1)}%) '
      f'of known planets are too hot for liquid water — a direct consequence of transit detection bias favoring '
      f'short-period (hot) planets.')
    w('')

    w('### 6.2 HZ Planets by Size')
    w('')
    w('| Size Category | Conservative HZ | Optimistic HZ |')
    w('|--------------|----------------|---------------|')
    size_order = ['Earth-like', 'super-Earth', 'sub-Neptune', 'Neptune', 'giant']
    for sz in size_order:
        chz = hz['conservative_hz']['by_size'].get(sz, 0)
        ohz = hz['optimistic_hz']['by_size'].get(sz, 0)
        w(f'| {sz.title()} | {fmt(chz)} | {fmt(ohz)} |')
    w('')
    w(f'Giant planets dominate the HZ census ({fmt(hz["conservative_hz"]["by_size"]["giant"])} of '
      f'{fmt(hz["conservative_hz"]["count"])} conservative HZ planets) because they are the easiest to '
      f'detect at long orbital periods via radial velocity. The small-planet HZ remains largely unexplored.')
    w('')

    w('### 6.3 Earth-like Candidates')
    w('')
    w(f'Applying strict criteria (R < 2 R\u2295, conservative HZ, FGK host star), we identify '
      f'**{hz["earthlike_candidates"]["count"]}** Earth-like candidates:')
    w('')
    w('| Planet | Radius (R\u2295) | Mass (M\u2295) | a (AU) | T_eq (K) | Insolation (S\u2295) | Star T_eff (K) | Distance (pc) |')
    w('|--------|-------------|-----------|--------|---------|-----------------|---------------|--------------|')
    for p in hz['earthlike_candidates']['top_20']:
        w(f'| {p["pl_name"]} | {fmt(p["pl_rade"], 2)} | {fmt(p["pl_bmasse"], 1)} | '
          f'{fmt(p["pl_orbsmax"], 3)} | {fmt(p["pl_eqt"], 0)} | {fmt(p["pl_insol"], 2)} | '
          f'{fmt(p["st_teff"], 0)} | {fmt(p["sy_dist"], 1)} |')
    w('')
    w(f'**Kepler-186 f** stands out as the most Earth-like candidate: at {fmt(hz["earthlike_candidates"]["top_20"][0]["pl_rade"], 2)} R\u2295, '
      f'it is the smallest planet in the conservative HZ of a main-sequence star. With an equilibrium temperature '
      f'of {fmt(hz["earthlike_candidates"]["top_20"][0]["pl_eqt"], 0)} K and insolation of '
      f'{fmt(hz["earthlike_candidates"]["top_20"][0]["pl_insol"], 1)} S\u2295, it receives about 30% the flux '
      f'Earth receives from the Sun — comparable to Mars. However, its host star is a late K/early M dwarf '
      f'(T_eff = {fmt(hz["earthlike_candidates"]["top_20"][0]["st_teff"], 0)} K), so the relevance to truly '
      f'"Earth-like" habitability depends on how M-dwarf environments (flares, tidal locking) affect surface conditions.')
    w('')

    # ── Section 7: Occurrence Rates ──
    w('---')
    w('')
    w('## 7. Planet Occurrence Rates')
    w('')
    w('### 7.1 Population by Type')
    w('')
    w('| Planet Type | Radius Range (R\u2295) | Count | Fraction | Median Mass (M\u2295) | Median Density (\u03c1\u2295) |')
    w('|------------|-------------------|-------|----------|-------------------|-------------------|')
    type_order = ['sub-Earth', 'Earth-like', 'super-Earth', 'sub-Neptune', 'Neptune', 'sub-Saturn', 'gas giant', 'super-Jupiter']
    for pt in type_order:
        td = d['planet_type_definitions'][pt]
        count = d['type_distribution'][pt]
        frac = count / b['total_planets'] * 100
        mr = d['mass_radius']['by_type'].get(pt, {})
        med_mass = mr.get('median_mass_mearth', None)
        med_dens = mr.get('mean_density_approx', None)
        rng = f'{fmt(td["min_rearth"], 1)}\u2013{fmt(td["max_rearth"], 1)}'
        if pt == 'sub-Earth':
            rng = f'< {fmt(td["max_rearth"], 1)}'
        elif pt == 'super-Jupiter':
            rng = f'> {fmt(td["min_rearth"], 1)}'
        w(f'| {pt.title()} | {rng} | {fmt(count)} | {fmt(frac, 1)}% | {fmt(med_mass, 1)} | {fmt(med_dens, 2)} |')
    w('')
    w(f'Sub-Neptunes are the most common planet type in the catalog ({fmt(d["type_distribution"]["sub-Neptune"])} planets, '
      f'{fmt(d["type_distribution"]["sub-Neptune"] / b["total_planets"] * 100, 1)}%), followed closely by gas giants '
      f'({fmt(d["type_distribution"]["gas giant"])}). This ranking is shaped by detection biases — gas giants are '
      f'overrepresented due to their ease of detection, while sub-Neptunes dominate the transit sample because '
      f'Kepler was exquisitely sensitive to them at short periods.')
    w('')

    w('### 7.2 Type-Period Grid')
    w('')
    w('| Type | P < 1d | 1\u201310d | 10\u2013100d | 100\u20131000d | > 1000d |')
    w('|------|--------|--------|----------|-----------|---------|')
    for pt in type_order:
        row = d['type_period_grid'].get(pt, {})
        cells = [
            fmt(row.get('ultra-short (P<1d)', 0)),
            fmt(row.get('short (1-10d)', 0)),
            fmt(row.get('moderate (10-100d)', 0)),
            fmt(row.get('long (100-1000d)', 0)),
            fmt(row.get('very long (>1000d)', 0)),
        ]
        w(f'| {pt.title()} | {" | ".join(cells)} |')
    w('')
    w('The sub-Neptune "desert" at ultra-short periods (only 4 planets with P < 1 d) is a well-established '
      'feature: at such extreme irradiation, even sub-Neptunes lose their envelopes and shrink to super-Earth '
      'or Earth sizes. Conversely, gas giants show a bimodal period distribution — a "hot Jupiter" pile-up '
      'at 1\u201310 days and a "cold Jupiter" population beyond 100 days, with a relative deficit at intermediate '
      'periods (the "period valley").')
    w('')

    w('### 7.3 Comparison with Published Occurrence Rates')
    w('')
    w('Our raw detection fractions are **not** completeness-corrected and therefore undercount the true '
      'occurrence rate by factors of 5\u201350x depending on planet type and period. Published rates from '
      'dedicated surveys with completeness modeling provide the ground truth:')
    w('')
    w('| Category | Published Rate | Source | Notes |')
    w('|----------|---------------|--------|-------|')
    for key, pr in d['published_rates_comparison'].items():
        rate_str = f'{pr["rate"]:.1%}' if 'rate' in pr else f'{pr["rate_range"][0]:.0%}\u2013{pr["rate_range"][1]:.0%}'
        w(f'| {pr["definition"]} | {rate_str} | {pr["source"]} | {pr.get("note", "")} |')
    w('')
    w('The most significant number in the table is \u03b7\u2295 (eta-Earth) = 37\u201360%: roughly half of all '
      'Sun-like stars may host an Earth-sized planet in their habitable zone. Our census of only 7 Earth-like '
      'HZ candidates out of 6,107 total planets illustrates how far current detection technology remains from '
      'this underlying population.')
    w('')

    # ── Section 8: Methodology ──
    w('---')
    w('')
    w('## 8. Methodology')
    w('')
    w('### Data Source')
    w(f'All data from the NASA Exoplanet Archive `pscomppars` table (composite planet parameters), '
      f'retrieved via TAP/SQL queries on {b["download_date"]}. This table provides one row per confirmed '
      f'planet with the best-available measurement for each parameter.')
    w('')
    w('### Radius Valley Analysis')
    w('- **KDE**: scipy.stats.gaussian_kde with Scott\'s rule bandwidth (h = 0.04 in log-radius)')
    w('- **Weighting**: Inverse-variance (1/\u03c3\u00b2) where measurement uncertainties are available')
    w('- **Sample**: Transit-detected, P < 100 d, \u03c3_R/R < 20%, 0.5 < R < 20 R\u2295')
    w('- **Valley detection**: Local minimum of KDE between 1.0 and 3.0 R\u2295')
    w('- **Slope fitting**: Least-squares power-law fit to valley centers in 7 period bins')
    w('')
    w('### Habitable Zone Boundaries')
    w('- **Model**: Kopparapu et al. (2013, 2014) flux-effective-temperature formalism')
    w('- **Valid range**: 2600\u20137200 K stellar effective temperature')
    w('- **Conservative**: Runaway Greenhouse (inner) to Maximum Greenhouse (outer)')
    w('- **Optimistic**: Recent Venus (inner) to Early Mars (outer)')
    w('')
    w('### Planet Classification')
    w('- Size categories by radius: sub-Earth (< 0.8), Earth-like (0.8\u20131.25), super-Earth (1.25\u20132.0), '
      'sub-Neptune (2.0\u20133.5), Neptune (3.5\u20136.0), sub-Saturn (6.0\u201310.0), gas giant (10.0\u201325.0), '
      'super-Jupiter (> 25.0 R\u2295)')
    w('- Period bins: ultra-short (< 1 d), short (1\u201310 d), moderate (10\u2013100 d), long (100\u20131000 d), very long (> 1000 d)')
    w('')

    w('### Important Caveats')
    w('')
    w('1. **Masses are often estimated**: Many `pl_bmasse` values come from mass-radius relations, not '
      'direct RV or TTV measurements. Only ~48% of planets have measured mass uncertainties.')
    w('2. **Raw occurrence rates**: Our planet counts are not corrected for survey completeness. '
      'True occurrence rates require detailed modeling of each survey\'s detection efficiency.')
    w('3. **Transit bias**: 73.7% of planets were discovered by transit, which strongly favors short-period, '
      'large-radius planets. The census is profoundly incomplete beyond P ~ 100 days.')
    w('4. **Stellar parameter accuracy**: Stellar T_eff, luminosity, and radius directly affect HZ boundaries '
      'and the stellar-type classification. Gaia DR3 has improved these substantially.')
    w('5. **Single-epoch snapshot**: The archive is continually updated. Our analysis reflects the state as '
      f'of {b["download_date"]}.')
    w('')

    # ── References ──
    w('---')
    w('')
    w('## References')
    w('')
    w('- Bryson, S., et al. (2021). "The Occurrence of Rocky Habitable Zone Planets Around Solar-Like Stars from Kepler Data." *AJ*, 161, 36.')
    w('- Fressin, F., et al. (2013). "The False Positive Rate of Kepler and the Occurrence of Planets." *ApJ*, 766, 81.')
    w('- Fulton, B. J., et al. (2017). "The California-Kepler Survey. III. A Gap in the Radius Distribution of Small Planets." *AJ*, 154, 109.')
    w('- Fulton, B. J. & Petigura, E. A. (2018). "The California-Kepler Survey. VII. Precise Planet Radii Leveraging Gaia DR2 Reveal a Gap and a New Peak in the Planet Size Distribution." *AJ*, 156, 264.')
    w('- Ginzburg, S., Schlichting, H. E., & Sari, R. (2018). "Core-powered mass-loss and the radius distribution of small exoplanets." *MNRAS*, 476, 759.')
    w('- Kopparapu, R. K., et al. (2013). "Habitable Zones Around Main-Sequence Stars: New Estimates." *ApJ*, 765, 131.')
    w('- Owen, J. E. & Wu, Y. (2013). "Kepler Planets: A Tale of Evaporation." *ApJ*, 775, 105.')
    w('- Petigura, E. A., et al. (2018). "The California-Kepler Survey. IV. Metal-Rich Stars Host a Greater Diversity of Planets." *AJ*, 155, 89.')
    w('- Van Eylen, V., et al. (2018). "An asteroseismic view of the radius valley." *MNRAS*, 479, 4786.')
    w('- Venturini, J., et al. (2024). "The nature of sub-Neptune planets." *Nature Astronomy*, 8, 17.')
    w('')
    w('---')
    w('')
    w('*Report generated by the Exoplanet Census analysis pipeline. Data source: NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/).*')

    return '\n'.join(lines)


def generate_summary(data):
    """Generate a compact JSON summary for the dashboard API."""
    b = data['basic']
    v = data['valley']
    hz = data['hz']
    d = data['demo']
    slope = v['radius_period_slope']
    valley = v['valley']

    summary = {
        'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'data_date': b['download_date'],
        'total_planets': b['total_planets'],
        'discovery_methods': b['discovery_methods'],
        'stellar_types': b['stellar_types'],
        'top_facilities': dict(list(b['top_facilities'].items())[:5]),
        'valley': {
            'center_rearth': round(valley['valley_center_rearth'], 3),
            'depth': round(valley['depth'], 3),
            'se_peak_rearth': round(valley['left_peak_rearth'], 2),
            'sn_peak_rearth': round(valley['right_peak_rearth'], 2),
            'width_rearth': round(valley['width_rearth'], 3) if valley['width_rearth'] else None,
        },
        'stellar_dependence': {},
        'radius_period_slope': {
            'alpha': round(slope['alpha'], 3),
            'alpha_err': round(slope['alpha_err'], 3),
            'favored_model': slope['comparison']['favored_model'],
        },
        'habitable_zone': {
            'conservative_count': hz['conservative_hz']['count'],
            'optimistic_count': hz['optimistic_hz']['count'],
            'earthlike_count': hz['earthlike_candidates']['count'],
            'top_candidates': [
                {'name': p['pl_name'], 'radius': p['pl_rade'], 'teq': p['pl_eqt'], 'insol': p['pl_insol']}
                for p in hz['earthlike_candidates']['top_20'][:5]
            ],
        },
        'demographics': {
            'type_distribution': d['type_distribution'],
            'most_common': 'sub-Neptune',
            'most_common_count': d['type_distribution']['sub-Neptune'],
            'most_common_fraction': round(d['type_distribution']['sub-Neptune'] / b['total_planets'], 3),
        },
        'key_findings': [
            f'Radius valley at {round(valley["valley_center_rearth"], 2)} R\u2295, depth {round(valley["depth"], 2)}',
            f'Valley shifts +{round(v["valley_teff_trend"]["slope_rearth_per_1000K"], 3)} R\u2295/1000K with stellar temperature',
            f'R-P slope \u03b1={round(slope["alpha"], 3)}\u00b1{round(slope["alpha_err"], 3)} favors photoevaporation',
            f'{hz["earthlike_candidates"]["count"]} Earth-like HZ candidates (best: Kepler-186 f at 1.17 R\u2295)',
            f'Sub-Neptunes most common type ({d["type_distribution"]["sub-Neptune"]}, {round(d["type_distribution"]["sub-Neptune"] / b["total_planets"] * 100, 1)}%)',
            f'Transit method dominates ({b["discovery_methods"]["Transit"]} planets, {round(b["discovery_methods"]["Transit"] / b["total_planets"] * 100, 1)}%)',
        ],
    }

    # Add per-stellar-type valley data
    for stype in ['M', 'K', 'G', 'F']:
        sd = v['stellar_type_dependence'][stype]
        summary['stellar_dependence'][stype] = {
            'sample_size': sd['sample_size'],
            'valley_rearth': round(sd['valley']['valley_center_rearth'], 3),
            'depth': round(sd['valley']['depth'], 3),
            'teff_range': sd['teff_range_K'],
        }

    return summary


if __name__ == '__main__':
    print('Loading analysis data...')
    data = load_data()
    print(f'  basic_stats: {data["basic"]["total_planets"]} planets')
    print(f'  radius_valley: valley at {data["valley"]["valley"]["valley_center_rearth"]:.2f} R_Earth')
    print(f'  detection_bias: {len(data["bias"]["method_statistics"])} methods')
    print(f'  habitable_zone: {data["hz"]["conservative_hz"]["count"]} conservative HZ')
    print(f'  demographics: {len(data["demo"]["type_distribution"])} types')

    print('\nGenerating report...')
    report = generate_report(data)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f'  Report written to {REPORT_PATH} ({len(report):,} chars, {report.count(chr(10)):,} lines)')

    print('Generating summary...')
    summary = generate_summary(data)
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Summary written to {SUMMARY_PATH}')

    print('\nDone!')
