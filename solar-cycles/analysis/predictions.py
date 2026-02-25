"""Solar Cycle 25 prediction assessment.

Compares observed SSN data against NOAA's official SC25 predictions.
Computes bias, RMSE, and confidence interval containment.
"""

import json
import numpy as np
from pathlib import Path


def parse_predictions(records):
    """Parse NOAA prediction records.

    Returns dict with arrays: time_yr, predicted_ssn, high_ssn, low_ssn
    """
    times = []
    predicted = []
    high = []
    low = []

    for rec in records:
        tag = rec["time-tag"]  # "YYYY-MM"
        yr, mo = int(tag[:4]), int(tag[5:7])
        t = yr + (mo - 0.5) / 12.0
        times.append(t)
        predicted.append(float(rec["predicted_ssn"]))
        high.append(float(rec["high_ssn"]))
        low.append(float(rec["low_ssn"]))

    return {
        "time_yr": np.array(times),
        "predicted_ssn": np.array(predicted),
        "high_ssn": np.array(high),
        "low_ssn": np.array(low),
    }


def parse_daily_ssn(records):
    """Parse daily SSN records from NOAA.

    Returns dict with arrays: time_yr, ssn
    """
    times = []
    ssn = []

    for rec in records:
        date_str = rec["Obsdate"]  # "YYYY-MM-DD 00:00:00"
        yr = int(date_str[:4])
        mo = int(date_str[5:7])
        dy = int(date_str[8:10])
        t = yr + (mo - 0.5) / 12.0 + (dy - 15) / 365.25
        val = rec.get("swpc_ssn")
        if val is not None:
            times.append(t)
            ssn.append(float(val))

    return {
        "time_yr": np.array(times),
        "ssn": np.array(ssn),
    }


def monthly_from_daily(daily_data):
    """Aggregate daily SSN to monthly means.

    Returns dict with time_yr, ssn arrays (monthly).
    """
    t = daily_data["time_yr"]
    s = daily_data["ssn"]

    # Group by year-month
    monthly = {}
    for i in range(len(t)):
        yr = int(t[i])
        mo = int((t[i] - yr) * 12) + 1
        if mo > 12:
            mo = 12
        key = (yr, mo)
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(s[i])

    times = []
    means = []
    for (yr, mo) in sorted(monthly.keys()):
        times.append(yr + (mo - 0.5) / 12.0)
        means.append(np.mean(monthly[(yr, mo)]))

    return {
        "time_yr": np.array(times),
        "ssn": np.array(means),
    }


def sc25_assessment(monthly_data, prediction_data, daily_data=None):
    """Comprehensive SC25 prediction assessment.

    Args:
        monthly_data: dict from cycles.parse_monthly_data (observed monthly SSN)
        prediction_data: dict from parse_predictions (NOAA predictions)
        daily_data: dict from parse_daily_ssn (optional, for higher-res analysis)

    Returns:
        dict with assessment metrics
    """
    obs_t = monthly_data["time_yr"]
    obs_ssn = monthly_data["ssn"]
    pred_t = prediction_data["time_yr"]
    pred_ssn = prediction_data["predicted_ssn"]
    pred_high = prediction_data["high_ssn"]
    pred_low = prediction_data["low_ssn"]

    # SC25 start: ~Dec 2019 (cycle minimum)
    sc25_start = 2019.9
    sc25_mask_obs = obs_t >= sc25_start
    sc25_t = obs_t[sc25_mask_obs]
    sc25_ssn = obs_ssn[sc25_mask_obs]

    # Find overlap period (where we have both observed and predicted)
    pred_start = pred_t[0]
    pred_end = pred_t[-1]
    obs_end = obs_t[-1]

    overlap_start = pred_start
    overlap_end = min(pred_end, obs_end)

    # Match observed to predicted by nearest month
    overlap_obs = []
    overlap_pred = []
    overlap_high = []
    overlap_low = []
    overlap_times = []

    for i, pt in enumerate(pred_t):
        if pt > obs_end:
            break
        # Find nearest observed month
        idx = np.argmin(np.abs(obs_t - pt))
        if abs(obs_t[idx] - pt) < 0.1:  # within ~1 month
            overlap_obs.append(float(obs_ssn[idx]))
            overlap_pred.append(float(pred_ssn[i]))
            overlap_high.append(float(pred_high[i]))
            overlap_low.append(float(pred_low[i]))
            overlap_times.append(float(pt))

    overlap_obs = np.array(overlap_obs)
    overlap_pred = np.array(overlap_pred)
    overlap_high = np.array(overlap_high)
    overlap_low = np.array(overlap_low)
    n_overlap = len(overlap_obs)

    # Compute metrics
    if n_overlap > 0:
        residuals = overlap_obs - overlap_pred
        bias = float(np.mean(residuals))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        # Confidence interval containment
        in_ci = np.sum((overlap_obs >= overlap_low) & (overlap_obs <= overlap_high))
        ci_containment = float(in_ci / n_overlap)

        # Direction: is SC25 running above or below predictions?
        above = np.sum(overlap_obs > overlap_pred)
        below = np.sum(overlap_obs < overlap_pred)
    else:
        bias = rmse = mae = ci_containment = 0.0
        above = below = 0

    # SC25 peak analysis
    # Find observed peak so far
    peak_idx = np.argmax(sc25_ssn)
    obs_peak_ssn = float(sc25_ssn[peak_idx])
    obs_peak_time = float(sc25_t[peak_idx])

    # NOAA predicted peak
    pred_peak_idx = np.argmax(pred_ssn)
    pred_peak_ssn = float(pred_ssn[pred_peak_idx])
    pred_peak_time = float(pred_t[pred_peak_idx])
    pred_peak_high = float(pred_high[pred_peak_idx])
    pred_peak_low = float(pred_low[pred_peak_idx])

    # Smoothed SSN peak (from monthly data)
    smoothed = monthly_data.get("smoothed_ssn")
    smoothed_peak = None
    if smoothed is not None:
        sc25_smoothed = smoothed[sc25_mask_obs]
        valid = ~np.isnan(sc25_smoothed)
        if np.any(valid):
            sp_idx = np.argmax(sc25_smoothed[valid])
            valid_indices = np.where(valid)[0]
            smoothed_peak = {
                "ssn": round(float(sc25_smoothed[valid_indices[sp_idx]]), 1),
                "time": round(float(sc25_t[valid_indices[sp_idx]]), 2),
            }

    # Compare with recent cycles (SC23, SC24)
    # SC23: ~1996.6-2008.9, peak ~2001.9 at 180.3
    # SC24: ~2008.9-2019.9, peak ~2014.3 at 116.4
    # SC25: ~2019.9-ongoing
    cycle_comparison = {
        "SC23": {"peak_ssn": 180.3, "peak_time": 2001.9, "period_yr": 12.33},
        "SC24": {"peak_ssn": 116.4, "peak_time": 2014.3, "period_yr": 11.0},
        "SC25_observed": {
            "peak_ssn_so_far": obs_peak_ssn,
            "peak_time_so_far": round(obs_peak_time, 2),
            "months_since_start": int(round((obs_t[-1] - sc25_start) * 12)),
            "current_ssn": round(float(obs_ssn[-1]), 1),
            "current_time": round(float(obs_t[-1]), 2),
        },
    }

    # Prediction vs reality trajectory
    # Sample SC25 observed at yearly intervals
    trajectory = []
    for yr in range(2020, int(obs_t[-1]) + 1):
        yr_mask = (sc25_t >= yr) & (sc25_t < yr + 1)
        if np.any(yr_mask):
            yr_mean = float(np.mean(sc25_ssn[yr_mask]))
            yr_max = float(np.max(sc25_ssn[yr_mask]))
            trajectory.append({
                "year": yr,
                "mean_ssn": round(yr_mean, 1),
                "max_ssn": round(yr_max, 1),
                "n_months": int(np.sum(yr_mask)),
            })

    # Daily SSN analysis (if available)
    daily_metrics = None
    if daily_data is not None:
        d_t = daily_data["time_yr"]
        d_ssn = daily_data["ssn"]
        sc25_d_mask = d_t >= sc25_start
        sc25_daily = d_ssn[sc25_d_mask]
        sc25_daily_t = d_t[sc25_d_mask]

        if len(sc25_daily) > 0:
            # Find daily peak
            dpeak_idx = np.argmax(sc25_daily)
            # Count days above threshold
            days_above_100 = int(np.sum(sc25_daily >= 100))
            days_above_200 = int(np.sum(sc25_daily >= 200))
            total_days = len(sc25_daily)

            daily_metrics = {
                "daily_peak_ssn": round(float(sc25_daily[dpeak_idx]), 1),
                "daily_peak_date": round(float(sc25_daily_t[dpeak_idx]), 3),
                "days_above_100": days_above_100,
                "days_above_200": days_above_200,
                "pct_above_100": round(100 * days_above_100 / total_days, 1),
                "pct_above_200": round(100 * days_above_200 / total_days, 1),
                "total_days": total_days,
                "daily_mean_ssn": round(float(np.mean(sc25_daily)), 1),
                "daily_std_ssn": round(float(np.std(sc25_daily)), 1),
            }

    results = {
        "overlap_period": {
            "start": round(overlap_start, 2) if n_overlap > 0 else None,
            "end": round(overlap_end, 2) if n_overlap > 0 else None,
            "n_months": n_overlap,
        },
        "prediction_skill": {
            "bias": round(bias, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "ci_containment": round(ci_containment, 3),
            "months_above_prediction": int(above),
            "months_below_prediction": int(below),
            "interpretation": _interpret_skill(bias, rmse, ci_containment, n_overlap),
        },
        "peak_comparison": {
            "observed_peak_so_far": {
                "ssn": round(obs_peak_ssn, 1),
                "time": round(obs_peak_time, 2),
            },
            "predicted_peak": {
                "ssn": round(pred_peak_ssn, 1),
                "time": round(pred_peak_time, 2),
                "high": round(pred_peak_high, 1),
                "low": round(pred_peak_low, 1),
            },
            "smoothed_peak": smoothed_peak,
            "peak_ssn_ratio": round(obs_peak_ssn / pred_peak_ssn, 3) if pred_peak_ssn > 0 else None,
        },
        "cycle_comparison": cycle_comparison,
        "yearly_trajectory": trajectory,
        "daily_metrics": daily_metrics,
    }

    return results


def _interpret_skill(bias, rmse, ci_containment, n_months):
    """Generate human-readable interpretation of prediction skill."""
    parts = []

    if n_months == 0:
        return "No overlap between observations and predictions."

    if bias > 10:
        parts.append(f"SC25 is running well above NOAA predictions (mean bias +{bias:.1f})")
    elif bias > 0:
        parts.append(f"SC25 is running slightly above NOAA predictions (mean bias +{bias:.1f})")
    elif bias > -10:
        parts.append(f"SC25 is running slightly below NOAA predictions (mean bias {bias:.1f})")
    else:
        parts.append(f"SC25 is running well below NOAA predictions (mean bias {bias:.1f})")

    if ci_containment >= 0.9:
        parts.append(f"Observations well within confidence intervals ({ci_containment:.0%} containment)")
    elif ci_containment >= 0.7:
        parts.append(f"Observations mostly within confidence intervals ({ci_containment:.0%} containment)")
    elif ci_containment >= 0.5:
        parts.append(f"Observations partially outside confidence intervals ({ci_containment:.0%} containment)")
    else:
        parts.append(f"Observations frequently outside confidence intervals ({ci_containment:.0%} containment)")

    return ". ".join(parts) + "."


def run(monthly_records, prediction_records, daily_records, output_dir):
    """Run SC25 prediction assessment.

    Args:
        monthly_records: raw NOAA monthly index records
        prediction_records: raw NOAA prediction records
        daily_records: raw NOAA daily SSN records
        output_dir: Path to save results

    Returns:
        dict with prediction assessment results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .cycles import parse_monthly_data

    print("=" * 60)
    print("SC25 PREDICTION ASSESSMENT")
    print("=" * 60)

    print("\nParsing data...")
    monthly_data = parse_monthly_data(monthly_records)
    pred_data = parse_predictions(prediction_records)
    daily_data = parse_daily_ssn(daily_records)

    print(f"  Monthly observed: {monthly_data['time_yr'][0]:.1f} to {monthly_data['time_yr'][-1]:.1f} ({len(monthly_data['time_yr'])} months)")
    print(f"  Predictions: {pred_data['time_yr'][0]:.1f} to {pred_data['time_yr'][-1]:.1f} ({len(pred_data['time_yr'])} months)")
    print(f"  Daily SSN: {daily_data['time_yr'][0]:.1f} to {daily_data['time_yr'][-1]:.1f} ({len(daily_data['time_yr'])} days)")

    print("\nRunning assessment...")
    results = sc25_assessment(monthly_data, pred_data, daily_data)

    # Print summary
    skill = results["prediction_skill"]
    print(f"\nPrediction skill ({results['overlap_period']['n_months']} months overlap):")
    print(f"  Bias: {skill['bias']:+.1f}")
    print(f"  RMSE: {skill['rmse']:.1f}")
    print(f"  MAE: {skill['mae']:.1f}")
    print(f"  CI containment: {skill['ci_containment']:.1%}")
    print(f"  Above/below: {skill['months_above_prediction']}/{skill['months_below_prediction']}")

    peak = results["peak_comparison"]
    print(f"\nPeak comparison:")
    print(f"  Observed peak so far: {peak['observed_peak_so_far']['ssn']:.1f} at {peak['observed_peak_so_far']['time']:.1f}")
    print(f"  Predicted peak: {peak['predicted_peak']['ssn']:.1f} at {peak['predicted_peak']['time']:.1f}")
    print(f"  Peak ratio (obs/pred): {peak['peak_ssn_ratio']:.3f}")
    if peak['smoothed_peak']:
        print(f"  Smoothed SSN peak: {peak['smoothed_peak']['ssn']:.1f} at {peak['smoothed_peak']['time']:.1f}")

    comp = results["cycle_comparison"]
    print(f"\nCycle comparison:")
    print(f"  SC23 peak: {comp['SC23']['peak_ssn']:.1f}")
    print(f"  SC24 peak: {comp['SC24']['peak_ssn']:.1f}")
    print(f"  SC25 peak so far: {comp['SC25_observed']['peak_ssn_so_far']:.1f}")

    if results["daily_metrics"]:
        dm = results["daily_metrics"]
        print(f"\nDaily SSN metrics:")
        print(f"  Daily peak: {dm['daily_peak_ssn']:.1f}")
        print(f"  Days above 100: {dm['days_above_100']} ({dm['pct_above_100']:.1f}%)")
        print(f"  Days above 200: {dm['days_above_200']} ({dm['pct_above_200']:.1f}%)")

    print(f"\n  {skill['interpretation']}")

    # Save
    out_path = output_dir / "predictions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results
