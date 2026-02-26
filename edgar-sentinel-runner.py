#!/usr/bin/env python3
"""
EDGAR Sentinel Pipeline Runner

Executes the full pipeline (ingestion -> analysis -> signals -> backtest)
with progress tracking. Called by the Express dashboard backend.

Usage: python3 edgar-sentinel-runner.py <job_dir>

The job_dir must contain:
  - config.json: Pipeline configuration from the frontend
  - status.json: Initial job status (created by Express)

The script updates status.json as it progresses through stages and writes
final results including benchmark comparisons.
"""

import asyncio
import json
import os
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

# Add edgar-sentinel to path
EDGAR_DIR = Path("/agent/edgar-sentinel")
sys.path.insert(0, str(EDGAR_DIR / "src"))


def update_status(job_dir: Path, updates: dict):
    """Atomically update the job status file."""
    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text())
    status.update(updates)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.rename(status_path)


def update_stage(job_dir: Path, stage_name: str, stage_updates: dict):
    """Update a specific stage within the status."""
    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text())
    for s in status["stages"]:
        if s["stage"] == stage_name:
            s.update(stage_updates)
            break
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.rename(status_path)


async def run_ingestion(store, config, ing_config):
    """Stage 1: Ingest SEC filings from EDGAR."""
    from edgar_sentinel.ingestion.client import EdgarClient
    from edgar_sentinel.core.models import FormType

    tickers = [t.strip().upper() for t in ing_config["tickers"].split(",") if t.strip()]
    form_type = ing_config.get("formType", "both")
    start_year = ing_config.get("startYear", 2024)
    end_year = ing_config.get("endYear", 2026)

    form_types = []
    if form_type in ("10-K", "both"):
        form_types.append(FormType("10-K"))
    if form_type in ("10-Q", "both"):
        form_types.append(FormType("10-Q"))

    edgar_config_dict = {
        "user_agent": "EdgarSentinel research@example.com",
        "rate_limit": 8,
        "cache_dir": str(EDGAR_DIR / "data" / "filings"),
        "request_timeout": 30,
    }

    # Try to load from edgar-sentinel.yml
    yml_path = EDGAR_DIR / "edgar-sentinel.yml"
    if yml_path.exists():
        try:
            import yaml
            with open(yml_path) as f:
                yml = yaml.safe_load(f)
            if yml and "edgar" in yml:
                edgar_config_dict.update(yml["edgar"])
        except ImportError:
            pass

    from edgar_sentinel.core.config import EdgarConfig
    edgar_cfg = EdgarConfig(**edgar_config_dict)

    new_fetched = 0
    from_cache = 0
    failures = []
    ticker_results = {}

    async with EdgarClient(edgar_cfg) as client:
        for ticker in tickers:
            ticker_new = 0
            ticker_cached = 0
            try:
                start_date = date(start_year, 1, 1)
                end_date = date(end_year, 12, 31)
                filings = await client.get_filings_for_ticker(ticker, form_types, start_date, end_date)

                for filing in filings:
                    # Check if already ingested
                    existing = await store.get_filing(filing.accession_number)
                    if existing:
                        from_cache += 1
                        ticker_cached += 1
                        continue

                    # Fetch and parse
                    try:
                        html = await client.get_filing_document(filing.url)
                        from edgar_sentinel.ingestion.parser import FilingParser
                        parser = FilingParser()
                        sections = parser.parse(html, filing.form_type, filing.accession_number)

                        from edgar_sentinel.core.models import Filing
                        f = Filing(
                            metadata=filing,
                            sections=sections,
                        )
                        await store.save_filing(f)
                        new_fetched += 1
                        ticker_new += 1
                    except Exception as e:
                        failures.append(f"{ticker}/{filing.accession_number}: {e}")
                        print(f"Warning: Failed to fetch {ticker} {filing.accession_number}: {e}", file=sys.stderr)
            except Exception as e:
                failures.append(f"{ticker}: {e}")
                print(f"Warning: Failed to process {ticker}: {e}", file=sys.stderr)

            ticker_results[ticker] = {"new": ticker_new, "cached": ticker_cached}

    # Query DB for total count in the configured scope
    from edgar_sentinel.ingestion.store import SqliteStore as _S
    total_in_db = 0
    try:
        all_meta = await store.list_filings()
        # Filter to tickers in config
        ticker_set = set(tickers)
        total_in_db = sum(1 for fm in all_meta if fm.ticker in ticker_set)
    except Exception:
        pass

    total_via_api = new_fetched + from_cache
    return {
        "filings_count": total_via_api,
        "new_fetched": new_fetched,
        "from_cache": from_cache,
        "total_in_db": total_in_db,
        "tickers": tickers,
        "ticker_results": ticker_results,
        "failures": len(failures),
        "failure_details": failures[:10],  # first 10 only
    }


async def run_analysis(store, config, ana_config):
    """Stage 2: Run sentiment and similarity analyzers.

    list_filings() returns FilingMetadata (no sections). We load full Filing
    objects so we can iterate their sections.
    """
    new_sentiment = 0
    cached_sentiment = 0
    new_similarity = 0
    cached_similarity = 0

    # Load metadata list first, then fetch full Filing objects with sections
    filing_metas = await store.list_filings()

    # Load full filings (with sections) for all metadata entries.
    # Loaded lazily per-filing to avoid holding all HTML in memory at once.
    full_filings = {}
    for fm in filing_metas:
        try:
            filing = await store.get_filing(fm.accession_number)
            if filing and filing.sections:
                full_filings[fm.accession_number] = filing
        except Exception as e:
            print(f"Warning: Could not load filing {fm.accession_number}: {e}", file=sys.stderr)

    if ana_config.get("dictionary", True):
        from edgar_sentinel.analyzers.dictionary import DictionaryAnalyzer
        try:
            dict_cfg_dict = {"enabled": True}
            lm_path = EDGAR_DIR / "data" / "lm_dictionary.csv"
            if lm_path.exists():
                dict_cfg_dict["dictionary_path"] = str(lm_path)

            from edgar_sentinel.core.config import DictionaryAnalyzerConfig
            dict_cfg = DictionaryAnalyzerConfig(**dict_cfg_dict)
            analyzer = DictionaryAnalyzer(dict_cfg)

            for acc_num, filing in full_filings.items():
                # filing.sections is dict[SectionName, FilingSection]
                for section in filing.sections.values():
                    try:
                        existing = await store.get_sentiments(
                            acc_num,
                            section_name=section.section_name,
                            analyzer_name="dictionary"
                        )
                        if existing:
                            cached_sentiment += 1
                            continue

                        result = analyzer.analyze(section)
                        if result:
                            await store.save_sentiment(result)
                            new_sentiment += 1
                    except Exception as e:
                        print(f"Warning: Dict analysis failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Dictionary analyzer init failed: {e}", file=sys.stderr)

    if ana_config.get("similarity", True):
        from edgar_sentinel.analyzers.similarity import SimilarityAnalyzer
        try:
            from edgar_sentinel.core.config import SimilarityAnalyzerConfig
            sim_cfg = SimilarityAnalyzerConfig(enabled=True)
            sim_analyzer = SimilarityAnalyzer(sim_cfg)

            # Group full filings by ticker and form type for sequential comparison
            from collections import defaultdict
            grouped = defaultdict(list)
            for acc_num, filing in full_filings.items():
                key = (filing.metadata.ticker, filing.metadata.form_type)
                grouped[key].append(filing)

            for key, group in grouped.items():
                group.sort(key=lambda f: str(f.metadata.filed_date))
                for i in range(1, len(group)):
                    current = group[i]
                    prior = group[i - 1]
                    for section in current.sections.values():
                        prior_section = prior.sections.get(section.section_name)
                        if prior_section:
                            try:
                                existing = await store.get_similarity(
                                    current.metadata.accession_number,
                                    section_name=section.section_name
                                )
                                if existing:
                                    cached_similarity += 1
                                    continue
                                result = sim_analyzer.analyze(section, prior_section)
                                if result:
                                    await store.save_similarity(result)
                                    new_similarity += 1
                            except Exception as e:
                                print(f"Warning: Similarity analysis failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Similarity analyzer init failed: {e}", file=sys.stderr)

    total = new_sentiment + cached_sentiment + new_similarity + cached_similarity
    return {
        "analysis_results": total,
        "new_sentiment": new_sentiment,
        "cached_sentiment": cached_sentiment,
        "new_similarity": new_similarity,
        "cached_similarity": cached_similarity,
        "filings_with_sections": len(full_filings),
    }


async def run_signals(store, config, sig_config, bt_config):
    """Stage 3: Generate composite signals for each rebalance date."""
    from edgar_sentinel.signals.builder import SignalBuilder, FilingDateMapping
    from edgar_sentinel.signals.composite import SignalComposite
    from edgar_sentinel.backtest.portfolio import generate_rebalance_dates

    buffer_days = sig_config.get("bufferDays", 2)
    half_life = sig_config.get("decayHalfLife", 90)
    method = sig_config.get("compositeMethod", "equal")

    start_str = f"{bt_config.get('startYear', config['ingestion']['startYear'])}-01-01"
    end_str = f"{bt_config.get('endYear', config['ingestion']['endYear'])}-12-31"
    start = date.fromisoformat(start_str) if isinstance(start_str, str) else start_str
    end = date.fromisoformat(end_str) if isinstance(end_str, str) else end_str

    # Use backtest date range
    bt_start = bt_config.get("startDate", start_str)
    bt_end = bt_config.get("endDate", end_str)
    if isinstance(bt_start, str):
        bt_start = date.fromisoformat(bt_start)
    if isinstance(bt_end, str):
        bt_end = date.fromisoformat(bt_end)

    freq = bt_config.get("rebalanceFrequency", "quarterly")
    rebalance_dates = generate_rebalance_dates(bt_start, bt_end, freq)

    from edgar_sentinel.core.config import SignalsConfig
    from edgar_sentinel.core.models import CompositeMethod
    method_enum = CompositeMethod.EQUAL
    if method == "ic_weighted":
        method_enum = CompositeMethod.IC_WEIGHTED

    sig_cfg = SignalsConfig(
        buffer_days=buffer_days,
        decay_half_life=half_life,
        composite_method=method_enum,
    )

    builder = SignalBuilder(sig_cfg)
    composite = SignalComposite(method=method_enum)

    # Collect all analysis data once (outside rebalance loop) for efficiency.
    # store.get_sentiments/get_similarity require a filing_id, so we iterate
    # over all filings and aggregate.
    all_filings_meta = await store.list_filings()
    all_sentiments = []
    all_similarities = []
    from datetime import timedelta
    filing_dates = {}
    for f in all_filings_meta:
        fd = f.filed_date
        if isinstance(fd, str):
            fd = date.fromisoformat(fd)
        filing_dates[f.accession_number] = FilingDateMapping(
            ticker=f.ticker,
            filing_id=f.accession_number,
            filed_date=fd,
            signal_date=fd + timedelta(days=buffer_days),
        )
        sents = await store.get_sentiments(f.accession_number)
        all_sentiments.extend(sents)
        sims = await store.get_similarity(f.accession_number)
        all_similarities.extend(sims)

    from edgar_sentinel.analyzers.base import AnalysisResults
    results = AnalysisResults(
        sentiment_results=all_sentiments,
        similarity_results=all_similarities,
    )

    total_signals = 0
    all_composites = []

    for as_of in rebalance_dates:
        signals = builder.build(results, filing_dates, as_of_date=as_of)
        composites = composite.combine(signals, as_of_date=as_of)

        for c in composites:
            await store.save_composite(c)
            all_composites.append(c)
            total_signals += 1

    return {"signals_generated": total_signals, "rebalance_dates": len(rebalance_dates), "composites": all_composites}


def _compound_period_return(monthly_map: dict, period_start, period_end) -> float:
    """Compound monthly benchmark returns over a holding period (period_start, period_end].

    For quarterly rebalancing, compounds 3 monthly returns rather than
    looking up just the endpoint month. Falls back to single-month lookup
    when period_start is None.
    """
    import pandas as pd

    if period_start is None:
        month_str = period_end.strftime("%Y-%m") if hasattr(period_end, "strftime") else str(period_end)[:7]
        return monthly_map.get(month_str, 0.0)

    compound = 1.0
    start_ts = pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end)
    month_dates = pd.date_range(start=start_ts, end=end_ts, freq="ME")
    for dt in month_dates:
        month_str = dt.strftime("%Y-%m")
        r = monthly_map.get(month_str, 0.0)
        compound *= (1 + r)
    return compound - 1.0


async def run_backtest(store, config, bt_config, all_composites):
    """Stage 4: Run backtest with benchmark comparisons."""
    from edgar_sentinel.backtest.engine import BacktestEngine
    from edgar_sentinel.backtest.returns import YFinanceProvider

    tickers = [t.strip().upper() for t in config["ingestion"]["tickers"].split(",") if t.strip()]

    start_str = f"{config['ingestion']['startYear']}-01-01"
    end_str = f"{config['ingestion']['endYear']}-12-31"
    start = date.fromisoformat(start_str)
    end = date.today() if date.fromisoformat(end_str) > date.today() else date.fromisoformat(end_str)

    from edgar_sentinel.core.models import BacktestConfig, RebalanceFrequency

    freq = RebalanceFrequency.QUARTERLY
    if bt_config.get("rebalanceFrequency") == "monthly":
        freq = RebalanceFrequency.MONTHLY

    bt_cfg = BacktestConfig(
        start_date=start,
        end_date=end,
        universe=tickers,
        rebalance_frequency=freq,
        num_quantiles=bt_config.get("numQuantiles", 5),
        signal_buffer_days=config.get("signals", {}).get("bufferDays", 2),
        long_quantile=bt_config.get("longQuantile", 1),
        short_quantile=bt_config.get("shortQuantile"),
        transaction_cost_bps=bt_config.get("transactionCostBps", 10),
    )

    db_path = str(EDGAR_DIR / "data" / "edgar_sentinel.db")
    provider = YFinanceProvider(cache_db_path=db_path)
    # BacktestEngine auto-configures MetricsCalculator annualization_factor
    # from config.rebalance_frequency — no need to pass explicit MetricsCalculator
    engine = BacktestEngine(
        config=bt_cfg,
        returns_provider=provider,
    )

    # Get composites from store
    composites = all_composites
    if not composites:
        composites = await store.get_composites()

    result = engine.run(signals=composites)

    # Compute benchmarks
    benchmarks = await compute_benchmarks(provider, tickers, start, end)

    # Build period returns comparison (correctly compounded over the full holding period)
    monthly = []
    for mr in getattr(result, "monthly_returns", []):
        period_end_val = mr.period_end
        period_start_val = mr.period_start  # Now populated from engine
        month_str = period_end_val.strftime("%Y-%m") if hasattr(period_end_val, "strftime") else str(period_end_val)[:7]
        # Compound benchmark returns over the full holding period, not just endpoint month
        spy_ret = _compound_period_return(benchmarks["spy_monthly"], period_start_val, period_end_val)
        ew_ret = _compound_period_return(benchmarks["ew_monthly"], period_start_val, period_end_val)
        strat_ret = mr.long_return
        monthly.append({
            "month": month_str,
            "strategy": strat_ret,
            "spy": spy_ret,
            "equalWeight": ew_ret,
        })

    # Build signal rankings (latest quarter)
    rankings = []
    if composites:
        latest_date = max(c.signal_date for c in composites)
        latest = sorted(
            [c for c in composites if c.signal_date == latest_date],
            key=lambda c: c.composite_score,
            reverse=True,
        )
        for i, c in enumerate(latest, 1):
            rankings.append({
                "ticker": c.ticker,
                "compositeScore": c.composite_score,
                "rank": i,
            })

    # Build equity curve: daily portfolio/SPY/EW values starting at $10,000
    equity_curve = []
    for pt in getattr(result, "equity_curve", []):
        equity_curve.append({
            "date": pt.date.isoformat() if hasattr(pt.date, "isoformat") else str(pt.date),
            "portfolio": pt.portfolio_value,
            "spy": pt.spy_value,
            "equalWeight": pt.ew_value,
        })

    # Build portfolioHistory: per-rebalance snapshot with positions and dollar values
    portfolio_history = []
    portfolio_value = 10_000.0  # starting account value
    snapshots = engine.portfolio_history.snapshots
    for snapshot, mr in zip(snapshots, result.monthly_returns):
        positions_list = []
        for pos in snapshot.positions:
            leg = "long" if pos.weight >= 0 else "short"
            dollar_value = round(portfolio_value * abs(pos.weight), 2)
            positions_list.append({
                "ticker": pos.ticker,
                "weight": round(pos.weight, 6),
                "quantile": pos.quantile,
                "signalScore": round(pos.signal_score, 4),
                "leg": leg,
                "dollarValue": dollar_value,
            })
        portfolio_history.append({
            "rebalanceDate": snapshot.rebalance_date.isoformat()
                if hasattr(snapshot.rebalance_date, "isoformat")
                else str(snapshot.rebalance_date),
            "portfolioValue": round(portfolio_value, 2),
            "turnover": round(snapshot.turnover, 4),
            "transactionCost": round(snapshot.transaction_cost, 6),
            "positions": positions_list,
            "nLong": snapshot.n_long,
            "nShort": snapshot.n_short,
        })
        # Advance portfolio value to the next period using net return
        # For long-only: long_return minus transaction_cost
        # For long-short: long_short_return already includes transaction cost
        if mr.long_short_return is not None:
            period_net = mr.long_short_return
        else:
            period_net = mr.long_return - snapshot.transaction_cost
        portfolio_value *= (1.0 + period_net)

    # Build signalHistory: per-date composite scores for all tickers
    from collections import defaultdict
    signal_by_date: dict = defaultdict(list)
    for c in composites:
        date_str = c.signal_date.isoformat() if hasattr(c.signal_date, "isoformat") else str(c.signal_date)
        signal_by_date[date_str].append(c)

    signal_history = []
    for date_str in sorted(signal_by_date.keys()):
        tickers_on_date = sorted(
            signal_by_date[date_str],
            key=lambda c: c.composite_score,
            reverse=True,
        )
        signal_history.append({
            "date": date_str,
            "signals": [
                {
                    "ticker": c.ticker,
                    "compositeScore": c.composite_score,
                    "rank": i + 1,
                }
                for i, c in enumerate(tickers_on_date)
            ],
        })

    return {
        "strategy": {
            "totalReturn": result.total_return,
            "annualizedReturn": result.annualized_return,
            "sharpeRatio": result.sharpe_ratio,
            "sortinoRatio": getattr(result, "sortino_ratio", 0),
            "maxDrawdown": result.max_drawdown,
            "winRate": getattr(result, "win_rate", 0),
            "informationCoefficient": getattr(result, "information_ratio", 0),
        },
        "benchmarks": benchmarks["summary"],
        "monthlyReturns": monthly,
        "signalRankings": rankings,
        "equityCurve": equity_curve,
        "portfolioHistory": portfolio_history,
        "signalHistory": signal_history,
    }


async def compute_benchmarks(provider, tickers, start, end):
    """Compute SPY and equal-weight buy-and-hold benchmarks."""
    import pandas as pd

    spy_monthly = {}
    ew_monthly = {}
    spy_summary = {"totalReturn": 0, "annualizedReturn": 0, "sharpeRatio": 0}
    ew_summary = {"totalReturn": 0, "annualizedReturn": 0, "sharpeRatio": 0}

    try:
        # SPY benchmark
        spy_returns = provider.get_returns(["SPY"], start, end, frequency="monthly")
        if spy_returns is not None and not spy_returns.empty and "SPY" in spy_returns.columns:
            spy_series = spy_returns["SPY"].dropna()
            for idx, val in spy_series.items():
                month_str = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)[:7]
                spy_monthly[month_str] = float(val)

            total = float((1 + spy_series).prod() - 1)
            years = max((end - start).days / 365.25, 0.01)
            annualized = (1 + total) ** (1 / years) - 1
            sharpe = float(spy_series.mean() / spy_series.std() * (12 ** 0.5)) if spy_series.std() > 0 else 0
            spy_summary = {"totalReturn": total, "annualizedReturn": annualized, "sharpeRatio": sharpe}

        # Equal-weight benchmark
        ew_returns_df = provider.get_returns(tickers, start, end, frequency="monthly")
        if ew_returns_df is not None and not ew_returns_df.empty:
            available = [t for t in tickers if t in ew_returns_df.columns]
            if available:
                ew_series = ew_returns_df[available].mean(axis=1).dropna()
                for idx, val in ew_series.items():
                    month_str = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)[:7]
                    ew_monthly[month_str] = float(val)

                total = float((1 + ew_series).prod() - 1)
                years = max((end - start).days / 365.25, 0.01)
                annualized = (1 + total) ** (1 / years) - 1
                sharpe = float(ew_series.mean() / ew_series.std() * (12 ** 0.5)) if ew_series.std() > 0 else 0
                ew_summary = {"totalReturn": total, "annualizedReturn": annualized, "sharpeRatio": sharpe}
    except Exception as e:
        print(f"Warning: Benchmark computation error: {e}", file=sys.stderr)

    return {
        "summary": {
            "spy": spy_summary,
            "equalWeight": ew_summary,
        },
        "spy_monthly": spy_monthly,
        "ew_monthly": ew_monthly,
    }


async def main():
    if len(sys.argv) < 2:
        print("Usage: edgar-sentinel-runner.py <job_dir>", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1])
    config = json.loads((job_dir / "config.json").read_text())

    # Update status to running
    update_status(job_dir, {"status": "running", "currentStage": "ingestion"})

    # Initialize store
    from edgar_sentinel.ingestion.store import SqliteStore
    from edgar_sentinel.core.config import StorageConfig

    db_path = str(EDGAR_DIR / "data" / "edgar_sentinel.db")
    store_cfg = StorageConfig(sqlite_path=db_path)
    store = SqliteStore(store_cfg)
    await store.initialize()

    try:
        # Stage 1: Ingestion
        update_stage(job_dir, "ingestion", {"status": "running"})
        update_status(job_dir, {"currentStage": "ingestion"})
        ing_result = await run_ingestion(store, config, config["ingestion"])
        ing_new = ing_result["new_fetched"]
        ing_cached = ing_result["from_cache"]
        ing_total_db = ing_result["total_in_db"]
        ing_failures = ing_result["failures"]
        if ing_new > 0 or ing_cached > 0:
            ing_summary = (
                f"{ing_new} new + {ing_cached} cached filings fetched via EDGAR API "
                f"({ing_total_db} total in DB for {len(ing_result['tickers'])} tickers)"
            )
        else:
            ing_summary = (
                f"EDGAR API returned 0 filings via API "
                f"({ing_total_db} available in DB for {len(ing_result['tickers'])} tickers"
                + (f", {ing_failures} fetch failures" if ing_failures else "")
                + ")"
            )
        update_stage(job_dir, "ingestion", {
            "status": "completed",
            "summary": ing_summary,
            "detail": ing_result,
        })

        # Stage 2: Analysis
        update_stage(job_dir, "analysis", {"status": "running"})
        update_status(job_dir, {"currentStage": "analysis"})
        ana_result = await run_analysis(store, config, config["analysis"])
        ana_new = ana_result["new_sentiment"] + ana_result["new_similarity"]
        ana_cached = ana_result["cached_sentiment"] + ana_result["cached_similarity"]
        ana_summary = (
            f"{ana_new} new + {ana_cached} cached analysis results "
            f"({ana_result['filings_with_sections']} filings with sections)"
        )
        update_stage(job_dir, "analysis", {
            "status": "completed",
            "summary": ana_summary,
            "detail": ana_result,
        })

        # Stage 3: Signals
        update_stage(job_dir, "signals", {"status": "running"})
        update_status(job_dir, {"currentStage": "signals"})

        # Build backtest config for signal generation
        tickers = [t.strip().upper() for t in config["ingestion"]["tickers"].split(",") if t.strip()]
        start_str = f"{config['ingestion']['startYear']}-01-01"
        end_str = f"{config['ingestion']['endYear']}-12-31"
        bt_config_for_signals = {
            "startDate": start_str,
            "endDate": end_str,
            "rebalanceFrequency": config["backtest"].get("rebalanceFrequency", "quarterly"),
        }
        sig_result = await run_signals(store, config, config["signals"], bt_config_for_signals)
        update_stage(job_dir, "signals", {
            "status": "completed",
            "summary": f"Generated {sig_result['signals_generated']} signals across {sig_result['rebalance_dates']} rebalance dates",
            "detail": {"signals_generated": sig_result["signals_generated"], "rebalance_dates": sig_result["rebalance_dates"]},
        })

        # Stage 4: Backtest
        update_stage(job_dir, "backtest", {"status": "running"})
        update_status(job_dir, {"currentStage": "backtest"})
        bt_result = await run_backtest(store, config, config["backtest"], sig_result.get("composites", []))
        update_stage(job_dir, "backtest", {
            "status": "completed",
            "summary": f"Backtest complete — Sharpe: {bt_result['strategy']['sharpeRatio']:.3f}, Return: {bt_result['strategy']['totalReturn']:.1%}",
        })

        # Final status
        update_status(job_dir, {
            "status": "completed",
            "currentStage": "complete",
            "results": bt_result,
            "completedAt": datetime.utcnow().isoformat() + "Z",
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)

        # Find which stage failed
        status = json.loads((job_dir / "status.json").read_text())
        for s in status["stages"]:
            if s["status"] == "running":
                update_stage(job_dir, s["stage"], {
                    "status": "failed",
                    "error": str(e),
                })
                break

        update_status(job_dir, {
            "status": "failed",
            "error": str(e),
            "completedAt": datetime.utcnow().isoformat() + "Z",
        })
        sys.exit(1)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
