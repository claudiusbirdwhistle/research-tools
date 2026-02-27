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
    """Stage 1: Ingest SEC filings from EDGAR.

    Performance notes:
    - Pre-loads the full set of known accession numbers in one DB query so
      each per-filing existence check is an O(1) in-memory set lookup rather
      than a round-trip that loads the full filing + all sections.
    - Processes tickers concurrently (up to 8 at once) so EDGAR API calls
      and file-cache reads overlap, subject to the AsyncLimiter rate cap.
    """
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

    # Pre-load the set of already-ingested accession numbers in one query.
    # This avoids N individual get_filing() calls (which load all sections).
    existing_accessions = await store.get_filing_accession_numbers(tickers)

    new_fetched = 0
    from_cache = 0
    failures = []
    ticker_results = {}

    # Semaphore limits concurrent EDGAR coroutines to avoid overwhelming
    # the rate limiter queue when many tickers are processed in parallel.
    _CONCURRENCY = 8
    sem = asyncio.Semaphore(_CONCURRENCY)
    results_lock = asyncio.Lock()

    async def process_ticker(ticker, client):
        nonlocal new_fetched, from_cache
        ticker_new = 0
        ticker_cached = 0
        ticker_failures = []

        async with sem:
            try:
                start_date = date(start_year, 1, 1)
                end_date = date(end_year, 12, 31)
                filings = await client.get_filings_for_ticker(ticker, form_types, start_date, end_date)

                for filing in filings:
                    # O(1) set lookup — much cheaper than loading full Filing
                    if filing.accession_number in existing_accessions:
                        ticker_cached += 1
                        continue

                    # Fetch and parse (document may still come from file cache)
                    try:
                        html = await client.get_filing_document(filing.url)
                        from edgar_sentinel.ingestion.parser import FilingParser
                        parser = FilingParser()
                        sections = parser.parse(html, filing.form_type, filing.accession_number)

                        from edgar_sentinel.core.models import Filing
                        f = Filing(metadata=filing, sections=sections)
                        await store.save_filing(f)
                        # Track in existing set so sibling tasks don't re-fetch
                        existing_accessions.add(filing.accession_number)
                        ticker_new += 1
                    except Exception as e:
                        ticker_failures.append(f"{ticker}/{filing.accession_number}: {e}")
                        print(
                            f"Warning: Failed to fetch {ticker} {filing.accession_number}: {e}",
                            file=sys.stderr,
                        )
            except Exception as e:
                ticker_failures.append(f"{ticker}: {e}")
                print(f"Warning: Failed to process {ticker}: {e}", file=sys.stderr)

        async with results_lock:
            new_fetched += ticker_new
            from_cache += ticker_cached
            failures.extend(ticker_failures)
            ticker_results[ticker] = {"new": ticker_new, "cached": ticker_cached}

    async with EdgarClient(edgar_cfg) as client:
        tasks = [process_ticker(ticker, client) for ticker in tickers]
        await asyncio.gather(*tasks)

    # Total in DB for configured tickers (fast: reuse existing_accessions set)
    total_in_db = len(existing_accessions)

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

    Performance notes:
    - Filters list_filings() to configured tickers so prior runs with a
      different universe do not inflate the workload.
    - Pre-loads all existing sentiment/similarity keys in two bulk queries
      (one per result type) rather than one DB query per section per filing.
    - Accumulates new results in memory and batch-saves with a single commit
      instead of committing after every single row.
    """
    new_sentiment = 0
    cached_sentiment = 0
    new_similarity = 0
    cached_similarity = 0

    # Filter to tickers in config when available; otherwise process all DB filings.
    # The ticker filter avoids mixing in stale data from prior runs with a
    # different universe — this is the common production path.
    ingestion_cfg = config.get("ingestion", {}) if isinstance(config, dict) else {}
    ingestion_tickers_str = ingestion_cfg.get("tickers", "") if ingestion_cfg else ""
    if ingestion_tickers_str:
        tickers = [t.strip().upper() for t in ingestion_tickers_str.split(",") if t.strip()]
        filing_metas = []
        for ticker in tickers:
            metas = await store.list_filings(ticker=ticker)
            filing_metas.extend(metas)
    else:
        # Fallback: process all filings in the DB (backward-compatible)
        filing_metas = await store.list_filings()

    # Load full filings (with sections) for all metadata entries.
    full_filings = {}
    for fm in filing_metas:
        try:
            filing = await store.get_filing(fm.accession_number)
            if filing and filing.sections:
                full_filings[fm.accession_number] = filing
        except Exception as e:
            print(f"Warning: Could not load filing {fm.accession_number}: {e}", file=sys.stderr)

    # Build list of filing IDs once for bulk existence queries
    filing_ids = list(full_filings.keys())

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

            # One bulk query instead of one query per (filing, section)
            existing_sent_keys = await store.get_existing_sentiment_keys(filing_ids)

            new_sent_results = []
            for acc_num, filing in full_filings.items():
                for section in filing.sections.values():
                    try:
                        key = (acc_num, section.section_name, "dictionary")
                        if key in existing_sent_keys:
                            cached_sentiment += 1
                            continue

                        result = analyzer.analyze(section)
                        if result:
                            new_sent_results.append(result)
                            existing_sent_keys.add(key)
                            new_sentiment += 1
                    except Exception as e:
                        print(f"Warning: Dict analysis failed: {e}", file=sys.stderr)

            # Single bulk commit for all new sentiment results
            await store.save_sentiments_batch(new_sent_results)
        except Exception as e:
            print(f"Warning: Dictionary analyzer init failed: {e}", file=sys.stderr)

    if ana_config.get("similarity", True):
        from edgar_sentinel.analyzers.similarity import SimilarityAnalyzer
        try:
            from edgar_sentinel.core.config import SimilarityAnalyzerConfig
            sim_cfg = SimilarityAnalyzerConfig(enabled=True)
            sim_analyzer = SimilarityAnalyzer(sim_cfg)

            # One bulk query instead of one query per (filing, section)
            existing_sim_keys = await store.get_existing_similarity_keys(filing_ids)

            # Group full filings by ticker and form type for sequential comparison
            from collections import defaultdict
            grouped = defaultdict(list)
            for acc_num, filing in full_filings.items():
                key = (filing.metadata.ticker, filing.metadata.form_type)
                grouped[key].append(filing)

            new_sim_results = []
            for key, group in grouped.items():
                group.sort(key=lambda f: str(f.metadata.filed_date))
                for i in range(1, len(group)):
                    current = group[i]
                    prior = group[i - 1]
                    for section in current.sections.values():
                        prior_section = prior.sections.get(section.section_name)
                        if prior_section:
                            try:
                                sim_key = (current.metadata.accession_number, section.section_name)
                                if sim_key in existing_sim_keys:
                                    cached_similarity += 1
                                    continue
                                result = sim_analyzer.analyze(section, prior_section)
                                if result:
                                    new_sim_results.append(result)
                                    existing_sim_keys.add(sim_key)
                                    new_similarity += 1
                            except Exception as e:
                                print(f"Warning: Similarity analysis failed: {e}", file=sys.stderr)

            # Single bulk commit for all new similarity results
            await store.save_similarities_batch(new_sim_results)
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
    # IMPORTANT: Filter to only the selected tickers so that stale data from
    # previous runs with different universe configurations is excluded.
    selected_tickers = {t.strip().upper() for t in config["ingestion"]["tickers"].split(",") if t.strip()}
    all_filings_meta_unfiltered = await store.list_filings()
    all_filings_meta = [f for f in all_filings_meta_unfiltered if f.ticker.upper() in selected_tickers]
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
    from edgar_sentinel.backtest.universe import (
        Sp100HistoricalProvider,
        Sp500HistoricalProvider,
        Sp50HistoricalProvider,
        StaticUniverseProvider,
    )

    tickers = [t.strip().upper() for t in config["ingestion"]["tickers"].split(",") if t.strip()]

    start_str = f"{config['ingestion']['startYear']}-01-01"
    end_str = f"{config['ingestion']['endYear']}-12-31"
    start = date.fromisoformat(start_str)
    end = date.today() if date.fromisoformat(end_str) > date.today() else date.fromisoformat(end_str)

    from edgar_sentinel.core.models import BacktestConfig, RebalanceFrequency, UniverseSource

    freq = RebalanceFrequency.QUARTERLY
    if bt_config.get("rebalanceFrequency") == "monthly":
        freq = RebalanceFrequency.MONTHLY

    universe_source_str = bt_config.get("universeSource", "static")
    _SOURCE_MAP = {
        "sp500_historical": UniverseSource.SP500_HISTORICAL,
        "sp100_historical": UniverseSource.SP100_HISTORICAL,
        "sp50_historical": UniverseSource.SP50_HISTORICAL,
    }
    universe_source = _SOURCE_MAP.get(universe_source_str, UniverseSource.STATIC)

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
        universe_source=universe_source,
    )

    # Build the appropriate universe provider for survivorship-bias control
    if universe_source == UniverseSource.SP500_HISTORICAL:
        universe_provider = Sp500HistoricalProvider()
    elif universe_source == UniverseSource.SP100_HISTORICAL:
        universe_provider = Sp100HistoricalProvider()
    elif universe_source == UniverseSource.SP50_HISTORICAL:
        universe_provider = Sp50HistoricalProvider()
    else:
        universe_provider = StaticUniverseProvider(tickers)

    db_path = str(EDGAR_DIR / "data" / "edgar_sentinel.db")
    provider = YFinanceProvider(cache_db_path=db_path)
    # BacktestEngine auto-configures MetricsCalculator annualization_factor
    # from config.rebalance_frequency — no need to pass explicit MetricsCalculator
    engine = BacktestEngine(
        config=bt_cfg,
        returns_provider=provider,
        universe_provider=universe_provider,
    )

    # Get composites, filtering to only selected tickers to avoid pollution
    # from previous runs with different universe configurations.
    ticker_set = set(tickers)
    composites = [c for c in all_composites if c.ticker.upper() in ticker_set]
    if not composites:
        all_db_composites = await store.get_composites()
        composites = [c for c in all_db_composites if c.ticker.upper() in ticker_set]

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


async def run_validation(config, composites, tickers, start, end):
    """Stage 5: Run statistical validation suite on signal scores.

    Builds signal_df from composites and returns_df from daily prices,
    then runs the full validation pipeline.
    """
    import pandas as pd
    from edgar_sentinel.signal_validation import run_full_validation

    db_path = str(EDGAR_DIR / "data" / "edgar_sentinel.db")

    # Build signal_df from composites
    signal_rows = []
    for c in composites:
        # Extract sentiment and similarity sub-scores from components
        sim_vals = [v for k, v in c.components.items() if "similarity" in k.lower()]
        sent_vals = [v for k, v in c.components.items() if "dictionary" in k.lower() or "sentiment" in k.lower()]
        similarity_score = sum(sim_vals) / len(sim_vals) if sim_vals else 0.0
        sentiment_score = sum(sent_vals) / len(sent_vals) if sent_vals else 0.0
        signal_rows.append({
            "entity_id": c.ticker,
            "filing_date": c.signal_date,
            "similarity_score": similarity_score,
            "sentiment_score": sentiment_score,
            "composite_score": c.composite_score,
        })

    if not signal_rows:
        return {"error": "No composites available for validation", "skipped": True}

    signal_df = pd.DataFrame(signal_rows)

    # Get daily returns for all tickers
    from edgar_sentinel.backtest.returns import YFinanceProvider
    provider = YFinanceProvider(cache_db_path=db_path)
    daily_returns = provider.get_returns(tickers, start, end, frequency="daily")

    if daily_returns is None or daily_returns.empty:
        return {"error": "Could not fetch daily returns for validation", "skipped": True}

    # Build returns_df with monthly forward-looking horizons
    # Trading-day approximations: 1m≈21d, 2m≈42d, 3m≈63d, 6m≈126d, 9m≈189d, 12m≈252d
    HORIZON_DAYS = {"ret_1m": 21, "ret_2m": 42, "ret_3m": 63, "ret_6m": 126, "ret_9m": 189, "ret_12m": 252}
    returns_rows = []
    for ticker in tickers:
        if ticker not in daily_returns.columns:
            continue
        series = daily_returns[ticker].dropna()
        n = len(series)
        for idx in series.index:
            pos = series.index.get_loc(idx)
            row = {
                "entity_id": ticker,
                "date": idx.date() if hasattr(idx, "date") else idx,
            }
            for col, days in HORIZON_DAYS.items():
                row[col] = float((1 + series.iloc[pos:pos+days]).prod() - 1) if pos + days <= n else None
            returns_rows.append(row)

    if not returns_rows:
        return {"error": "No returns data available for validation", "skipped": True}

    returns_df = pd.DataFrame(returns_rows).dropna(subset=["ret_1m"])

    # Run validation with reduced permutations for speed
    try:
        results = run_full_validation(
            signal_df=signal_df,
            returns_df=returns_df,
            signal_col="composite_score",
            return_horizons=["ret_1m", "ret_2m", "ret_3m", "ret_6m", "ret_9m", "ret_12m"],
            n_quantiles=min(5, len(set(signal_df["entity_id"]))),
            n_placebo_permutations=50,
            oos_strategy="expanding",
        )
        result_dict = results.to_dict()
        result_dict["summary_text"] = results.summary()
        result_dict["skipped"] = False
        return result_dict
    except Exception as e:
        return {"error": str(e), "skipped": True, "summary_text": f"Validation failed: {e}"}


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
        all_composites = sig_result.get("composites", [])
        bt_result = await run_backtest(store, config, config["backtest"], all_composites)
        update_stage(job_dir, "backtest", {
            "status": "completed",
            "summary": f"Backtest complete — Sharpe: {bt_result['strategy']['sharpeRatio']:.3f}, Return: {bt_result['strategy']['totalReturn']:.1%}",
        })

        # Stage 5: Signal Validation
        update_stage(job_dir, "validation", {"status": "running"})
        update_status(job_dir, {"currentStage": "validation"})
        start_dt = date.fromisoformat(f"{config['ingestion']['startYear']}-01-01")
        end_dt = date.today() if date.fromisoformat(f"{config['ingestion']['endYear']}-12-31") > date.today() else date.fromisoformat(f"{config['ingestion']['endYear']}-12-31")
        val_result = await run_validation(config, all_composites, tickers, start_dt, end_dt)
        if val_result.get("skipped"):
            update_stage(job_dir, "validation", {
                "status": "completed",
                "summary": f"Skipped: {val_result.get('error', 'insufficient data')}",
                "detail": val_result,
            })
        else:
            n_tests = len(val_result.get("ols_results", {})) + len(val_result.get("ic_results", {})) + len(val_result.get("portfolio_sort_results", {}))
            update_stage(job_dir, "validation", {
                "status": "completed",
                "summary": f"Validation complete — {n_tests} test results across return horizons",
                "detail": {"tests_run": n_tests},
            })

        bt_result["signalValidation"] = val_result

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
