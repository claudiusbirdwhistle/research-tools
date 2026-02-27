#!/usr/bin/env python3
"""
EDGAR Sentinel Batch Backtest Runner

Runs a comprehensive comparison study across multiple backtest configurations:
  - 3 analysis types: similarity, sentiment, composite
  - 3 filing types: 10-K, 10-Q, both
  - 2 rebalance frequencies: monthly, quarterly

= 18 total configurations on the survivorship-bias-controlled SP500 universe.

Designed for multi-invocation execution:
  Phase 1: ingest  — Fetch SEC filings for all SP500 tickers (2020-present)
  Phase 2: analyze — Run sentiment + similarity analysis on all filings
  Phase 3: backtest — Run all 18 configurations
  Phase 4: report  — Generate comprehensive comparison report

Progress is tracked in /state/batch-progress.json for cross-invocation resumability.

Usage:
  python3 backtest-batch-runner.py ingest [--chunk-size 50]
  python3 backtest-batch-runner.py analyze
  python3 backtest-batch-runner.py backtest
  python3 backtest-batch-runner.py report
  python3 backtest-batch-runner.py status
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

# Add edgar-sentinel to path
EDGAR_DIR = Path("/agent/edgar-sentinel")
sys.path.insert(0, str(EDGAR_DIR / "src"))

PROGRESS_FILE = Path("/state/batch-progress.json")
RESULTS_DIR = Path("/output/backtest-comparison")
DB_PATH = str(EDGAR_DIR / "data" / "edgar_sentinel.db")

# The 18 backtest configurations
ANALYSIS_TYPES = ["similarity", "sentiment", "composite"]
FILING_TYPES = ["10-K", "10-Q", "both"]
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]

# Backtest parameters
START_DATE = date(2020, 1, 1)
END_DATE = date.today()
UNIVERSE_SOURCE = "sp500_historical"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {
        "phase": "not_started",
        "ingestion": {
            "total_tickers": 0,
            "completed_tickers": [],
            "failed_tickers": [],
            "filings_ingested": 0,
        },
        "analysis": {
            "completed": False,
            "sentiment_count": 0,
            "similarity_count": 0,
        },
        "backtests": {},  # config_key -> result summary
        "started_at": None,
        "last_updated": None,
    }


def save_progress(progress: dict):
    progress["last_updated"] = datetime.utcnow().isoformat() + "Z"
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2, default=str))
    tmp.rename(PROGRESS_FILE)


def get_sp500_superset() -> list[str]:
    """Compute the superset of all SP500 tickers from 2020 to present."""
    from edgar_sentinel.backtest.universe import Sp500HistoricalProvider
    from edgar_sentinel.backtest.portfolio import generate_rebalance_dates

    provider = Sp500HistoricalProvider()
    all_tickers = set()

    # Sample quarterly to catch all members
    dates = generate_rebalance_dates(START_DATE, END_DATE, "quarterly")
    for d in dates:
        tickers = provider.get_tickers(d)
        all_tickers.update(tickers)

    return sorted(all_tickers)


async def run_ingest_phase(chunk_size: int = 50):
    """Phase 1: Ingest SEC filings for all SP500 tickers."""
    from edgar_sentinel.ingestion.client import EdgarClient
    from edgar_sentinel.ingestion.store import SqliteStore
    from edgar_sentinel.ingestion.parser import FilingParser
    from edgar_sentinel.core.config import EdgarConfig, StorageConfig
    from edgar_sentinel.core.models import FormType, Filing

    progress = load_progress()
    progress["phase"] = "ingestion"
    if not progress["started_at"]:
        progress["started_at"] = datetime.utcnow().isoformat() + "Z"

    all_tickers = get_sp500_superset()
    progress["ingestion"]["total_tickers"] = len(all_tickers)

    # Filter to tickers not yet completed
    completed = set(progress["ingestion"]["completed_tickers"])
    remaining = [t for t in all_tickers if t not in completed]

    print(f"SP500 superset: {len(all_tickers)} tickers")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All tickers ingested!")
        save_progress(progress)
        return

    # Take a chunk
    chunk = remaining[:chunk_size]
    print(f"Processing chunk of {len(chunk)} tickers: {chunk[:10]}...")

    # Initialize store
    store_cfg = StorageConfig(sqlite_path=DB_PATH)
    store = SqliteStore(store_cfg)
    await store.initialize()

    # EDGAR config
    edgar_config_dict = {
        "user_agent": "EdgarSentinel research@example.com",
        "rate_limit": 8,
        "cache_dir": str(EDGAR_DIR / "data" / "filings"),
        "request_timeout": 30,
    }
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

    edgar_cfg = EdgarConfig(**edgar_config_dict)
    form_types = [FormType("10-K"), FormType("10-Q")]

    # Pre-load existing accessions
    existing_accessions = await store.get_filing_accession_numbers(all_tickers)

    new_total = 0
    cached_total = 0
    sem = asyncio.Semaphore(8)
    parser = FilingParser()

    async def process_ticker(ticker, client):
        nonlocal new_total, cached_total
        new_count = 0
        cached_count = 0

        async with sem:
            try:
                start_dt = date(2020, 1, 1)
                end_dt = date(2026, 12, 31)
                filings = await client.get_filings_for_ticker(
                    ticker, form_types, start_dt, end_dt
                )

                for filing in filings:
                    if filing.accession_number in existing_accessions:
                        cached_count += 1
                        continue

                    try:
                        html = await client.get_filing_document(filing.url)
                        sections = parser.parse(html, filing.form_type, filing.accession_number)
                        f = Filing(metadata=filing, sections=sections)
                        await store.save_filing(f)
                        existing_accessions.add(filing.accession_number)
                        new_count += 1
                    except Exception as e:
                        print(f"  Warning: {ticker}/{filing.accession_number}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"  Error processing {ticker}: {e}", file=sys.stderr)
                progress["ingestion"]["failed_tickers"].append(ticker)

        new_total += new_count
        cached_total += cached_count
        print(f"  {ticker}: {new_count} new, {cached_count} cached")
        progress["ingestion"]["completed_tickers"].append(ticker)

    try:
        async with EdgarClient(edgar_cfg) as client:
            tasks = [process_ticker(t, client) for t in chunk]
            await asyncio.gather(*tasks)
    finally:
        await store.close()

    progress["ingestion"]["filings_ingested"] += new_total
    done = len(progress["ingestion"]["completed_tickers"])
    total = progress["ingestion"]["total_tickers"]
    print(f"\nChunk complete: {new_total} new, {cached_total} cached")
    print(f"Overall progress: {done}/{total} tickers ({done/total*100:.1f}%)")
    save_progress(progress)


async def run_analyze_phase():
    """Phase 2: Run sentiment + similarity analysis on all filings."""
    from edgar_sentinel.ingestion.store import SqliteStore
    from edgar_sentinel.core.config import StorageConfig

    progress = load_progress()
    progress["phase"] = "analysis"

    store_cfg = StorageConfig(sqlite_path=DB_PATH)
    store = SqliteStore(store_cfg)
    await store.initialize()

    try:
        # Get all filings for SP500 tickers
        all_tickers = get_sp500_superset()
        filing_metas = []
        for ticker in all_tickers:
            metas = await store.list_filings(ticker=ticker)
            filing_metas.extend(metas)

        print(f"Total filings to analyze: {len(filing_metas)}")

        # Load full filings
        full_filings = {}
        for fm in filing_metas:
            try:
                filing = await store.get_filing(fm.accession_number)
                if filing and filing.sections:
                    full_filings[fm.accession_number] = filing
            except Exception:
                pass

        print(f"Filings with sections: {len(full_filings)}")
        filing_ids = list(full_filings.keys())

        # Sentiment analysis (dictionary)
        from edgar_sentinel.analyzers.dictionary import DictionaryAnalyzer
        from edgar_sentinel.core.config import DictionaryAnalyzerConfig

        dict_cfg_dict = {"enabled": True}
        lm_path = EDGAR_DIR / "data" / "lm_dictionary.csv"
        if lm_path.exists():
            dict_cfg_dict["dictionary_path"] = str(lm_path)

        dict_cfg = DictionaryAnalyzerConfig(**dict_cfg_dict)
        analyzer = DictionaryAnalyzer(dict_cfg)

        existing_sent_keys = await store.get_existing_sentiment_keys(filing_ids)
        new_sent_results = []
        cached_sent = 0

        for acc_num, filing in full_filings.items():
            for section in filing.sections.values():
                key = (acc_num, section.section_name, "dictionary")
                if key in existing_sent_keys:
                    cached_sent += 1
                    continue
                try:
                    result = analyzer.analyze(section)
                    if result:
                        new_sent_results.append(result)
                        existing_sent_keys.add(key)
                except Exception:
                    pass

        await store.save_sentiments_batch(new_sent_results)
        print(f"Sentiment: {len(new_sent_results)} new, {cached_sent} cached")

        # Similarity analysis
        from edgar_sentinel.analyzers.similarity import SimilarityAnalyzer
        from edgar_sentinel.core.config import SimilarityAnalyzerConfig
        from collections import defaultdict

        sim_cfg = SimilarityAnalyzerConfig(enabled=True)
        sim_analyzer = SimilarityAnalyzer(sim_cfg)
        existing_sim_keys = await store.get_existing_similarity_keys(filing_ids)

        grouped = defaultdict(list)
        for acc_num, filing in full_filings.items():
            key = (filing.metadata.ticker, filing.metadata.form_type)
            grouped[key].append(filing)

        new_sim_results = []
        cached_sim = 0

        for key, group in grouped.items():
            group.sort(key=lambda f: str(f.metadata.filed_date))
            for i in range(1, len(group)):
                current = group[i]
                prior = group[i - 1]
                for section in current.sections.values():
                    prior_section = prior.sections.get(section.section_name)
                    if prior_section:
                        sim_key = (current.metadata.accession_number, section.section_name)
                        if sim_key in existing_sim_keys:
                            cached_sim += 1
                            continue
                        try:
                            result = sim_analyzer.analyze(section, prior_section)
                            if result:
                                new_sim_results.append(result)
                                existing_sim_keys.add(sim_key)
                        except Exception:
                            pass

        await store.save_similarities_batch(new_sim_results)
        print(f"Similarity: {len(new_sim_results)} new, {cached_sim} cached")

        progress["analysis"]["completed"] = True
        progress["analysis"]["sentiment_count"] = len(new_sent_results) + cached_sent
        progress["analysis"]["similarity_count"] = len(new_sim_results) + cached_sim

    finally:
        await store.close()

    save_progress(progress)
    print("Analysis phase complete.")


def config_key(analysis_type: str, filing_type: str, rebalance_freq: str) -> str:
    """Generate a unique key for a backtest configuration."""
    return f"{analysis_type}__{filing_type}__{rebalance_freq}"


async def run_single_backtest(
    analysis_type: str,
    filing_type: str,
    rebalance_freq: str,
) -> dict:
    """Run a single backtest configuration.

    Returns a dict with strategy metrics and signal validation summary.
    """
    from edgar_sentinel.ingestion.store import SqliteStore
    from edgar_sentinel.core.config import StorageConfig, SignalsConfig
    from edgar_sentinel.core.models import (
        BacktestConfig, CompositeMethod, CompositeSignal,
        RebalanceFrequency, UniverseSource,
    )
    from edgar_sentinel.signals.builder import SignalBuilder, FilingDateMapping
    from edgar_sentinel.signals.composite import SignalComposite
    from edgar_sentinel.backtest.engine import BacktestEngine
    from edgar_sentinel.backtest.returns import YFinanceProvider
    from edgar_sentinel.backtest.universe import Sp500HistoricalProvider
    from edgar_sentinel.backtest.portfolio import generate_rebalance_dates
    from edgar_sentinel.analyzers.base import AnalysisResults

    key = config_key(analysis_type, filing_type, rebalance_freq)
    print(f"\n{'='*60}")
    print(f"Running: {key}")
    print(f"  Analysis: {analysis_type} | Filing: {filing_type} | Rebalance: {rebalance_freq}")
    print(f"{'='*60}")

    store_cfg = StorageConfig(sqlite_path=DB_PATH)
    store = SqliteStore(store_cfg)
    await store.initialize()

    try:
        # Get all filings, filtering by form type
        all_tickers = get_sp500_superset()
        filing_metas = []
        for ticker in all_tickers:
            metas = await store.list_filings(ticker=ticker)
            filing_metas.extend(metas)

        # Filter by filing type
        if filing_type == "10-K":
            filing_metas = [f for f in filing_metas if f.form_type == "10-K"]
        elif filing_type == "10-Q":
            filing_metas = [f for f in filing_metas if f.form_type == "10-Q"]
        # "both" = no filter

        print(f"  Filings after form type filter: {len(filing_metas)}")

        # Collect analysis results from DB for these filings
        all_sentiments = []
        all_similarities = []
        filing_dates = {}

        for fm in filing_metas:
            fd = fm.filed_date
            if isinstance(fd, str):
                fd = date.fromisoformat(fd)
            filing_dates[fm.accession_number] = FilingDateMapping(
                ticker=fm.ticker,
                filing_id=fm.accession_number,
                filed_date=fd,
                signal_date=fd + timedelta(days=2),
            )

            # Load analysis results based on analysis_type
            if analysis_type in ("sentiment", "composite"):
                sents = await store.get_sentiments(fm.accession_number)
                all_sentiments.extend(sents)
            if analysis_type in ("similarity", "composite"):
                sims = await store.get_similarity(fm.accession_number)
                all_similarities.extend(sims)

        print(f"  Sentiments: {len(all_sentiments)}, Similarities: {len(all_similarities)}")

        # Build signals
        sig_cfg = SignalsConfig(
            buffer_days=2,
            decay_half_life=90,
            composite_method=CompositeMethod.EQUAL,
        )
        builder = SignalBuilder(sig_cfg)
        composite = SignalComposite(method=CompositeMethod.EQUAL)

        results = AnalysisResults(
            sentiment_results=all_sentiments,
            similarity_results=all_similarities,
        )

        # Generate rebalance dates
        freq = rebalance_freq
        rebalance_dates = generate_rebalance_dates(START_DATE, END_DATE, freq)
        print(f"  Rebalance dates: {len(rebalance_dates)}")

        all_composites: list[CompositeSignal] = []
        for as_of in rebalance_dates:
            signals = builder.build(results, filing_dates, as_of_date=as_of)
            composites = composite.combine(signals, as_of_date=as_of)
            all_composites.extend(composites)

        print(f"  Composite signals: {len(all_composites)}")

        if not all_composites:
            return {
                "key": key,
                "error": "No composite signals generated",
                "analysis_type": analysis_type,
                "filing_type": filing_type,
                "rebalance_freq": rebalance_freq,
            }

        # Run backtest
        rb_freq = (
            RebalanceFrequency.MONTHLY
            if rebalance_freq == "monthly"
            else RebalanceFrequency.QUARTERLY
        )

        bt_cfg = BacktestConfig(
            start_date=START_DATE,
            end_date=END_DATE,
            universe=all_tickers,
            rebalance_frequency=rb_freq,
            num_quantiles=5,
            signal_buffer_days=2,
            long_quantile=1,
            short_quantile=None,
            transaction_cost_bps=10,
            universe_source=UniverseSource.SP500_HISTORICAL,
        )

        provider = YFinanceProvider(cache_db_path=DB_PATH)
        universe_provider = Sp500HistoricalProvider()

        engine = BacktestEngine(
            config=bt_cfg,
            returns_provider=provider,
            universe_provider=universe_provider,
        )

        result = engine.run(signals=all_composites)

        # Extract key metrics
        metrics = {
            "key": key,
            "analysis_type": analysis_type,
            "filing_type": filing_type,
            "rebalance_freq": rebalance_freq,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "information_ratio": result.information_ratio,
            "turnover": result.turnover,
            "n_periods": len(result.monthly_returns),
            "n_composites": len(all_composites),
            "n_filings": len(filing_metas),
            "n_sentiments": len(all_sentiments),
            "n_similarities": len(all_similarities),
        }

        # Compute win rate
        wins = sum(1 for mr in result.monthly_returns if mr.long_return > 0)
        metrics["win_rate"] = wins / len(result.monthly_returns) if result.monthly_returns else 0

        # Monthly returns for analysis
        monthly_returns = [
            {
                "period_start": mr.period_start.isoformat() if mr.period_start else None,
                "period_end": mr.period_end.isoformat(),
                "long_return": mr.long_return,
            }
            for mr in result.monthly_returns
        ]
        metrics["monthly_returns"] = monthly_returns

        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Annualized Return: {result.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")

        return metrics

    finally:
        await store.close()


async def run_backtest_phase():
    """Phase 3: Run all 18 backtest configurations."""
    progress = load_progress()
    progress["phase"] = "backtest"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (at, ft, rf)
        for at in ANALYSIS_TYPES
        for ft in FILING_TYPES
        for rf in REBALANCE_FREQUENCIES
    ]

    total = len(configs)
    completed = 0

    for at, ft, rf in configs:
        key = config_key(at, ft, rf)

        # Skip already completed
        if key in progress["backtests"] and "error" not in progress["backtests"][key]:
            print(f"Skipping {key} (already completed)")
            completed += 1
            continue

        try:
            result = await run_single_backtest(at, ft, rf)
            progress["backtests"][key] = result

            # Save individual result
            result_file = RESULTS_DIR / f"{key}.json"
            result_file.write_text(json.dumps(result, indent=2, default=str))

            completed += 1
            print(f"\nProgress: {completed}/{total} backtests complete")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR in {key}: {e}", file=sys.stderr)
            print(tb, file=sys.stderr)
            progress["backtests"][key] = {
                "key": key,
                "error": str(e),
                "analysis_type": at,
                "filing_type": ft,
                "rebalance_freq": rf,
            }

        save_progress(progress)

    # Save summary
    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(progress["backtests"], indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"All {total} backtests complete!")
    print(f"Results saved to {RESULTS_DIR}/")
    save_progress(progress)


def generate_report():
    """Phase 4: Generate comprehensive comparison report."""
    progress = load_progress()
    results = progress.get("backtests", {})

    if not results:
        print("No backtest results found. Run 'backtest' phase first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect valid results
    valid = {k: v for k, v in results.items() if "error" not in v}
    failed = {k: v for k, v in results.items() if "error" in v}

    if not valid:
        print("No valid results to report on.")
        return

    # Sort by Sharpe ratio
    by_sharpe = sorted(valid.values(), key=lambda x: x.get("sharpe_ratio", 0), reverse=True)

    report_lines = []
    report_lines.append("# EDGAR Sentinel Backtest Comparison Study")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    report_lines.append(f"**Universe:** S&P 500 Historical (survivorship-bias controlled)")
    report_lines.append(f"**Period:** {START_DATE.isoformat()} to {END_DATE.isoformat()}")
    report_lines.append(f"**Configurations Tested:** {len(valid)} successful, {len(failed)} failed")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    best = by_sharpe[0]
    worst = by_sharpe[-1]
    report_lines.append(f"**Best Configuration:** {best['key']}")
    report_lines.append(f"  - Sharpe: {best['sharpe_ratio']:.3f}, Ann. Return: {best['annualized_return']:.2%}, Max DD: {best['max_drawdown']:.2%}")
    report_lines.append("")
    report_lines.append(f"**Worst Configuration:** {worst['key']}")
    report_lines.append(f"  - Sharpe: {worst['sharpe_ratio']:.3f}, Ann. Return: {worst['annualized_return']:.2%}, Max DD: {worst['max_drawdown']:.2%}")
    report_lines.append("")

    # Full ranking table
    report_lines.append("## Full Results Ranking (by Sharpe Ratio)")
    report_lines.append("")
    report_lines.append("| Rank | Configuration | Analysis | Filing | Rebal. | Ann. Return | Sharpe | Max DD | Win Rate | Periods |")
    report_lines.append("|------|--------------|----------|--------|--------|-------------|--------|--------|----------|---------|")

    for i, r in enumerate(by_sharpe, 1):
        report_lines.append(
            f"| {i} | {r['key']} | {r['analysis_type']} | {r['filing_type']} | "
            f"{r['rebalance_freq']} | {r['annualized_return']:.2%} | {r['sharpe_ratio']:.3f} | "
            f"{r['max_drawdown']:.2%} | {r['win_rate']:.1%} | {r['n_periods']} |"
        )

    report_lines.append("")

    # Analysis by dimension
    report_lines.append("## Analysis by Dimension")
    report_lines.append("")

    # By analysis type
    report_lines.append("### By Analysis Type")
    report_lines.append("")
    for at in ANALYSIS_TYPES:
        subset = [v for v in valid.values() if v["analysis_type"] == at]
        if subset:
            avg_sharpe = sum(v["sharpe_ratio"] for v in subset) / len(subset)
            avg_ret = sum(v["annualized_return"] for v in subset) / len(subset)
            avg_dd = sum(v["max_drawdown"] for v in subset) / len(subset)
            report_lines.append(f"**{at.title()}** (n={len(subset)})")
            report_lines.append(f"  - Avg Sharpe: {avg_sharpe:.3f}, Avg Ann. Return: {avg_ret:.2%}, Avg Max DD: {avg_dd:.2%}")
            report_lines.append("")

    # By filing type
    report_lines.append("### By Filing Type")
    report_lines.append("")
    for ft in FILING_TYPES:
        subset = [v for v in valid.values() if v["filing_type"] == ft]
        if subset:
            avg_sharpe = sum(v["sharpe_ratio"] for v in subset) / len(subset)
            avg_ret = sum(v["annualized_return"] for v in subset) / len(subset)
            avg_dd = sum(v["max_drawdown"] for v in subset) / len(subset)
            report_lines.append(f"**{ft}** (n={len(subset)})")
            report_lines.append(f"  - Avg Sharpe: {avg_sharpe:.3f}, Avg Ann. Return: {avg_ret:.2%}, Avg Max DD: {avg_dd:.2%}")
            report_lines.append("")

    # By rebalance frequency
    report_lines.append("### By Rebalance Frequency")
    report_lines.append("")
    for rf in REBALANCE_FREQUENCIES:
        subset = [v for v in valid.values() if v["rebalance_freq"] == rf]
        if subset:
            avg_sharpe = sum(v["sharpe_ratio"] for v in subset) / len(subset)
            avg_ret = sum(v["annualized_return"] for v in subset) / len(subset)
            avg_dd = sum(v["max_drawdown"] for v in subset) / len(subset)
            report_lines.append(f"**{rf.title()}** (n={len(subset)})")
            report_lines.append(f"  - Avg Sharpe: {avg_sharpe:.3f}, Avg Ann. Return: {avg_ret:.2%}, Avg Max DD: {avg_dd:.2%}")
            report_lines.append("")

    # Interaction effects
    report_lines.append("## Interaction Effects")
    report_lines.append("")
    report_lines.append("### Analysis Type × Filing Type (Avg Sharpe)")
    report_lines.append("")
    report_lines.append("| | 10-K | 10-Q | Both |")
    report_lines.append("|---|------|------|------|")
    for at in ANALYSIS_TYPES:
        row = f"| **{at.title()}** |"
        for ft in FILING_TYPES:
            subset = [v for v in valid.values() if v["analysis_type"] == at and v["filing_type"] == ft]
            if subset:
                avg = sum(v["sharpe_ratio"] for v in subset) / len(subset)
                row += f" {avg:.3f} |"
            else:
                row += " N/A |"
        report_lines.append(row)
    report_lines.append("")

    report_lines.append("### Analysis Type × Rebalance Freq (Avg Sharpe)")
    report_lines.append("")
    report_lines.append("| | Monthly | Quarterly |")
    report_lines.append("|---|---------|-----------|")
    for at in ANALYSIS_TYPES:
        row = f"| **{at.title()}** |"
        for rf in REBALANCE_FREQUENCIES:
            subset = [v for v in valid.values() if v["analysis_type"] == at and v["rebalance_freq"] == rf]
            if subset:
                avg = sum(v["sharpe_ratio"] for v in subset) / len(subset)
                row += f" {avg:.3f} |"
            else:
                row += " N/A |"
        report_lines.append(row)
    report_lines.append("")

    # Key findings
    report_lines.append("## Key Findings")
    report_lines.append("")

    # Find best in each category
    for at in ANALYSIS_TYPES:
        subset = [v for v in valid.values() if v["analysis_type"] == at]
        if subset:
            best_in = max(subset, key=lambda x: x["sharpe_ratio"])
            report_lines.append(f"- **Best {at.title()}:** {best_in['filing_type']}/{best_in['rebalance_freq']} "
                              f"(Sharpe: {best_in['sharpe_ratio']:.3f}, Return: {best_in['annualized_return']:.2%})")

    report_lines.append("")

    # Determine if composite > individual
    comp_sharpes = [v["sharpe_ratio"] for v in valid.values() if v["analysis_type"] == "composite"]
    sent_sharpes = [v["sharpe_ratio"] for v in valid.values() if v["analysis_type"] == "sentiment"]
    sim_sharpes = [v["sharpe_ratio"] for v in valid.values() if v["analysis_type"] == "similarity"]

    if comp_sharpes and sent_sharpes and sim_sharpes:
        avg_comp = sum(comp_sharpes) / len(comp_sharpes)
        avg_sent = sum(sent_sharpes) / len(sent_sharpes)
        avg_sim = sum(sim_sharpes) / len(sim_sharpes)

        if avg_comp > max(avg_sent, avg_sim):
            report_lines.append("- **Composite signals outperform individual signals** on average, suggesting "
                              "complementary information between sentiment and similarity analysis.")
        else:
            best_type = "sentiment" if avg_sent > avg_sim else "similarity"
            report_lines.append(f"- **{best_type.title()} analysis outperforms composite**, suggesting "
                              f"that the other analysis type may add noise rather than signal.")

    report_lines.append("")

    # Failed configurations
    if failed:
        report_lines.append("## Failed Configurations")
        report_lines.append("")
        for k, v in failed.items():
            report_lines.append(f"- **{k}**: {v.get('error', 'Unknown error')}")
        report_lines.append("")

    # Methodology
    report_lines.append("## Methodology")
    report_lines.append("")
    report_lines.append("- **Universe:** S&P 500 historical constituents with survivorship-bias control")
    report_lines.append("- **Period:** 2020-01-01 to present")
    report_lines.append("- **Signal buffer:** 2 business days after filing date")
    report_lines.append("- **Signal decay:** Exponential, 90-day half-life")
    report_lines.append("- **Portfolio construction:** Top quintile (Q1) long-only, equal-weighted within quantile")
    report_lines.append("- **Transaction costs:** 10 bps per rebalance")
    report_lines.append("- **Quantiles:** 5 (quintile sort)")
    report_lines.append("")
    report_lines.append("### Analysis Types")
    report_lines.append("- **Similarity:** Measures textual change between consecutive filings (same ticker, same form type). Higher change = stronger signal.")
    report_lines.append("- **Sentiment:** Dictionary-based sentiment scoring using Loughran-McDonald financial dictionary.")
    report_lines.append("- **Composite:** Equal-weighted combination of similarity and sentiment signals.")
    report_lines.append("")
    report_lines.append("### Filing Types")
    report_lines.append("- **10-K:** Annual reports only. More comprehensive but lower frequency (1/year).")
    report_lines.append("- **10-Q:** Quarterly reports only. Higher frequency (3/year) but less comprehensive.")
    report_lines.append("- **Both:** Combines 10-K and 10-Q signals for maximum data coverage.")
    report_lines.append("")

    report = "\n".join(report_lines)
    report_path = RESULTS_DIR / "backtest-comparison-report.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")

    # Also save as the main output document
    output_report = Path("/output/backtest-comparison-report.md")
    output_report.write_text(report)
    print(f"Report also at {output_report}")


def show_status():
    """Show current progress."""
    progress = load_progress()
    print(f"Phase: {progress['phase']}")
    print(f"Started: {progress.get('started_at', 'N/A')}")
    print(f"Last updated: {progress.get('last_updated', 'N/A')}")

    ing = progress["ingestion"]
    done = len(ing["completed_tickers"])
    total = ing["total_tickers"]
    pct = done / total * 100 if total > 0 else 0
    print(f"\nIngestion: {done}/{total} tickers ({pct:.1f}%)")
    print(f"  Filings ingested: {ing['filings_ingested']}")
    print(f"  Failed tickers: {len(ing['failed_tickers'])}")

    ana = progress["analysis"]
    print(f"\nAnalysis: {'Complete' if ana['completed'] else 'Pending'}")
    print(f"  Sentiment results: {ana['sentiment_count']}")
    print(f"  Similarity results: {ana['similarity_count']}")

    bt = progress["backtests"]
    completed = sum(1 for v in bt.values() if "error" not in v)
    failed = sum(1 for v in bt.values() if "error" in v)
    print(f"\nBacktests: {completed}/18 complete, {failed} failed")

    if bt:
        print("\nResults:")
        for key, v in sorted(bt.items()):
            if "error" in v:
                print(f"  {key}: FAILED - {v['error']}")
            else:
                print(f"  {key}: Sharpe={v['sharpe_ratio']:.3f}, Return={v['annualized_return']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="EDGAR Sentinel Batch Backtest Runner")
    parser.add_argument("phase", choices=["ingest", "analyze", "backtest", "report", "status"],
                       help="Which phase to run")
    parser.add_argument("--chunk-size", type=int, default=50,
                       help="Number of tickers to ingest per invocation (default: 50)")
    args = parser.parse_args()

    if args.phase == "status":
        show_status()
    elif args.phase == "ingest":
        asyncio.run(run_ingest_phase(args.chunk_size))
    elif args.phase == "analyze":
        asyncio.run(run_analyze_phase())
    elif args.phase == "backtest":
        asyncio.run(run_backtest_phase())
    elif args.phase == "report":
        generate_report()


if __name__ == "__main__":
    main()
