"""Command-line entry point: load data, train the LSTM, backtest, save a report.

Examples
--------
    python train.py --ticker AAPL --start 2015-01-01
    python train.py --csv data/my_prices.csv --epochs 40 --threshold-mode tuned
    python train.py --synthetic            # offline demo on ARTIFICIAL data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import PipelineConfig
from src.data_collection import DataError, load_data
from src.eda import run_eda
from src.model import TrainingError
from src.pipeline import run_pipeline
from src.synthetic import generate_synthetic_ohlcv


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and backtest the deep learning trading agent.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ticker", default="AAPL", help="Yahoo Finance symbol (default AAPL)")
    src.add_argument("--csv", help="Path to an OHLCV CSV instead of downloading")
    src.add_argument("--synthetic", action="store_true", help="Use artificial demo data (offline)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--threshold-mode", choices=["absolute", "quantile", "tuned"], default="quantile")
    p.add_argument("--stop-loss", type=float, default=None, help="e.g. 0.08 for an 8%% stop")
    p.add_argument("--out", default="reports/latest", help="Directory for the model and report files")
    p.add_argument("--no-eda", action="store_true", help="Skip saving EDA figures")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = PipelineConfig()
    cfg.data.ticker, cfg.data.start, cfg.data.end, cfg.data.interval = args.ticker, args.start, args.end, args.interval
    m = cfg.model
    m.lookback, m.epochs, m.batch_size, m.learning_rate = args.lookback, args.epochs, args.batch_size, args.lr
    m.hidden_size, m.seed = args.hidden_size, args.seed
    s = cfg.strategy
    s.initial_capital, s.fee_rate, s.slippage_bps = args.capital, args.fee_rate, args.slippage_bps
    s.threshold_mode, s.stop_loss_pct = args.threshold_mode, args.stop_loss

    try:
        if args.synthetic:
            print("NOTE: using ARTIFICIAL synthetic data; results say nothing about real markets.")
            df = generate_synthetic_ohlcv()
        else:
            df, report = load_data(
                ticker=args.ticker, csv=args.csv, start=args.start, end=args.end,
                interval=args.interval, cache_dir=cfg.data.cache_dir, min_rows=cfg.data.min_rows,
            )
            print(f"Loaded {report['rows_out']} rows ({report['first_date']} -> {report['last_date']}) from {report['source']}.")

        out = Path(args.out)
        if not args.no_eda:
            run_eda(df, out / "eda", cfg.data.periods_per_year)

        def progress(epoch, total, tr, va):
            print(f"  epoch {epoch:3d}/{total}  train {tr:.4f}  val {va:.4f}")

        result = run_pipeline(df, cfg, save_dir=out, progress=progress)
    except (DataError, TrainingError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nChronological splits:\n", result.split_summary.to_string())
    print("\nTest-set model comparison:\n", result.comparison.round(5).to_string())
    print("\n" + result.verdict)
    print(f"\nThresholds ({result.threshold_info['method']}): "
          f"buy > {result.threshold_info['buy_threshold']:.5f}, sell < {result.threshold_info['sell_threshold']:.5f}")
    print("\nBacktest (test period only):\n", result.backtest.summary_table().to_string())
    print(f"\nSaved model + report to {out.resolve()}")
    print("Educational research only - not financial advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
