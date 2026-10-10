from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

sys.modules.setdefault("yfinance", types.SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame()))
from aktien_oop import backtest
from aktien_oop.backtest import BTConfig, Backtester


def test_backtest_writes_empty_position_snapshot_with_stable_schema(tmp_path, monkeypatch):
    tickers_path = tmp_path / "tickers.txt"
    meta_path = tmp_path / "meta.csv"
    save_dir = tmp_path / "out"
    tickers_path.write_text("AAA\n", encoding="utf-8")
    meta_path.write_text("ticker,sector\nAAA,Tech\n", encoding="utf-8")

    dates = pd.bdate_range("2024-01-01", periods=320)

    def fake_download_close(tickers, *args, **kwargs):
        requested = [tickers] if isinstance(tickers, str) else list(tickers)
        return pd.DataFrame(
            {ticker: np.linspace(100.0, 120.0, len(dates)) for ticker in requested},
            index=dates,
        )

    monkeypatch.setattr(backtest, "download_close", fake_download_close)
    monkeypatch.setattr(
        backtest,
        "calculate_portfolio",
        lambda *args, **kwargs: ({}, pd.Series(dtype=float)),
    )

    cfg = BTConfig(
        tickers_file=str(tickers_path),
        sector_meta=str(meta_path),
        save_dir=str(save_dir),
        as_of="2025-03-21",
        start="2024-01-01",
        end="2025-03-21",
        frequency="monthly",
        top_k=1,
        buffer_k=1,
        use_sector_limits=False,
        max_per_sector=1,
        cost_bps=0.0,
        slippage_bps=0.0,
        min_history_days=1,
        benchmark="SPY",
    )

    Backtester(cfg).run()

    positions = pd.read_csv(save_dir / "bt_monthly_1x1_positions.csv", comment="#")

    assert positions.empty
    assert positions.columns.tolist() == [
        "as_of",
        "ticker",
        "weight",
        "score",
        "rank",
        "sector",
    ]
