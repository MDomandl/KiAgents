from __future__ import annotations

from argparse import Namespace

import pandas as pd

from aktien_oop.backtest import download_close
from aktien_oop.data_client import DataClient
from aktien_oop.market_data_cache import MarketDataCache, MarketDataRequest, build_cache_key


def _close_frame(columns=("AAA", "BBB")) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=5, freq="D")
    return pd.DataFrame({col: range(10, 15) for col in columns}, index=idx, dtype=float)


def _ohlc_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [9, 10, 11, 12, 13],
            "High": [11, 12, 13, 14, 15],
            "Low": [8, 9, 10, 11, 12],
            "Close": [10, 11, 12, 13, 14],
        },
        index=idx,
        dtype=float,
    )


def test_cache_key_is_stable_for_ticker_order():
    left = MarketDataRequest.build(
        data_kind="close",
        tickers=["BBB", "AAA"],
        start="2025-01-01",
        end="2025-01-10",
        adjusted=True,
    )
    right = MarketDataRequest.build(
        data_kind="close",
        tickers=["AAA", "BBB"],
        start="2025-01-01",
        end="2025-01-10",
        adjusted=True,
    )

    assert build_cache_key(left) == build_cache_key(right)


def test_cache_miss_loads_and_stores(tmp_path):
    cache = MarketDataCache(tmp_path)
    request = MarketDataRequest.build(
        data_kind="close",
        tickers=["AAA", "BBB"],
        start="2025-01-01",
        end="2025-01-10",
        adjusted=True,
    )
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return _close_frame()

    result = cache.get_or_load_frame(request, loader)

    assert calls["count"] == 1
    assert cache.cache_path(request).exists()
    pd.testing.assert_frame_equal(result, _close_frame())


def test_cache_hit_reads_without_download(tmp_path):
    cache = MarketDataCache(tmp_path)
    request = MarketDataRequest.build(
        data_kind="close",
        tickers=["AAA", "BBB"],
        start="2025-01-01",
        end="2025-01-10",
        adjusted=True,
    )
    cache.get_or_load_frame(request, _close_frame)

    def fail_loader():
        raise AssertionError("download should not be called on cache hit")

    result = cache.get_or_load_frame(request, fail_loader)

    pd.testing.assert_frame_equal(result, _close_frame())


def test_incomplete_cache_is_discarded_and_reloaded(tmp_path):
    cache = MarketDataCache(tmp_path)
    request = MarketDataRequest.build(
        data_kind="close",
        tickers=["AAA", "BBB"],
        start="2025-01-01",
        end="2025-01-10",
        adjusted=True,
    )

    incomplete = cache.get_or_load_frame(request, lambda: _close_frame(columns=("AAA",)))
    assert list(incomplete.columns) == ["AAA"]
    assert not cache.cache_path(request).exists()

    result = cache.get_or_load_frame(request, _close_frame)

    assert cache.cache_path(request).exists()
    pd.testing.assert_frame_equal(result, _close_frame())


def test_backtest_download_close_cache_preserves_matrix_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("AKTIEN_OOP_MARKET_CACHE_DIR", str(tmp_path))
    calls = {"count": 0}

    def fake_download(ticker, **kwargs):
        calls["count"] += 1
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=pd.date_range("2025-01-02", periods=3))

    monkeypatch.setattr("aktien_oop.backtest.yf.download", fake_download)

    first = download_close(["AAA", "BBB"], "2025-01-01", "2025-01-08")
    second = download_close(["BBB", "AAA"], "2025-01-01", "2025-01-08")

    assert calls["count"] == 2
    assert list(first.columns) == ["AAA", "BBB"]
    assert list(second.columns) == ["AAA", "BBB"]
    pd.testing.assert_frame_equal(first, second)


def test_data_client_get_prices_cache_preserves_runner_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("AKTIEN_OOP_MARKET_CACHE_DIR", str(tmp_path))
    client = DataClient(Namespace(period="10d"))
    calls = {"count": 0}

    def fake_download(ticker, **kwargs):
        calls["count"] += 1
        return _ohlc_frame()

    monkeypatch.setattr("aktien_oop.data_client.yf.download", fake_download)

    first = client.get_prices(["AAA", "BBB"], "2025-01-05", "10d", True)
    second = client.get_prices(["BBB", "AAA"], "2025-01-05", "10d", True)

    assert calls["count"] == 2
    assert list(first.columns) == ["AAA", "BBB"]
    assert list(second.columns) == ["AAA", "BBB"]
    pd.testing.assert_frame_equal(first, second)
