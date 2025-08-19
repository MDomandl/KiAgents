from typing import Optional
from pathlib import Path
import yfinance as yf
import pandas as pd
from .config import Config, normalize_ticker
from .utils import as_series
import logging

class DataClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _ensure_ohlc(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)
        cols = set(df.columns)
        if "Close" not in cols and "Adj Close" in cols:
            df["Close"] = df["Adj Close"]; cols = set(df.columns)
        if "High" not in cols and "Close" in cols:
            df["High"] = df["Close"]
        if "Low" not in cols and "Close" in cols:
            df["Low"] = df["Close"]
        if not {"Close","High","Low"}.issubset(df.columns):
            return None
        df.attrs["_ticker"] = ticker
        return df

    def download_ohlc(self, ticker: str) -> Optional[pd.DataFrame]:
        t = normalize_ticker(ticker)
        df = yf.download(t, period=self.cfg.period, interval="1d",
                         progress=False, auto_adjust=self.cfg.adjusted, threads=False)
        df = self._ensure_ohlc(df, ticker)
        if df is not None: return df
        df = yf.download(ticker, period=self.cfg.period, interval="1d",
                         progress=False, auto_adjust=True, threads=False)
        return self._ensure_ohlc(df, ticker)

    def sp500_above_200dma(self) -> bool:
        df = yf.download("^GSPC", period="250d", interval="1d",
                         progress=False, auto_adjust=self.cfg.adjusted, threads=False)
        if df is None or df.empty or "Close" not in df.columns:
            logging.warning("S&P 500: keine Daten erhalten."); return False
        close = as_series(df["Close"]).dropna()
        if len(close) < 200:
            logging.warning("S&P 500: zu wenige Close-Werte für 200DMA."); return False
        sma200 = close.rolling(200).mean().dropna()
        last_close, last_sma = float(close.iloc[-1]), float(sma200.iloc[-1])
        logging.info(f"S&P 500 → Close: {last_close:.2f} | 200DMA: {last_sma:.2f} | Markt "
                     f"{'über' if last_close>last_sma else 'unter'} 200DMA")
        return last_close > last_sma
