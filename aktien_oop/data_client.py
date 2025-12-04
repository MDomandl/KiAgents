from typing import Optional
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

        # --- As-of Fenster bestimmen (wenn gesetzt) ---
        as_of = getattr(self.cfg, "as_of", None)
        if as_of:
            as_of_ts = pd.Timestamp(as_of).normalize()
            lookback = int(getattr(self.cfg, "max_lookback_days", 360))
            start_s = (as_of_ts - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
            end_s = as_of_ts.strftime("%Y-%m-%d")

            # 1) normalisierter Ticker
            df = yf.download(t, start=start_s, end=end_s, interval="1d",
                             progress=False, auto_adjust=self.cfg.adjusted, threads=False)
        else:
            # 1) normalisierter Ticker (period-basiert)
            df = yf.download(t, period=self.cfg.period, interval="1d",
                             progress=False, auto_adjust=self.cfg.adjusted, threads=False)

        df = self._ensure_ohlc(df, ticker)
        if df is not None:
            return df

        # Fallback: Roh-Ticker (z. B. wenn Normalisierung fehlschlägt)
        if as_of:
            df = yf.download(ticker, start=start_s, end=end_s, interval="1d",
                             progress=False, auto_adjust=self.cfg.adjusted, threads=False)
        else:
            df = yf.download(ticker, period=self.cfg.period, interval="1d",
                             progress=False, auto_adjust=self.cfg.adjusted, threads=False)

        return self._ensure_ohlc(df, ticker)

    def sp500_above_200dma(self) -> bool:
        kw = dict(interval="1d", progress=False, auto_adjust=self.cfg.adjusted, threads=False)

        if getattr(self.cfg, "as_of", None):
            as_of = pd.to_datetime(self.cfg.as_of).tz_localize(None)
            end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            start = (as_of - pd.Timedelta(days=max(self.cfg.max_lookback_days, 300))).strftime("%Y-%m-%d")
            df = yf.download("^GSPC", start=start, end=end, **kw)
        else:
            # mind. 250d für 200DMA; nimm notfalls self.cfg.period wenn größer
            per = self.cfg.period if int(self.cfg.period.rstrip("d")) >= 250 else "250d"
            df = yf.download("^GSPC", period=per, **kw)

        if df is None or df.empty or "Close" not in df.columns:
            logging.warning("S&P 500: keine Daten erhalten.");
            return False
        close = as_series(df["Close"]).dropna()
        if len(close) < 200:
            logging.warning("S&P 500: zu wenige Close-Werte für 200DMA.");
            return False
        sma200 = close.rolling(200).mean().dropna()
        last_close, last_sma = float(close.iloc[-1]), float(sma200.iloc[-1])
        logging.info(f"S&P 500 → Close: {last_close:.2f} | 200DMA: {last_sma:.2f} | Markt "
                     f"{'über' if last_close > last_sma else 'unter'} 200DMA")
        return last_close > last_sma

    def _period_to_days(period: str) -> int:
        """Akzeptiert '800d', '36m', '5y' etc. und liefert Tage (grobe Umrechnung)."""
        if period is None:
            return 800
        p = str(period).strip().lower()
        try:
            if p.endswith("d"):
                return int(p[:-1])
            if p.endswith("m"):
                return int(p[:-1]) * 30
            if p.endswith("y"):
                return int(p[:-1]) * 365
            # nackte Zahl → Tage
            return int(p)
        except Exception:
            return 800

    def get_prices(self, universe, as_of, period, adjusted=True) -> pd.DataFrame:
        def _period_to_days(p: str) -> int:
            if p is None: return 800
            s = str(p).strip().lower()
            try:
                if s.endswith("d"): return int(s[:-1])
                if s.endswith("m"): return int(s[:-1]) * 30
                if s.endswith("y"): return int(s[:-1]) * 365
                return int(s)
            except Exception:
                return 800

        cutoff = pd.Timestamp(as_of).tz_localize(None) if as_of else None
        lookback_days = _period_to_days(period)

        series_list = []
        for t in universe:
            df = self.download_ohlc(t)
            if df is None or df.empty: continue
            idx = pd.to_datetime(df.index).tz_localize(None)
            df = df.set_index(idx)
            if adjusted and "Adj Close" in df.columns:
                col = "Adj Close"
            elif "Close" in df.columns:
                col = "Close"
            else:
                continue
            s = df[col].rename(t)
            if cutoff is not None:
                start = cutoff - pd.Timedelta(days=lookback_days)
                s = s.loc[(s.index >= start) & (s.index <= cutoff)]
            if s.empty: continue
            series_list.append(s)

        if not series_list:
            return pd.DataFrame()

        mat = pd.concat(series_list, axis=1).sort_index()
        mat = mat.dropna(how="all")
        return mat

    def load_prices(self, universe, as_of, period, adjusted=True):
        return self.get_prices(universe, as_of, period, adjusted)