import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import Config, normalize_ticker
from .utils import as_series


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
            df["Close"] = df["Adj Close"]
            cols = set(df.columns)
        if "High" not in cols and "Close" in cols:
            df["High"] = df["Close"]
        if "Low" not in cols and "Close" in cols:
            df["Low"] = df["Close"]
        if not {"Close", "High", "Low"}.issubset(df.columns):
            return None
        df.attrs["_ticker"] = ticker
        return df

    # ---------- FIX 1: period parsing (static) ----------
    @staticmethod
    def _period_to_days(period: str | None) -> int:
        """Akzeptiert '800d', '36m', '5y' etc."""
        if period is None:
            return 800
        s = str(period).strip().lower()
        try:
            if s.endswith("d"):
                return int(s[:-1])
            if s.endswith("m"):
                return int(s[:-1]) * 30
            if s.endswith("y"):
                return int(s[:-1]) * 365
            return int(s)
        except Exception:
            return 800

    # ---------- FIX 2: OHLC loader uses as_of/period/adjusted ----------
    def download_ohlc(
        self,
        ticker: str,
        *,
        as_of: str | pd.Timestamp | None = None,
        period: str | None = None,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Lädt OHLC-Daten für EINEN Ticker.

        Wichtig:
        - Wenn as_of gesetzt ist, ist yfinance 'end' EXKLUSIV -> end = as_of + 1 Tag,
          damit der as_of-Handelstag sicher enthalten ist.
        - Lookback wird aus 'period' abgeleitet (nicht aus cfg.max_lookback_days),
          damit Runner und BT identische Fenster benutzen können.
        """
        t_norm = normalize_ticker(ticker)
        period = period or getattr(self.cfg, "period", "800d")
        adjusted = bool(adjusted)

        kw = dict(interval="1d", progress=False, auto_adjust=adjusted, threads=False)

        if as_of is not None:
            cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
            days = self._period_to_days(str(period))
            start_s = (cutoff - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            end_s = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end exklusiv!
            df = yf.download(t_norm, start=start_s, end=end_s, **kw)
        else:
            df = yf.download(t_norm, period=str(period), **kw)

        df = self._ensure_ohlc(df, ticker)
        if df is not None:
            return df

        # Fallback: Roh-Ticker
        if as_of is not None:
            df = yf.download(ticker, start=start_s, end=end_s, **kw)
        else:
            df = yf.download(ticker, period=str(period), **kw)

        return self._ensure_ohlc(df, ticker)

    def get_prices(self, universe, as_of, period, adjusted=True) -> pd.DataFrame:
        """
        Baut eine Close-Matrix (Index=Date, Spalten=Ticker) für genau das Fenster:
        [as_of - period, as_of] (inklusive).
        """
        cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of else None
        lookback_days = self._period_to_days(str(period))
        start = (cutoff - pd.Timedelta(days=lookback_days)) if cutoff is not None else None

        series_list = []
        for t in universe:
            df = self.download_ohlc(t, as_of=cutoff, period=period, adjusted=adjusted)
            if df is None or df.empty:
                continue

            idx = pd.to_datetime(df.index).tz_localize(None)
            df = df.set_index(idx).sort_index()

            # Nach _ensure_ohlc existiert "Close" sicher
            s = df["Close"].astype(float).rename(t)

            if cutoff is not None:
                s = s.loc[(s.index >= start) & (s.index <= cutoff)]

            if not s.empty:
                series_list.append(s)

        if not series_list:
            return pd.DataFrame()

        mat = pd.concat(series_list, axis=1).sort_index()
        mat = mat.dropna(how="all")
        return mat

    # ---------- FIX 3: Series/Scalar robust (keine ambige truth values) ----------
    def sp500_above_200dma(self) -> bool:
        """
        True, wenn S&P500 (ˆGSPC) über seinem 200DMA liegt – as_of-korrekt.
        Robust gegen yfinance MultiIndex/Series-Probleme.
        """
        kw = dict(
            interval="1d",
            progress=False,
            auto_adjust=bool(getattr(self.cfg, "adjusted", True)),
            threads=False,
        )

        cfg_as_of = getattr(self.cfg, "as_of", None)
        cfg_period = getattr(self.cfg, "period", "800d")

        if cfg_as_of:
            cutoff = pd.to_datetime(cfg_as_of).tz_localize(None).normalize()

            # Fenster: aus period ableiten (Lockstep-Idee), aber mind. ~300d,
            # damit 200DMA sicher berechenbar bleibt.
            days = max(self._period_to_days(cfg_period), 300)

            start = (cutoff - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            end = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance 'end' ist exklusiv
            df = yf.download("^GSPC", start=start, end=end, **kw)
        else:
            # ohne as_of: period-basiert, aber mindestens 300d
            days = max(self._period_to_days(cfg_period), 300)
            df = yf.download("^GSPC", period=f"{days}d", **kw)

        if df is None or df.empty or "Close" not in df.columns:
            logging.warning("S&P 500: keine Daten erhalten.")
            return False

        close = df["Close"]
        # yfinance kann hier DataFrame (MultiIndex) oder Series liefern → auf Series normieren
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < 200:
            logging.warning("S&P 500: zu wenige Close-Werte für 200DMA.")
            return False

        sma200 = close.rolling(200, min_periods=200).mean().dropna()
        if sma200.empty:
            logging.warning("S&P 500: 200DMA nicht berechenbar (empty).")
            return False

        last_close = float(close.iloc[-1])
        last_sma = float(sma200.iloc[-1])

        logging.info(
            f"S&P 500 → Close: {last_close:.2f} | 200DMA: {last_sma:.2f} | Markt "
            f"{'über' if last_close > last_sma else 'unter'} 200DMA"
        )
        return last_close > last_sma

    def load_prices(self, universe, as_of, period, adjusted=True):
        return self.get_prices(universe, as_of, period, adjusted)