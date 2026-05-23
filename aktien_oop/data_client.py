import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from dataclasses import dataclass

from .config import Config, normalize_ticker
from typing import Any
from .utils import as_series
from .market_data_cache import MarketDataRequest, get_default_market_data_cache

@dataclass(frozen=True)
class RegimeDecision:
    ok: bool
    reason: str | None = None

class DataClient:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._market_cache = get_default_market_data_cache()

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
    def _download_ohlc_uncached(
        self,
        ticker: str,
        *,
        as_of: str | pd.Timestamp | None = None,
        period: str | None = None,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        t_norm = normalize_ticker(ticker)
        period = period or getattr(self.cfg, "period", "800d")
        adjusted = bool(adjusted)

        kw = dict(interval="1d", progress=False, auto_adjust=adjusted, threads=False)

        if as_of is not None:
            cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
            days = self._period_to_days(str(period))
            start_s = (cutoff - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            end_s = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(t_norm, start=start_s, end=end_s, **kw)
        else:
            start_s = None
            end_s = None
            df = yf.download(t_norm, period=str(period), **kw)

        df = self._ensure_ohlc(df, ticker)
        if df is not None:
            return df

        if as_of is not None:
            df = yf.download(ticker, start=start_s, end=end_s, **kw)
        else:
            df = yf.download(ticker, period=str(period), **kw)

        return self._ensure_ohlc(df, ticker)

    def download_ohlc(
        self,
        ticker: str,
        *,
        as_of: str | pd.Timestamp | None = None,
        period: str | None = None,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        period = period or getattr(self.cfg, "period", "800d")
        cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else None
        start_s = None
        end_s = None
        if cutoff is not None:
            days = self._period_to_days(str(period))
            start_s = (cutoff - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            end_s = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        request = MarketDataRequest.build(
            data_kind="ohlc",
            tickers=[ticker],
            start=start_s,
            end=end_s,
            period=period if cutoff is None else None,
            as_of=cutoff,
            adjusted=adjusted,
        )
        frame = self._market_cache.get_or_load_frame(
            request,
            lambda: self._download_ohlc_uncached(
                ticker,
                as_of=cutoff,
                period=period,
                adjusted=adjusted,
            ),
        )
        return self._ensure_ohlc(frame, ticker) if frame is not None else None

    def _get_prices_uncached(self, universe, as_of, period, adjusted=True) -> pd.DataFrame:
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

    def get_prices(self, universe, as_of, period, adjusted=True) -> pd.DataFrame:
        cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of else None
        start_s = None
        end_s = None
        if cutoff is not None:
            lookback_days = self._period_to_days(str(period))
            start_s = (cutoff - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end_s = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        request = MarketDataRequest.build(
            data_kind="close",
            tickers=universe,
            start=start_s,
            end=end_s,
            period=period if cutoff is None else None,
            as_of=cutoff,
            adjusted=adjusted,
        )
        frame = self._market_cache.get_or_load_frame(
            request,
            lambda: self._get_prices_uncached(universe, cutoff, period, adjusted),
        )
        return frame if frame is not None else pd.DataFrame()

    # ---------- FIX 3: Series/Scalar robust (keine ambige truth values) ----------
    def sp500_above_200dma(self, as_of: str, sma_days: int = 200, period: str = "800d") -> bool:
        as_of_ts = pd.Timestamp(as_of).normalize()

        days = self._period_to_days(str(period))
        start_s = (as_of_ts - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        end_s = (as_of_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        request = MarketDataRequest.build(
            data_kind="benchmark",
            tickers=["^GSPC"],
            start=start_s,
            end=end_s,
            as_of=as_of_ts,
            adjusted=False,
            benchmark_symbol="^GSPC",
        )
        px = self._market_cache.get_or_load_frame(
            request,
            lambda: yf.download("^GSPC", start=start_s, end=end_s, auto_adjust=False, progress=False),
        )
        if px is None or getattr(px, "empty", True):
            return True  # defensiv

        # ---- Close als 1D-Series extrahieren (robust) ----
        close = None

        # Fall A: normale Spalten ("Open", "High", ..., "Close")
        if isinstance(px, pd.DataFrame) and "Close" in px.columns:
            close = px["Close"]

        # Fall B: MultiIndex-Spalten (z.B. ("Close", "^GSPC"))
        if close is None and isinstance(px, pd.DataFrame) and isinstance(px.columns, pd.MultiIndex):
            if ("Close", "^GSPC") in px.columns:
                close = px[("Close", "^GSPC")]
            else:
                # fallback: nimm alle "Close"-Spalten und squeeze auf eine
                try:
                    close = px.xs("Close", axis=1, level=0)
                except Exception:
                    close = None

        if close is None:
            return True

        # close kann jetzt Series ODER DataFrame sein -> auf Series bringen
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return True
            close = close.iloc[:, 0]

        close = close.dropna()
        close = close[close.index <= as_of_ts]
        if close.empty:
            return True

        sma = close.rolling(sma_days).mean()

        # ---- Scalars erzwingen ----
        last_close = close.iloc[-1]
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        last_close = float(last_close)

        last_sma = sma.iloc[-1]
        if isinstance(last_sma, pd.Series):
            last_sma = last_sma.iloc[0]

        if pd.isna(last_sma):
            return True

        last_sma = float(last_sma)

        return last_close > last_sma

    # NOTE:
    # This is the single authoritative regime decision logic.
    # Do NOT duplicate this elsewhere (BT / utils / runner).
    def regime_decision(self, cfg, as_of: str) -> dict:
        require = bool(getattr(cfg, "require_above_sma", False))

        # Action immer normieren (Default HOLD), damit Caller immer ein Feld hat

        action_below = str(getattr(cfg, "regime_below_action", "HOLD") or "HOLD").upper()
        if action_below not in ("HOLD", "SELL"):
            action_below = "HOLD"

        # Wenn Regime nicht aktiv: immer proceed
        if not require:
            return {"ok": True, "reason": "", "action": "PROCEED", "below_action": action_below}

        sma_days = int(getattr(cfg, "regime_sma_days", 200) or 200)
        period = str(getattr(cfg, "period", "800d") or "800d")

        ok = self.sp500_above_200dma(as_of=as_of, sma_days=sma_days, period=period)
        if ok:
            # Hier explizit action setzen, damit Caller nicht raten muss
            return {"ok": True, "reason": "", "action": "PROCEED", "below_action": action_below}

        # Regime ist aktiv und nicht ok -> below_action anwenden
        return {"ok": False, "reason": "sp500_below_200dma", "action": action_below, "below_action": action_below}

    def load_prices(self, universe, as_of, period, adjusted=True):
        return self.get_prices(universe, as_of, period, adjusted)