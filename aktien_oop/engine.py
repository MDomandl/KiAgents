from typing import Optional, Dict
import numpy as np
import pandas as pd
from .config import Config
from .data_client import DataClient
from .indicators import Indicators
from .models import TickerSignal

class SignalEngine:
    def __init__(self, cfg: Config, data: DataClient):
        self.cfg, self.data = cfg, data
        self.fail_counts: Dict[str, int] = {}

    def _fail(self, key: str): self.fail_counts[key] = self.fail_counts.get(key, 0) + 1

    def compute_for_ticker(self, ticker: str) -> Optional[TickerSignal]:
        df_full = self.data.download_ohlc(ticker)
        if df_full is None or df_full.empty: self._fail("no_data"); return None

        adv = Indicators.avg_dollar_volume(df_full, 20)
        if adv is not None and adv < self.cfg.adv_min_dollars: self._fail("illiquid"); return None

        m121 = Indicators.mom_12_1(df_full["Close"])
        if m121 is None: self._fail("mom121_nan")

        df = df_full.dropna(subset=["Close","High","Low"]).tail(self.cfg.days_win)
        if len(df) < self.cfg.days_win: self._fail("too_few_days"); return None

        close = df["Close"].dropna()
        sma100 = close.rolling(100).mean().dropna()
        if sma100.empty or not (float(close.iloc[-1]) > float(sma100.iloc[-1])): self._fail("under_sma"); return None

        series_for_gap = df["Adj Close"] if "Adj Close" in df.columns else close
        ok_gap, _ = Indicators.no_big_gap(series_for_gap, self.cfg.gap_th)
        if not ok_gap: self._fail("gap"); return None

        slope, r2, score_lin = Indicators.linear_trend_log(close)
        atr = Indicators.atr14(df)
        if atr is None: self._fail("atr_nan"); return None

        last_close = float(close.iloc[-1])
        stop_loss_pct = (3 * atr / last_close) * 100 if last_close else None
        vol = Indicators.annual_vol(close)

        return TickerSignal(
            ticker=ticker, score_lin=round(score_lin,6), mom_12_1=(np.nan if m121 is None else round(m121,4)),
            slope=round(slope,6), r2=round(r2,4),
            volatility=(None if vol is None else round(vol,4)),
            stop_loss_pct=(None if stop_loss_pct is None else round(stop_loss_pct,2)),
        )
