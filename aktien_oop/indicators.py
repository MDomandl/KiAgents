from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import as_series

class Indicators:
    @staticmethod
    def avg_dollar_volume(df: pd.DataFrame, win: int = 20) -> Optional[float]:
        if "Close" not in df.columns or "Volume" not in df.columns: return None
        ser = (as_series(df["Close"]) * as_series(df["Volume"])).rolling(win).mean().dropna()
        return None if ser.empty else float(ser.iloc[-1])

    @staticmethod
    def mom_12_1(close: pd.Series) -> Optional[float]:
        c = as_series(close).dropna()
        if len(c) < 252: return None
        start, end = float(c.iloc[-252]), float(c.iloc[-21])
        if start == 0: return None
        return end / start - 1.0

    @staticmethod
    def no_big_gap(series: pd.Series, threshold: float) -> Tuple[bool, Optional[float]]:
        s = as_series(series).dropna()
        if s.empty: return False, None
        gaps = s.pct_change().abs().dropna()
        if gaps.empty: return True, 0.0
        mg = float(gaps.max())
        return (mg < threshold), mg

    @staticmethod
    def linear_trend_log(close: pd.Series):
        s = as_series(close).dropna()
        df = pd.DataFrame({"Close": s}).copy()
        df["LogPrice"], df["Day"] = np.log(df["Close"]), np.arange(len(df), dtype=float)
        X, Y = df["Day"].values.reshape(-1,1), df["LogPrice"].values.reshape(-1,1)
        reg = LinearRegression().fit(X, Y)
        slope, r2 = float(reg.coef_[0][0]), float(reg.score(X, Y))
        return slope, r2, slope * r2

    @staticmethod
    def atr14(df: pd.DataFrame) -> Optional[float]:
        w = df.copy()
        w["H-L"]  = w["High"] - w["Low"]
        w["H-PC"] = (w["High"] - w["Close"].shift(1)).abs()
        w["L-PC"] = (w["Low"]  - w["Close"].shift(1)).abs()
        w["TR"]   = w[["H-L","H-PC","L-PC"]].max(axis=1)
        atr = w["TR"].rolling(14).mean().dropna()
        return None if atr.empty else float(atr.iloc[-1])

    @staticmethod
    def annual_vol(close: pd.Series) -> Optional[float]:
        s = as_series(close).dropna()
        lr = np.log(s / s.shift(1)).dropna()
        return None if lr.empty else float(lr.std() * np.sqrt(252))
