#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from collections import Counter
from datetime import datetime
from typing import Optional

# =====================
# Konfiguration
# =====================
ADJUSTED = True              # Bereinigte Kurse (Splits/Dividenden)
PERIOD   = "400d"            # Für 12-1 Momentum ausreichend lang
DAYS_WIN = 100               # Fenster für OLS-Trend/ATR/Vol
GAP_TH   = 0.18              # Max. Tagesgap
ADV_MIN_DOLLARS = 2_000_000  # Mindestdurchschnittsumsatz (Close*Volume, 20d)

TOP_K    = 8                 # Zielanzahl Titel
BUFFER_K = 12                # Turnover-Puffer: bestehende Titel bleiben bis Rang <= BUFFER_K
FORCE_REBALANCE = False      # Unabhängig vom Monat neu gewichten
SAVE_DIR = Path(".")         # Ausgabeordner (relative Pfade)
VERBOSE  = False

FAIL = Counter()             # Zähler, warum Titel rausfallen

# =====================
# Hilfsfunktionen
# =====================
def _as_series(x, name="Close") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return pd.Series(x, name=name)

def ensure_ohlc(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    # MultiIndex flachziehen (z. B. wenn mehrere Ticker als Spaltenebene geliefert werden)
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

def download_ohlc(ticker: str, period: str = PERIOD, adjusted: bool = ADJUSTED) -> Optional[pd.DataFrame]:
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=adjusted, threads=False)
    df = ensure_ohlc(df, ticker)
    if df is not None:
        return df

    # Fallback (zur Sicherheit): nochmals mit auto_adjust=True
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True, threads=False)
    return ensure_ohlc(df, ticker)

def kein_großes_gap_series(series, threshold: float = GAP_TH):
    """Rückgabe: (ok: bool, max_gap: float|None)"""
    s = _as_series(series, "Close").dropna()
    if s.empty:
        return False, None
    gaps = s.pct_change().abs().dropna()
    if gaps.empty:
        return True, 0.0
    mg = float(gaps.max())
    return (mg < threshold), mg

def avg_dollar_volume(df: pd.DataFrame, win: int = 20) -> Optional[float]:
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    ser = (_as_series(df["Close"], "Close") * _as_series(df["Volume"], "Volume")).rolling(win).mean().dropna()
    if ser.empty:
        return None
    return float(ser.iloc[-1])

def mom_12_1(close: pd.Series) -> Optional[float]:
    """12-1 Momentum: Rendite der letzten 12 Monate ohne den letzten Monat."""
    c = _as_series(close, "Close").dropna()
    if len(c) < 252:
        return None
    try:
        start = float(c.iloc[-252])
        end   = float(c.iloc[-21])
        if start == 0:
            return None
        return end / start - 1.0
    except Exception:
        return None

def sp500_above_200dma() -> bool:
    df = yf.download("^GSPC", period="250d", interval="1d",
                     progress=False, auto_adjust=ADJUSTED, threads=False)
    if df is None or df.empty or "Close" not in df.columns:
        print("S&P 500: keine Daten erhalten.")
        return False

    close = _as_series(df["Close"], "Close").dropna()
    if len(close) < 200:
        print(f"S&P 500: nur {len(close)} gültige Close-Werte — 200DMA nicht möglich.")
        return False

    sma200 = close.rolling(200).mean().dropna()
    last_close = float(close.iloc[-1])
    last_sma   = float(sma200.iloc[-1])

    print(f"S&P 500 → Close: {last_close:.2f} | 200DMA: {last_sma:.2f} | Markt {'über' if last_close>last_sma else 'unter'} 200DMA")
    return last_close > last_sma

# =====================
# Signal- & Score-Berechnung
# =====================
def calculate_signals_for_ticker(ticker: str):
    """Gibt dict mit Signalen/Features zurück oder None bei Ausschluss."""
    try:
        df_full = download_ohlc(ticker, period=PERIOD, adjusted=ADJUSTED)
        if df_full is None or df_full.empty:
            FAIL["no_data"] += 1
            return None

        # (a) Liquidität
        adv = avg_dollar_volume(df_full, 20)
        if adv is not None and adv < ADV_MIN_DOLLARS:
            FAIL["illiquid"] += 1
            return None

        # (b) 12-1 Momentum
        m121 = mom_12_1(df_full["Close"])
        if m121 is None:
            FAIL["mom121_nan"] += 1

        # Kürzen auf Auswertefenster
        df = df_full.dropna(subset=["Close", "High", "Low"])
        df = df[-DAYS_WIN:]
        if len(df) < DAYS_WIN:
            FAIL["too_few_days"] += 1
            return None

        # Filter 1: über 100er SMA?
        close = _as_series(df["Close"], "Close").dropna()
        sma100 = close.rolling(100).mean().dropna()
        if sma100.empty or not (float(close.iloc[-1]) > float(sma100.iloc[-1])):
            FAIL["under_sma"] += 1
            return None

        # Filter 2: keine großen Gaps (bereinigte Serie bevorzugen)
        series_for_gap = df["Adj Close"] if "Adj Close" in df.columns else close
        ok_gap, max_gap = kein_großes_gap_series(series_for_gap, threshold=GAP_TH)
        if not ok_gap:
            FAIL["gap"] += 1
            return None

        # Trend (OLS auf Log-Preisen)
        df = df.copy()
        df["LogPrice"] = np.log(_as_series(df["Close"], "Close"))
        df["Day"] = np.arange(len(df), dtype=float)
        X = df["Day"].values.reshape(-1, 1)
        y = df["LogPrice"].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        slope = float(reg.coef_[0][0])
        r2 = float(reg.score(X, y))
        score_lin = slope * r2

        # ATR(14)
        df["H-L"]  = df["High"] - df["Low"]
        df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
        df["L-PC"] = (df["Low"]  - df["Close"].shift(1)).abs()
        df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        atr_series = df["TR"].rolling(window=14).mean().dropna()
        if atr_series.empty:
            FAIL["atr_nan"] += 1
            return None
        atr = float(atr_series.iloc[-1])

        last_close = float(close.iloc[-1])
        stop_loss_pct = (3 * atr / last_close) * 100 if last_close != 0 else None

        # Historische Volatilität
        log_ret = np.log(close / close.shift(1)).dropna()
        vol_annual = float(log_ret.std() * np.sqrt(252)) if not log_ret.empty else None

        return {
            "ticker": ticker,
            "score_lin": round(score_lin, 6),
            "mom_12_1": round(m121, 4) if m121 is not None else np.nan,
            "slope": round(slope, 6),
            "r2": round(r2, 4),
            "volatility": round(vol_annual, 4) if vol_annual is not None else None,
            "stop_loss_pct": round(stop_loss_pct, 2) if stop_loss_pct is not None else None
        }
    except Exception:
        FAIL["exception"] += 1
        return None

def load_tickers(path: str = "sp500_tickers.txt") -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
        if not tickers:
            raise ValueError("Datei leer")
        return tickers
    except Exception as e:
        print(f"Warnung: Konnte '{path}' nicht laden ({e}). Nutze Fallback (AAPL, MSFT, NVDA).")
        return ["AAPL", "MSFT", "NVDA"]

# =====================
# Turnover-Puffer & Rebalancing
# =====================
def current_month_key() -> str:
    return datetime.now().strftime("%Y-%m")

def read_prev_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["as_of", "ticker", "allocation_pct", "rank", "score"])
    df = pd.read_csv(path)
    for col in ["as_of", "ticker", "allocation_pct", "rank", "score"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def select_with_buffer(ranked_df: pd.DataFrame, prev_positions: pd.DataFrame,
                       top_k: int = TOP_K, buffer_k: int = BUFFER_K) -> pd.DataFrame:
    """Behält alte Titel mit Rang <= buffer_k, füllt Rest mit besten neuen bis top_k."""
    keep = []
    if not prev_positions.empty:
        prev_tickers = set(prev_positions["ticker"].astype(str))
        sub = ranked_df[ranked_df["ticker"].isin(prev_tickers)]
        keep = sub[sub["rank"] <= buffer_k].sort_values("rank").head(top_k)

    need = top_k - len(keep)
    filler = ranked_df[~ranked_df["ticker"].isin(keep["ticker"] if len(keep) else [])]
    filler = filler.sort_values("rank").head(max(0, need))

    sel = pd.concat([keep, filler], ignore_index=True)
    sel = sel.sort_values("rank").head(top_k).reset_index(drop=True)
    return sel

def inverse_vol_allocation(df_sel: pd.DataFrame) -> pd.Series:
    vols = df_sel["volatility"].replace(0, np.nan)
    inv = 1.0 / vols
    total = float(inv.sum())
    if total == 0 or np.isnan(total):
        return pd.Series([np.nan]*len(df_sel), index=df_sel.index)
    return (inv / total * 100).round(2)

def should_rebalance(prev_positions_path: Path) -> bool:
    if FORCE_REBALANCE:
        return True
    if not prev_positions_path.exists():
        return True
    df_prev = pd.read_csv(prev_positions_path)
    if df_prev.empty:
        return True
    last_date = pd.to_datetime(df_prev["as_of"].max(), errors="coerce")
    if pd.isna(last_date):
        return True
    return last_date.strftime("%Y-%m") != current_month_key()

def append_csv(path: Path, df: pd.DataFrame):
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")

# =====================
# Hauptprogramm
# =====================
def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    positions_path = SAVE_DIR / "portfolio_positions.csv"
    rankings_log   = SAVE_DIR / "rankings_log.csv"
    runs_log       = SAVE_DIR / "runs_log.csv"
    top8_log       = SAVE_DIR / "top8_log.csv"

    tickers = load_tickers()
    print(f"Starte Bewertung ({len(tickers)} Ticker)...")

    if not sp500_above_200dma():
        print("⚠️  Abbruch: S&P 500 unter 200-Tage-Linie (kein Long-Markt).")
        return

    # Signale sammeln
    rows = []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...")
        sig = calculate_signals_for_ticker(t)
        if sig:
            rows.append(sig)

    if not rows:
        print("⚠️  Keine Ergebnisse nach Filtern/Berechnung.")
        if FAIL:
            print("Filter-Statistik:", dict(FAIL))
        run_row = pd.DataFrame([{
            "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "adjusted": ADJUSTED, "period": PERIOD, "days_win": DAYS_WIN,
            "gap_th": GAP_TH, "adv_min": ADV_MIN_DOLLARS,
            "top_k": TOP_K, "buffer_k": BUFFER_K,
            "num_universe": len(tickers),
            "num_pass": 0, "fail_counts": dict(FAIL)
        }])
        append_csv(runs_log, run_row)
        return

    df = pd.DataFrame(rows)

    # Kombi-Score (Ranks)
    df["rank_lin"]  = df["score_lin"].rank(pct=True)
    df["rank_m121"] = df["mom_12_1"].rank(pct=True).fillna(df["rank_lin"])
    df["score"] = 0.5*df["rank_lin"] + 0.5*df["rank_m121"]

    # Gesamtrang (1=best)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1)

    # Rankings-Log (vollständig)
    full_rank_log = df.copy()
    full_rank_log.insert(0, "as_of", pd.Timestamp.now().strftime("%Y-%m-%d"))
    append_csv(rankings_log, full_rank_log)

    # Turnover-Puffer anwenden / monatlich rebalancen
    prev_positions = read_prev_positions(positions_path)
    rebalance_now = should_rebalance(positions_path)
    if not rebalance_now:
        print("ℹ️  Diesen Monat bereits rebalanced – nutze bestehende Positionen (FORCE_REBALANCE=True zum Erzwingen).")
        print(prev_positions)
        return

    sel = select_with_buffer(df[["ticker", "rank", "score", "volatility", "stop_loss_pct"]],
                             prev_positions, TOP_K, BUFFER_K)

    # Allocation
    sel["allocation_pct"] = inverse_vol_allocation(sel)
    sel.insert(0, "as_of", pd.Timestamp.now().strftime("%Y-%m-%d"))

    # Ausgabe
    print("\nNeues Portfolio (Turnover-Puffer aktiv):\n")
    print(sel)

    # Logs schreiben
    append_csv(top8_log, sel)

    run_row = pd.DataFrame([{
        "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "adjusted": ADJUSTED, "period": PERIOD, "days_win": DAYS_WIN,
        "gap_th": GAP_TH, "adv_min": ADV_MIN_DOLLARS,
        "top_k": TOP_K, "buffer_k": BUFFER_K,
        "num_universe": len(tickers),
        "num_pass": len(df),
        "fail_counts": dict(FAIL)
    }])
    append_csv(runs_log, run_row)

    # Portfolio-Status aktualisieren (überschreiben = aktueller Bestand)
    sel[["as_of","ticker","allocation_pct","rank","score"]].to_csv(positions_path, index=False, encoding="utf-8")

    print("\nFilter-Statistik:", dict(FAIL))
    print(f"\nGespeichert unter:\n  - {positions_path}\n  - {top8_log}\n  - {rankings_log}\n  - {runs_log}")

if __name__ == "__main__":
    main()
