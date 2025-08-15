#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from collections import Counter


# =====================
# Konfiguration
# =====================
ADJUSTED = True          # Bereinigte Kurse (Splits/Dividenden) verwenden
PERIOD   = "250d"        # Downloadzeitraum (etwas länger = stabilere SMAs/Regression)
DAYS_WIN = 100           # Fenster für die Auswertung
GAP_TH   = 0.18          # Max. erlaubtes Tagesgap (18%)
VERBOSE  = False         # Mehr Diagnoseausgaben

FAIL = Counter()         # Zähler, warum Titel rausfallen


# =====================
# Hilfsfunktionen
# =====================
def _as_series(x, name="Close"):
    """Akzeptiert Series oder 1-Spalten-DataFrame und gibt garantiert eine Series zurück."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return pd.Series(x, name=name)


def ensure_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalisiert ein von yfinance geliefertes DataFrame auf Spalten: Open, High, Low, Close, Volume."""
    if df is None or df.empty:
        return None

    # MultiIndex flachziehen (z. B. wenn mehrere Ticker als Spaltenebene geliefert werden)
    if isinstance(df.columns, pd.MultiIndex):
        # Falls die unterste Ebene die Ticker enthält, wähle den passenden Teilbaum
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            # ansonsten erste Ebene nehmen (Open/High/Low/Close/Volume)
            df.columns = df.columns.get_level_values(0)

    cols = set(df.columns)

    # Close aus Adj Close ableiten, falls nötig
    if "Close" not in cols and "Adj Close" in cols:
        df["Close"] = df["Adj Close"]
        cols = set(df.columns)

    # High/Low notfalls aus Close füllen (damit Pipeline weiterläuft; ATR wird dann konservativ klein)
    if "High" not in cols and "Close" in cols:
        df["High"] = df["Close"]
    if "Low" not in cols and "Close" in cols:
        df["Low"] = df["Close"]

    if not {"Close", "High", "Low"}.issubset(df.columns):
        return None

    # Für spätere Debugs den Ticker mitschreiben (optional)
    df.attrs["_ticker"] = ticker
    return df


def download_ohlc(ticker: str, period: str = PERIOD, adjusted: bool = ADJUSTED) -> pd.DataFrame | None:
    """Robuster Download mit Normalisierung und kleinem Fallback."""
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=adjusted, threads=False)
    df = ensure_ohlc(df, ticker)
    if df is not None:
        return df

    # Fallback (zur Sicherheit): nochmal ohne Adjust laden
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True, threads=False)
    return ensure_ohlc(df, ticker)


def kein_großes_gap_series(series, threshold: float = GAP_TH) -> tuple[bool, float | None]:
    """Prüft, ob das maximale Tagesgap (abs. prozentuale Änderung) unterhalb threshold liegt.
    Rückgabe: (ok, max_gap)
    """
    s = _as_series(series, "Close").dropna()
    if s.empty:
        return False, None
    gaps = s.pct_change().abs().dropna()
    if gaps.empty:
        return True, 0.0
    mg = float(gaps.max())
    return (mg < threshold), mg


# =====================
# Markt-Filter
# =====================
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
# Score-Berechnung pro Ticker
# =====================
def calculate_clenow_score(ticker: str, days: int = DAYS_WIN):
    try:
        df = download_ohlc(ticker, period=PERIOD, adjusted=ADJUSTED)
        if df is None or df.empty:
            if VERBOSE: print(f"{ticker}: keine OHLC-Daten.")
            FAIL["no_data"] += 1
            return None

        df = df.dropna(subset=["Close", "High", "Low"])
        df = df[-days:]
        if len(df) < days:
            if VERBOSE: print(f"{ticker}: zu wenige Tage ({len(df)}/{days}).")
            FAIL["too_few_days"] += 1
            return None

        # Filter 1: über 100-Tage-SMA?
        close = _as_series(df["Close"], "Close").dropna()
        sma100 = close.rolling(100).mean().dropna()
        if sma100.empty:
            FAIL["sma_nan"] += 1
            return None

        if not (float(close.iloc[-1]) > float(sma100.iloc[-1])):
            if VERBOSE: print(f"{ticker}: unter 100-Tage-Linie ❌")
            FAIL["under_sma"] += 1
            return None

        # Filter 2: keine großen Gaps (auf bereinigter Serie, falls vorhanden)
        series_for_gap = df["Adj Close"] if "Adj Close" in df.columns else close
        ok_gap, max_gap = kein_großes_gap_series(series_for_gap, threshold=GAP_TH)
        if not ok_gap:
            if VERBOSE: print(f"{ticker}: großes Gap {max_gap:.2%} ❌")
            FAIL["gap"] += 1
            return None

        # --- Regression auf Log-Preisen ---
        df = df.copy()
        df["LogPrice"] = np.log(_as_series(df["Close"], "Close"))
        df["Day"] = np.arange(len(df), dtype=float)

        X = df["Day"].values.reshape(-1, 1)
        y = df["LogPrice"].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        slope = float(reg.coef_[0][0])
        r2 = float(reg.score(X, y))
        score = slope * r2

        # --- ATR (Average True Range, 14) ---
        df["H-L"]  = df["High"] - df["Low"]
        df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
        df["L-PC"] = (df["Low"]  - df["Close"].shift(1)).abs()
        df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        atr_series = df["TR"].rolling(window=14).mean().dropna()
        if atr_series.empty:
            if VERBOSE: print(f"{ticker}: ATR nicht berechenbar (zu wenige Zeilen).")
            FAIL["atr_nan"] += 1
            return None
        atr = float(atr_series.iloc[-1])

        last_close = float(close.iloc[-1])
        stop_loss_pct = (3 * atr / last_close) * 100 if last_close != 0 else None

        # --- Historische Volatilität ---
        log_ret = np.log(close / close.shift(1)).dropna()
        vol_annual = float(log_ret.std() * np.sqrt(252)) if not log_ret.empty else None

        return {
            "ticker": ticker,
            "slope": round(slope, 6),
            "r2": round(r2, 4),
            "score": round(score, 6),
            "stop_loss_pct": round(stop_loss_pct, 2) if stop_loss_pct is not None else None,
            "volatility": round(vol_annual, 4) if vol_annual is not None else None
        }

    except Exception as e:
        if VERBOSE:
            print(f"{ticker}: Fehler [{e}]")
        FAIL["exception"] += 1
        return None


# =====================
# Ticker-Liste laden
# =====================
def load_tickers(path: str = "sp500_tickers.txt") -> list[str]:
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
# Hauptprogramm
# =====================
def main():
    tickers = load_tickers()
    print(f"Starte Bewertung ({len(tickers)} Ticker)...")

    if not sp500_above_200dma():
        print("⚠️  Abbruch: S&P 500 unter 200-Tage-Linie (kein Long-Markt).")
        return

    results = []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...")
        res = calculate_clenow_score(t)
        if res:
            results.append(res)

    if not results:
        print("⚠️  Keine Ergebnisse nach Filtern/Berechnung.")
        if FAIL:
            print("Filter-Statistik:", dict(FAIL))
        return

    df = pd.DataFrame(results)
    top_8 = df.sort_values(by="score", ascending=False).head(8).reset_index(drop=True)

    # Inverse-Volatilitäts-Allokation
    inv_vols = 1.0 / top_8["volatility"].replace(0, np.nan)
    total_inv = float(inv_vols.sum())
    if total_inv == 0 or np.isnan(total_inv):
        top_8["allocation_pct"] = np.nan
    else:
        top_8["allocation_pct"] = ((inv_vols / total_inv) * 100).round(2)

    print("\nTop 8 Momentum-Aktien (Clenow + Filter):\n")
    print(top_8)

    print("\nFilter-Statistik:", dict(FAIL))

    print("\nStrategiehinweise")
    print("------------------")
    print("• Neubewertung: Jeden Mittwoch. Aktien, die nicht mehr führend sind, werden verkauft und ersetzt.")
    print("• Stop-Loss-Anpassung: Ebenfalls mittwochs (3× ATR).")
    print("• Rebalancing: Alle 6 Monate Allokation prüfen und ggf. anpassen.")


if __name__ == "__main__":
    main()
