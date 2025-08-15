import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------------------------
# Hilfsfunktionen
# ----------------------------------------

def load_tickers(path="sp500_tickers.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def sp500_above_200dma():
    df = yf.download("^GSPC", period="250d", progress=False)
    df = df.dropna()
    sma200 = df["Close"].rolling(window=200).mean()
    last_close = float(df["Close"].iloc[-1])
    last_sma = float(sma200.iloc[-1])


    print(f"\nS&P 500: letzter Schlusskurs = {round(last_close, 2)} | 200-Tage-Linie = {round(last_sma, 2)}")
    return last_close > last_sma


def aktie_ueber_100dma(df):
    if "Close" not in df.columns:
        return False

    sma100 = df["Close"].rolling(window=100).mean()
    close = df["Close"]

    try:
        sma_val = float(sma100.iloc[-1])
        close_val = float(close.iloc[-1])
    except:
        return False

    return close_val > sma_val


def kein_großes_gap(df, threshold=0.15):
    try:
        df["Gap"] = df["Close"].pct_change().abs()
        max_gap = df["Gap"].max()
        return bool(max_gap < threshold)
    except:
        return False

# ----------------------------------------
# Hauptfunktion zur Score-Berechnung
# ----------------------------------------

def calculate_clenow_score(ticker, days=100):
    try:
        df = yf.download(ticker, period="130d", progress=False)
        df = df[-days:]

        if len(df) < days or df.isnull().values.any():
            return None

        # Filter 1: Aktie > 100 SMA
        if not aktie_ueber_100dma(df):
            print(f"{ticker}: unter 100-Tage-Linie ❌")
            return None

        # Filter 2: Kein Tagesgap > 15 %
        gap_ok = kein_großes_gap(df)
        if gap_ok is False:
            print(f"{ticker}: hat großes Gap ❌")
            return None

        # Log-Regression
        df["LogPrice"] = np.log(df["Close"])
        df["Day"] = np.arange(len(df))
        X = df["Day"].values.reshape(-1, 1)
        y = df["LogPrice"].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0][0]
        r2 = reg.score(X, y)
        score = slope * r2

        # Berechne ATR (Average True Range)
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
        df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        atr = df["TR"].rolling(window=14).mean().iloc[-1]

        close_val = float(df["Close"].iloc[-1])
        stop_loss_pct = (3 * atr / close_val) * 100 if close_val != 0 else None

        # Berechne historische Volatilität
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        volatility = df["log_return"].std() * np.sqrt(252)

        return {
        "ticker": ticker,
        "slope": round(slope, 6),
        "r2": round(r2, 4),
        "score": round(score, 6),
        "stop_loss_pct": round(stop_loss_pct, 2) if stop_loss_pct else None,
        "volatility": round(volatility, 4) if volatility else None
}

    except Exception as e:
        print(f"{ticker}: Fehler [{e}]")
        return None

# ----------------------------------------
# Hauptprogramm
# ----------------------------------------

def main():
    tickers = load_tickers()
    results = []

    if not sp500_above_200dma():
        print("⚠️ Kein Long-Markt. S&P 500 unter 200-Tage-Linie.")
        return

    print("\nStarte Bewertung aller Aktien mit Filtersystem...\n")

    for i, t in enumerate(tickers):
        print(f"{i + 1}/{len(tickers)}: {t}")
        result = calculate_clenow_score(t)
        if result:
            results.append(result)

    if not results:
        print("⚠️ Kein Ergebnis nach Anwendung der Filter.")
        return

    df = pd.DataFrame(results)
    top_8 = df.sort_values(by="score", ascending=False).head(8).reset_index(drop=True)

    # Risikobasierte Allokation auf Basis der inversen Volatilität
    inv_vols = 1 / top_8["volatility"].replace(0, np.nan)
    total_inv_vol = inv_vols.sum()
    top_8["allocation_pct"] = round((inv_vols / total_inv_vol) * 100, 2)

    print("Top 8 Momentum-Aktien nach Clenow + Filter:\n")
    print(top_8)
    return top_8

    print("\nStrategiehinweise")
    print("------------------")
    print("Neubewertung: Jeden Mittwoch. Aktien, die nicht mehr im Ranking sind, werden vollständig verkauft und durch neue ersetzt.")
    print("Stop-Loss-Anpassung: Ebenfalls mittwochs. Die Stop-Loss-Order wird anhand der aktuellen Werte neu gesetzt.")
    print("Rebalancing: Alle 6 Monate wird die Gewichtung des Portfolios überprüft und angepasst.")



if __name__ == "__main__":
    main()
