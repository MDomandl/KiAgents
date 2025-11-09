from pathlib import Path
import pandas as pd

def load_sp500_tickers(tickers_file: str) -> list[str]:
    p = Path(tickers_file)
    if p.suffix.lower() in {".csv"}:
        df = pd.read_csv(p)
        # akzeptiere Spalten: symbol|ticker|Ticker
        for col in ["symbol","ticker","Ticker"]:
            if col in df.columns:
                return df[col].dropna().astype(str).unique().tolist()
        raise ValueError("CSV ohne Spalte symbol/ticker/Ticker")
    else:
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def load_sp500_meta(meta_file: str) -> pd.DataFrame:
    return pd.read_csv(meta_file)
