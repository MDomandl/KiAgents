from pathlib import Path
import hashlib
import pandas as pd


def load_tickers(path: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        for col in ["symbol", "ticker", "Ticker"]:
            if col in df.columns:
                return df[col].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()
        raise ValueError("CSV ohne Spalte symbol/ticker/Ticker")

    return [
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def universe_hash(tickers: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(map(str, tickers))).encode("utf-8")).hexdigest()


def load_meta(meta_file: str) -> pd.DataFrame:
    return pd.read_csv(meta_file)


def load_sp500_tickers(tickers_file: str) -> list[str]:
    return load_tickers(tickers_file)


def load_sp500_meta(meta_file: str) -> pd.DataFrame:
    return load_meta(meta_file)
