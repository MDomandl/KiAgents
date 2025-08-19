#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from .config import normalize_ticker as normalize, Config
from pathlib import Path
import json

import pandas as pd
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT_OK   = Path("sp500_tickers.txt")
OUT_BAD  = Path("sp500_invalid.txt")
OUT_META = Path("sp500_meta.csv")

ALIAS_FILE = Path("alias_map.json")
# manuelle Umbenennungen / Alias
DEFAULT_ALIASES  = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    "BF.B":  "BF-B",
    "BF.A":  "BF-A",
    "BFB":   "BF-B",    # falls mal so gelistet
    "FB":    "META",    # Meta Platforms
}

def _load_aliases() -> dict[str, str]:
    aliases = {k.upper(): v.upper() for k, v in DEFAULT_ALIASES.items()}
    if ALIAS_FILE.exists():
        try:
            user = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                aliases.update({k.upper(): v.upper() for k, v in user.items()})
        except Exception:
            pass
    return aliases

ALIASES = _load_aliases()

def normalize_ticker(sym: str) -> str:
    s = (sym or "").strip().upper().replace(".", "-")
    return ALIASES.get(s, s)

def fetch_sp500_table() -> pd.DataFrame:
    # Wikipedia-Tabelle (enthält 'Symbol','Security','GICS Sector','GICS Sub-Industry')
    tables = pd.read_html(WIKI_URL, flavor="bs4")
    df = tables[0].copy()
    df.rename(columns={
        "Symbol":"ticker_raw",
        "Security":"security",
        "GICS Sector":"sector",
        "GICS Sub-Industry":"sub_industry",
    }, inplace=True)
    df["ticker"] = df["ticker_raw"].apply(normalize)
    return df[["ticker","security","sector","sub_industry"]]

def fetch_sp500_symbols() -> List[str]:
    tables = pd.read_html(WIKI_URL, flavor="bs4")
    # erste Tabelle enthält die Constituents
    tbl = tables[0]
    symbols = [str(x) for x in tbl["Symbol"].tolist()]
    return [normalize(s) for s in symbols]

def is_valid_yf(sym: str) -> bool:
    try:
        df = yf.download(sym, period="5d", interval="1d",
                         progress=False, auto_adjust=True, threads=False)
        return (not df.empty) and ("Close" in df.columns) and df["Close"].dropna().shape[0] > 0
    except Exception:
        return False

def validate_symbols(symbols: List[str], workers: int = 8) -> Tuple[List[str], List[str]]:
    good, bad = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(is_valid_yf, s): s for s in symbols}
        for f in as_completed(fut):
            s = fut[f]
            ok = False
            try:
                ok = f.result()
            except Exception:
                ok = False
            (good if ok else bad).append(s)
    # stabil sortieren
    good.sort()
    bad.sort()
    return good, bad

def main():
    # Nimmt alle Pfade aus der Config (werden in __post_init__ aufgelöst)
    cfg = Config()

    print("Hole S&P-500 Liste…")
    meta = fetch_sp500_table().dropna(subset=["ticker", "sector"])

    # Ticker normalisieren (Großschreibung, Punkte -> Bindestriche, etc.)
    meta["ticker"] = meta["ticker"].astype(str).map(normalize_ticker)
    syms = sorted(set(meta["ticker"].tolist()))

    print(f"Validiere {len(syms)} Ticker bei Yahoo Finance…")
    good, bad = [], []
    for s in syms:
        (good if is_valid_yf(s) else bad).append(s)

    good_set = set(good)

    # --- Ausgabepfade aus cfg ---
    out_ok   = cfg.tickers_file                 # z. B. aktien_oop/sp500_tickers.txt
    out_meta = cfg.sector_meta_file             # z. B. aktien_oop/sp500_meta.csv
    out_bad  = cfg.save_dir / "sp500_bad_tickers.txt"

    # Verzeichnisse sicherstellen
    out_ok.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_bad.parent.mkdir(parents=True, exist_ok=True)

    # Nur valide Ticker in die Meta-CSV übernehmen
    meta_valid = meta[meta["ticker"].isin(good_set)].drop_duplicates(subset=["ticker"])
    meta_valid.to_csv(out_meta, index=False, encoding="utf-8")

    # Tickerlisten schreiben
    out_ok.write_text("\n".join(good) + "\n", encoding="utf-8")
    out_bad.write_text("\n".join(bad)  + "\n", encoding="utf-8")

    print(f"OK: {len(good)}  |  Ungültig: {len(bad)}")
    print(f"Geschrieben: {out_ok}  /  {out_meta}")
    if bad:
        print(f"Ungültige Ticker: {out_bad}")


if __name__ == "__main__":
    main()
