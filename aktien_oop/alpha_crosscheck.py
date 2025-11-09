#!/usr/bin/env python3
# alpha_crosscheck.py
# Aufruf
# python alpha_crosscheck.py ^
#   --equity_csv "aktien_oop\bt_monthly_8x2_equity_curve.csv" ^
#   --bm_csv     "aktien_oop\bt_monthly_8x2_benchmark.csv"

import argparse, math
from pathlib import Path
import pandas as pd
import numpy as np

EPS = 1e-12
TRADING_DAYS = 252

def read_csv_series(path: Path, col_hint: str | None = None):
    """
    Erwartet CSV mit Datumsindex (erste Spalte) und mindestens einer Daten-Spalte.
    Ignoriert Kommentarzeilen (z. B. '# run_id=...').
    """
    df = pd.read_csv(path, comment="#", index_col=0, parse_dates=[0])
    # Spalte wählen
    if col_hint and col_hint in df.columns:
        s = df[col_hint]
    else:
        # Wenn 'equity' existiert → nehmen; sonst erste Spalte
        s = df[df.columns[0]] if "equity" not in df.columns else df["equity"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.sort_index().dropna()
    return s, df  # Serie + kompletter DF (für BM-Spalten-Suche)

def cagr_from_indexed(series: pd.Series) -> float:
    """Series ist wie bei dir auf 1.0 normiert (equity)."""
    if series.empty: return float("nan")
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    days = max(1, (series.index[-1] - series.index[0]).days)
    return (end ** (365.0 / days)) - 1.0

def total_return(series: pd.Series) -> float:
    if series.empty: return float("nan")
    return float(series.iloc[-1] - 1.0)

def ann_vol_from_series(series: pd.Series) -> float:
    rets = series.pct_change().dropna()
    return float(rets.std() * math.sqrt(TRADING_DAYS))

def print_block(title: str):
    print("\n" + title)
    print("-" * len(title))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity_csv", required=True, help="Pfad zu *_equity_curve.csv")
    ap.add_argument("--bm_csv",     required=True, help="Pfad zu *_benchmark.csv")
    ap.add_argument("--bm_col",     default="",   help="Optional: BM-Spaltenname (z.B. BM1_SPY); auto-detect wenn leer")
    args = ap.parse_args()

    eq_path = Path(args.equity_csv)
    bm_path = Path(args.bm_csv)

    # 1) Equity laden
    eq, eq_df = read_csv_series(eq_path, col_hint="equity")

    # 2) Benchmark laden und BM-Spalte finden
    bm_series, bm_df = read_csv_series(bm_path, col_hint="")
    # Falls benchmark.csv mehrere Spalten hat: nimm die Nicht-'equity'-Spalte
    bm_candidates = [c for c in bm_df.columns if c.lower() != "equity"]
    if args.bm_col:
        bm_col = args.bm_col
    elif len(bm_candidates) >= 1:
        bm_col = bm_candidates[0]
    else:
        # Fallback: nimm genau die Serie, die read_csv_series geliefert hat
        bm_col = bm_df.columns[0]
    bm = bm_df[bm_col].copy()
    bm.index = pd.to_datetime(bm.index).tz_localize(None)
    bm = bm.sort_index().dropna()

    # 3) Auf gleichen Index bringen (Forward-Fill)
    idx = eq.index
    eq_aligned = eq.reindex(idx).ffill().dropna()
    bm_aligned = bm.reindex(idx).ffill().dropna()
    common_idx = eq_aligned.index.intersection(bm_aligned.index)
    eq_aligned = eq_aligned.reindex(common_idx)
    bm_aligned = bm_aligned.reindex(common_idx)

    # 4) Kennzahlen
    port_tr  = total_return(eq_aligned)
    port_cagr = cagr_from_indexed(eq_aligned)
    bm_tr    = total_return(bm_aligned)
    bm_cagr  = cagr_from_indexed(bm_aligned)
    alpha    = port_cagr - bm_cagr

    port_vol = ann_vol_from_series(eq_aligned)
    bm_vol   = ann_vol_from_series(bm_aligned)
    # Rel.Vol + Corr
    port_rets = eq_aligned.pct_change().dropna()
    bm_rets   = bm_aligned.pct_change().dropna()
    corr = float(np.corrcoef(port_rets.align(bm_rets, join="inner")[0],
                             port_rets.align(bm_rets, join="inner")[1])[0,1])
    rel_vol = float(port_vol / (bm_vol + EPS))

    # 5) Ausgabe
    print_block("Alpha Crosscheck")
    print(f"Equity CSV:    {eq_path}")
    print(f"Benchmark CSV: {bm_path}  (BM-Spalte: {bm_col})")
    print()
    print(f"Portfolio Total Ret: {port_tr:7.2%} | Port CAGR: {port_cagr:7.2%}")
    print(f"BM Total Ret:        {bm_tr:7.2%} | BM  CAGR:  {bm_cagr:7.2%}")
    print(f"Alpha (ann., CAGR-Δ): {alpha:7.2%}")
    print(f"Vol (ann): Port={port_vol:6.2%}  BM={bm_vol:6.2%}  |  Rel.Vol={rel_vol:4.2f}x  Corr={corr:4.2f}")

    # Optional: Diagnose der "Mean-Diff × 252"-Alpha (nur als Vergleich)
    mean_alpha = (port_rets.mean() - bm_rets.mean()) * TRADING_DAYS if len(port_rets)*len(bm_rets) else float("nan")
    print(f"(Diag) Mean-Diff×252 Alpha: {mean_alpha:7.2%}  ← sollte sich von CAGR-Δ unterscheiden, wenn Sampling ungleichmäßig ist")

if __name__ == "__main__":
    main()
