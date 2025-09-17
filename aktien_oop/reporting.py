# aktien_oop/reporting.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class EquityArtifacts:
    equity_csv: str
    drawdown_csv: str
    equity_png: str
    drawdown_png: str
    summary_txt: str
    params_json: str | None = None


def _infer_periods_per_year(frequency: str) -> int:
    f = (frequency or "").lower()
    if f.startswith("week"):
        return 52
    return 12  # default: monthly


def _compute_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet entweder Spalten:
      - date, equity
    ODER
      - date, period_return  (wird zu equity kumuliert)
    """
    out = df.copy()
    if "equity" not in out.columns:
        if "period_return" not in out.columns:
            raise ValueError("Equity-Input muss 'equity' oder 'period_return' enthalten.")
        out["equity"] = (1.0 + out["period_return"]).cumprod()
    if "date" in out.columns:
        out = out.sort_values("date")
    out["running_peak"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["running_peak"] - 1.0
    return out[["date", "equity", "running_peak", "drawdown"]]


def _annual_metrics(eqd: pd.DataFrame, periods_per_year: int) -> dict:
    # Periodenrenditen aus equity ableiten
    rets = eqd["equity"].pct_change().fillna(0.0).to_numpy()
    n = len(rets)
    start, end = float(eqd["equity"].iloc[0]), float(eqd["equity"].iloc[-1])

    # CAGR robust, falls n==0
    if n == 0 or start <= 0:
        cagr = float("nan")
    else:
        cagr = (end / start) ** (periods_per_year / max(n, 1)) - 1.0

    ann_vol = float(np.std(rets, ddof=1) * math.sqrt(periods_per_year)) if n > 1 else float("nan")
    sharpe = cagr / ann_vol if (ann_vol and ann_vol > 0) else float("nan")
    max_dd = float(eqd["drawdown"].min()) if n > 0 else float("nan")

    # einfache DD-Periode näherungsweise bestimmen
    dd_idx = int(eqd["drawdown"].idxmin()) if n > 0 else None
    dd_date = str(eqd["date"].iloc[dd_idx].date()) if dd_idx is not None else None

    return {
        "samples": n,
        "final_equity": end,
        "CAGR": cagr,
        "ann_vol": ann_vol,
        "Sharpe_0pct": sharpe,
        "MaxDrawdown": max_dd,
        "MaxDrawdown_at": dd_date,
    }


def write_equity_artifacts(
    equity_or_returns_df: pd.DataFrame,
    out_prefix: Path | str,
    *,
    frequency: str = "monthly",
    params: dict | None = None,
    write_params_json: bool = True,
) -> EquityArtifacts:
    """
    Erstellt CSVs + PNGs + Summary für Equity/Drawdown.
    - equity_or_returns_df: enthält entweder 'equity' ODER 'period_return' (und immer 'date').
    - out_prefix: Basis-Pfad ohne Suffix, z. B. save_dir / f"bt_{frequency}_{top_k}x{buffer_k}"
    - frequency: 'monthly' oder 'weekly' (bestimmt Annualisierung)
    - params: optionale Parameter (werden in _params.json weggeschrieben)
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    eq_csv = f"{out_prefix}_equity_curve.csv"
    dd_csv = f"{out_prefix}_drawdown.csv"
    eq_png = f"{out_prefix}_equity_curve.png"
    dd_png = f"{out_prefix}_drawdown.png"
    summ   = f"{out_prefix}_summary.txt"
    params_json = f"{out_prefix}_params.json" if write_params_json and params is not None else None

    # Equity/Drawdown berechnen
    eqd = _compute_equity_df(equity_or_returns_df)

    # CSVs
    eqd[["date", "equity"]].to_csv(eq_csv, index=False)
    eqd[["date", "drawdown"]].to_csv(dd_csv, index=False)

    # Kennzahlen
    ppy = _infer_periods_per_year(frequency)
    metrics = _annual_metrics(eqd, ppy)

    # Plots
    plt.figure()
    plt.plot(eqd["date"], eqd["equity"])
    plt.title("Equity-Kurve (Start = 1.0)")
    plt.xlabel("Datum")
    plt.ylabel("Depotwert (relativ)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(eq_png, dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(eqd["date"], eqd["drawdown"])
    plt.title("Drawdown-Verlauf")
    plt.xlabel("Datum")
    plt.ylabel("Drawdown (relativ)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dd_png, dpi=140, bbox_inches="tight")
    plt.close()

    # Summary
    with open(summ, "w", encoding="utf-8") as f:
        f.write("Backtest Summary\n")
        f.write("================\n")
        f.write(f"Frequency      : {frequency}\n")
        for k, v in metrics.items():
            f.write(f"{k:<15}: {v}\n")
        f.write("\nArtefakte:\n")
        f.write(f"  Equity CSV   : {eq_csv}\n")
        f.write(f"  Drawdown CSV : {dd_csv}\n")
        f.write(f"  Equity PNG   : {eq_png}\n")
        f.write(f"  Drawdown PNG : {dd_png}\n")

    # Params (optional)
    if params_json:
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

    return EquityArtifacts(
        equity_csv=eq_csv,
        drawdown_csv=dd_csv,
        equity_png=eq_png,
        drawdown_png=dd_png,
        summary_txt=summ,
        params_json=params_json,
    )
