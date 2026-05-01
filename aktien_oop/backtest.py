from __future__ import annotations

import argparse
import hashlib
import json, sys, platform, getpass, socket

from aktien_oop.universe import (
    load_tickers as load_universe_tickers,
    load_meta,
    universe_hash,
)

try:
    from zoneinfo import ZoneInfo  # Py>=3.9
except Exception:
    ZoneInfo = None
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from aktien_oop.core_calc import CalcParams, calculate_portfolio, _rank_desc_stable
from aktien_oop.data_client import DataClient
from .config import Config

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import tomllib  # Py 3.11+
except Exception:
    tomllib = None

# optionales Reporting (falls dein Modul existiert)
try:
    from aktien_oop.reporting import write_equity_artifacts  # noqa: F401
    _HAS_REPORTING = True
except Exception:
    _HAS_REPORTING = False

__BT_VERSION__ = "BT-2025-11-11-a"
TRADING_DAYS = 252.0
DAYS_PER_YEAR = 365.25
EPS = 1e-12

# python -m aktien_oop.backtest --config aktien_oop/backtest_config.toml
# python -m aktien_oop.backtest --config aktien_oop/backtest_config.toml --start 2020-01-01 --end 2025-08-16
# ---------------------------------------
# Konfig (Backtest)
# ---------------------------------------
@dataclass
class BTConfig:
    tickers_file: str
    sector_meta: Optional[str]
    save_dir: str

    as_of: str

    start: str               # "YYYY-MM-DD"
    end: str                 # "YYYY-MM-DD"
    frequency: str           # "monthly" | "weekly"

    top_k: int
    buffer_k: int
    use_sector_limits: bool
    max_per_sector: Optional[int]   # None = aus
    cost_bps: float                 # Transaktionskosten (bps) pro Turnover
    slippage_bps: float             # Slippage (bps) pro Turnover
    min_history_days: int = 260     # Mindesthistorie je Ticker

    use_equal_weight: bool = False
    gap_filter: float = 0.0
    friction_eps: float = 0.0
    friction_eps_pct: float = 0.0
    weight_round_step: float = 0.0
    max_turnover_cap: float = 0.0
    rebalance_every_n: int = 1

    benchmark: str = "SXR8.DE"
    benchmark_ticker: Optional[str] = "SPY"
    dual_benchmark: bool = False
    benchmark2: str = ""

    require_above_sma: bool = False
    regime_below_action: str = "HOLD"
    regime_sma_days: int = 200  # z.B. 200 Kalendertage
    regime_exposure_low: float = 0.50  # z.B. 50% Exposure bei "unter SMA"

    vol_target_ann: Optional[float] = None  # z.B. 0.20 für 20% p.a.; None = aus
    vol_lookback_days: int = 20  # Roll-Fenster (Handelstage) für Sigma

    min_position_weight: float = 0.0
    max_active_names: int = 0

    include_cash: bool = False
    cash_yield_annual: float = 0.0

    dump_decision_bundles: bool = False
    dump_selection: bool = False
    dump_weights: bool = False
    decisions_dir: str = "aktien_oop/decisions"

    # NEU: Fenster-Konfiguration
    score_days: int = 252
    vol_days: int = 63

    verbose: bool = False
    universe_name: str = "sp500"

# ---------------------------------------
# Hilfen
# ---------------------------------------
def _bt_write_decision_bundle(prefix, decisions_dir, as_of_str,
                              weights, ranks=None, scores=None, vol=None,
                              run_id=None, **extras):
    Path(decisions_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    fname = f"{prefix}_{(run_id + '_' ) if run_id else ''}{ts}_{as_of_str}.json"
    # weights -> dict[str,float]
    if hasattr(weights, "items"):
        wdict = {str(k): float(v) for k, v in weights.items()}
    else:
        try:
            # z. B. pandas.Series
            wdict = {str(k): float(v) for k, v in weights.to_dict().items()}
        except Exception:
            wdict = {}
    bundle = {
        "as_of": as_of_str,
        "run_id": run_id,
        "timestamp": ts,
        "weights": wdict,
        "ranks":   ranks or {},
        "scores":  scores or {},
        "vol":     vol or {},
    }
    bundle.update(extras)
    out = Path(decisions_dir) / fname

    out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DECISION] wrote {out}")


def _round_and_renorm(w: pd.Series, step: float) -> pd.Series:
    if step and step > 0:
        w = (w / step).round() * step
    s = float(w.sum())
    return w / s if s > 0 else w

def _to_csv_with_runid(path: Path, df: pd.DataFrame, index: bool = True, run_id: Optional[str] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_id:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(f"# run_id={run_id}\n")
            df.to_csv(f, index=index, lineterminator="\n")
    else:
        df.to_csv(path, index=index, lineterminator="\n")

def _sanitize_for_fs(s: str) -> str:
    # Entfernt Windows-kritische Zeichen
    for ch in r'\/:*?"<>|':
        s = s.replace(ch, "_")
    return s

def load_tickers(path: str) -> List[str]:
    return sorted(set(load_universe_tickers(path)))

def load_sector_map(path: Optional[str]) -> Dict[str, str]:
    """
    Robustes Laden von Ticker→Sector aus CSV.
    Erwartet idealerweise Spalten: ticker, sector
    akzeptiert aber auch gängige Alternativen (Symbol, GICS Sector, etc.)
    und normalisiert Ticker (z.B. BRK.B -> BRK-B).
    """
    if not path:
        return {}

    p = Path(path)
    if not p.is_absolute():
        # relativ zur Projekt-Root / Working-Dir
        p = Path.cwd() / p

    if not p.exists():
        return {}

    df = pd.read_csv(p)

    # Spalten erkennen
    cols = {c.lower(): c for c in df.columns}

    def _pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_ticker = _pick("ticker", "symbol", "sym")
    c_sector = _pick("sector", "gics sector", "gics_sector", "gics")

    if not c_ticker or not c_sector:
        return {}

    def _norm_ticker(x: str) -> str:
        s = str(x).strip().upper()
        # Yahoo/Runner-Normalisierung: Punkt wird oft zu Bindestrich
        s = s.replace(".", "-")
        return s

    tmp = df[[c_ticker, c_sector]].copy()
    tmp.columns = ["ticker", "sector"]
    tmp["ticker"] = tmp["ticker"].map(_norm_ticker)
    tmp["sector"] = tmp["sector"].astype(str).str.strip()

    tmp = tmp.dropna(subset=["ticker", "sector"])
    tmp = tmp.drop_duplicates(subset=["ticker"])

    return dict(zip(tmp["ticker"], tmp["sector"]))


def _cfg_get(cfg, dotted: str, default=None):
    cur = cfg
    for key in dotted.split('.'):
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
            else:
                return default
        else:
            if hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                return default
    return cur

def _resolve_universe_paths(cfg):
    # Erst Versuch: neue Struktur [universe]
    t = _cfg_get(cfg, "universe.tickers_file")
    m = _cfg_get(cfg, "universe.meta_file")
    # Fallbacks für ältere Configs
    if not t:
        t = _cfg_get(cfg, "aktien_oop.tickers_file") or _cfg_get(cfg, "tickers_file")
    if not m:
        m = _cfg_get(cfg, "aktien_oop.meta_file") or _cfg_get(cfg, "meta_file") or _cfg_get(cfg, "sector_meta")
    return t, m

def download_close(tickers, start, end, verbose=False) -> pd.DataFrame:
    import time
    cols = {}
    failed = []

    for i, t in enumerate(tickers, 1):
        if verbose:
            print(f"[{i}/{len(tickers)}] DL {t} ...", flush=True)

        last_err = None
        df = None
        for attempt in range(3):
            try:
                df = yf.download(
                    t, start=start, end=end,
                    auto_adjust=True, progress=False,
                    timeout=30
                )
                if df is not None and not df.empty and "Close" in df.columns:
                    break
            except Exception as e:
                last_err = e
            time.sleep(1.0 + attempt)

        if df is None or df.empty or "Close" not in df.columns:
            failed.append(t)
            if verbose and last_err:
                print(f"   {t}: fehlgeschlagen ({last_err})", flush=True)
            continue

        s = df["Close"].astype(float)
        s.name = t                     # <<< hier der wichtige Fix
        cols[t] = s

    if verbose:
        print(f"Erfolg: {len(cols)} / {len(tickers)}   Fehlgeschlagen: {len(failed)}")
        if failed:
            print("Fehler bei:", ", ".join(failed[:20]) + (" …" if len(failed) > 20 else ""))

    if not cols:
        return pd.DataFrame()

    return pd.concat(cols, axis=1).sort_index()



def rebal_dates(px: pd.DataFrame, start: str, end: str, freq: str) -> List[pd.Timestamp]:
    """
    Liefert Rebalancing-Daten (Indexpunkte, die es in px wirklich gibt).
    freq: 'monthly' => Monatsultimo ('ME'), 'weekly' => Freitage ('W-FRI')
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    if freq == "monthly":
        raw = pd.date_range(s, e, freq="ME")      # Month End – statt 'M'
    elif freq == "weekly":
        raw = pd.date_range(s, e, freq="W-FRI")   # Wochentakt: Freitag
    else:
        raise ValueError(f"Unsupported freq={freq}")

    # auf echte Handelstage in px mappen
    px_idx = pd.DatetimeIndex(px.index).tz_localize(None)
    out = []
    for d in raw:
        # Nächster <= d liegender Handelstag
        ix = px_idx.searchsorted(d, side="right") - 1
        if ix >= 0:
            out.append(px_idx[ix])
    # Deduplikate + Sort
    out = sorted(pd.Index(out).unique())
    # Letztes Datum sicherstellen (falls nicht ohnehin abgedeckt)
    if len(out) == 0 or out[-1] < px_idx[-1]:
        out.append(px_idx[-1])
    return [pd.Timestamp(x) for x in out]



def compute_features(px: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Feature-Vorbereitung: tägliche Returns, SMA100, Vol20, Mom63/126/252 und Preise."""
    rets = px.pct_change()
    sma100 = px.rolling(100, min_periods=100).mean()
    vol20 = rets.rolling(20, min_periods=20).std()

    mom63 = px / px.shift(63) - 1.0
    mom126 = px / px.shift(126) - 1.0
    mom252 = px / px.shift(252) - 1.0

    mom12_1 = px.shift(21) / px.shift(252) - 1.0

    return {
        "px": px,
        "rets": rets,
        "sma100": sma100,
        "vol20": vol20,
        "mom63": mom63,
        "mom126": mom126,
        "mom252": mom252,
        "mom12_1": mom12_1,
    }


def score_universe(feat: Dict[str, pd.DataFrame],
                   as_of: pd.Timestamp,
                   universe: List[str]) -> pd.DataFrame:
    """Ranking & Hilfsspalten am Stichtag."""
    cols = universe
    # letzte Werte <= as_of
    row = {
        "mom63":  feat["mom63"].loc[:as_of, cols].iloc[-1],
        "mom126": feat["mom126"].loc[:as_of, cols].iloc[-1],
        "mom252": feat["mom252"].loc[:as_of, cols].iloc[-1],
        "mom12_1": feat["mom12_1"].loc[:as_of, cols].iloc[-1],
        "sma100": feat["sma100"].loc[:as_of, cols].iloc[-1],
        "vol20":  feat["vol20"].loc[:as_of, cols].iloc[-1],
    }

    # 1d Return zum Gap-Check
    rets_slice = feat["rets"].loc[:as_of, cols]
    if len(rets_slice) == 0:
        return pd.DataFrame()
    ret1d = rets_slice.iloc[-1]
    has_gap = ret1d.abs() > 0.12  # 12%

    # Under-SMA mit echten Preisen
    last_close = feat["px"].loc[:as_of, cols].iloc[-1]
    under_sma = (last_close < row["sma100"])

    # Cross-Section-Rank 0..1 (höher besser)
    def xs_rank01(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    r63 = xs_rank01(row["mom63"])
    r126 = xs_rank01(row["mom126"])
    r252 = xs_rank01(row["mom252"])
    r12_1 = xs_rank01(row["mom12_1"])
    score = (0.50 * r12_1 + 0.25 * r126 + 0.25 * r63)

    vol = row["vol20"]

    df = pd.DataFrame({
        "ticker": score.index,
        "score": score.values,
        "mom63": r63.values,
        "mom126": r126.values,
        "mom252": r252.values,
        "mom12_1": r12_1.values,
        "volatility": vol.values,
        "gap": has_gap.reindex(score.index).fillna(False).values,
        "under_sma": under_sma.reindex(score.index).fillna(True).values,
    })
    penalty = 0.15 * df["under_sma"].astype(int)  # 0.15 = 15%-Punkte Rank-Penalty
    df["score_adj"] = df["score"] - penalty
    df = df.sort_values(["score_adj", "ticker"], ascending=[False, True], kind="mergesort")
    df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    mask = (~df["gap"])
    out = df[mask].copy()
    stats = {"gap": int(df["gap"].sum()), "under_sma": int(df["under_sma"].sum())}
    return out, stats


def enforce_sector_limits(df_sorted: pd.DataFrame,
                          sector_map: Dict[str, str],
                          max_per_sector: Optional[int],
                          top_k: int) -> pd.DataFrame:
    if (max_per_sector is None) or (max_per_sector <= 0):
        return df_sorted.head(top_k).copy()
    picks = []
    counts: Dict[str, int] = {}
    for _, row in df_sorted.iterrows():
        tic = row["ticker"]
        sec = sector_map.get(tic, "UNKNOWN")
        if counts.get(sec, 0) < max_per_sector:
            picks.append(row)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= top_k:
                break
    return pd.DataFrame(picks)


def turnover_buffer(df_sorted: pd.DataFrame, prev: List[str], top_k: int, buffer_k: int) -> List[str]:
    """Behalte Vorpositionen, wenn sie innerhalb top_k+buffer_k bleiben."""
    keep: List[str] = []
    universe_rank = dict(zip(df_sorted["ticker"], df_sorted["rank"]))
    for t in prev:
        r = universe_rank.get(t, 1_000_000_000)
        if r <= top_k + buffer_k:
            keep.append(t)
    need = top_k - len(keep)
    add = [t for t in df_sorted["ticker"].tolist() if t not in keep]
    return keep + add[:max(0, need)]

def inverse_vol_weights(sel: pd.DataFrame) -> pd.Series:
    v = sel["volatility"].replace([np.inf, 0], np.nan)
    if v.isna().all():
        v = pd.Series(1.0, index=sel.index)
    v = v.fillna(v.median())
    inv = 1.0 / v
    w = inv / inv.sum()
    # WICHTIG: Index = Ticker
    w.index = sel["ticker"].values
    return w

def build_weights(sel: pd.DataFrame, use_equal_weight: bool) -> pd.Series:
    if use_equal_weight:
        w = pd.Series(1.0 / len(sel), index=sel["ticker"].values)
    else:
        w = inverse_vol_weights(sel)
    return w


def calc_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = dd.min()
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return float(mdd), start, end


def _apply_exposures(series: pd.Series,
                     regime_on: bool, bm_s: Optional[pd.Series], bm_sma: Optional[pd.Series], regime_low: float,
                     vt_on: bool, vt_exp: Optional[pd.Series]) -> pd.Series:
    """Wendet Regime- und VolTarget-Exposures (multiplikativ) auf eine Return-Serie an."""
    if series.empty:
        return series
    exp = pd.Series(1.0, index=series.index)
    if regime_on and bm_s is not None and bm_sma is not None:
        sig = (bm_s.reindex(exp.index) >= bm_sma.reindex(exp.index)).fillna(True)
        exp = exp * (sig.astype(float) * (1.0 - float(regime_low)) + float(regime_low))
    if vt_on and vt_exp is not None:
        exp = exp * vt_exp.reindex(exp.index).fillna(1.0)
    return series * exp

def apply_exposures_and_cash(
    base_port_rets: pd.Series,
    regime_on: bool, bm_s: Optional[pd.Series], bm_sma: Optional[pd.Series], regime_low: float,
    vt_on: bool, vt_exp: Optional[pd.Series],
    cash_weight: float, cash_yield_annual: float
) -> pd.Series:
    """
    Wendet Regime/VolTarget auf den Aktien-Teil an und addiert den Cash-Ertrag.
    Gibt eine tägliche Return-Serie des Gesamtportfolios zurück.
    """
    if base_port_rets is None or base_port_rets.empty:
        stock_part = pd.Series(0.0, index=pd.Index([], dtype='object'))
    else:
        stock_part = _apply_exposures(base_port_rets, regime_on, bm_s, bm_sma, regime_low, vt_on, vt_exp)
    cash_part = float(cash_weight) * _cash_daily_return(float(cash_yield_annual))
    return stock_part + cash_part

def _period_to_days(period) -> int:
    """'800d'/'252' → int Tage (robust)."""
    try:
        value = str(period).strip()
        if value.endswith("d"):
            value = value[:-1] + "D"
        return int(pd.to_timedelta(value) / pd.Timedelta(days=1))
    except Exception:
        import re
        m = re.search(r"\d+", str(period))
        return int(m.group(0)) if m else 252

def _calc_turnover(old_w: dict[str, float], new_w: dict[str, float]) -> float:
    old = pd.Series(old_w, dtype=float)
    new = pd.Series(new_w, dtype=float)
    aligned = pd.concat([old, new], axis=1).fillna(0.0)
    aligned.columns = ["old", "new"]
    return float((aligned["old"] - aligned["new"]).abs().sum() / 2.0)

# ---------------------------------------
# Backtester
# ---------------------------------------
class Backtester:
    def __init__(self, cfg: BTConfig):
        self.cfg = cfg
        self.data = DataClient(cfg)
        # -- Universe-Loader: zentral & identisch zum Runner --
        ufile, mfile = _resolve_universe_paths(self.cfg)
        self.tickers = load_universe_tickers(str(ufile))
        self.meta = load_meta(str(mfile))
        self.universe_name = str(getattr(self.cfg, "universe_name", "sp500") or "sp500")
        self.universe_file = str(ufile)
        self.universe_hash = universe_hash(self.tickers)
        self.cfg.tickers_file = str(ufile)
        self.cfg.sector_meta = str(mfile)
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    def _dbg(self, msg: str):
        if getattr(self.cfg, "verbose", False):
            print(msg)

    def run(self):
        print(f"[BT] version={__BT_VERSION__}")
        # --- Self-fingerprint: sicherstellen, dass wir denselben Code sehen ---
        try:
            this_path = Path(__file__)
            bt_sha1 = hashlib.sha1(this_path.read_bytes()).hexdigest()
            print(f"[BT] file_sha1={bt_sha1}  py={sys.version.split()[0]}  os={platform.system()}-{platform.release()}")
        except Exception:
            pass

        # --- /Self-fingerprint ---

        def _first(*vals, default=None):
            for v in vals:
                if v is not None:
                    return v
            return default

        def _get(obj, name, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)
        # --- NORMALIZE (Backtester) ---
        def _assign(obj, name, value):
            try:
                object.__setattr__(obj, name, value)
            except Exception:
                setattr(obj, name, value)

        cfg = self.cfg

        windows = _get(cfg, "windows")
        data = _get(cfg, "data")
        limits = _get(cfg, "limits")
        topk = _get(cfg, "topk")

        top_k = getattr(cfg, "top_k", None) or getattr(getattr(cfg, "topk", object()), "top_k", None) or 12
        buffer_k = getattr(cfg, "buffer_k", None) or getattr(getattr(cfg, "topk", object()), "buffer_k", None) or 3
        max_per_sector = _first(
            _get(limits, "max_per_sector"),
            _get(cfg, "max_per_sector"),
            3,
        )
        use_sector_limits = bool(_first(
            _get(limits, "use_sector_limits"),
            _get(cfg, "use_sector_limits"),
            True,
        ))
        gap_filter = float(_first(
            _get(limits, "gap_filter"),
            _get(cfg, "gap_filter"),
            0.0,
        ) or 0.0)

        # Fenster direkt aus BTConfig (kann über [windows] oder Root kommen)
        score_days = int(getattr(cfg, "score_days", 252) or 252)
        vol_days = int(getattr(cfg, "vol_days", 63) or 63)
        adjusted = getattr(cfg, "adjusted", None)
        if adjusted is None:
            adjusted = getattr(getattr(cfg, "data", object()), "adjusted", True)

        _assign(cfg, "top_k", int(top_k))
        _assign(cfg, "buffer_k", int(buffer_k))
        _assign(cfg, "max_per_sector", int(max_per_sector) if max_per_sector is not None else None)
        _assign(cfg, "use_sector_limits", bool(use_sector_limits))
        _assign(cfg, "gap_filter", float(gap_filter))
        _assign(cfg, "score_days", int(score_days))
        _assign(cfg, "vol_days", int(vol_days))
        _assign(cfg, "adjusted", bool(adjusted))
        # --- /NORMALIZE ---

        self._dbg(
            "[CFG] "
            f"freq={self.cfg.frequency}  top_k={self.cfg.top_k}  buffer_k={self.cfg.buffer_k}  "
            f"max_per_sector={self.cfg.max_per_sector}  max_active_names={getattr(self.cfg, 'max_active_names', 0)}  "
            f"include_cash={getattr(self.cfg, 'include_cash', False)}  "
            f"eps={getattr(self.cfg, 'friction_eps', 0.0)}  eps_pct={getattr(self.cfg, 'friction_eps_pct', 0.0)}  "
            f"cap={getattr(self.cfg, 'max_turnover_cap', 1.0)}  round={getattr(self.cfg, 'weight_round_step', 0.0)}  "
            f"bm1={getattr(self.cfg, 'benchmark', 'SXR8.DE')}  bm2={getattr(self.cfg, 'benchmark2', '')}"
        )

        tickers = load_tickers(self.cfg.tickers_file)
        if self.cfg.verbose:
            print(f"{len(tickers)} Ticker aus Datei '{self.cfg.tickers_file}' geladen. "
                  f"Beispiele: {', '.join(tickers[:10])}")

        assert self.cfg.top_k > 0, "top_k must be > 0"
        assert self.cfg.buffer_k >= 0, "buffer_k must be >= 0"
        if getattr(self.cfg, "max_active_names", 0):
            assert self.cfg.max_active_names <= self.cfg.top_k, "max_active_names <= top_k empfohlen"
        if self.cfg.weight_round_step:
            assert 0.0 <= self.cfg.weight_round_step <= 0.05, "round step looks large; did you mean e.g. 0.01?"

        # 1) Daten laden
        px = download_close(tickers, self.cfg.start, self.cfg.end, verbose=self.cfg.verbose)
        # Index säubern und sortieren
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = px[~px.index.duplicated()].sort_index()
        # Spalten normalisieren (MultiIndex → Ticker), dann str
        px = _normalize_price_columns(px)
        # Spaltennamen als str (verhindert seltsame Joins)
        px.columns = px.columns.astype(str)

        if px.empty:
            print("Keine Preisdaten geladen.")
            return
        feats = compute_features(px)

        # --- Regime-Filter vorbereiten (optional) ---
        regime_on = bool(getattr(self.cfg, "require_above_sma", False))
        bm_s = None
        bm_sma = None
        if regime_on:
            bm_tk = self.cfg.benchmark_ticker or "SXR8.DE"
            bm_px = download_close([bm_tk], self.cfg.start, self.cfg.end, verbose=False)
            bm_px = _normalize_price_columns(bm_px)
            if not bm_px.empty:
                s = bm_px.iloc[:, 0].astype(float).dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s = s[~s.index.duplicated()].sort_index()
                # Zeitbasiertes Rolling: letzte N Kalendertage
                N = int(self.cfg.regime_sma_days)
                sma = s.rolling(window=N).mean()
                bm_s, bm_sma = s, sma
            else:
                regime_on = False

        vt_on = bool(getattr(self.cfg, "vol_target_ann", None))
        vt_exp = None
        if vt_on and bm_s is not None:
            bm_ret = bm_s.pct_change().replace([np.inf, -np.inf], np.nan)
            roll = bm_ret.rolling(self.cfg.vol_lookback_days).std()
            target_daily = float(self.cfg.vol_target_ann) / np.sqrt(TRADING_DAYS)
            vt_exp = (target_daily / roll).clip(upper=1.0)  # nur runterregeln, nie hebeln

        # 2) Rebalance-Termine
        first_valid = px.index.min() + pd.Timedelta(days=self.cfg.min_history_days)
        rdates = [d for d in rebal_dates(px, self.cfg.start, self.cfg.end, self.cfg.frequency) if d >= first_valid]
        if not rdates:
            print("Keine Rebalance-Termine nach Mindesthistorie.")
            return

        out_prefix = Path(self.cfg.save_dir) / f"bt_{self.cfg.frequency}_{self.cfg.top_k}x{self.cfg.buffer_k}"
        eq_path   = Path(str(out_prefix) + "_equity_curve.csv")
        pos_path  = Path(str(out_prefix) + "_positions.csv")
        trd_path  = Path(str(out_prefix) + "_trades.csv")
        summ_path = Path(str(out_prefix) + "_summary.txt")

        equity_rows: List[Dict[str, Any]] = []
        positions_log: List[pd.DataFrame] = []
        trades_log: List[Dict[str, Any]] = []

        # Laufender Portfoliozustand
        cur_weights = pd.Series(dtype=float)  # index=ticker, val=weight
        cur_holdings: List[str] = []
        equity_val = 1.0

        # === PREPARE UNIVERSE, SECTORS, PRELOAD PRICES/RETS ===
        # 2.1) Universe ermitteln (alles, was du Backtesten willst)
        try:
            universe_all = list(self.tickers)  # falls du self.tickers setzt
        except Exception:
            universe_all = load_universe_tickers(str(self.cfg.tickers_file))

        # 2.2) Sector-Map (Ticker -> Sektor) – MUSS lockstep zum Runner sein
        secmap_loaded = load_sector_map(self.cfg.sector_meta)  # z.B. aktien_oop/sp500_meta.csv

        # fallback nur wenn wirklich nichts geladen werden konnte
        if not secmap_loaded:
            try:
                meta_df = load_meta(str(self.cfg.sector_meta))
                if isinstance(meta_df, dict):
                    # meta_df: { "AAPL": {"sector": "..."} , ... }
                    secmap_loaded = {
                        str(t).strip().upper().replace(".", "-"): str(meta_df.get(t, {}).get("sector", "UNKNOWN"))
                        for t in universe_all
                    }
            except Exception:
                secmap_loaded = {}

        # Universe-spezifische Map (und UNKNOWN wenn nicht vorhanden)
        secmap_pre = {
            str(t).strip().upper().replace(".", "-"): secmap_loaded.get(str(t).strip().upper().replace(".", "-"),
                                                                        "UNKNOWN")
            for t in universe_all
        }

        # 2.3) Preload-Fenster bestimmen
        asof0 = pd.Timestamp(rdates[0])
        asofN = pd.Timestamp(rdates[-1])
        _days = max(800, _period_to_days(getattr(self.cfg, "period", "800D")))
        bt_start = (asof0 - pd.Timedelta(days=_days)).normalize()
        bt_end = asofN

        # 2.4) Preise EINMAL laden & normalisieren (breit: Spalten = Ticker)
        prices_all = download_close(
            universe_all,
            start=bt_start.date().isoformat(),
            end=(bt_end + pd.Timedelta(days=1)).date().isoformat(),  # <- +1 Tag
            verbose=False
        )
        prices_all = _normalize_price_columns(prices_all)
        prices_all.index = pd.to_datetime(prices_all.index).tz_localize(None)
        prices_all = prices_all[~prices_all.index.duplicated()].sort_index().ffill()

        # 2.5) Returns & SMAs EINMAL berechnen
        rets_all = prices_all.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sma200_all = prices_all.rolling(200, min_periods=1).mean()
        sma100_all = prices_all.rolling(100, min_periods=1).mean()  # falls du 100d brauchst

        # 2.6) valid_cols_preload ableiten (genug Historie + nicht nur NaN)
        min_hist_days = 252  # anpassen, falls du eine andere Mindesthistorie willst
        have_hist = (prices_all.notna().rolling(min_hist_days, min_periods=min_hist_days).sum().iloc[
                         -1] >= min_hist_days)
        valid_cols_preload = [c for c in prices_all.columns if bool(have_hist.get(c, False))]

        # 2.7) Alles auf valid_cols_preload „trimmen“ (Konsistenz)
        prices_all = prices_all[valid_cols_preload]
        rets_all = rets_all[valid_cols_preload]
        sma200_all = sma200_all[valid_cols_preload]
        sma100_all = sma100_all[valid_cols_preload]
        secmap = {t: secmap_pre.get(t, "UNKNOWN") for t in valid_cols_preload}  # <- ab hier 'secmap' verfügbar
        # === /PREPARE ===

        # 3) Backtest-Schleife
        for i, d in enumerate(rdates):
            # Nächsten Stichtag gleich bestimmen
            d_next = rdates[i + 1] if i + 1 < len(rdates) else pd.Timestamp(self.cfg.end)

            # Nur jeden n-ten Termin handeln?
            n = int(getattr(self.cfg, "rebalance_every_n", 1) or 1)
            if n > 1 and (i % n) != 0:
                # KEIN Reweighting: Kosten = 0, mit aktuellen Gewichten weiterlaufen
                trade_cost = 0.0
                equity_rows.append({"date": pd.Timestamp(d), "equity": equity_val})
                if cur_holdings:
                    ret_mask = (feats["rets"].index > d) & (feats["rets"].index <= d_next)
                    # Nur Aktien-Renditen ziehen – CASH gibt es nicht in feats["rets"]
                    stock_holdings = [t for t in cur_holdings if t != "CASH"]
                    if stock_holdings:
                        rets_sub = feats["rets"].loc[ret_mask, stock_holdings].replace([np.inf, -np.inf],
                                                                                       np.nan).fillna(0.0)
                    else:
                        rets_sub = pd.DataFrame(index=feats["rets"].index[ret_mask])

                    w_stocks = cur_weights.reindex(stock_holdings).fillna(0.0) if stock_holdings else pd.Series(
                        dtype=float)

                    # Basis-Return der Aktien
                    base_port_rets = rets_sub.dot(w_stocks) if not rets_sub.empty else pd.Series(0.0,
                                                                                                 index=rets_sub.index)

                    cash_w = float(cur_weights.get("CASH", 0.0))
                    port_rets = apply_exposures_and_cash(
                        base_port_rets,
                        regime_on, bm_s, bm_sma, self.cfg.regime_exposure_low,
                        vt_on, vt_exp,
                        cash_w, self.cfg.cash_yield_annual
                    )

                    for idx, r in port_rets.items():
                        equity_val *= (1.0 + float(r))
                        equity_rows.append({"date": pd.Timestamp(idx), "equity": equity_val})

                # Logs schreiben
                if cur_holdings:
                    tmp = pd.DataFrame({"ticker": cur_holdings})
                    tmp["allocation_pct"] = (cur_weights.reindex(cur_holdings).values * 100.0).round(2)
                    if secmap:
                        tmp["sector"] = tmp["ticker"].map(secmap).fillna("UNKNOWN")
                    tmp.insert(0, "as_of", d.strftime("%Y-%m-%d"))
                    positions_log.append(tmp)

                if self.cfg.verbose:
                    print(f"{d.date()} [SKIP n={n}] turnover=0.000 cost=0.0000 eq={equity_val:.4f}")

                if getattr(self.cfg, "dump_decision_bundles", False):
                    _asof_ts = pd.Timestamp(d)

                    def _series_to_dict(_s):
                        try:
                            return {str(k): float(v) for k, v in _s.astype(float).items()}
                        except Exception:
                            try:
                                return {str(k): float(v) for k, v in _s.items()}
                            except Exception:
                                return {}

                    _ddir = Path(getattr(self.cfg, "decisions_dir", "aktien_oop/decisions"))
                    _ddir.mkdir(parents=True, exist_ok=True)
                    _prefix = str(getattr(self.cfg, "decision_prefix", "BT"))
                    _rid = getattr(self, "_run_id", None)
                    _rid_str = _sanitize_for_fs(str(_rid)) if _rid else ""

                    # bei Skip: keine Änderungen → eff_new = aktuelle Gewichte
                    eff_new = cur_weights.copy()
                    # --- LOCKSTEP: BT-Parameter/Universe-Check (Skip-Zweig) ---
                    _cfg = self.cfg
                    _univ = list(self.tickers)  # Backtester lädt sie in __init__
                    _univ_sha = universe_hash(_univ)
                    _asof_dbg = _asof_ts.strftime("%Y-%m-%d")

                    print(
                        f"[LOCKSTEP][BT ] as_of={_asof_dbg} "
                        f"top_k={getattr(_cfg, 'top_k', None)} buffer_k={getattr(_cfg, 'buffer_k', None)} "
                        f"use_sector_limits={bool(getattr(_cfg, 'use_sector_limits', True))} "
                        f"max_per_sector={getattr(_cfg, 'max_per_sector', None)} "
                        f"score_days={int(score_days)} vol_days={int(vol_days)} adjusted={bool(adjusted)} "
                        f"universe_name={self.universe_name} universe_file={self.universe_file} "
                        f"universe_len={len(_univ)} universe_hash={_univ_sha}"
                    )

                    # --- /LOCKSTEP ---

                    print(f"[DBG][BT] write bundle as_of={_asof_ts:%Y-%m-%d} dir={_ddir} prefix={_prefix}")
                    _bt_write_decision_bundle(
                        prefix=_prefix,
                        decisions_dir=str(_ddir),
                        as_of_str=_asof_ts.strftime("%Y-%m-%d"),
                        weights=_series_to_dict(eff_new),
                        run_id=_rid_str or None,
                        top_k=int(self.cfg.top_k),
                        buffer_k=int(self.cfg.buffer_k),
                        max_per_sector=(getattr(self.cfg, "max_per_sector", None)
                                        or getattr(getattr(self.cfg, "limits", object()), "max_per_sector", None)
                                        or getattr(getattr(self.cfg, "filters", object()), "max_per_sector", None)),
                        max_active_names=int(getattr(self.cfg, "max_active_names", 0) or 0),
                        include_cash=bool(getattr(self.cfg, "include_cash", False)),
                        params={
                            "friction_eps": float(getattr(self.cfg, "friction_eps", 0.0) or 0.0),
                            "friction_eps_pct": float(getattr(self.cfg, "friction_eps_pct", 0.0) or 0.0),
                            "max_turnover_cap": float(getattr(self.cfg, "max_turnover_cap", 1.0) or 1.0),
                            "weight_round_step": float(getattr(self.cfg, "weight_round_step", 0.0) or 0.0),
                        },
                        old_weights=_series_to_dict(cur_weights),  # vor Skip == aktuell
                        new_weights=_series_to_dict(eff_new),  # nach Skip == unverändert
                        turnover_raw=0.0,
                        turnover_eff=0.0,
                        holdings=list(eff_new.sort_values(ascending=False).index),
                        universe_name=self.universe_name,
                        universe_file=self.universe_file,
                        universe_len=len(_univ),
                        universe_hash=_univ_sha,
                    )
                continue

            # Universum mit genügend Historie
            valid_cols_asof = []
            for t in px.columns:
                seg = px.loc[:d, t].dropna()
                if len(seg) >= self.cfg.min_history_days:
                    valid_cols_asof.append(t)
            if not valid_cols_asof:
                continue

            # Scoring + Filter
            # === Rebalance via Core (einheitlicher Pfad) ===

            # 1) Universe & Snapshots
            universe = list(valid_cols_asof)  # valid_cols_asof kommt direkt oberhalb aus deinem Preload
            # 1) Universe & Snapshots (STATEFUL für echten Backtest)
            _old_weights_snapshot = (
                cur_weights.to_dict()
                if isinstance(cur_weights, pd.Series) and not cur_weights.empty
                else {}
            )
            prev_holdings = [
                t for t, w in _old_weights_snapshot.items()
                if float(w) > 0.0 and t != "CASH"
            ]

            # _old_weights_snapshot = cur_weights.to_dict() if isinstance(cur_weights, pd.Series) else {}
           # prev_holdings = [t for t, w in _old_weights_snapshot.items() if float(w) > 0.0 and t != "CASH"]

            # 2) Preis-/Sektor-Callbacks für Core
            def _get_prices_bt(tickers, as_of, period, adjusted=True):
                """
                Vom Core aufgerufen als get_prices(universe, p.as_of, p.period, p.adjusted)
                → as_of: Enddatum (YYYY-MM-DD), period: Lookback z.B. '800d'
                """
                _idx = prices_all.index
                _asof = pd.Timestamp(as_of)
                _days = _period_to_days(period)  # '800d' → 800
                _start = _asof - pd.Timedelta(days=_days)

                # auf vorhandene Indizes klemmen
                _start = _idx[_idx.get_indexer([_start], method="backfill")[0]]
                _end = _idx[_idx.get_indexer([_asof], method="pad")[0]]

                frame = prices_all.loc[_start:_end, list(tickers)].copy()
                return frame

            def _to_ts(x):
                return pd.Timestamp(x).normalize() if x is not None else None

            def _get_sectors_bt(tickers):
                return {t: secmap.get(t, "UNKNOWN") for t in tickers}

            # 3) CalcParams aus cfg
            _asof_str = d.strftime("%Y-%m-%d")

            cp = CalcParams(
                as_of=_asof_str,
                period=getattr(self.cfg, "period", "800D"),
                adjusted=bool(adjusted),
                score_days=int(score_days),
                vol_days=int(vol_days),

                # Filter
                use_under_sma=bool(getattr(self.cfg, "use_under_sma", False)),
                sma_days=int(getattr(self.cfg, "sma_days", 200)),
                gap_filter=float(self.cfg.gap_filter),
                min_price=float(getattr(self.cfg, "min_price", 0.0)),
                min_volume=float(getattr(self.cfg, "min_volume", 0.0)),

                # Limits & Auswahl
                use_sector_limits=bool(use_sector_limits),
                max_per_sector=int(self.cfg.max_per_sector),
                top_k=int(self.cfg.top_k),
                buffer_k=int(self.cfg.buffer_k),
                max_active_names=int(getattr(self.cfg, "max_active_names", 0) or 0),

                # Finalisierung
                include_cash=bool(getattr(self.cfg, "include_cash", False)),
                weight_round_step=float(getattr(self.cfg, "weight_round_step", 0.0)),
                max_turnover_cap=float(getattr(self.cfg, "max_turnover_cap", 1.0)),
                friction_eps=float(getattr(self.cfg, "friction_eps", 0.0)),
                friction_eps_pct=float(getattr(self.cfg, "friction_eps_pct", 0.0)),

                dump_scores=True,
                dump_selection=bool(getattr(self.cfg, "dump_selection", False)),
                dump_weights=bool(getattr(self.cfg, "dump_weights", False)),
                dump_tag="BT",
            )

            decision = self.data.regime_decision(self.cfg, _asof_str)
            print(f"[DBG][REGIME] as_of={_asof_str} decision={decision}")
            if not decision["ok"]:
                if decision["action"] == "HOLD":
                    new_w_dict = _old_weights_snapshot.copy()
                    scores_ser = None  # kann None bleiben; optional
                    # dann Turnover=0, trade_cost=0, usw. (oder einfach im Turnover-Block rauskommt)
                elif decision["action"] == "SELL":
                    if bool(getattr(self.cfg, "include_cash", False)):
                        new_w_dict = {"CASH": 1.0}
                    else:
                        new_w_dict = {}
                # und dann NICHT calculate_portfolio callen
            else:
                # 4) Core-Call (Selektion/Ranks/Scores)
                new_w_dict, scores_ser = calculate_portfolio(
                    universe,
                    cp,
                    _get_prices_bt,
                    _get_sectors_bt,
                    prev_holdings=prev_holdings,
                    prev_weights=_old_weights_snapshot,
                )

            self._dbg(
                f"[BT/DBG] core_new_w_len={len(new_w_dict)} "
                f"tickers={sorted(new_w_dict.keys())[:15]}"
            )

            # >>> DEBUG-DUMP NUR FÜR LOCKSTEP-TAG 2025-10-08
            if d.strftime("%Y-%m-%d") == "2025-10-08":
                self._dbg(f"[BT/DBG] as_of={d} universe_len={len(universe)} examples={list(universe)[:10]}")
                self._dbg(
                    f"[BT/DBG] scores_nonNa={scores_ser.dropna().sort_values(ascending=False).head(15).to_dict()}")
                self._dbg(f"[BT/DBG] new_w_dict={dict(sorted(new_w_dict.items(), key=lambda x: -x[1])[:15])}")
            # <<< DEBUG-DUMP ENDE

            # 5) Turnover/Cost EINMAL (alt vs. neu) — vor cur_weights-Update!
            raw_turnover = _calc_turnover(_old_weights_snapshot, new_w_dict)

            _cap = float(self.cfg.max_turnover_cap or 0.0)
            turnover_eff = min(raw_turnover, _cap) if _cap > 0.0 else raw_turnover

            _cost_bps = float(getattr(self.cfg, "cost_bps", 0.0) or 0.0)
            _slip_bps = float(getattr(self.cfg, "slippage_bps", 0.0) or 0.0)
            trade_cost = turnover_eff * ((_cost_bps + _slip_bps) / 10000.0)

            # 6) cur_weights setzen (nach Turnover-Berechnung)
            cur_weights = pd.Series(new_w_dict, dtype=float)
            cur_weights = cur_weights.sort_index()
            cur_holdings = list(cur_weights.index)

            # Renditefenster d -> d_next für aktuell gewählte Aktien
            ret_mask = (feats["rets"].index > d) & (feats["rets"].index <= d_next)
            stock_holdings = [t for t in cur_holdings if t != "CASH"]

            if stock_holdings:
                rets_sub = feats["rets"].loc[ret_mask, stock_holdings].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                rets_sub = pd.DataFrame(index=feats["rets"].index[ret_mask])

            # 7) Rankings/Scores/Vol fürs Bundle
            try:
                # Holdings aus den neuen Gewichten
                cur_holdings = list(new_w_dict.keys())

                # Scores aus dem Core-Call auf genau diese Holdings abbilden
                _scores = None
                try:
                    if scores_ser is not None and len(scores_ser) > 0:
                        _scores = scores_ser.reindex(cur_holdings).astype(float)
                except Exception:
                    _scores = None

                # Dicts robust bauen
                if _scores is not None:
                    scores_dict = _scores.round(6).to_dict()
                    _scores.index = _scores.index.astype(str)
                    ranks_dict = _rank_desc_stable(_scores).to_dict()
                else:
                    scores_dict = {t: float("nan") for t in cur_holdings}
                    ranks_dict = {t: int(1) for t in cur_holdings}  # Fallback
                # Vol kann später ergänzt werden
                vol_dict = {}

                # Filter-Stats für Debug/Prints defensiv bereitstellen
                filt_stats = locals().get("filt_stats", {})

            except Exception:
                sel_idx = None

            # Kosten einmalig am Rebalance-Tag buchen
            equity_val *= (1.0 - trade_cost)
            equity_rows.append({"date": pd.Timestamp(d), "equity": equity_val})

            # --- Trades-Log (nur Rebalance-Tag) ---
            _old_ws = _old_weights_snapshot
            _new_ws = new_w_dict
            old_set = {t for t, w in _old_ws.items() if t != "CASH" and float(w) > 0.0}
            new_set = {t for t, w in _new_ws.items() if t != "CASH" and float(w) > 0.0}
            date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            turn_for_log = float(locals().get("turnover_eff", locals().get("turnover", 0.0)))

            trades_log.append({
                "date": date_str,
                "turnover": turn_for_log,
                "trade_cost": float(trade_cost),
                "enter": ",".join(sorted(new_set - old_set)),
                "exit": ",".join(sorted(old_set - new_set)),
            })

            if getattr(self.cfg, "verbose", False):
                print(f"{date_str} [REBAL] turnover={turn_for_log:.3%} cost={float(trade_cost):.4f} "
                      f"enter=[{','.join(sorted(new_set - old_set))}] exit=[{','.join(sorted(old_set - new_set))}]")

            # saubere, label-sichere Gewichtung
            w_stocks = (cur_weights.reindex(stock_holdings).fillna(0.0) if stock_holdings
                        else pd.Series(dtype=float))
            rets_sub = rets_sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Basis-Return der Aktien
            base_port_rets = (rets_sub.dot(w_stocks) if not rets_sub.empty
                              else pd.Series(0.0, index=rets_sub.index))

            cash_w = float(cur_weights.get("CASH", 0.0))
            port_rets = apply_exposures_and_cash(
                base_port_rets,
                regime_on, bm_s, bm_sma, self.cfg.regime_exposure_low,
                vt_on, vt_exp,
                cash_w, self.cfg.cash_yield_annual
            )

            for idx, r in port_rets.items():
                    equity_val *= (1.0 + float(r))
                    equity_rows.append({"date": pd.Timestamp(idx), "equity": equity_val})

            # Quelle: new_w_dict (Gewichte), scores_dict, ranks_dict
            pos_rows = []
            for t, w in new_w_dict.items():
                pos_rows.append({
                    "ticker": t,
                    "weight": float(w),
                    "score": float(scores_dict.get(t, float("nan"))) if scores_dict else float("nan"),
                    "rank": int(ranks_dict.get(t, 0)) if ranks_dict else 0,
                })

            tmp = pd.DataFrame(pos_rows)
            # as_of-Spalte vorn einfügen
            tmp.insert(0, "as_of", pd.Timestamp(d).strftime("%Y-%m-%d"))

            # Sektoren zuordnen (falls secmap existiert)
            if secmap:
                tmp["sector"] = tmp["ticker"].map(secmap).fillna("UNKNOWN")
            else:
                tmp["sector"] = "UNKNOWN"

            # CASH sauber labeln
            tmp.loc[tmp["ticker"] == "CASH", "sector"] = "CASH"

            # optional schön sortieren (erst Gewicht, dann Rank)
            # tmp = tmp.sort_values(by=["weight", "rank"], ascending=[False, True], ignore_index=True)

            positions_log.append(tmp)

            # Debug-Ausgabe (self._dbg checkt verbose selbst)
            filt_stats = locals().get("filt_stats", {"gap": None, "under_sma": None})
            self._dbg(
                f"{d.date()} holdings={cur_holdings} turnover_eff={turn_for_log:.3f} "
                f"cost={trade_cost:.4f} eq={equity_val:.4f} filt={filt_stats}"
            )

            # --- Decision-Bundle (pro Stichtag) ---
            if getattr(self.cfg, "dump_decision_bundles", False):
                # As-of: wir sind im Rebalance-Loop -> d ist gesetzt
                _asof_ts = pd.Timestamp(d)

                # Helfer: Series -> dict[str -> float]
                def _series_to_dict(_s):
                    try:
                        return {str(k): float(v) for k, v in _s.astype(float).items()}
                    except Exception:
                        try:
                            return {str(k): float(v) for k, v in _s.items()}
                        except Exception:
                            return {}

                # Roh-/Eff-Turnover robust aus locals
                _t_raw = float(locals().get("raw_turnover", float("nan")))
                _t_eff = float(locals().get("turnover_eff", float("nan")))

                # Alte Gewichte als Series für das Bundle (vor dem Rebalance)
                try:
                    _old_ws_dict = _old_weights_snapshot if isinstance(_old_weights_snapshot, dict) else {}
                except NameError:
                    _old_ws_dict = {}
                old_sorted = pd.Series(_old_ws_dict, dtype=float)
                old_sorted = old_sorted.reindex(sorted(old_sorted.index)).fillna(0.0).sort_values(ascending=False)

                # finale Gewichte als sortierte Series fürs Bundle
                eff_new = cur_weights.sort_values(ascending=False)

                bundle = {
                    "run_id": getattr(self, "_run_id", None),
                    "as_of": _asof_ts.strftime("%Y-%m-%d"),
                    "top_k": int(self.cfg.top_k),
                    "buffer_k": int(self.cfg.buffer_k),
                    "max_per_sector": getattr(self.cfg, "max_per_sector", None)
                        or getattr(getattr(self.cfg, "limits", object()), "max_per_sector", None)
                        or getattr(getattr(self.cfg, "filters", object()), "max_per_sector", None),
                    "max_active_names": int(getattr(self.cfg, "max_active_names", 0) or 0),
                    "include_cash": bool(getattr(self.cfg, "include_cash", False)),
                    "params": {
                        "friction_eps": float(getattr(self.cfg, "friction_eps", 0.0) or 0.0),
                        "friction_eps_pct": float(getattr(self.cfg, "friction_eps_pct", 0.0) or 0.0),
                        "max_turnover_cap": float(getattr(self.cfg, "max_turnover_cap", 1.0) or 1.0),
                        "weight_round_step": float(getattr(self.cfg, "weight_round_step", 0.0) or 0.0),
                    },
                    # diese Variablen sind an der Stelle vorhanden:
                    "old_weights": _series_to_dict(old_sorted),  # vor dem Rebalance
                    "new_weights": _series_to_dict(eff_new),  # finale Gewichte
                    "turnover_raw": _t_raw,
                    "turnover_eff": _t_eff,
                    "holdings": list(eff_new.sort_values(ascending=False).index),
                }

                _ddir = Path(getattr(self.cfg, "decisions_dir", "aktien_oop/decisions"))
                _ddir.mkdir(parents=True, exist_ok=True)
                _prefix = str(getattr(self.cfg, "decision_prefix", "BT"))
                _rid = getattr(self, "_run_id", None)
                _rid_str = _sanitize_for_fs(str(_rid)) if _rid else ""
                _ts = datetime.now().strftime("%y%m%d_%H%M%S")

                # Variante A (kurz): run_id falls vorhanden, sonst ts
                _core = f"{_rid}_{_ts}" if _rid else _ts
                # --- LOCKSTEP: BT-Parameter/Universe-Check (Trade-Zweig) ---
                _cfg = self.cfg
                _univ = list(self.tickers)
                _univ_sha = universe_hash(_univ)
                _asof_dbg = _asof_ts.strftime("%Y-%m-%d")

                print(
                    f"[LOCKSTEP][BT ] as_of={_asof_dbg} "
                    f"top_k={getattr(_cfg, 'top_k', None)} buffer_k={getattr(_cfg, 'buffer_k', None)} "
                    f"use_sector_limits={bool(getattr(_cfg, 'use_sector_limits', True))} "
                    f"max_per_sector={getattr(_cfg, 'max_per_sector', None)} "
                    f"score_days={int(score_days)} vol_days={int(vol_days)} adjusted={bool(adjusted)} "
                    f"universe_name={self.universe_name} universe_file={self.universe_file} "
                    f"universe_len={len(_univ)} universe_hash={_univ_sha}"
                )
                # --- /LOCKSTEP ---

                _bt_write_decision_bundle(
                    prefix=_prefix,
                    decisions_dir=str(_ddir),
                    as_of_str=_asof_ts.strftime("%Y-%m-%d"),
                    weights=_series_to_dict(eff_new),
                    ranks=(ranks_dict if isinstance(ranks_dict, dict) and ranks_dict else None),
                    scores=(scores_dict if isinstance(scores_dict, dict) and scores_dict else None),
                    vol=(vol_dict if isinstance(vol_dict, dict) and vol_dict else None),
                    run_id=_rid_str or None,
                    # alles darunter sind deine bisherigen Zusatzfelder aus "bundle":
                    top_k=int(self.cfg.top_k),
                    buffer_k=int(self.cfg.buffer_k),
                    max_per_sector=(getattr(self.cfg, "max_per_sector", None)
                                    or getattr(getattr(self.cfg, "limits", object()), "max_per_sector", None)
                                    or getattr(getattr(self.cfg, "filters", object()), "max_per_sector", None)),
                    max_active_names=int(getattr(self.cfg, "max_active_names", 0) or 0),
                    include_cash=bool(getattr(self.cfg, "include_cash", False)),
                    params={
                        "friction_eps": float(getattr(self.cfg, "friction_eps", 0.0) or 0.0),
                        "friction_eps_pct": float(getattr(self.cfg, "friction_eps_pct", 0.0) or 0.0),
                        "max_turnover_cap": float(getattr(self.cfg, "max_turnover_cap", 1.0) or 1.0),
                        "weight_round_step": float(getattr(self.cfg, "weight_round_step", 0.0) or 0.0),
                    },
                    old_weights=_series_to_dict(old_sorted),
                    new_weights=_series_to_dict(eff_new),
                    turnover_raw=float(locals().get("raw_turnover", float("nan"))),
                    turnover_eff=float(locals().get("turnover_eff", float("nan"))),
                    holdings=list(eff_new.sort_values(ascending=False).index),
                    universe_name=self.universe_name,
                    universe_file=self.universe_file,
                    universe_len=len(_univ),
                    universe_hash=_univ_sha,
                )

        # === POST-LOOP: Equity/Benchmark/Summary/META (einmalig) ===
        # Equity-Kurve robust bauen (aus equity_rows)
        if equity_rows:
            eq_df = pd.DataFrame(equity_rows)
            eq_df["date"] = pd.to_datetime(eq_df["date"]).dt.tz_localize(None)
            eq_df = eq_df.sort_values("date").drop_duplicates("date")
            eq = eq_df.set_index("date")["equity"].astype(float)
        else:
            print("Keine Equity-Curve erzeugt (möglicherweise kein Rebalance-Intervall mit Daten).")
            return

        # Basismetriken (für Summary & META)
        total_return = float(eq.iloc[-1] - 1.0)
        days = max(1, (eq.index[-1] - eq.index[0]).days)
        cagr = float(eq.iloc[-1] ** (DAYS_PER_YEAR / days) - 1.0)
        port_rets = eq.pct_change().fillna(0.0)
        vol_ann = float(port_rets.std() * np.sqrt(TRADING_DAYS))
        sharpe = float((port_rets.mean() * TRADING_DAYS) / (vol_ann + EPS))
        mdd, dd_start, dd_end = calc_drawdown(eq)
        avg_turnover = float(np.mean([t["turnover"] for t in trades_log]) if trades_log else 0.0)
        avg_cost     = float(np.mean([t["trade_cost"] for t in trades_log]) if trades_log else 0.0)

        print(f"Total Return: {total_return:7.2%}   |  CAGR: {cagr:7.2%}")
        print(f"Volatility:   {vol_ann:6.2%} |  Sharpe(0%): {sharpe:4.2f}")
        print(f"Max DD:       {mdd:7.2%}   [{dd_start.date()} -> {dd_end.date()}]")
        print(f"Avg Turnover: {avg_turnover:6.2%} |  Avg Cost: {avg_cost:6.4f}")

        # CSVs (ggf. erneut/erstmalig) schreiben – einheitlich inkl. run_id
        _to_csv_with_runid(eq_path,  eq_df.set_index("date"), index=True,  run_id=getattr(self, "_run_id", None))
        _to_csv_with_runid(pos_path, pd.concat(positions_log, ignore_index=True) if positions_log else pd.DataFrame(),
                           index=False, run_id=getattr(self, "_run_id", None))
        _to_csv_with_runid(trd_path, pd.DataFrame(trades_log), index=False, run_id=getattr(self, "_run_id", None))

        # Einmalige Benchmark-Helferfunktion (außerhalb der Schleife)
        def _bench_metrics(ticker: str, eq_series: pd.Series):
            """
            Lädt Benchmark-Schlusskurse für 'ticker' im Zeitfenster von eq_series,
            normiert auf 1.0 am Start und liefert Series + Kennzahlen zurück.
            Robust ggü. download_close-Varianten (Series/DataFrame/MultiIndex/Dict/Tuple).
            """
            # 1) Fenster
            start_dt = pd.Timestamp(eq_series.index.min()).normalize()
            end_dt   = pd.Timestamp(eq_series.index.max()).normalize()
            start_s, end_s = str(start_dt.date()), str(end_dt.date())

            # 2) Download immer als Liste aufrufen (sonst wird "SPY" zu ["S","P","Y"])
            bm_px = download_close([ticker], start=start_s, end=end_s)

            # 3) Rückgabe-Typen entpacken
            if isinstance(bm_px, tuple) and len(bm_px) >= 1:
                bm_px = bm_px[0]
            if isinstance(bm_px, dict):
                bm_px = bm_px.get(ticker, list(bm_px.values())[0])

            # 4) Auf Series normalisieren (Close/Adj Close wählen)
            if isinstance(bm_px, pd.DataFrame):
                df = bm_px
                # MultiIndex-Columns abwickeln
                if isinstance(df.columns, pd.MultiIndex):
                    s = None
                    try:
                        if 'Close' in df.columns.get_level_values(0):
                            close_block = df.xs('Close', axis=1, level=0)
                            s = close_block[ticker] if ticker in getattr(close_block, "columns", []) else close_block.iloc[:, 0]
                    except Exception:
                        pass
                    if s is None:
                        try:
                            if 'Adj Close' in df.columns.get_level_values(0):
                                adj_block = df.xs('Adj Close', axis=1, level=0)
                                s = adj_block[ticker] if ticker in getattr(adj_block, "columns", []) else adj_block.iloc[:, 0]
                        except Exception:
                            pass
                    bm_s = s if isinstance(s, pd.Series) else df.iloc[:, 0]
                else:
                    # Single-Level: prefer Adj Close/Close heuristisch
                    cols = list(df.columns)
                    pick = None
                    for cand in ["Adj Close", "Close", ticker]:
                        if cand in cols:
                            pick = cand; break
                    bm_s = df[pick] if pick in cols else df.iloc[:, 0]
            else:
                bm_s = pd.Series(bm_px)

            bm_s.index = pd.to_datetime(bm_s.index).tz_localize(None)
            bm_s = bm_s.sort_index().dropna()
            # Auf eq-Fenster trimmen
            bm_loc = bm_s.reindex(eq_series.index).ffill().dropna()
            if bm_loc.empty:
                return {"series": pd.Series([], dtype=float), "total": None, "cagr": None, "vol": None, "sharpe": None}

            bm_eq_loc = (bm_loc / float(bm_loc.iloc[0])).astype(float)
            bm_rets   = bm_eq_loc.pct_change().fillna(0.0)
            bm_total  = float(bm_eq_loc.iloc[-1] - 1.0)
            dt_days   = max(1, (bm_eq_loc.index[-1] - bm_eq_loc.index[0]).days)
            bm_cagr   = float(bm_eq_loc.iloc[-1] ** (DAYS_PER_YEAR / dt_days) - 1.0)
            bm_vol_ann= float(bm_rets.std() * np.sqrt(TRADING_DAYS))
            bm_sharpe = float((bm_rets.mean() * TRADING_DAYS) / (bm_vol_ann + EPS))
            return {"series": bm_eq_loc, "total": bm_total, "cagr": bm_cagr, "vol": bm_vol_ann, "sharpe": bm_sharpe}

        # Benchmark 1 (obligatorisch)
        bm_ticker = str(getattr(self.cfg, "benchmark", getattr(self.cfg, "benchmark_ticker", "SXR8.DE")))
        bm1 = _bench_metrics(bm_ticker, eq)
        bm_eq = bm1["series"].rename("BM1_" + bm_ticker)

        # Alpha/Rel.Vol/Corr gegen BM1
        bm1_rets = bm1["series"].pct_change().fillna(0.0)
        corr   = float(np.corrcoef(port_rets, bm1_rets)[0, 1])
        rel_vol = float(port_rets.std() / (bm1_rets.std() + EPS))
        alpha = float(cagr - bm1["cagr"])

        # Benchmark CSV schreiben
        bm_path = Path(str(out_prefix) + "_benchmark.csv")
        bm_df_legacy = pd.concat([eq, bm_eq], axis=1)
        _to_csv_with_runid(bm_path, bm_df_legacy, index=True, run_id=getattr(self, "_run_id", None))

        # Optionaler BM2
        bm2_label = str(getattr(self.cfg, "benchmark2", "") or "").strip()
        bm2_enabled = bool(bm2_label) or bool(getattr(self.cfg, "dual_benchmark", False))
        self._dbg(f"[BM] dual_benchmark={getattr(self.cfg,'dual_benchmark',False)} bm1={bm_ticker} bm2='{bm2_label}' -> bm2_enabled={bm2_enabled}")

        bm2_total = bm2_cagr = bm2_vol_ann = bm2_sharpe = alpha2 = corr2 = rel_vol2 = None
        if bm2_enabled and bm2_label:
            bm2 = _bench_metrics(bm2_label, eq)
            bm2_rets = bm2["series"].pct_change().fillna(0.0)
            corr2   = float(np.corrcoef(port_rets, bm2_rets)[0, 1])
            rel_vol2 = float(port_rets.std() / (bm2_rets.std() + EPS))
            alpha2 = float(cagr - bm2["cagr"])
            bm2_total, bm2_cagr, bm2_vol_ann, bm2_sharpe = bm2["total"], bm2["cagr"], bm2["vol"], bm2["sharpe"]

        # Einmalige Prints
        print(f"Portfolio Total Ret: {total_return:7.2%} | Port CAGR: {cagr:7.2%}")
        print(f"Benchmark:     {bm_ticker}")
        print(f"BM Total Ret:   {bm1['total']:7.2%}   |  BM CAGR:  {bm1['cagr']:7.2%}")
        print(f"BM Volatility:  {bm1['vol']:6.2%} |  BM Sharpe(0%):  {bm1['sharpe']:4.2f}")
        print(f"Alpha (ann.):   {alpha:7.2%}     |  Corr(EQ,BM):  {corr:4.2f}")
        print(f"Rel. Vol (Port/BM):  {rel_vol:4.2f}x")
        if bm2_enabled and bm2_label and bm2_total is not None:
            print(f"Benchmark 2:   {bm2_label}")
            print(f"BM2 Total Ret:  {bm2_total:7.2%}   |  BM2 CAGR:  {bm2_cagr:7.2%}")
            print(f"BM2 Volatility: {bm2_vol_ann:6.2%} |  BM2 Sharpe(0%):  {bm2_sharpe:4.2f}")
            print(f"BM2 Alpha (ann.): {alpha2:7.2%}     |  Corr(EQ,BM2):  {corr2:4.2f}")
            print(f"BM2 Rel. Vol (Port/BM2):  {rel_vol2:4.2f}x")

        # Summary-Text (neu aufbauen; summary_lines könnte zuvor nicht existiert haben)
        summary_lines = [
            f"Total Return: {total_return:7.2%}   |  CAGR: {cagr:7.2%}",
            f"Volatility:   {vol_ann:6.2%} |  Sharpe(0%): {sharpe:4.2f}",
            f"Max DD:       {mdd:7.2%}   [{dd_start.date()} -> {dd_end.date()}]",
            f"Avg Turnover: {avg_turnover:6.2%} |  Avg Cost: {avg_cost:6.4f}",
            "",
            f"Benchmark:     {bm_ticker}",
            f"BM Total Ret:   {bm1['total']:7.2%}   |  BM CAGR:  {bm1['cagr']:7.2%}",
            f"BM Volatility:  {bm1['vol']:6.2%} |  BM Sharpe(0%):  {bm1['sharpe']:4.2f}",
            f"Alpha (ann.):   {alpha:7.2%}     |  Corr(EQ,BM):  {corr:4.2f}",
            f"Rel. Vol (Port/BM):  {rel_vol:4.2f}x",
        ]
        if bm2_enabled and bm2_label and bm2_total is not None:
            summary_lines += [
                f"Benchmark 2:   {bm2_label}",
                f"BM2 Total Ret:  {bm2_total:7.2%}   |  BM2 CAGR:  {bm2_cagr:7.2%}",
                f"BM2 Volatility: {bm2_vol_ann:6.2%} |  BM2 Sharpe(0%):  {bm2_sharpe:4.2f}",
                f"BM2 Alpha (ann.): {alpha2:7.2%}     |  Corr(EQ,BM2):  {corr2:4.2f}",
                f"BM2 Rel. Vol (Port/BM2):  {rel_vol2:4.2f}x",
            ]
        summary_lines += [
            f"Equity Curve: {eq_path}",
            f"Positions:    {pos_path}",
            f"Trades:       {trd_path}",
            f"Benchmark CSV:{bm_path}",
        ]
        txt = "\n".join(summary_lines)
        if hasattr(self, "_run_id"):
            txt = f"# run_id={self._run_id}\n" + txt
        Path(summ_path).write_text(txt, encoding="utf-8")

        # META am Ende (konsolidiert)
        try:
            if hasattr(self, "_meta_path") and self._meta_path:
                iso_end, _ = _now_local_str("Europe/Berlin")
                end_meta = {
                    "finished_at": iso_end,
                    "summary": {
                        "total_return": float(total_return),
                        "cagr": float(cagr),
                        "volatility": float(vol_ann),
                        "sharpe": float(sharpe),
                        "max_drawdown": float(mdd),
                        "max_drawdown_start": str(dd_start),
                        "max_drawdown_end": str(dd_end),
                        "avg_turnover": float(avg_turnover),
                        "avg_cost": float(avg_cost),
                        "bm_total": float(bm1['total']),
                        "bm_cagr": float(bm1['cagr']),
                        "bm_volatility": float(bm1['vol']),
                        "bm_sharpe": float(bm1['sharpe']),
                        **({
                            "bm2_total": float(bm2_total),
                            "bm2_cagr": float(bm2_cagr),
                            "bm2_volatility": float(bm2_vol_ann),
                            "bm2_sharpe": float(bm2_sharpe)
                        } if (bm2_enabled and bm2_label and bm2_total is not None) else {})
                    },
                }
                p = Path(self._meta_path)
                base = json.loads(p.read_text(encoding="utf-8"))
                base.update(end_meta)
                _dump_run_meta(p, base)
                print(f"[META] updated {p}")
        except Exception as e:
            print(f"[META] update failed: {e}")
        # === /POST-LOOP ===
        # Params-Snapshot (einmalig, Post-Loop)
        Path(str(out_prefix) + "_params.json").write_text(
            json.dumps(vars(self.cfg), indent=2, default=str), encoding="utf-8"
        )

        # Kurze Abschlussmeldung mit Pfaden
        print(f"Equity:   {eq_path}")
        print(f"Positions:{pos_path}")
        print(f"Trades:   {trd_path}")
        print(f"Bench:    {bm_path}")
        print(f"Summary:  {summ_path}")


def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bringt yfinance-Frames sicher auf Single-Level-Spalten mit Tickernamen.
    Bevorzugt 'Adj Close', sonst 'Close'. Entfernt ggf. übrige Level.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 1) Versuche, eine Ebene 'Adj Close' oder 'Close' zu picken – egal auf welchem Level
        picked = False
        for lvl in range(df.columns.nlevels):
            vals = df.columns.get_level_values(lvl)
            if "Adj Close" in vals:
                df = df.xs("Adj Close", axis=1, level=lvl, drop_level=True)
                picked = True
                break
            if "Close" in vals:
                df = df.xs("Close", axis=1, level=lvl, drop_level=True)
                picked = True
                break
        # 2) Falls immer noch MultiIndex (z. B. ('Ticker','something')), letzte Ebene als Ticker behalten
        if isinstance(df.columns, pd.MultiIndex):
            # nimm die letzte Ebene als Spalten (Tickerebene) und drople die anderen
            keep_level = df.columns.nlevels - 1
            drop_levels = [i for i in range(df.columns.nlevels) if i != keep_level]
            df = df.droplevel(drop_levels, axis=1)
    # 3) Strings + Dedupe
    df.columns = df.columns.astype(str)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _now_local_str(tz_name: str = "Europe/Berlin"):
    """Returns (iso_string, run_id_string) in local tz if available."""
    if ZoneInfo is not None:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        iso = now.strftime("%Y-%m-%d %H:%M:%S%z")
    else:
        now = datetime.now()
        iso = now.strftime("%Y-%m-%d %H:%M:%S")
    run_id = now.strftime("%Y%m%d_%H%M%S")
    return iso, run_id

def _to_jsonable(obj):
    """Make dataclass / numpy / pandas types JSON-serializable."""
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # pandas / numpy fallbacks
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if obj is np.nan:
            return None
    except Exception:
        pass
    try:
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, (pd.Series,)):
            return obj.astype(float).to_dict()
        if isinstance(obj, (pd.Index,)):
            return obj.astype(str).tolist()
        if isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="list")
    except Exception:
        pass
    return obj

def _dump_run_meta(path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _cash_daily_return(annual: float) -> float:
    a = float(annual)
    return (1.0 + a) ** (1.0 / TRADING_DAYS) - 1.0 if a > 0.0 else 0.0


# ---------------------------------------
# Config laden
# ---------------------------------------
def _load_toml(path: str) -> dict:
    if not path:
        return {}
    if tomllib is None:
        raise RuntimeError("tomllib nicht verfügbar (Python < 3.11). Bitte Python 3.11+ nutzen oder per CLI-Flags konfigurieren.")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _first_not_none(*vals):
    """Gibt den ersten Wert zurück, der nicht None ist (ansonsten None)."""
    for v in vals:
        if v is not None:
            return v
    return None

def _build_cfg_from_config_and_cli(a: argparse.Namespace) -> BTConfig:
    cfg_toml = _load_toml(a.config) if a.config else {}

    # --- Fenster aus [windows] lesen, falls vorhanden ---
    win = cfg_toml.get("windows") or {}
    if isinstance(win, dict):
        win_score_days = win.get("score_days")
        win_vol_days = win.get("vol_days")
    else:
        win_score_days = getattr(win, "score_days", None)
        win_vol_days = getattr(win, "vol_days", None)

    # CLI-False (store_true default) nicht TOML überstimmen lassen:
    cli_dual = getattr(a, "dual_benchmark", None)
    if cli_dual is False:  # Flag nicht gesetzt → wie "None" behandeln
        cli_dual = None

    # --- Normalize benchmark keys from CLI/TOML (accept - and _ variants) ---
    bm_primary = _first_not_none(
        getattr(a, "benchmark", None),
        getattr(a, "benchmark_ticker", None),
        cfg_toml.get("benchmark"),
        cfg_toml.get("benchmark_ticker"),
        cfg_toml.get("benchmark-ticker"),
        "SXR8.DE",
    )

    bm_dual = _first_not_none(
        cli_dual,
        cfg_toml.get("dual_benchmark"),
        cfg_toml.get("dual-benchmark"),
        False,
    )

    bm_secondary = _first_not_none(
        getattr(a, "benchmark2", None),
        cfg_toml.get("benchmark2"),
        cfg_toml.get("benchmark_2"),
        cfg_toml.get("benchmark-2"),
        "",
    )

    cli_dump = getattr(a, "dump_decisions", None)
    if cli_dump is False:  # Flag nicht gesetzt → wie None behandeln
        cli_dump = None

    dump_bundles = _first_not_none(
        getattr(a, "dump_decisions", None),
        cfg_toml.get("dump_decision_bundles"),
        cfg_toml.get("dump-decisions"),
        False,
    )
    dump_selection = _first_not_none(
        getattr(a, "dump_selection", None),
        cfg_toml.get("dump_selection"),
        cfg_toml.get("dump-selection"),
        False,
    )
    dump_weights = _first_not_none(
        getattr(a, "dump_weights", None),
        cfg_toml.get("dump_weights"),
        cfg_toml.get("dump-weights"),
        False,
    )
    dec_dir = _first_not_none(
        getattr(a, "decisions_dir", None),
        cfg_toml.get("decisions_dir"),
        cfg_toml.get("decisions-dir"),
        "aktien_oop/decisions",
    )

    regime_cfg = cfg_toml.get("regime") or {}
    limits_cfg = cfg_toml.get("limits") or {}
    universe_cfg = cfg_toml.get("universe") or {}

    raw = (
            regime_cfg.get("regime_below_action")
            or regime_cfg.get("regime-below-action")
            or "HOLD"
    )
    act = str(raw).strip().upper()
    if act not in ("HOLD", "SELL"):
        act = "HOLD"

    return BTConfig(
        tickers_file     = _coalesce(a.tickers,           universe_cfg.get("tickers_file"), cfg_toml.get("tickers_file"), "aktien_oop/sp500_tickers.txt"),
        sector_meta      = _coalesce(a.sector_meta,       universe_cfg.get("meta_file"), cfg_toml.get("meta_file"), cfg_toml.get("sector_meta"), None),
        save_dir         = _coalesce(a.save_dir,          cfg_toml.get("save_dir"),       "aktien_oop"),

        start            = _coalesce(a.start,             cfg_toml.get("start"),          "2018-01-01"),
        end              = _coalesce(a.end,               cfg_toml.get("end"),            "2025-08-16"),
        frequency        = _coalesce(a.frequency,         cfg_toml.get("frequency"),      "monthly"),

        # as_of=_coalesce(a.start, cfg_toml.get("as_of"), None),
        as_of=_coalesce(getattr(a, "as_of", None), cfg_toml.get("as_of"), None),

        top_k            = _coalesce(a.top_k,             cfg_toml.get("top_k"),          8),
        buffer_k         = _coalesce(a.buffer_k,          cfg_toml.get("buffer_k"),       2),
        use_sector_limits=bool(_coalesce(
            getattr(a, "use_sector_limits", None),
            (limits_cfg.get("use_sector_limits") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "use_sector_limits", None)),
            cfg_toml.get("use_sector_limits"),
            True,
        )),
        max_per_sector   = _coalesce(
            a.max_per_sector,
            (limits_cfg.get("max_per_sector") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "max_per_sector", None)),
            cfg_toml.get("max_per_sector"),
            None,
        ),
        cost_bps         = _coalesce(a.cost_bps,          cfg_toml.get("cost_bps"),       10.0),
        slippage_bps     = _coalesce(a.slippage_bps,      cfg_toml.get("slippage_bps"),   5.0),
        min_history_days = _coalesce(a.min_history_days,  cfg_toml.get("min_history_days"), 260),

        use_equal_weight=_coalesce(a.use_equal_weight if a.use_equal_weight else None, cfg_toml.get("use_equal_weight"), False),
        gap_filter=float(_coalesce(
            getattr(a, "gap_filter", None),
            (limits_cfg.get("gap_filter") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "gap_filter", None)),
            cfg_toml.get("gap_filter"),
            0.0,
        ) or 0.0),
        friction_eps=_coalesce(
            a.friction_eps,
            (limits_cfg.get("friction_eps") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "friction_eps", None)),
            cfg_toml.get("friction_eps"),
            0.0,
        ),
        friction_eps_pct=_coalesce(
            getattr(a, "friction_eps_pct", None),
            cfg_toml.get("friction_eps_pct"),
            0.0
        ),
        weight_round_step=_coalesce(
            getattr(a, "weight_round_step", None),
            (limits_cfg.get("weight_round_step") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "weight_round_step", None)),
            cfg_toml.get("weight_round_step"), 0.0),
        max_turnover_cap=_coalesce(getattr(a, "max_turnover_cap", None), cfg_toml.get("max_turnover_cap"), 0.0),
        rebalance_every_n=_coalesce(getattr(a, "rebalance_every_n", None), cfg_toml.get("rebalance_every_n"), 1),
        # --- Normalize benchmark keys from CLI/TOML (accept - and _ variants) ---
        benchmark=bm_primary,
        benchmark_ticker=bm_primary,  # (nur setzen, wenn das Feld in BTConfig existiert)
        dual_benchmark=bool(bm_dual),
        benchmark2=bm_secondary,
        dump_decision_bundles=bool(dump_bundles),
        dump_selection=bool(dump_selection),
        dump_weights=bool(dump_weights),
        decisions_dir=str(dec_dir),

        min_position_weight=_coalesce(
            getattr(a, "min_position_weight", None),
            (limits_cfg.get("min_position_weight") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "min_position_weight", None)),
            cfg_toml.get("min_position_weight"), 0.0),

        max_active_names=_coalesce(
            getattr(a, "max_active_names", None),
            (limits_cfg.get("max_active_names") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "max_active_names", None)),
            cfg_toml.get("max_active_names", cfg_toml.get("names_limit")), 0),

        # NEU: Fenster-Werte aus [windows] oder Root
        score_days=_coalesce(getattr(a, "score_days", None),
                             win_score_days,
                             cfg_toml.get("score_days"),
                             252),
        vol_days=_coalesce(getattr(a, "vol_days", None),
                           win_vol_days,
                           cfg_toml.get("vol_days"),
                           63),

        # Regime
        require_above_sma=_coalesce(getattr(regime_cfg, "require_above_sma", None), regime_cfg.get("require_above_sma"), False),
        regime_below_action=act,
        regime_sma_days=_coalesce(
            getattr(regime_cfg, "regime_sma_days", None),
            regime_cfg.get("regime_sma_days"),
            200
        ),
        regime_exposure_low=_coalesce(
            getattr(a, "regime_exposure_low", None),
            cfg_toml.get("regime_exposure_low"),
            0.50
        ),

        # Vol-Targeting
        vol_target_ann=_coalesce(
            getattr(a, "vol_target_ann", None),
            cfg_toml.get("vol_target_ann"),
            None
        ),
        vol_lookback_days=_coalesce(
            getattr(a, "vol_lookback_days", None),
            cfg_toml.get("vol_lookback_days"),
            20
        ),
        include_cash=_coalesce(
            getattr(a, "include_cash", None),
            (limits_cfg.get("include_cash") if isinstance(limits_cfg, dict) else getattr(limits_cfg, "include_cash", None)),
            cfg_toml.get("include_cash"),
            False
        ),
        cash_yield_annual=_coalesce(getattr(a, "cash_yield_annual", None),
                                    cfg_toml.get("cash_yield_annual"), 0.0),

        verbose=_coalesce(a.verbose if a.verbose else None, cfg_toml.get("verbose"), False),
        universe_name=str(universe_cfg.get("name", cfg_toml.get("universe_name", "sp500")) or "sp500"),
    )


# ---------------------------------------
# CLI
# ---------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Momentum Backtester")
    ap.add_argument("--config", help="TOML-Konfigdatei (z. B. aktien_oop/backtest_config.toml)")

    ap.add_argument("--tickers")
    ap.add_argument("--sector-meta")
    ap.add_argument("--save-dir")

    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--frequency", choices=["monthly", "weekly"])

    ap.add_argument("--top-k", type=int)
    ap.add_argument("--buffer-k", type=int)
    ap.add_argument("--use-sector-limits", dest="use_sector_limits", action="store_true", default=None)
    ap.add_argument("--no-sector-limits", dest="use_sector_limits", action="store_false", default=None)
    ap.add_argument("--max-per-sector", type=int)
    ap.add_argument("--cost-bps", type=float)
    ap.add_argument("--slippage-bps", type=float)
    ap.add_argument("--min-history-days", type=int)

    ap.add_argument("--use-equal-weight", action="store_true", default=None)
    ap.add_argument("--gap-filter", type=float)
    ap.add_argument("--friction-eps", type=float)
    ap.add_argument("--friction-eps-pct", type=float)
    ap.add_argument("--weight-round-step", type=float)
    ap.add_argument("--max-turnover-cap", type=float)
    ap.add_argument("--rebalance-every-n", type=int)

    ap.add_argument("--benchmark", type=str)
    ap.add_argument("--benchmark-ticker", type=str)  # Alias
    ap.add_argument("--dual-benchmark", action="store_true", default=None)
    ap.add_argument("--benchmark2", type=str)

    ap.add_argument("--regime-use-filter", action="store_true")
    ap.add_argument("--regime-sma-days", type=int)
    ap.add_argument("--regime-exposure-low", type=float)

    ap.add_argument("--vol-target-ann", type=float)  # None = aus
    ap.add_argument("--vol-lookback-days", type=int)

    ap.add_argument("--include-cash", action="store_true", default=None)
    ap.add_argument("--cash-yield-annual", type=float)

    ap.add_argument("--dump-decisions", action="store_true", default=None)
    ap.add_argument("--decisions-dir", type=str)

    ap.add_argument("--verbose", action="store_true", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = _build_cfg_from_config_and_cli(args)
    print(f"[CFG] benchmark={cfg.benchmark} dual_benchmark={cfg.dual_benchmark} benchmark2={cfg.benchmark2}")
    print(f"[CFG] dump_decision_bundles={cfg.dump_decision_bundles} decisions_dir={cfg.decisions_dir}")
    print(f"[CFG] universe_name={cfg.universe_name} universe_file={cfg.tickers_file}")

    # --- Run meta (global) ---
    iso, run_id = _now_local_str("Europe/Berlin")
    out_dir = Path("aktien_oop")
    meta_path = out_dir / f"bt_meta_{run_id}.json"
    meta = {
        "run_id": run_id,
        "started_at": iso,
        "tz": "Europe/Berlin",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "argv": sys.argv,
        "cfg": _to_jsonable(cfg),
    }
    _dump_run_meta(meta_path, meta)
    print(f"[META] wrote {meta_path}")

    # --- Backtest ---
    bt = Backtester(cfg)
    bt._meta_path = meta_path  # für Endzeit/Ergebnis-Update
    bt._run_id = run_id        # optional nutzbar für Dateinamen
    bt.run()



if __name__ == "__main__":
    main()
