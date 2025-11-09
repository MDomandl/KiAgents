# aktien_oop/core_calc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Optional, Tuple, Callable, Iterable
import pandas as pd

# ======= Kern-Parameter (wirken auf Ergebnis) =======
@dataclass(frozen=True)
class CalcParams:
    as_of: str                 # "YYYY-MM-DD"
    period: str                # "800d"
    adjusted: bool             # True = Adjusted Prices
    score_days: int            # z.B. 252
    vol_days: int              # z.B. 63

    # Filter
    use_under_sma: bool = False
    sma_days: int = 200
    gap_filter: float = 0.0
    min_price: float = 0.0
    min_volume: float = 0.0

    # Limits & Auswahl
    use_sector_limits: bool = True
    max_per_sector: Optional[int] = 3
    top_k: int = 12
    buffer_k: int = 3

    # Finalisierung / Sizing (optional)
    include_cash: bool = False
    weight_round_step: float = 0.0
    max_turnover_cap: float = 1.0
    friction_eps: float = 0.0
    friction_eps_pct: float = 0.0


# Signaturen für die Injektionsfunktionen
GetPricesFn  = Callable[[Sequence[str], str, str, bool], pd.DataFrame]
GetSectorsFn = Callable[[Sequence[str]], Dict[str, str]]

def _rank_desc(scores: pd.Series) -> pd.Series:
    """Ränge 1..N (1 = bester Score) – stabil wie im BT."""
    return scores.rank(ascending=False, method="first").astype(int)

def _sector_ok(picked_count_by_sec: Dict[str,int], sec: str, p: CalcParams) -> bool:
    if not p.use_sector_limits or not p.max_per_sector:
        return True
    return picked_count_by_sec.get(sec, 0) < int(p.max_per_sector or 0)

def select_topk_buffer(
    scores: pd.Series,
    keep: pd.Index,
    sectors: Dict[str, str],
    p: CalcParams,
    prev_holdings: Optional[Iterable[str]] = None,
) -> Sequence[str]:
    """
    Top-K Auswahl mit Turnover-Buffer (BT-kompatibel):
      1) sortiere nach Score desc
      2) 'Keepers' = vorherige Holdings, deren Rang <= top_k + buffer_k
      3) fülle mit besten Resten auf top_k auf
      4) Sektor-Limits werden in beiden Schritten erzwungen
    """
    if prev_holdings is None:
        prev_holdings = []

    # Kandidaten & Ränge
    s = scores.loc[keep].dropna().sort_values(ascending=False)
    if s.empty:
        return []
    ranks = _rank_desc(s)  # 1..N (1 = top)

    top_k = int(p.top_k)
    buf_k = max(int(p.buffer_k or 0), 0)
    cutoff = top_k + buf_k

    picked: list[str] = []
    per_sec: Dict[str,int] = {}

    # 1) Keepers: frühere Holdings, die innerhalb des Buffers liegen
    for t in prev_holdings:
        if t not in s.index:
            continue
        if int(ranks[t]) <= cutoff:
            sec = sectors.get(t, "UNKNOWN")
            if _sector_ok(per_sec, sec, p):
                picked.append(t)
                per_sec[sec] = per_sec.get(sec, 0) + 1
                if len(picked) >= top_k:
                    return picked  # fertig

    # 2) Mit besten Resten auffüllen (unter Wahrung der Limits)
    for t in s.index:
        if t in picked:
            continue
        sec = sectors.get(t, "UNKNOWN")
        if _sector_ok(per_sec, sec, p):
            picked.append(t)
            per_sec[sec] = per_sec.get(sec, 0) + 1
            if len(picked) >= top_k:
                break

    return picked

# ======= Pure Scoring (→ exakt wie im BT!) =======
def _to_close_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Akzeptiert:
      - MultiIndex-Spalten (Yahoo-Style): ('Close', Ticker) / ('Adj Close', Ticker)
      - Single-Column 'Close' / 'Adj Close' (ggf. mit MultiIndex-Index (date,ticker))
      - Wide-Format: Spalten = Ticker, Werte = Schlusskurse
    Gibt zurück:
      - DataFrame mit Index=DatetimeIndex (naiv) und Spalten = Ticker (Close-Preise)
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame()

    df = prices.copy()

    # 1) Falls MultiIndex COLUMNS (Yahoo-Download etc.)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        if any(x in ("close", "adj close") for x in lvl0):
            # bevorzuge 'Close', sonst 'Adj Close'
            if "close" in lvl0:
                close = df.xs("Close", axis=1, level=0, drop_level=True)
            else:
                close = df.xs("Adj Close", axis=1, level=0, drop_level=True)
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.sort_index().ffill().dropna(how="all")
            return close

    # 2) Falls MultiIndex INDEX (date,ticker) mit Spalte 'Close'/'Adj Close'
    if isinstance(df.index, pd.MultiIndex):
        cols_lower = [c.lower() for c in df.columns.astype(str)]
        if "close" in cols_lower:
            close = df[df.columns[cols_lower.index("close")]].unstack("ticker")
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.sort_index().ffill().dropna(how="all")
            return close
        if "adj close" in cols_lower:
            close = df[df.columns[cols_lower.index("adj close")]].unstack("ticker")
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.sort_index().ffill().dropna(how="all")
            return close

    # 3) Falls Single-Column 'Close'/'Adj Close' (breit oder schmal)
    cols_lower = [c.lower() for c in df.columns.astype(str)]
    if "close" in cols_lower:
        close = df[df.columns[cols_lower.index("close")]]
        # falls das schon wide ist, einfach zurückgeben
        if isinstance(close, pd.DataFrame):
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.sort_index().ffill().dropna(how="all")
            return close
        # andernfalls zu wide pivoten (wenn Index MultiIndex hat)
        if isinstance(df.index, pd.MultiIndex):
            close = close.unstack("ticker")
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.sort_index().ffill().dropna(how="all")
            return close

    if "adj close" in cols_lower:
        adj = df[df.columns[cols_lower.index("adj close")]]
        if isinstance(adj, pd.DataFrame):
            adj.index = pd.to_datetime(adj.index).tz_localize(None)
            adj = adj.sort_index().ffill().dropna(how="all")
            return adj
        if isinstance(df.index, pd.MultiIndex):
            adj = adj.unstack("ticker")
            adj.index = pd.to_datetime(adj.index).tz_localize(None)
            adj = adj.sort_index().ffill().dropna(how="all")
            return adj

    # 4) Fallback: **Wide-Format** (Spalten = Ticker, keine 'Close'-Spalte vorhanden)
    # Heuristik: Spaltennamen sehen wie Ticker aus (Großbuchstaben/.-) und Werte sind numerisch.
    sample_cols = list(df.columns[: min(5, len(df.columns))])
    looks_like_tickers = all(isinstance(c, str) for c in sample_cols)
    if looks_like_tickers:
        # sicherstellen, dass numerisch
        try:
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            pass
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index().ffill().dropna(how="all")
        return df

    # Wenn wir hier landen, kennen wir das Format nicht.
    raise ValueError("compute_scores: Keine 'Close'/'Adj Close' gefunden und Wide-Format-Heuristik schlug fehl.")


def _xs_rank01(s: pd.Series) -> pd.Series:
    # Cross-Section-Rank 0..1 (höher besser) – wie im BT
    return s.rank(pct=True)

def compute_scores(prices: pd.DataFrame, p: CalcParams) -> pd.Series:
    """
    BT-kompatible Score-Berechnung:
      score = 0.50 * rank(mom12_1) + 0.25 * rank(mom126) + 0.25 * rank(mom63)
      under_sma = Close < SMA(100)  → penalty = 0.15
      score_adj = score - penalty
    Rückgabe: score_adj (Series: index=ticker, value=float)
    """
    close = _to_close_matrix(prices)
    if close.empty:
        return pd.Series(dtype=float)

    # bis as_of clippen (robust falls as_of nicht exakt im Index ist)
    as_of_ts = pd.Timestamp(p.as_of).tz_localize(None)
    close = close.loc[:as_of_ts]
    if len(close) < 253:  # 252 + 1 für mom12_1
        return pd.Series(index=close.columns, dtype=float)

    # Features wie im BT:
    # mom63 / mom126 / mom252
    mom63  = close / close.shift(63)  - 1.0
    mom126 = close / close.shift(126) - 1.0
    # mom12_1: 12M Momentum ohne letzten Monat
    mom12_1 = close.shift(21) / close.shift(252) - 1.0
    # SMA100 für under_sma
    sma100 = close.rolling(100, min_periods=100).mean()

    # Letzte verfügbare Zeile (<= as_of)
    r63    = _xs_rank01(mom63.iloc[-1])
    r126   = _xs_rank01(mom126.iloc[-1])
    r12_1  = _xs_rank01(mom12_1.iloc[-1])

    score = 0.50 * r12_1 + 0.25 * r126 + 0.25 * r63

    last_close = close.iloc[-1]
    sma_last   = sma100.iloc[-1]
    # Ticker-Index festlegen (Scores ist bei uns die Referenz)
    # Referenzindex
    tickers = list(score.index)  # oder score.index, je nach Variable bei dir

    # Align beider Eingaben
    last_close_s = (last_close if isinstance(last_close, pd.Series)
                    else pd.Series(last_close, index=tickers)).reindex(tickers)
    sma_last_s = (sma_last if isinstance(sma_last, pd.Series)
                  else pd.Series(sma_last, index=tickers)).reindex(tickers)

    # Vergleich -> sicherstellen, dass es eine pandas Series ist
    comp = (last_close_s < sma_last_s)
    if not isinstance(comp, pd.Series):
        comp = pd.Series(comp, index=tickers)

    # Nullable-Boolean erzwingen, dann fehlende als True auffüllen
    under_sma = comp.astype("boolean").fillna(True)

    penalty   = 0.15 * under_sma.astype(int)  # exakt wie im BT
    score_adj = (score - penalty).astype(float)

    return score_adj


def apply_filters(scores: pd.Series, prices: pd.DataFrame, p: CalcParams) -> pd.Index:
    keep = scores.dropna().index

    # Close-Matrix (wie oben)
    if "Close" in prices.columns and isinstance(prices.index, pd.MultiIndex):
        close = prices["Close"].unstack("ticker").ffill()
    elif "Close" in prices.columns:
        close = prices["Close"].ffill()
    elif "Adj Close" in prices.columns and isinstance(prices.index, pd.MultiIndex):
        close = prices["Adj Close"].unstack("ticker").ffill()
    else:
        close = prices.get("Adj Close", None)
        if close is None:
            return keep

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().loc[:pd.Timestamp(p.as_of).tz_localize(None)]

    # 1d Return für Gap-Check (12% – BT-Logik)
    if len(close) >= 2:
        ret1d = close.pct_change().iloc[-1]
        gap_mask = ret1d.abs() > 0.12   # 12% Schwellwert wie im BT
        keep = keep.difference(gap_mask[gap_mask].index)

    # optionale Preis/Volumen-Grenzen (falls du sie weiterhin nutzen willst)
    min_px = float(getattr(p, "min_price", 0.0) or 0.0)
    if min_px > 0:
        last_px = close.iloc[-1].dropna()
        keep = keep.intersection(last_px[last_px >= min_px].index)

    if "Volume" in prices.columns:
        vol = None
        if isinstance(prices.index, pd.MultiIndex):
            vol = prices["Volume"].unstack("ticker").mean()
        else:
            try:
                vol = prices["Volume"].mean()
            except Exception:
                vol = None
        min_vol = float(getattr(p, "min_volume", 0.0) or 0.0)
        if vol is not None and min_vol > 0:
            keep = keep.intersection(vol[vol >= min_vol].index)

    # WICHTIG: Under-SMA ist im BT KEIN Filter (nur Penalty) → hier NICHT filtern.
    return keep


def select_topk(scores: pd.Series, keep: pd.Index,
                sectors: Dict[str, str], p: CalcParams) -> Sequence[str]:
    s = scores.loc[keep].sort_values(ascending=False)
    if not p.use_sector_limits or not p.max_per_sector:
        return list(s.index[:p.top_k])

    picked, per_sec = [], {}
    for t in s.index:
        sec = sectors.get(t, "UNKNOWN")
        if per_sec.get(sec, 0) >= int(p.max_per_sector):
            continue
        picked.append(t)
        per_sec[sec] = per_sec.get(sec, 0) + 1
        if len(picked) >= p.top_k:
            break
    return picked


def size_weights(names: Sequence[str], p: CalcParams) -> Dict[str, float]:
    if not names:
        return {}
    w = 1.0 / len(names)
    weights = {t: float(w) for t in names}
    # TODO: Falls BT Caps/Rundung/Turnover-Logik in der Finalisierung nutzt, hier spiegeln.
    return weights

def _to_series(weights) -> pd.Series:
    """Erzwingt eine float-Series aus dict/Series, ohne Indexverluste."""
    if isinstance(weights, pd.Series):
        return weights.astype(float)
    return pd.Series(weights, dtype=float)

def _safe_reindex_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Gibt df nur mit den vorhandenen 'cols' zurück, fügt fehlende als 0.0 hinzu
    und stellt am Ende die gewünschte Spaltenreihenfolge sicher.
    """
    have = [c for c in cols if c in df.columns]
    out = df.loc[:, have].copy()
    # fehlende Spalten als 0.0 auffüllen
    missing = [c for c in cols if c not in df.columns]
    for m in missing:
        out[m] = 0.0
    # gewünschte Reihenfolge wiederherstellen
    return out.reindex(columns=cols)


def calculate_portfolio(
    universe: Sequence[str],
    p: CalcParams,
    get_prices: GetPricesFn,
    get_sectors: GetSectorsFn,
    prev_holdings: Optional[Iterable[str]] = None,   # <— NEU
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Ein-Schuss-Berechnung für EIN as_of.
    Liefert: (weights_dict, scores_series)
    """
    prices  = get_prices(universe, p.as_of, p.period, p.adjusted)
    sectors = get_sectors(universe)
    scores  = compute_scores(prices, p)
    keep    = apply_filters(scores, prices, p)

    if prev_holdings is not None:
        names = select_topk_buffer(scores, keep, sectors, p, prev_holdings=prev_holdings)
    else:
        names = select_topk(scores, keep, sectors, p)

    # einfache Equal-Weights (falls du Caps/Rundung willst: in size_weights spiegeln)
    weights = size_weights(names, p)
    return weights, scores

