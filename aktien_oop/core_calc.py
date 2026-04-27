# aktien_oop/core_calc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Optional, Tuple, Callable, Iterable
import pandas as pd
from pathlib import Path

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
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    rebalance: str = "monthly"
    max_lookback_days: int  = None
    max_active_names: int = 8

    dump_scores: bool = False
    dump_selection: bool = False
    dump_weights: bool = False
    dump_tag: str = ""

# Signaturen für die Injektionsfunktionen
GetPricesFn  = Callable[[Sequence[str], str, str, bool], pd.DataFrame]
GetSectorsFn = Callable[[Sequence[str]], Dict[str, str]]

def _sort_score_frame(scores: pd.Series) -> pd.DataFrame:
    return (
        pd.DataFrame({"ticker": scores.index.astype(str), "score": scores.astype(float).values})
        .sort_values(["score", "ticker"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def _sort_scores_desc_ticker(scores: pd.Series) -> pd.Series:
    sorted_df = _sort_score_frame(scores.dropna())
    return pd.Series(
        sorted_df["score"].to_numpy(),
        index=sorted_df["ticker"].to_numpy(),
        dtype=float,
    )


def _rank_desc_stable(scores: pd.Series) -> pd.Series:
    sorted_scores = _sort_scores_desc_ticker(scores)
    ranked = pd.Series(
        range(1, len(sorted_scores) + 1),
        index=sorted_scores.index,
        dtype=int,
    )
    return ranked.reindex(scores.index).astype(int)


def _rank_desc(scores: pd.Series) -> pd.Series:
    """R?nge 1..N (1 = bester Score) mit deterministischem Tie-Breaker via Ticker."""
    return _rank_desc_stable(scores)

def _period_to_days(period: str | None) -> int:
    if period is None:
        return 800
    s = str(period).strip().lower()
    try:
        if s.endswith("d"):
            return int(s[:-1])
        if s.endswith("m"):
            return int(s[:-1]) * 30
        if s.endswith("y"):
            return int(s[:-1]) * 365
        return int(s)
    except Exception:
        return 800


def slice_to_window(prices: pd.DataFrame, as_of: str, period: str) -> pd.DataFrame:
    """
    Einheitliches, deterministisches Slicing/Clamping:
    - tz-naiv
    - sortiert, dedupliziert
    - start = backfill auf nächsten verfügbaren Handelstag
    - end   = pad auf letzten verfügbaren Handelstag <= as_of
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    df = prices.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated()].sort_index()

    idx = df.index
    asof = pd.Timestamp(as_of).tz_localize(None)
    days = _period_to_days(period)
    start_target = asof - pd.Timedelta(days=days)

    # Clamp start/end auf vorhandene Indizes
    start_pos = idx.get_indexer([start_target], method="backfill")[0]
    end_pos   = idx.get_indexer([asof], method="pad")[0]

    if start_pos < 0:
        start_pos = 0
    if end_pos < 0:
        end_pos = len(idx) - 1

    start = idx[start_pos]
    end   = idx[end_pos]

    return df.loc[start:end].copy()

def _norm_sector(sec) -> str | None:
    """
    Normalisiert Sektor-Strings, damit BT/RUN identische Keys verwenden.
    None => "unbekannt" => NICHT sektor-limitiert.
    """
    if sec is None:
        return None
    s = str(sec).strip()
    if not s:
        return None
    u = s.upper()
    if u in ("UNKNOWN", "N/A", "NA", "NONE", "NULL"):
        return None
    return u

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

    # Kandidaten: scores nur für keep, ohne NaNs
    s0 = scores.reindex(keep).dropna()

    # Deterministische Sortierung: score desc, ticker asc
    s = _sort_scores_desc_ticker(pd.Series(s0.values, index=s0.index.astype(str), dtype=float))

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
            sec = _norm_sector(sectors.get(t))  # None, wenn unbekannt

            # Alles was "unbekannt" ist, soll NICHT sektor-limitiert werden
            if sec is None or str(sec).strip().upper() in ("", "UNKNOWN", "N/A", "NA", "NONE"):
                picked.append(t)
                if len(picked) >= top_k:
                    return picked
            else:
                if _sector_ok(per_sec, sec, p):
                    picked.append(t)
                    per_sec[sec] = per_sec.get(sec, 0) + 1
                    if len(picked) >= top_k:
                        return picked

                if len(picked) >= top_k:
                    return picked  # fertig

    # 2) Mit besten Resten auffüllen (unter Wahrung der Limits)
    for t in s.index:
        if t in picked:
            continue
        sec = _norm_sector(sectors.get(t))  # None, wenn unbekannt

        # Alles was "unbekannt" ist, soll NICHT sektor-limitiert werden
        if sec is None or str(sec).strip().upper() in ("", "UNKNOWN", "N/A", "NA", "NONE"):
            picked.append(t)
            if len(picked) >= top_k:
                return picked if len(picked) >= top_k else picked
        else:
            if _sector_ok(per_sec, sec, p):
                picked.append(t)
                per_sec[sec] = per_sec.get(sec, 0) + 1
                if len(picked) >= top_k:
                    return picked

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

def compute_scores(close: pd.DataFrame, p: CalcParams) -> pd.Series:
    """
    Scoring-Logik für BT und Runner:
    - Verwendet p.score_days dynamisch (statt starr 252)
    - Kombiniert drei Momenta (kurz/mittel/lang) + Under-SMA-Penalty
    """

    # ---- Minimal benötigte Historie ----
    # Lange Window = score_days (z.B. 200 oder 252)
    long_win = max(p.score_days, 63)
    mid_win  = max(p.score_days // 2, 21)
    short_win = max(p.score_days // 4, 10)

    # +21 Tage, da wir für das "12M ohne letzten Monat" eine Verschiebung brauchen
    min_rows = long_win + 21
    if len(close) < min_rows:
        # Zu wenig Historie -> alle Scores NaN -> wird später rausgefiltert
        return pd.Series(index=close.columns, dtype=float)

    # ---- Momentum-Features ----
    # Kurz: z.B. ~score_days/4
    mom_short = close / close.shift(short_win) - 1.0
    # Mittel: z.B. ~score_days/2
    mom_mid   = close / close.shift(mid_win)   - 1.0
    # Lang: "12M ohne letzten Monat" analog, aber mit long_win statt fest 252
    mom_long  = close.shift(21) / close.shift(long_win) - 1.0

    # Letzte verfügbare Zeile
    r_short = _xs_rank01(mom_short.iloc[-1])
    r_mid   = _xs_rank01(mom_mid.iloc[-1])
    r_long  = _xs_rank01(mom_long.iloc[-1])

    # Gewichtung wie im BT (lang dominiert)
    score = 0.50 * r_long + 0.25 * r_mid + 0.25 * r_short

    # ---- Under-SMA-Filter (für Penalty) ----
    # SMA-Fenster: mindestens 100, sonst halbe score_days
    sma_win = max(100, mid_win)
    sma = close.rolling(sma_win, min_periods=sma_win).mean()

    last_close = close.iloc[-1]
    sma_last   = sma.iloc[-1]

    tickers = score.index

    last_close_s = last_close.reindex(tickers)
    sma_last_s   = sma_last.reindex(tickers)

    comp = (last_close_s < sma_last_s)
    if not isinstance(comp, pd.Series):
        comp = pd.Series(comp, index=tickers)

    under_sma = comp.astype("boolean").fillna(True)
    penalty   = 0.15 * under_sma.astype(int)

    score_adj = (score - penalty).astype(float)
    return score_adj


def apply_filters(scores: pd.Series, prices: pd.DataFrame, p: CalcParams) -> pd.Index:
    keep = scores.dropna().index

    # Close-Matrix (wie oben)
    close = _to_close_matrix(prices)
    close = close.loc[:pd.Timestamp(p.as_of).tz_localize(None)]

    # 1d Return für Gap-Check – Schwellwert aus p.gap_filter
    gap_thr = float(getattr(p, "gap_filter", 0.0) or 0.0)
    if len(close) >= 2 and gap_thr > 0.0:
        ret1d = close.pct_change().iloc[-1]
        gap_mask = ret1d.abs() > gap_thr
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
    # Kandidaten: scores nur für keep, ohne NaNs
    s0 = scores.reindex(keep).dropna()

    # Deterministische Sortierung: score desc, ticker asc
    s = _sort_scores_desc_ticker(pd.Series(s0.values, index=s0.index.astype(str), dtype=float))

    if not p.use_sector_limits or not p.max_per_sector:
        return list(s.index[:p.top_k])

    picked, per_sec = [], {}
    for t in s.index:
        sec = _norm_sector(sectors.get(t))
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


def _apply_friction_eps(
    weights: Dict[str, float],
    prev_weights: Optional[Dict[str, float]],
    p: CalcParams,
) -> Dict[str, float]:
    eps = float(getattr(p, "friction_eps", 0.0) or 0.0)
    if eps <= 0.0 or not prev_weights:
        return dict(weights)

    merged: Dict[str, float] = {}
    keys = {str(k) for k in weights} | {str(k) for k in prev_weights}
    for ticker in sorted(keys):
        old_weight = float(prev_weights.get(ticker, 0.0))
        new_weight = float(weights.get(ticker, 0.0))
        kept_weight = old_weight if abs(new_weight - old_weight) < eps else new_weight
        if kept_weight > 0.0:
            merged[ticker] = kept_weight

    total = float(sum(merged.values()))
    if total <= 0.0:
        return {}

    return {ticker: float(weight / total) for ticker, weight in merged.items()}

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

def dump_scores(scores, as_of_str, tag):
    Path("aktien_oop/decisions").mkdir(parents=True, exist_ok=True)

    sorted_scores = _sort_scores_desc_ticker(scores)
    df = sorted_scores.to_frame(name=as_of_str)
    df.index.name = "ticker"

    path = f"aktien_oop/decisions/scores_{tag}_{as_of_str}.csv"
    df.to_csv(path)


def _build_selection_dump(scores: pd.Series, keep: pd.Index, names: Sequence[str], sectors: Dict[str, str], p: CalcParams, prev_holdings: Optional[Iterable[str]] = None) -> pd.DataFrame:
    prev_list = [str(t) for t in (prev_holdings or [])]
    prev_set = set(prev_list)
    selected_set = {str(t) for t in names}

    ranked = _sort_score_frame(pd.Series(scores.reindex(keep).dropna().values, index=scores.reindex(keep).dropna().index.astype(str), dtype=float))

    selection_df = ranked.copy()
    selection_df["rank"] = range(1, len(selection_df) + 1)
    selection_df["sector"] = selection_df["ticker"].map(lambda t: sectors.get(t, "UNKNOWN"))
    selection_df["selected"] = selection_df["ticker"].isin(selected_set)

    top_k = int(getattr(p, "top_k", 0) or 0)
    buffer_k = max(int(getattr(p, "buffer_k", 0) or 0), 0)
    cutoff = top_k + buffer_k
    ranks = {str(row.ticker): int(row.rank) for row in selection_df.itertuples(index=False)}

    reasons: dict[str, str] = {}
    picked: list[str] = []
    per_sec: Dict[str, int] = {}

    def _mark_reason(ticker: str, reason: str) -> None:
        if ticker not in reasons:
            reasons[ticker] = reason

    for ticker in prev_list:
        if ticker not in ranks:
            continue
        rank = ranks[ticker]
        if rank > cutoff:
            _mark_reason(ticker, "cutoff")
            continue

        sec = _norm_sector(sectors.get(ticker))
        if sec is not None and not _sector_ok(per_sec, sec, p):
            _mark_reason(ticker, "sector_limit")
            continue

        if len(picked) >= top_k:
            _mark_reason(ticker, "cutoff")
            continue

        picked.append(ticker)
        if sec is not None:
            per_sec[sec] = per_sec.get(sec, 0) + 1
        _mark_reason(ticker, "buffer")

    for row in selection_df.itertuples(index=False):
        ticker = str(row.ticker)
        rank = int(row.rank)
        if ticker in picked:
            continue

        sec = _norm_sector(sectors.get(ticker))
        if sec is not None and not _sector_ok(per_sec, sec, p):
            _mark_reason(ticker, "sector_limit")
            continue

        if len(picked) >= top_k:
            _mark_reason(ticker, "cutoff")
            continue

        picked.append(ticker)
        if sec is not None:
            per_sec[sec] = per_sec.get(sec, 0) + 1
        _mark_reason(ticker, "top_k" if rank <= top_k else "selected")

    selection_df["cutoff_rank"] = cutoff
    selection_df["within_cutoff"] = selection_df["rank"] <= cutoff
    selection_df["within_top_20"] = selection_df["rank"] <= 20
    selection_df["is_prev_holding"] = selection_df["ticker"].isin(prev_set)
    selection_df["reason"] = selection_df["ticker"].map(lambda ticker: reasons.get(str(ticker), "cutoff"))

    return selection_df.loc[:, [
        "ticker",
        "score",
        "rank",
        "cutoff_rank",
        "within_cutoff",
        "within_top_20",
        "is_prev_holding",
        "sector",
        "selected",
        "reason",
    ]]


def _dump_selection(scores: pd.Series, keep: pd.Index, names: Sequence[str], sectors: Dict[str, str], p: CalcParams, prev_holdings: Optional[Iterable[str]] = None) -> None:
    dump_dir = Path("aktien_oop/dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)

    as_of = str(p.as_of)[:10]
    tag = getattr(p, "dump_tag", "X")
    out_sel = dump_dir / f"selection_{tag}_{as_of}.csv"

    selection_df = _build_selection_dump(scores, keep, names, sectors, p, prev_holdings=prev_holdings)
    selection_df.to_csv(out_sel, index=False)


def _dump_weights(weight_raw: Dict[str, float], weight_after_round: Dict[str, float], weight_final: Dict[str, float], p: CalcParams) -> None:
    dump_dir = Path("aktien_oop/dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)

    as_of = str(p.as_of)[:10]
    tag = getattr(p, "dump_tag", "X")
    out_weights = dump_dir / f"weights_{tag}_{as_of}.csv"

    tickers = sorted(
        {str(t) for t in weight_raw}
        | {str(t) for t in weight_after_round}
        | {str(t) for t in weight_final}
    )
    cash_weight = float(weight_final.get("CASH", 0.0)) if isinstance(weight_final, dict) else 0.0

    rows = []
    for ticker in tickers:
        rows.append({
            "ticker": ticker,
            "weight_raw": float(weight_raw.get(ticker, 0.0)),
            "weight_after_round": float(weight_after_round.get(ticker, 0.0)),
            "weight_final": float(weight_final.get(ticker, 0.0)),
            "cash_weight": cash_weight,
        })

    pd.DataFrame(
        rows,
        columns=["ticker", "weight_raw", "weight_after_round", "weight_final", "cash_weight"],
    ).to_csv(out_weights, index=False)


def calculate_portfolio(
    universe: Sequence[str],
    p: CalcParams,
    get_prices: GetPricesFn,
    get_sectors: GetSectorsFn,
    prev_holdings: Optional[Iterable[str]] = None,   # <— NEU
    prev_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Ein-Schuss-Berechnung für EIN as_of.
    Liefert: (weights_dict, scores_series)
    """
    prices  = get_prices(universe, p.as_of, p.period, p.adjusted)
    close = prices
    try:
        close = close.loc[:pd.Timestamp(p.as_of).tz_localize(None)]
    except Exception:
        pass
    # --- DEBUG: Price window check (Runner) ---
    if getattr(p, "dump_scores", False):
        print("=== PRICE WINDOW DEBUG (Runner) ===")
        print("p.as_of:", p.as_of)
        print("p.period:", p.period)
        print("TAG:", getattr(p, "dump_tag", "?"), "AS_OF:", p.as_of, "PERIOD:", p.period)
        print("ROWS:", len(close.index), "MIN:", close.index.min(), "MAX:", close.index.max())
        print("prices.index.min():", prices.index.min() if not prices.empty else None)
        print("prices.index.max():", prices.index.max() if not prices.empty else None)
        print("len(prices.index):", len(prices.index))
        print("LAST ROW DATE:", prices.index.max(), "AS_OF:", p.as_of)
        if not prices.empty:
            print("prices.index[-5:]:", list(prices.index[-5:]))
        print("===================================")

    prices = slice_to_window(prices, p.as_of, p.period)
    prices = slice_to_window(prices, p.as_of, p.period)

    if getattr(p, "dump_scores", False):
        print("=== PRICE WINDOW AFTER SLICE ===")
        print("prices.index.min():", prices.index.min())
        print("prices.index.max():", prices.index.max())
        print("len(prices.index):", len(prices.index))
        print("LAST ROW DATE:", prices.index.max(), "AS_OF:", p.as_of)
        print("================================")

    if prices is not None and not prices.empty:
        eff_asof = pd.Timestamp(prices.index.max()).tz_localize(None)
    else:
        eff_asof = pd.Timestamp(p.as_of).tz_localize(None)

    req_asof = pd.Timestamp(p.as_of).normalize()
    px = prices.loc[:req_asof]
    if len(px) == 0:
        raise ValueError(f"No price data on/before as_of={req_asof}")

    asof_eff = px.index.max()  # letzter verfügbarer Handelstag <= req_asof
    prices_eff = prices.loc[:asof_eff]  # konsistente Basis

    sectors = get_sectors(universe)
    if getattr(p, "dump_scores", False):
        watch = {"APH", "AVGO", "LRCX", "MPWR", "ORCL", "PLTR", "DASH", "EBAY", "GE", "HII", "IVZ", "TPR"}
        for t in sorted(watch):
            if t in universe:
                print(f"[SECTOR DBG] {getattr(p, 'dump_tag', 'X')} {t}: {sectors.get(t)!r}")

    scores  = compute_scores(prices_eff, p)
    if getattr(p, "dump_scores", False):
        dump_dir = Path("aktien_oop/dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)

        as_of = str(p.as_of)[:10]
        tag = getattr(p, "dump_tag", "X")

        out = dump_dir / f"scores_{tag}_{as_of}.csv"

        s = _sort_scores_desc_ticker(scores)
        df = pd.DataFrame({"ticker": s.index, "score": s.values})
        df.to_csv(out, index=False)

    keep    = apply_filters(scores, prices_eff, p)

    if prev_holdings is not None:
        names = select_topk_buffer(scores, keep, sectors, p, prev_holdings=prev_holdings)
    else:
        names = select_topk(scores, keep, sectors, p)
    if getattr(p, "dump_selection", False):
        _dump_selection(scores, keep, names, sectors, p, prev_holdings=prev_holdings)

    # einfache Equal-Weights (falls du Caps/Rundung willst: in size_weights spiegeln)
    weights_raw = size_weights(names, p)
    weights_after_round = dict(weights_raw)
    weights = dict(weights_after_round)

    # --- NEW: max_active_names (post-sizing, cash-aware) ---
    max_names = int(getattr(p, "max_active_names", 0) or 0)
    if max_names > 0:
        # CASH nicht mitzählen, Top-N nach Gewicht behalten
        cash = float(weights.get("CASH", 0.0)) if "CASH" in weights else 0.0
        items = [(k, float(v)) for k, v in weights.items() if k != "CASH" and float(v) > 0.0]

        if len(items) > max_names:
            items.sort(key=lambda kv: kv[1], reverse=True)
            kept = dict(items[:max_names])

            # Renorm auf (1 - cash)
            s = sum(kept.values())
            if s > 0:
                scale = (1.0 - cash) / s
                kept = {k: v * scale for k, v in kept.items()}
            if cash > 0.0:
                kept["CASH"] = cash

            weights = kept

    weights = _apply_friction_eps(weights, prev_weights, p)

    if getattr(p, "dump_weights", False):
        _dump_weights(weights_raw, weights_after_round, weights, p)
    # --- /NEW ---
    return weights, scores

def _apply_max_active_names(weights: dict[str, float], max_active_names: int) -> dict[str, float]:
    if not weights:
        return {}

    n = int(max_active_names or 0)
    if n <= 0:
        return weights

    # CASH nicht mitzählen
    cash = float(weights.get("CASH", 0.0)) if "CASH" in weights else None
    items = [(k, float(v)) for k, v in weights.items() if k != "CASH" and float(v) > 0.0]

    if len(items) <= n:
        return weights

    # Top-N nach Gewicht behalten
    items.sort(key=lambda kv: kv[1], reverse=True)
    kept = dict(items[:n])

    # CASH wieder hinzufügen (unverändert) – Renorm macht ggf. später ein anderer Schritt
    if cash is not None and cash > 0.0:
        kept["CASH"] = cash

    return kept
