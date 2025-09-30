from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

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

    start: str               # "YYYY-MM-DD"
    end: str                 # "YYYY-MM-DD"
    frequency: str           # "monthly" | "weekly"

    top_k: int
    buffer_k: int
    max_per_sector: Optional[int]   # None = aus
    cost_bps: float                 # Transaktionskosten (bps) pro Turnover
    slippage_bps: float             # Slippage (bps) pro Turnover
    min_history_days: int = 260     # Mindesthistorie je Ticker

    use_equal_weight: bool = False
    friction_eps: float = 0.0
    weight_round_step: float = 0.0
    max_turnover_cap: float = 0.0
    rebalance_every_n: int = 1
    benchmark_ticker: Optional[str] = "SPY"

    regime_use_filter: bool = False
    regime_sma_days: int = 200  # z.B. 200 Kalendertage
    regime_exposure_low: float = 0.50  # z.B. 50% Exposure bei "unter SMA"

    vol_target_ann: Optional[float] = None  # z.B. 0.20 für 20% p.a.; None = aus
    vol_lookback_days: int = 20  # Roll-Fenster (Handelstage) für Sigma

    min_position_weight: float = 0.0
    max_active_names: int = 0

    verbose: bool = False


# ---------------------------------------
# Hilfen
# ---------------------------------------
def load_tickers(path: str) -> List[str]:
    syms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                syms.append(s)
    return sorted(set(syms))


def load_sector_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    df = df.dropna(subset=["ticker", "sector"])
    return dict(zip(df["ticker"], df["sector"]))


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
                if df is not None and not df.empty and "Close" in df:
                    break
            except Exception as e:
                last_err = e
            time.sleep(1.0 + attempt)

        if df is None or df.empty or "Close" not in df:
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
    df = df.sort_values("score_adj", ascending=False)
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


# ---------------------------------------
# Backtester
# ---------------------------------------
class Backtester:
    def __init__(self, cfg: BTConfig):
        self.cfg = cfg
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        tickers = load_tickers(self.cfg.tickers_file)
        if self.cfg.verbose:
            print(f"{len(tickers)} Ticker aus Datei '{self.cfg.tickers_file}' geladen. "
                  f"Beispiele: {', '.join(tickers[:10])}")

        secmap = load_sector_map(self.cfg.sector_meta)

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
        regime_on = bool(getattr(self.cfg, "regime_use_filter", False))
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
                sma = s.rolling(window=f"{N}D").mean()
                bm_s, bm_sma = s, sma
            else:
                regime_on = False

        vt_on = bool(getattr(self.cfg, "vol_target_ann", None))
        vt_exp = None
        if vt_on and bm_s is not None:
            bm_ret = bm_s.pct_change().replace([np.inf, -np.inf], np.nan)
            roll = bm_ret.rolling(self.cfg.vol_lookback_days).std()
            target_daily = float(self.cfg.vol_target_ann) / np.sqrt(252.0)
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
                    rets_sub = feats["rets"].loc[ret_mask, cur_holdings].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if not rets_sub.empty:
                        w_ser = cur_weights.reindex(cur_holdings).fillna(0.0)
                        for idx, r in rets_sub.dot(w_ser).items():
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

                trades_log.append(
                    {"date": d.strftime("%Y-%m-%d"), "turnover": 0.0, "trade_cost": 0.0, "enter": "", "exit": ""})
                if self.cfg.verbose:
                    print(f"{d.date()} [SKIP n={n}] turnover=0.000 cost=0.0000 eq={equity_val:.4f}")
                continue

            # Universum mit genügend Historie
            valid_cols = []
            for t in px.columns:
                seg = px.loc[:d, t].dropna()
                if len(seg) >= self.cfg.min_history_days:
                    valid_cols.append(t)
            if not valid_cols:
                continue

            # Scoring + Filter
            rank_df = score_universe(feats, d, valid_cols)
            if rank_df.empty:
                continue
            filtered, filt_stats = apply_filters(rank_df)

            # Sektor-Limits
            # 1) Kandidaten: sector-capped, aber groß genug für den Buffer
            cand_df = enforce_sector_limits(
                filtered, secmap, self.cfg.max_per_sector, self.cfg.top_k + self.cfg.buffer_k
            )

            # 2) Puffer anwenden: alte Titel behalten, falls Rang ≤ top_k + buffer_k
            target_list = turnover_buffer(
                cand_df.sort_values("rank"), cur_holdings, self.cfg.top_k, self.cfg.buffer_k
            )

            # 3) Final: genau die top_k aus den Kandidaten
            sel = cand_df.set_index("ticker").loc[target_list].reset_index()
            sel = sel.sort_values("rank").reset_index(drop=True)

            # Gewichte
            w = build_weights(sel, self.cfg.use_equal_weight)
            sel["allocation_pct"] = (w.reindex(sel["ticker"]).values * 100.0).round(2)

            # Trades / Turnover / Kosten
            new_weights = w.copy()  # ← garantiert gesetzt
            old_weights = cur_weights.copy()

            # optionales Runden aus der TOML (z. B. 0.02 = 2 %-Punkte)
            step = float(getattr(self.cfg, "weight_round_step", 0.0) or 0.0)
            if step > 0:
                new_weights = (new_weights / step).round() * step
                s = float(new_weights.sum())
                if s > 0:
                    new_weights = new_weights / s

            # Ticker-Index als str
            new_weights.index = new_weights.index.map(str)
            old_weights.index = old_weights.index.map(str)

            # Union beider Seiten (damit Verkäufe gezählt werden)
            union = new_weights.index.union(old_weights.index)
            new_sorted = new_weights.reindex(union, fill_value=0.0).sort_index()
            old_sorted = old_weights.reindex(union, fill_value=0.0).sort_index()

            # Delta & Turnover
            delta = new_sorted - old_sorted
            raw_turnover = float(delta.abs().sum())

            # Friction: kleine Änderungen ignorieren (eps in Prozentpunkten)
            eps = float(self.cfg.friction_eps or 0.0)
            if eps > 0:
                delta = delta.where(delta.abs() >= eps, 0.0)

            turnover = float(delta.abs().sum())

            # VOR 'eff_new = old_sorted + delta' – direkt nach Friction und vor dem Cap:
            sell_zero = (new_sorted == 0) & (old_sorted > 0)
            buy_pos = (new_sorted > old_sorted)
            sell_pos = (new_sorted < old_sorted)

            # Reihenfolge: zuerst 'sell_zero', dann übrige sells, dann buys – alles im Cap-Budget
            budget = float(getattr(self.cfg, "max_turnover_cap", 0.0) or 0.0)
            if budget > 0:
                alloc = pd.Series(0.0, index=delta.index)

                # 1) Null-Ziel-Verkäufe priorisieren
                need = (-delta.where(sell_zero, 0.0)).abs()
                take = need.copy()
                if need.sum() > 0 and need.sum() > budget:
                    take *= budget / need.sum()
                alloc -= take
                budget -= float(take.sum())

                # 2) restliche Verkäufe
                if budget > 1e-12:
                    need = (-delta.where(sell_pos & ~sell_zero, 0.0)).abs()
                    if need.sum() > 0:
                        take = need * min(1.0, budget / need.sum())
                        alloc -= take
                        budget -= float(take.sum())

                # 3) Käufe (vom Restbudget)
                if budget > 1e-12:
                    need = (delta.where(buy_pos, 0.0)).abs()
                    if need.sum() > 0:
                        take = need * min(1.0, budget / need.sum())
                        alloc += take
                        budget -= float(take.sum())

                delta = alloc

            # Soft-Cap (uniform) – nur für die Cap-Entscheidung berechnen
            cap = float(getattr(self.cfg, "max_turnover_cap", 0.0) or 0.0)
            if cap > 0.0:
                t_for_cap = float(delta.abs().sum())
                if t_for_cap > cap:
                    scale = cap / t_for_cap
                    delta *= scale

            # Effektive neue Gewichte + Renorm
            eff_new = old_sorted + delta
            if eff_new.sum() > 0:
                eff_new = eff_new / eff_new.sum()

            # 1) Mini-Positionen auf 0 setzen (Dust)
            dust = float(getattr(self.cfg, "min_position_weight", 0.0) or 0.0)  # z.B. 0.01 = 1%-Punkt
            if dust > 0:
                eff_new = eff_new.where(eff_new >= dust, other=0.0)
                s = float(eff_new.sum())
                if s > 0:
                    eff_new = eff_new / s

            # 2) Max. aktive Titel begrenzen (z.B. = top_k)
            max_names = int(getattr(self.cfg, "max_active_names", 0) or 0)
            if max_names > 0 and (eff_new > 0).sum() > max_names:
                keep = eff_new.sort_values(ascending=False).head(max_names).index
                eff_new = eff_new.where(eff_new.index.isin(keep), other=0.0)
                eff_new = eff_new / eff_new.sum()

            # 3) Turnover/Kosten NACH der Finalisierung berechnen
            # --- FINALER Cap-Guard nach Dust/Limit ---
            delta_eff = eff_new - old_sorted
            turnover = float(delta_eff.abs().sum())

            cap = float(getattr(self.cfg, "max_turnover_cap", 0.0) or 0.0)
            if cap > 0.0 and turnover > cap:
                scale = cap / turnover
                eff_new = old_sorted + delta_eff * scale
                # (optional) numerisch renormalisieren:
                s = float(eff_new.sum())
                if s > 0:
                    eff_new = eff_new / s
                # Turnover neu ermitteln – jetzt garantiert ≤ cap
                delta_eff = eff_new - old_sorted
                turnover = float(delta_eff.abs().sum())

            # Kosten auf Basis des ***finalen*** Turnover
            trade_cost = turnover * (self.cfg.cost_bps + self.cfg.slippage_bps) / 10000.0

            # Zustand setzen
            cur_weights = eff_new[eff_new > 0].copy()
            cur_holdings = cur_weights.index.tolist()
            assert abs(cur_weights.sum() - 1.0) < 1e-9

            # Enter/Exit-Listen aus effektiven Trades
            # NEU: set-basierte Variante (typ-sicher)
            prev_idx = old_sorted[old_sorted > 0.0].index
            curr_idx = cur_weights[cur_weights > 0.0].index

            enter_list = [t for t in curr_idx if t not in prev_idx]
            exit_list = [t for t in prev_idx if t not in curr_idx]

            # Positions-Log aus cur_weights bauen
            sel = pd.DataFrame({"ticker": cur_holdings})
            sel["allocation_pct"] = (cur_weights.reindex(cur_holdings).values * 100.0).round(2)
            if secmap:
                sel["sector"] = sel["ticker"].map(secmap).fillna("UNKNOWN")

            trades_log.append({
                "date": d.strftime("%Y-%m-%d"),
                "turnover": turnover,
                "trade_cost": trade_cost,
                "enter": ",".join(enter_list),
                "exit": ",".join(exit_list),
            })

            # (Optional) Debug
            if self.cfg.verbose:
                print(f"[FRICTION] raw_turnover={raw_turnover:.3f}  eff_turnover={turnover:.3f}  eps={eps:.3f}")

            # 3d) Performance bis zum nächsten Rebalance-Datum (exklusiv)
            # exklusives Ende: alle Tage >= d und < d_next
            # (d, d_next] → Tage NACH dem Rebalance bis inkl. d_next
            ret_mask = (feats["rets"].index > d) & (feats["rets"].index <= d_next)
            rets_sub = feats["rets"].loc[ret_mask, cur_holdings]

            # Kosten einmalig am Rebalance-Tag buchen
            equity_val *= (1.0 - trade_cost)
            equity_rows.append({"date": pd.Timestamp(d), "equity": equity_val})

            if not rets_sub.empty:
                # saubere, label-sichere Gewichtung
                w_ser = cur_weights.reindex(cur_holdings).fillna(0.0)
                rets_sub = rets_sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # Tagesweise fortschreiben
                base_port_rets = rets_sub.dot(w_ser)

                exp = pd.Series(1.0, index=base_port_rets.index)
                if regime_on and bm_s is not None and bm_sma is not None:
                    sig = (bm_s.reindex(exp.index) >= bm_sma.reindex(exp.index)).fillna(True)
                    low = float(self.cfg.regime_exposure_low)
                    exp = exp * (sig.astype(float) * (1.0 - low) + low)

                if vt_on and vt_exp is not None:
                    exp = exp * vt_exp.reindex(exp.index).fillna(1.0)

                port_rets = base_port_rets * exp

                for idx, r in port_rets.items():
                    equity_val *= (1.0 + float(r))
                    equity_rows.append({"date": pd.Timestamp(idx), "equity": equity_val})

            # Positionen loggen
            tmp = sel.copy()
            tmp.insert(0, "as_of", d.strftime("%Y-%m-%d"))
            if secmap:
                tmp["sector"] = tmp["ticker"].map(secmap).fillna("UNKNOWN")
            positions_log.append(tmp)

            if self.cfg.verbose:
                print(f"{d.date()} holdings={cur_holdings} turnover_eff={turnover:.3f} cost={trade_cost:.4f} eq={equity_val:.4f} filt={filt_stats}")

        # 4) Ergebnisse schreiben
        if equity_rows:
            eq_df = pd.DataFrame(equity_rows).drop_duplicates(subset=["date"]).set_index("date").sort_index()
            eq_df.to_csv(eq_path)
        else:
            eq_df = pd.DataFrame(columns=["equity"])

        if positions_log:
            pos_df = pd.concat(positions_log, ignore_index=True)
            pos_df.to_csv(pos_path, index=False)
        else:
            pos_df = pd.DataFrame(columns=[])

        trd_df = pd.DataFrame(trades_log)
        trd_df.to_csv(trd_path, index=False)

        # 5) Kennzahlen + Summary
        if not eq_df.empty and "equity" in eq_df:
            eq = eq_df["equity"]
            total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
            n_years = max(1e-9, (eq.index[-1] - eq.index[0]).days / 365.25)
            base = float(eq.iloc[0])
            cagr = ((float(eq.iloc[-1]) / base) ** (1.0 / n_years) - 1.0)

            eq_ret = eq.pct_change(fill_method=None).dropna()
            if eq_ret.empty:
                print("Zu wenig Datenpunkte für Kennzahlen (Equity hat < 2 Zeitpunkte).")
            vol_ann = float(eq_ret.std() * np.sqrt(252)) if not eq_ret.empty else float("nan")
            sharpe = float(eq_ret.mean() / eq_ret.std() * np.sqrt(252)) if eq_ret.std() > 0 else float("nan")
            mdd, dd_start, dd_end = calc_drawdown(eq)

            avg_turnover = float(trd_df["turnover"].mean()) if not trd_df.empty else 0.0
            avg_cost = float(trd_df["trade_cost"].mean()) if not trd_df.empty else 0.0

            summary_lines = [
                f"Backtest  {self.cfg.start} → {self.cfg.end}  "
                f"({self.cfg.frequency}, top_k={self.cfg.top_k}, buffer_k={self.cfg.buffer_k}, "
                f"max_per_sector={self.cfg.max_per_sector})",
                f"Total Return: {total_return: .2%}",
                f"CAGR:         {cagr: .2%}",
                f"Volatility:   {vol_ann: .2%}",
                f"Sharpe(0%):   {sharpe: .2f}",
                f"MaxDD:        {mdd: .2%}  von {dd_start.date()} bis {dd_end.date()}",
                f"Avg Turnover: {avg_turnover: .2%}  |  Avg Cost: {avg_cost: .4%}",
                f"Equity Curve: {eq_path}",
                f"Positions:    {pos_path}",
                f"Trades:       {trd_path}",
            ]
            # --- Benchmark ---
            bm = self.cfg.benchmark_ticker
            if bm:
                bm_px = download_close([bm], self.cfg.start, self.cfg.end, verbose=False)
                if not bm_px.empty:
                    s = bm_px.iloc[:, 0].reindex(eq_df.index).dropna()
                    bm_eq = (s / s.iloc[0]).rename("benchmark")
                    bm_ret = bm_eq.pct_change(fill_method=None).dropna()

                    eq = eq_df["equity"].reindex(bm_eq.index).dropna()
                    eq_ret = eq.pct_change(fill_method=None).dropna()

                    # Kennzahlen
                    bm_total = float(bm_eq.iloc[-1] - 1.0)
                    years = max(1e-9, (bm_eq.index[-1] - bm_eq.index[0]).days / 365.25)
                    bm_cagr = float(bm_eq.iloc[-1] ** (1.0 / years) - 1.0)
                    # NEU: Benchmark-Volatilität & Sharpe(0%)
                    bm_vol_ann = float(bm_ret.std() * np.sqrt(252)) if not bm_ret.empty else float("nan")
                    bm_sharpe = float(bm_ret.mean() / bm_ret.std() * np.sqrt(252)) if bm_ret.std() > 0 else float("nan")

                    # Portfolio vs. Benchmark (optional, praktisch für Vergleich auf einen Blick)
                    rel_vol = float(vol_ann / bm_vol_ann) if bm_vol_ann > 0 else float("nan")
                    alpha = float((eq_ret.mean() - bm_ret.mean()) * 252)  # ~annualisierte Exzessrendite
                    corr = float(eq_ret.corr(bm_ret))

                    # speichern
                    bm_path = Path(str(out_prefix) + "_benchmark.csv")
                    pd.concat([eq, bm_eq], axis=1).to_csv(bm_path)

                    # Summary erweitern
                    summary_lines.extend([
                        f"Benchmark:     {bm}",
                        f"BM Total Ret:  {bm_total: .2%}   |  BM CAGR: {bm_cagr: .2%}",
                        f"BM Volatility: {bm_vol_ann: .2%} |  BM Sharpe(0%): {bm_sharpe: .2f}",
                        f"Alpha (ann.):  {alpha: .2%}     |  Corr(EQ,BM): {corr: .2f}",
                        f"Rel. Vol (Port/BM): {rel_vol: .2f}x",
                        f"Benchmark CSV: {bm_path}",
                    ])

            Path(summ_path).write_text("\n".join(summary_lines), encoding="utf-8")
            print("\n".join(summary_lines))

            # optional: extra Artefakte (Plots/JSON), nur wenn vorhanden
            if _HAS_REPORTING:
                try:
                    from aktien_oop.reporting import write_equity_artifacts
                    params = {
                        "tickers_file": self.cfg.tickers_file,
                        "sector_meta": self.cfg.sector_meta,
                        "save_dir": self.cfg.save_dir,
                        "start": self.cfg.start,
                        "end": self.cfg.end,
                        "frequency": self.cfg.frequency,
                        "top_k": self.cfg.top_k,
                        "buffer_k": self.cfg.buffer_k,
                        "max_per_sector": self.cfg.max_per_sector,
                        "cost_bps": self.cfg.cost_bps,
                        "slippage_bps": self.cfg.slippage_bps,
                        "min_history_days": self.cfg.min_history_days,
                        "verbose": self.cfg.verbose,
                    }
                    arts = write_equity_artifacts(
                        equity_or_returns_df=eq_df.reset_index()[["date", "equity"]],
                        out_prefix=out_prefix,
                        frequency=self.cfg.frequency,
                        params=params,
                        write_params_json=True,
                    )
                    print("\nArtefakte erzeugt:")
                    print(f"  {arts.equity_csv}")
                    print(f"  {arts.drawdown_csv}")
                    print(f"  {arts.equity_png}")
                    print(f"  {arts.drawdown_png}")
                    print(f"  {arts.summary_txt}")
                    if arts.params_json:
                        print(f"  {arts.params_json}")
                except Exception as e:
                    print(f"(Hinweis) Reporting übersprungen: {e}")

            # params.json zusätzlich
            Path(str(out_prefix) + "_params.json").write_text(
                json.dumps(vars(self.cfg), indent=2, default=str), encoding="utf-8"
            )
        else:
            print("Keine Equity-Curve erzeugt (möglicherweise kein Rebalance-Intervall mit Daten).")


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


def _coalesce(cli_val, cfg_val, default):
    return cli_val if cli_val is not None else (cfg_val if cfg_val is not None else default)


def _build_cfg_from_config_and_cli(a: argparse.Namespace) -> BTConfig:
    d = _load_toml(a.config) if a.config else {}

    return BTConfig(
        tickers_file     = _coalesce(a.tickers,           d.get("tickers_file"),   "aktien_oop/sp500_tickers.txt"),
        sector_meta      = _coalesce(a.sector_meta,       d.get("sector_meta"),    None),
        save_dir         = _coalesce(a.save_dir,          d.get("save_dir"),       "aktien_oop"),

        start            = _coalesce(a.start,             d.get("start"),          "2018-01-01"),
        end              = _coalesce(a.end,               d.get("end"),            "2025-08-16"),
        frequency        = _coalesce(a.frequency,         d.get("frequency"),      "monthly"),

        top_k            = _coalesce(a.top_k,             d.get("top_k"),          8),
        buffer_k         = _coalesce(a.buffer_k,          d.get("buffer_k"),       2),
        max_per_sector   = _coalesce(a.max_per_sector,    d.get("max_per_sector"), None),
        cost_bps         = _coalesce(a.cost_bps,          d.get("cost_bps"),       10.0),
        slippage_bps     = _coalesce(a.slippage_bps,      d.get("slippage_bps"),   5.0),
        min_history_days = _coalesce(a.min_history_days,  d.get("min_history_days"), 260),

        use_equal_weight=_coalesce(a.use_equal_weight if a.use_equal_weight else None, d.get("use_equal_weight"), False),
        friction_eps=_coalesce(a.friction_eps, d.get("friction_eps"), 0.0),
        weight_round_step=_coalesce(getattr(a, "weight_round_step", None), d.get("weight_round_step"), 0.0),
        max_turnover_cap=_coalesce(getattr(a, "max_turnover_cap", None), d.get("max_turnover_cap"), 0.0),
        rebalance_every_n=_coalesce(getattr(a, "rebalance_every_n", None), d.get("rebalance_every_n"), 1),
        benchmark_ticker=_coalesce(a.benchmark_ticker, d.get("benchmark_ticker"), "SPY"),

        min_position_weight=_coalesce(getattr(a, "min_position_weight", None), d.get("min_position_weight"), 0.0),
        max_active_names=_coalesce(getattr(a, "max_active_names", None), d.get("max_active_names"), 0),

        # Regime
        regime_use_filter=_coalesce(
            (a.regime_use_filter if getattr(a, "regime_use_filter", False) else None),
            d.get("regime_use_filter"),
            False
        ),
        regime_sma_days=_coalesce(
            getattr(a, "regime_sma_days", None),
            d.get("regime_sma_days"),
            200
        ),
        regime_exposure_low=_coalesce(
            getattr(a, "regime_exposure_low", None),
            d.get("regime_exposure_low"),
            0.50
        ),

        # Vol-Targeting
        vol_target_ann=_coalesce(
            getattr(a, "vol_target_ann", None),
            d.get("vol_target_ann"),
            None
        ),
        vol_lookback_days=_coalesce(
            getattr(a, "vol_lookback_days", None),
            d.get("vol_lookback_days"),
            20
        ),

        verbose=_coalesce(a.verbose if a.verbose else None, d.get("verbose"), False),
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
    ap.add_argument("--max-per-sector", type=int)
    ap.add_argument("--cost-bps", type=float)
    ap.add_argument("--slippage-bps", type=float)
    ap.add_argument("--min-history-days", type=int)

    ap.add_argument("--use-equal-weight", action="store_true")
    ap.add_argument("--friction-eps", type=float)
    ap.add_argument("--weight-round-step", type=float)
    ap.add_argument("--max-turnover-cap", type=float)
    ap.add_argument("--rebalance-every-n", type=int)
    ap.add_argument("--benchmark-ticker")

    ap.add_argument("--regime-use-filter", action="store_true")
    ap.add_argument("--regime-sma-days", type=int)
    ap.add_argument("--regime-exposure-low", type=float)

    ap.add_argument("--vol-target-ann", type=float)  # None = aus
    ap.add_argument("--vol-lookback-days", type=int)

    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = _build_cfg_from_config_and_cli(a)
    # <= 0 als deaktiviert interpretieren
    if isinstance(cfg.max_per_sector, int) and cfg.max_per_sector <= 0:
        cfg.max_per_sector = None
    Backtester(cfg).run()


if __name__ == "__main__":
    main()
