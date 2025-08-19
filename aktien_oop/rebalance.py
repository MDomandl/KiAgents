# rebalance.py
from typing import Optional, Dict
import pandas as pd
import numpy as np
from .store import PortfolioStore   # relativ, falls du als Paket startest

class Rebalancer:
    def __init__(self, store: PortfolioStore, top_k: int, buffer_k: int, force: bool):
        self.store, self.top_k, self.buffer_k, self.force = store, top_k, buffer_k, force

    @staticmethod
    def inverse_vol_allocation(df_sel: pd.DataFrame) -> pd.Series:
        vols = df_sel["volatility"].replace(0, np.nan)
        inv = 1.0 / vols
        total = float(inv.sum())
        if total == 0 or np.isnan(total):
            return pd.Series([np.nan] * len(df_sel), index=df_sel.index)
        return (inv / total * 100).round(2)

    def should_rebalance(self) -> bool:
        if self.force: return True
        prev = self.store.read_positions()
        if prev.empty: return True
        last = pd.to_datetime(prev["as_of"].max(), errors="coerce")
        if pd.isna(last): return True
        return last.strftime("%Y-%m") != pd.Timestamp.now().strftime("%Y-%m")

    @staticmethod
    def _sector_of(ticker: str, sector_map: dict[str, str]) -> str:
        return sector_map.get(str(ticker), "Unknown")

    @staticmethod
    def _allowed(sector: str, counts: dict[str, int],
                 max_per_sector: int | None,
                 sector_limits: dict[str, int] | None) -> bool:
        # spezifisches Limit hat Vorrang
        if sector_limits and sector in sector_limits:
            lim = sector_limits[sector]
            return counts.get(sector, 0) < lim
        # globales Limit (falls gesetzt)
        if max_per_sector is not None:
            return counts.get(sector, 0) < max_per_sector
        # kein Limit → immer erlaubt
        return True

    def _trim_keep_to_limits(self, keep: pd.DataFrame, ranked_df: pd.DataFrame,
                             sector_map: dict[str, str],
                             max_per_sector: int | None,
                             sector_limits: dict[str, int] | None) -> pd.DataFrame:
        if keep.empty: return keep
        counts: dict[str, int] = {}
        # Zähle je Sektor
        for t in keep["ticker"]:
            s = self._sector_of(t, sector_map)
            counts[s] = counts.get(s, 0) + 1
        # Entferne Überschüsse je Sektor (schlechtest-gerankte zuerst)
        changed = False
        for sector, cnt in list(counts.items()):
            # Ziel-Limit ermitteln
            lim = None
            if sector_limits and sector in sector_limits:
                lim = sector_limits[sector]
            elif max_per_sector is not None:
                lim = max_per_sector
            if lim is not None and cnt > lim:
                # schneide die schlechtesten in diesem Sektor ab
                sub = keep[keep["ticker"].map(lambda x: self._sector_of(x, sector_map) == sector)]
                drop_n = cnt - lim
                if drop_n > 0:
                    to_drop = sub.sort_values("rank", ascending=False).head(drop_n).index
                    keep = keep.drop(index=to_drop)
                    changed = True
        return keep.sort_values("rank").reset_index(drop=True) if changed else keep

    def select_with_buffer(
        self,
        ranked_df: pd.DataFrame,
        prev_positions: pd.DataFrame,
        top_k: Optional[int] = None,
        buffer_k: Optional[int] = None,
        *,
        sector_map: Optional[Dict[str, str]] = None,
        max_per_sector: Optional[int] = None,
        sector_limits: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        top_k = top_k or self.top_k
        buffer_k = buffer_k or self.buffer_k

        keep = pd.DataFrame(columns=ranked_df.columns)
        if prev_positions is not None and not prev_positions.empty:
            prev_tickers = set(prev_positions["ticker"].astype(str))
            sub = ranked_df[ranked_df["ticker"].astype(str).isin(prev_tickers)]
            keep = sub.loc[sub["rank"] <= buffer_k].sort_values("rank").head(top_k)

        # optional: Limits auf keep anwenden (falls du die Logik schon eingebaut hast)
        if sector_map and (sector_limits or (max_per_sector is not None)):
            keep = self._trim_keep_to_limits(keep, ranked_df, sector_map, max_per_sector, sector_limits)

        need = max(0, top_k - len(keep))
        filler_pool = ranked_df[~ranked_df["ticker"].isin(keep["ticker"])].sort_values("rank")

        if sector_map and (sector_limits or (max_per_sector is not None)):
            counts: Dict[str, int] = {}
            for t in keep["ticker"]:
                sec = sector_map.get(str(t), "Unknown")
                counts[sec] = counts.get(sec, 0) + 1

            chosen = []
            for _, row in filler_pool.iterrows():
                if len(chosen) >= need:
                    break
                sec = sector_map.get(str(row["ticker"]), "Unknown")
                # spezifisches Limit hat Vorrang
                lim = (sector_limits or {}).get(sec, max_per_sector)
                if lim is None or counts.get(sec, 0) < lim:
                    chosen.append(row)
                    counts[sec] = counts.get(sec, 0) + 1
            filler = pd.DataFrame(chosen)
        else:
            filler = filler_pool.head(need)

        sel = (
            pd.concat([keep, filler], ignore_index=True)
              .sort_values("rank")
              .head(top_k)
              .reset_index(drop=True)
        )
        return sel
