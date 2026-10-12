# store.py
from pathlib import Path
from dataclasses import dataclass
import json
import csv
import logging
import pandas as pd


@dataclass(frozen=True)
class PositionSnapshot:
    as_of: pd.Timestamp
    positions: pd.DataFrame


class PortfolioStore:

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.positions_path = self.save_dir / "portfolio_positions.csv"
        self.snapshots_path = self.save_dir / "portfolio_snapshots.csv"
        self.positions_pending_path = self.save_dir / "portfolio_positions.csv.pending"
        self.snapshots_pending_path = self.save_dir / "portfolio_snapshots.csv.pending"
        self.snapshots_staging_path = self.save_dir / "portfolio_snapshots.csv.pending.tmp"
        self._complete_pending_snapshot_write()
        self.rankings_log   = self.save_dir / "rankings_log.csv"
        self.runs_log       = self.save_dir / "runs_log.csv"
        self.topk_log       = self.save_dir / "topk_log.csv"
        self.runs_meta_jsonl = self.save_dir / "runs_meta.jsonl"

    @staticmethod
    def _replace_file(source: Path, target: Path) -> None:
        source.replace(target)

    def _snapshot_paths_for_read(self) -> tuple[Path, Path]:
        if self.snapshots_pending_path.exists():
            positions_path = (
                self.positions_pending_path
                if self.positions_pending_path.exists()
                else self.positions_path
            )
            return positions_path, self.snapshots_pending_path
        return self.positions_path, self.snapshots_path

    def _complete_pending_snapshot_write(self) -> None:
        if not self.snapshots_pending_path.exists():
            return
        if self.positions_pending_path.exists():
            self._replace_file(self.positions_pending_path, self.positions_path)
        self._replace_file(self.snapshots_pending_path, self.snapshots_path)

    def load_positions_snapshot_before(
        self,
        as_of: str | pd.Timestamp,
    ) -> PositionSnapshot | None:
        positions_path, snapshots_path = self._snapshot_paths_for_read()
        if positions_path.exists():
            df = pd.read_csv(positions_path)
        else:
            df = pd.DataFrame()

        position_dates = pd.Series(dtype="datetime64[ns]")
        if "as_of" in df.columns:
            df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.normalize()
            position_dates = df["as_of"].dropna()
        elif not df.empty:
            logging.warning("positions.csv hat keine 'as_of'-Spalte â€“ kann nicht 'before(as_of)' selektieren.")

        indexed_dates = pd.Series(dtype="datetime64[ns]")
        if snapshots_path.exists():
            snapshots = pd.read_csv(snapshots_path)
            if "as_of" in snapshots.columns:
                indexed_dates = pd.to_datetime(
                    snapshots["as_of"], errors="coerce"
                ).dropna().dt.normalize()
            elif not snapshots.empty:
                logging.warning("portfolio_snapshots.csv hat keine 'as_of'-Spalte und wird ignoriert.")

        snapshot_dates = pd.concat([position_dates, indexed_dates], ignore_index=True)
        cur = pd.Timestamp(as_of).normalize()
        previous_dates = snapshot_dates[snapshot_dates < cur]
        if previous_dates.empty:
            return None

        prev_asof = previous_dates.max()
        if "as_of" in df.columns:
            snap = df[df["as_of"] == prev_asof].copy()
        else:
            snap = df.iloc[0:0].copy()
        if "ticker" in snap.columns:
            snap = snap.sort_values("ticker", kind="mergesort").reset_index(drop=True)
        return PositionSnapshot(as_of=prev_asof, positions=snap)

    def load_positions_before(self, as_of: str | pd.Timestamp) -> pd.DataFrame | None:
        snapshot = self.load_positions_snapshot_before(as_of)
        return None if snapshot is None else snapshot.positions

    def load_positions(self) -> pd.DataFrame:
        if not self.positions_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.positions_path)
        if "as_of" in df.columns:
            df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
        return df

    def save_positions(
        self,
        df: pd.DataFrame,
        *,
        snapshot_as_of: str | pd.Timestamp | None = None,
    ) -> None:
        self._complete_pending_snapshot_write()
        df = df.copy()
        if "as_of" not in df.columns:
            df.insert(0, "as_of", pd.Timestamp.now())
        if "ticker" not in df.columns:
            raise ValueError("positions must include a 'ticker' column")

        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["as_of", "ticker"])
        df["ticker"] = df["ticker"].astype(str)

        replacement_as_of = set(df["as_of"])
        if snapshot_as_of is not None:
            explicit_as_of = pd.to_datetime(snapshot_as_of, errors="coerce")
            if pd.isna(explicit_as_of):
                raise ValueError("snapshot_as_of must be a valid timestamp")
            replacement_as_of.add(pd.Timestamp(explicit_as_of).normalize())

        if self.positions_path.exists():
            existing = pd.read_csv(self.positions_path)
            if not existing.empty:
                if "as_of" not in existing.columns or "ticker" not in existing.columns:
                    logging.warning("positions.csv schema is missing 'as_of' or 'ticker' - replacing with current snapshot.")
                    combined = df
                else:
                    existing["as_of"] = pd.to_datetime(existing["as_of"], errors="coerce").dt.normalize()
                    existing["ticker"] = existing["ticker"].astype(str)
                    existing = existing[~existing["as_of"].isin(replacement_as_of)]
                    combined = existing if df.empty else pd.concat([existing, df], ignore_index=True, sort=False)
            else:
                combined = df
        else:
            combined = df

        combined = combined.drop_duplicates(subset=["as_of", "ticker"], keep="last")
        combined = combined.sort_values(["as_of", "ticker"], kind="mergesort").reset_index(drop=True)

        snapshot_dates = pd.Series(dtype="datetime64[ns]")
        if self.snapshots_path.exists():
            snapshots = pd.read_csv(self.snapshots_path)
            if "as_of" in snapshots.columns:
                snapshot_dates = pd.to_datetime(
                    snapshots["as_of"], errors="coerce"
                ).dropna().dt.normalize()
            elif not snapshots.empty:
                logging.warning("portfolio_snapshots.csv hat keine 'as_of'-Spalte und wird ersetzt.")

        all_snapshot_dates = sorted(set(snapshot_dates) | replacement_as_of)
        combined.to_csv(self.positions_pending_path, index=False)
        pd.DataFrame({"as_of": all_snapshot_dates}).to_csv(
            self.snapshots_staging_path,
            index=False,
        )
        self._replace_file(self.snapshots_staging_path, self.snapshots_pending_path)
        self._replace_file(self.positions_pending_path, self.positions_path)
        self._replace_file(self.snapshots_pending_path, self.snapshots_path)

    def last_rebalance_time(self):
        """JÃ¼ngsten Timestamp aus runs_log oder positions ermitteln (robust)."""
        ts = None
        if self.runs_log.exists():
            try:
                r = pd.read_csv(self.runs_log, engine="python")
                if not r.empty and "as_of" in r.columns:
                    r["as_of"] = pd.to_datetime(r["as_of"], errors="coerce")
                    if not r["as_of"].isna().all():
                        ts = r["as_of"].max()
            except Exception:
                # Falls die Datei mal wieder â€žkrummâ€œ ist, einfach ignorieren.
                pass

        if ts is None and self.positions_path.exists():
            p = pd.read_csv(self.positions_path)
            if "as_of" in p.columns:
                p["as_of"] = pd.to_datetime(p["as_of"], errors="coerce")
                if not p["as_of"].isna().all():
                    ts = p["as_of"].max()
        return ts

    def append_run(self, **meta) -> None:
        """
        Stabil: feste Spalten, Rest als JSON.
        Verhindert CSV-Schema-Drift (unterschiedliche Spaltenanzahlen).
        """
        payload = {
            "as_of": (meta.get("as_of") or pd.Timestamp.now().isoformat()),
            "rebalance_frequency": meta.get("rebalance_frequency"),  # <- weekly / monthly
            # feste (optionale) Meta-Felder:
            "universe_size": meta.get("universe_size"),
            "top_k": meta.get("top_k"),
            "buffer_k": meta.get("buffer_k"),
            "max_per_sector": meta.get("max_per_sector"),
            "sector_limits_on": bool(meta.get("sector_limits_on", False)),
            "tickers_file": str(meta.get("tickers_file") or ""),
            "sector_meta_file": str(meta.get("sector_meta_file") or ""),
            # alles andere als JSON:
            "meta_json": json.dumps(
                {k: v for k, v in meta.items() if k not in {
                    "universe_size","top_k","buffer_k","max_per_sector",
                    "sector_limits_on","tickers_file","sector_meta_file"
                }},
                ensure_ascii=False
            )
        }
        df = pd.DataFrame([payload])
        header = not self.runs_log.exists()
        df.to_csv(self.runs_log, mode="a", header=header, index=False)

    def read_positions(self) -> pd.DataFrame:
        if not self.positions_path.exists():
            return pd.DataFrame(columns=["as_of","ticker","allocation_pct","rank","score"])
        return pd.read_csv(self.positions_path)

    def load_last_topk(self) -> pd.DataFrame:
        if not self.topk_log.exists():
            return pd.DataFrame()
        expected = ["Ticker", "Weight", "Rank", "Score", "Vol", "Sector", "Flags"]

        try:
            # 1. Versuch: normales CSV mit Quotes korrekt interpretieren
            df = pd.read_csv(
                self.topk_log,
                engine="python",
                sep=",",
                quotechar='"',
                header=0,
                names=expected,
                on_bad_lines="error"  # wechsle auf 'error', damit wir sauber in den Fallback springen
            )
        except Exception:
            # 2. Fallback: Zeilen manuell parsen und Ãœberlauf in 'Flags' zurÃ¼ckfÃ¼hren
            rows = []
            with open(self.topk_log, "r", encoding="utf-8", newline="") as f:
                rdr = csv.reader(f, delimiter=",", quotechar='"', doublequote=True)
                header = next(rdr, None)  # Header Ã¼berspringen
                for parts in rdr:
                    if len(parts) > 7:
                        parts = parts[:6] + [",".join(parts[6:])]  # Rest wieder an Flags anhÃ¤ngen
                    elif len(parts) < 7:
                        # fehlende Spalten auffÃ¼llen, damit der Frame passt
                        parts += [""] * (7 - len(parts))
                    rows.append(parts)
            df = pd.DataFrame(rows, columns=expected)
        if df.empty or "as_of" not in df.columns:
            return pd.DataFrame()
        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
        last_ts = df["as_of"].max()
        return df[df["as_of"] == last_ts].copy()

    @staticmethod
    def append_csv(path: Path, df: pd.DataFrame):
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")

    def write_positions(self, df: pd.DataFrame):
        df.to_csv(self.positions_path, index=False, encoding="utf-8")

    # â¬‡ï¸ neu
    def append_jsonl(self, path: Path, record: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

