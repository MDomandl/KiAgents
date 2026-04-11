# store.py
from pathlib import Path
import json
import csv
import logging
import pandas as pd

class PortfolioStore:

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.positions_path = self.save_dir / "portfolio_positions.csv"
        self.rankings_log   = self.save_dir / "rankings_log.csv"
        self.runs_log       = self.save_dir / "runs_log.csv"
        self.topk_log       = self.save_dir / "topk_log.csv"
        self.runs_meta_jsonl = self.save_dir / "runs_meta.jsonl"

    def load_positions_before(self, as_of: str | pd.Timestamp) -> pd.DataFrame | None:
        p = self.positions_path
        if not p.exists():
            return None

        df = pd.read_csv(p)
        if df.empty:
            return None

        if "as_of" not in df.columns:
            logging.warning("positions.csv hat keine 'as_of'-Spalte – kann nicht 'before(as_of)' selektieren.")
            return None

        df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
        cur = pd.Timestamp(as_of).normalize()

        df_prev = df[df["as_of"] < cur]
        if df_prev.empty:
            return None

        prev_asof = df_prev["as_of"].max()
        snap = df_prev[df_prev["as_of"] == prev_asof].copy()
        if "ticker" in snap.columns:
            snap = snap.sort_values("ticker", kind="mergesort").reset_index(drop=True)
        return snap

    def load_positions(self) -> pd.DataFrame:
        if not self.positions_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.positions_path)
        if "as_of" in df.columns:
            df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
        return df

    def save_positions(self, df: pd.DataFrame) -> None:
        df = df.copy()
        if "as_of" not in df.columns:
            df.insert(0, "as_of", pd.Timestamp.now())
        if "ticker" not in df.columns:
            raise ValueError("positions must include a 'ticker' column")

        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["as_of", "ticker"])
        df["ticker"] = df["ticker"].astype(str)

        if self.positions_path.exists():
            existing = pd.read_csv(self.positions_path)
            if not existing.empty:
                if "as_of" not in existing.columns or "ticker" not in existing.columns:
                    logging.warning("positions.csv schema is missing 'as_of' or 'ticker' - replacing with current snapshot.")
                    combined = df
                else:
                    existing["as_of"] = pd.to_datetime(existing["as_of"], errors="coerce").dt.normalize()
                    existing["ticker"] = existing["ticker"].astype(str)
                    combined = pd.concat([existing, df], ignore_index=True, sort=False)
            else:
                combined = df
        else:
            combined = df

        combined = combined.drop_duplicates(subset=["as_of", "ticker"], keep="last")
        combined = combined.sort_values(["as_of", "ticker"], kind="mergesort").reset_index(drop=True)
        combined.to_csv(self.positions_path, index=False)

    def last_rebalance_time(self):
        """Jüngsten Timestamp aus runs_log oder positions ermitteln (robust)."""
        ts = None
        if self.runs_log.exists():
            try:
                r = pd.read_csv(self.runs_log, engine="python")
                if not r.empty and "as_of" in r.columns:
                    r["as_of"] = pd.to_datetime(r["as_of"], errors="coerce")
                    if not r["as_of"].isna().all():
                        ts = r["as_of"].max()
            except Exception:
                # Falls die Datei mal wieder „krumm“ ist, einfach ignorieren.
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
            # 2. Fallback: Zeilen manuell parsen und Überlauf in 'Flags' zurückführen
            rows = []
            with open(self.topk_log, "r", encoding="utf-8", newline="") as f:
                rdr = csv.reader(f, delimiter=",", quotechar='"', doublequote=True)
                header = next(rdr, None)  # Header überspringen
                for parts in rdr:
                    if len(parts) > 7:
                        parts = parts[:6] + [",".join(parts[6:])]  # Rest wieder an Flags anhängen
                    elif len(parts) < 7:
                        # fehlende Spalten auffüllen, damit der Frame passt
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

    # ⬇️ neu
    def append_jsonl(self, path: Path, record: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
