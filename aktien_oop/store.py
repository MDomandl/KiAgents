# store.py
from pathlib import Path
import json
import pandas as pd

class PortfolioStore:

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.positions_path = self.save_dir / "portfolio_positions.csv"
        self.rankings_log   = self.save_dir / "rankings_log.csv"
        self.runs_log       = self.save_dir / "runs_log.csv"
        self.topk_log       = self.save_dir / "topk_log.csv"
        self.runs_meta_jsonl = self.save_dir / "runs_meta.jsonl"   # ⬅️ neu

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
        df.to_csv(self.positions_path, index=False)

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
            "as_of": pd.Timestamp.now().isoformat(),
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
        df = pd.read_csv(self.topk_log)
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
