# main.py
from pathlib import Path
import argparse
from .config import Config, PKG_ROOT
from .runner import Runner
from types import SimpleNamespace
from .universe import load_sp500_tickers, load_sp500_meta



def _resolve_universe_paths(cfg):
    """Bestimme Dateien für Universe. Unterstützt neue und alte Config-Felder."""
    u = getattr(cfg, "universe", None)
    tfile = getattr(u, "tickers_file", None) if u else None
    mfile = getattr(u, "meta_file", None) if u else None
    # mögliche CLI/Config-Felder
    tfile = tfile or getattr(cfg, "tickers", None)
    mfile = mfile or getattr(cfg, "sector_meta", None)
    # Fallback auf Package-Root
    tfile = tfile or (PKG_ROOT / "sp500_tickers.txt")
    mfile = mfile or (PKG_ROOT / "sp500_meta.csv")
    return Path(tfile), Path(mfile)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=Path, default=PKG_ROOT / "sp500_tickers.txt")
    p.add_argument("--sector-meta", type=Path, default=PKG_ROOT / "sp500_meta.csv")
    p.add_argument("--save-dir", type=Path, default=PKG_ROOT)   # alles hierhin schreiben
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--top-k", type=int)
    p.add_argument("--buffer-k", type=int)
    p.add_argument("--rebalance-frequency", choices=["monthly", "weekly"])
    p.add_argument("--force", "--force-rebalance",
                   action="store_true", dest="force_rebalance",
                   help="Rebalancing erzwingen (überspringt Perioden-Check)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--as-of", type=str, help="YYYY-MM-DD: nutze Preise bis inkl. diesem Handelstag")
    return p.parse_args()


def main():
    cfg = Config.from_cli()
    # Universe auflösen und anhängen
    tickers_file, meta_file = _resolve_universe_paths(cfg)
    tickers = load_sp500_tickers(str(tickers_file))
    meta = load_sp500_meta(str(meta_file))
    object.__setattr__(cfg, "universe", SimpleNamespace(
        tickers=tickers,
        meta=meta,
        tickers_file=str(tickers_file),
        meta_file=str(meta_file),
    ))
    cfg.universe.tickers = tickers
    cfg.universe.meta = meta
    cfg.universe.tickers_file = str(tickers_file)
    cfg.universe.meta_file = str(meta_file)
    Runner(cfg).run()

if __name__ == "__main__":
    main()
