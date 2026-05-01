# main.py
from pathlib import Path
import argparse
from .config import Config, PKG_ROOT
from .runner import Runner
from types import SimpleNamespace
from .universe import load_tickers, load_meta, universe_hash



def _resolve_universe_paths(cfg):
    """Bestimme Dateien für Universe. Unterstützt neue und alte Config-Felder."""
    u = getattr(cfg, "universe", None)
    tfile = getattr(u, "tickers_file", None) if u else None
    mfile = getattr(u, "meta_file", None) if u else None
    # mögliche CLI/Config-Felder
    tfile = tfile or getattr(cfg, "tickers_file", None) or getattr(cfg, "tickers", None)
    mfile = mfile or getattr(cfg, "sector_meta_file", None) or getattr(cfg, "sector_meta", None)
    return Path(tfile), Path(mfile)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=Path)
    p.add_argument("--sector-meta", type=Path)
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
    tickers = load_tickers(str(tickers_file))
    meta = load_meta(str(meta_file))
    object.__setattr__(cfg, "universe_data", SimpleNamespace(
        name=getattr(cfg.universe, "name", "sp500"),
        tickers=tickers,
        meta=meta,
        tickers_file=str(tickers_file),
        meta_file=str(meta_file),
        hash=universe_hash(tickers),
    ))
    Runner(cfg).run()

if __name__ == "__main__":
    main()
