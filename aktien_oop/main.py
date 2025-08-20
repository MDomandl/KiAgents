# main.py
from pathlib import Path
import argparse
from .config import Config, PKG_ROOT
from .runner import Runner

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
    return p.parse_args()

def main():
    cfg = Config.from_cli()
    Runner(cfg).run()

if __name__ == "__main__":
    main()
