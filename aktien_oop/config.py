# config.py
from dataclasses import dataclass
from pathlib import Path
import argparse, logging, json
from typing import Optional

# -----------------------
# Aliase (falls genutzt)
# -----------------------
ALIAS_FILE = Path("alias_map.json")
DEFAULT_ALIASES = {"BRK.B":"BRK-B","BRK.A":"BRK-A","BF.B":"BF-B","BF.A":"BF-A","FB":"META"}

PKG_ROOT = Path(__file__).resolve().parent           # <— aktien_oop/
DEFAULT_SAVE_DIR = PKG_ROOT                          # oder: PKG_ROOT / "runs"

def _load_aliases() -> dict[str,str]:
    aliases = {k.upper(): v.upper() for k,v in DEFAULT_ALIASES.items()}
    if ALIAS_FILE.exists():
        try:
            user = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                aliases.update({k.upper(): v.upper() for k,v in user.items()})
        except Exception:
            pass
    return aliases

ALIASES = _load_aliases()

def normalize_ticker(sym: str) -> str:
    s = (sym or "").strip().upper().replace(".", "-")
    return ALIASES.get(s, s)

# -----------------------
# Logging
# -----------------------
def setup_logging(verbose: bool, lib_debug: bool = False, log_file: Path | None = None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, format="%(asctime)s %(levelname)s: %(message)s")
    if not lib_debug:
        logging.getLogger("yfinance").setLevel(logging.WARNING)

# -----------------------
# Config mit DEFAULTS
# -----------------------
@dataclass(frozen=True)
class Config:
    tickers_file: Path = PKG_ROOT / "sp500_tickers.txt"
    sector_meta_file: Path = PKG_ROOT / "sp500_meta.csv" # Mapping: ticker,sector[,sub_industry]
    save_dir: Path = DEFAULT_SAVE_DIR

    # Strategie-Parameter
    adjusted: bool = True
    period: str = "400d"
    days_win: int = 100
    gap_th: float = 0.08
    adv_min_dollars: float = 5_000_000
    top_k: int = 8
    buffer_k: int = 4
    force_rebalance: bool = False
    verbose: bool = False
    lib_debug: bool = False

    # 🔽 DEFAULTS für Sektorsteuerung
    max_per_sector: int | None = 2                 # Global: max. 2 Titel je Sektor (None = aus)
    sector_limits: dict | None = None               # Spezifische Limits, z. B. {"Industrials":1}

    @classmethod
    def from_cli(cls):
        ap = argparse.ArgumentParser()
        ap.add_argument("--tickers", dest="tickers_file", type=Path)
        ap.add_argument("--sector-meta", dest="sector_meta_file", type=Path)
        ap.add_argument("--save-dir", dest="save_dir", type=Path)
        ap.add_argument("--verbose", action="store_true")
        ap.add_argument("--lib-debug", action="store_true")
        ap.add_argument("--force", dest="force_rebalance", action="store_true")  # <—
        args = ap.parse_args()
        return Config(**{k: v for k, v in vars(args).items() if v is not None})


def _parse_sector_limits(pairs: list[str] | None) -> dict[str,int] | None:
    """Erwartet z. B.: ['Information Technology=2','Industrials=1']"""
    if not pairs: return None
    out: dict[str,int] = {}
    for item in pairs:
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[k.strip()] = int(v)
            except ValueError:
                pass
    return out or None

def resolve_paths(self):
    self.tickers_file = Path(self.tickers_file).resolve()
    self.sector_meta_file = Path(self.sector_meta_file).resolve()
    self.save_dir = Path(self.save_dir).resolve()
    self.save_dir.mkdir(parents=True, exist_ok=True)

def __post_init__(self):
    self.resolve_paths()
    self.tickers_file = Path(self.tickers_file).resolve()
    self.sector_meta_file = Path(self.sector_meta_file).resolve()
    self.save_dir = Path(self.save_dir).resolve()
    self.save_dir.mkdir(parents=True, exist_ok=True)

def _coerce_limit(x: Optional[int]) -> Optional[int]:
    """<=0 oder None bedeutet 'deaktiviert'."""
    if x is None: return None
    return x if x > 0 else None

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Momentum Screener (OOP)")
    p.add_argument("--adjusted", action="store_true", default=True)
    p.add_argument("--no-adjusted", dest="adjusted", action="store_false")
    p.add_argument("--period", type=str, default=None)
    p.add_argument("--days-win", type=int, default=None)
    p.add_argument("--gap", type=float, default=None)
    p.add_argument("--adv-min", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--buffer-k", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--lib-debug", action="store_true")

    # 🔽 Sektor-Optionen (optional; überschreiben Defaults nur bei Angabe)
    p.add_argument("--sector-meta", type=str, default=None)
    p.add_argument("--max-per-sector", type=int, default=None,
                   help="Globales Limit je Sektor; <=0 deaktiviert.")
    p.add_argument("--sector-limit", action="append", default=[],
                   help='Wiederholbar, Format "Sektor=Anzahl" (z. B. --sector-limit "Industrials=1")')

    a = p.parse_args()
    sector_limits = _parse_sector_limits(a.sector_limit)

    # Ausgangspunkt: Defaults aus Config
    defaults = Config()

    return Config(
        adjusted=a.adjusted,
        period=a.period or defaults.period,
        days_win=a.days_win if a.days_win is not None else defaults.days_win,
        gap_th=a.gap if a.gap is not None else defaults.gap_th,
        adv_min_dollars=a.adv_min if a.adv_min is not None else defaults.adv_min_dollars,
        top_k=a.top_k if a.top_k is not None else defaults.top_k,
        buffer_k=a.buffer_k if a.buffer_k is not None else defaults.buffer_k,
        force_rebalance=a.force or defaults.force_rebalance,
        save_dir=Path(a.save_dir) if a.save_dir is not None else defaults.save_dir,
        verbose=a.verbose or defaults.verbose,
        tickers_file=Path(a.tickers) if a.tickers is not None else defaults.tickers_file,
        lib_debug=a.lib_debug or defaults.lib_debug,

        # 🔽 Defaults bleiben, bis CLI explizit überschreibt
        sector_meta_file=Path(a.sector_meta) if a.sector_meta is not None else defaults.sector_meta_file,
        max_per_sector=_coerce_limit(a.max_per_sector) if a.max_per_sector is not None else defaults.max_per_sector,
        sector_limits=sector_limits if sector_limits is not None else defaults.sector_limits,
    )
