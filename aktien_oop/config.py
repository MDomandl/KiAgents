# config.py
from dataclasses import dataclass
from pathlib import Path
import argparse, logging, json
from typing import Optional
try:
    import tomllib  # Py>=3.11
except Exception:
    tomllib = None

def _coalesce(cli_val, cfg_val, default):
    return cli_val if cli_val is not None else (cfg_val if cfg_val is not None else default)

def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _resolve_cfg_path(raw: str) -> Path:
    """
    Versucht mehrere Basisverzeichnisse:
    - wie angegeben (relativ zum aktuellen Arbeitsverzeichnis)
    - relativ zum Paket-Root
    - relativ zum Parent des Paket-Roots (Projektwurzel)
    """
    candidates = [
        Path(raw).expanduser(),
        (PKG_ROOT / raw),
        (PKG_ROOT.parent / raw),
    ]
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            pass
        if c.exists():
            return c
    # Fallback: erster Kandidat (wird später zum Fehler)
    return Path(raw)

def _load_toml_chain(paths: list[str]) -> dict:
    if not paths:
        return {}
    if tomllib is None:
        raise RuntimeError("Python 3.11+ (tomllib) benötigt für --config.")
    merged = {}
    for raw in paths:
        fp = _resolve_cfg_path(raw)
        if not fp.exists():
            raise FileNotFoundError(
                f"Config-Datei nicht gefunden: '{raw}'. "
                f"Versucht u.a.: '{fp}'. Bitte Pfad prüfen."
            )
        with open(fp, "rb") as f:
            d = tomllib.load(f) or {}
        _deep_merge(merged, d)
    return merged



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
    top_k: int = 10   # Anzahl Aktien im Portfolio
    buffer_k: int = 4  # Turnover-Puffer
    force_rebalance: bool = False # via CLI überschreibbar
    rebalance_frequency: str = "weekly"  # "monthly" oder "weekly"
    verbose: bool = False # via CLI überschreibbar
    lib_debug: bool = False
    use_equal_weight: bool = False
    weight_round_step: float = 0.0

    require_above_sma: bool = False
    regime_below_action: str = "HOLD"
    regime_sma_days: int = 200  # z.B. 200 Kalendertage

    show_plots: bool = False
    # Decision Bundles (für Comparator / Runner)
    dump_decision_bundles: bool = True
    decisions_dir: Path = PKG_ROOT / "decisions"
    decision_prefix: str = "RUN"  # Runner schreibt RUN_*.json
    as_of: str = ""
    max_lookback_days: int = 360  # Sicherheits-Puffer, falls as_of genutzt wird

    @property
    def force(self) -> bool:
        return self.force_rebalance

    # 🔽 DEFAULTS für Sektorsteuerung
    max_per_sector: int | None = 2                 # Global: max. 2 Titel je Sektor (None = aus)
    sector_limits: dict | None = None               # Spezifische Limits, z. B. {"Industrials":1}

    def __post_init__(self):
        # frozen=True → object.__setattr__
        object.__setattr__(self, "tickers_file", Path(self.tickers_file).resolve())
        object.__setattr__(self, "sector_meta_file", Path(self.sector_meta_file).resolve())
        object.__setattr__(self, "save_dir", Path(self.save_dir).resolve())
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cli(cls):
        ap = argparse.ArgumentParser()
        # im Argumentparser:
        ap.add_argument(
            "--config", dest="configs", action="append", type=str, default=[],
            help="TOML-Datei; mehrfach angeben für Overlays (Basis zuerst, Overlay danach)"
        )

        ap.add_argument("--tickers", dest="tickers_file", type=str)
        ap.add_argument("--sector-meta", dest="sector_meta_file", type=str)
        ap.add_argument("--save-dir", dest="save_dir", type=str)

        ap.add_argument("--top-k", dest="top_k", type=int)
        ap.add_argument("--buffer-k", dest="buffer_k", type=int)
        ap.add_argument("--rebalance", dest="rebalance_frequency", choices=["weekly", "monthly"])
        ap.add_argument("--max-per-sector", dest="max_per_sector", type=int)
        ap.add_argument("--equal-weight", dest="use_equal_weight", action="store_true", default=None)
        ap.add_argument("--weight-round-step", dest="weight_round_step", type=float)
        # Optional dict z.B. {"Energy":1,"IT":2} – zuerst weglassen, bei Bedarf parse_json ergänzen
        # ap.add_argument("--sector-limits", dest="sector_limits", type=str)

        # WICHTIG: Booleans mit default=None, damit „nicht angegeben“ ≠ False ist
        ap.add_argument("--verbose", dest="verbose", action="store_true", default=None)
        ap.add_argument("--lib-debug", dest="lib_debug", action="store_true", default=None)
        ap.add_argument("--force", dest="force_rebalance", action="store_true", default=None)

        # Decision-Bundles
        ap.add_argument("--dump-decisions", dest="dump_decision_bundles", action="store_true", default=None)
        ap.add_argument("--no-dump-decisions", dest="dump_decision_bundles", action="store_false", default=None)
        ap.add_argument("--decisions-dir", dest="decisions_dir", type=str)
        ap.add_argument("--prefix", dest="prefix", type=str)

        ap.add_argument("--as-of", type=str, default=None,
                        help="Stichtag YYYY-MM-DD; wenn gesetzt, lädt der Runner Daten mit start/end statt period")
        ap.add_argument("--period", type=str, default=None,
                        help="Fallback-Period (z. B. 400d), wenn --as-of nicht gesetzt ist")
        ap.add_argument("--max-lookback-days", dest="max_lookback_days", type=int)

        args = ap.parse_args()

        # --- TOML laden & Sections aufsplitten ---
        cfg_toml = _load_toml_chain(args.configs or [])

        core_cfg = cfg_toml.get("core", {}) or {}
        win_cfg = cfg_toml.get("windows", {}) or {}
        lim_cfg = cfg_toml.get("limits", {}) or {}
        reb_cfg = cfg_toml.get("rebalance", {}) or {}
        topk_cfg = cfg_toml.get("topk", {}) or {}
        regime_cfg = cfg_toml.get("regime", {}) or {}

        # --- Regime: kanonische Keys direkt im cfg nutzen ---
        # Legacy alias (falls früher regime_use_filter verwendet wurde)
        if "regime_use_filter" in regime_cfg and "require_above_sma" not in regime_cfg:
            regime_cfg["require_above_sma"] = bool(regime_cfg.get("regime_use_filter"))

        require_above_sma = bool(regime_cfg.get("require_above_sma", cfg_toml.get("require_above_sma", False)))
        regime_sma_days = int(regime_cfg.get("regime_sma_days", cfg_toml.get("regime_sma_days", 200)) or 200)

        _act = str(regime_cfg.get("regime_below_action", cfg_toml.get("regime_below_action", "HOLD")) or "HOLD").upper()
        if _act not in ("HOLD", "SELL"):
            _act = "HOLD"
        regime_below_action = _act

        # flaches Dict für „einfache“ Keys (Top-Level + core + limits etc.)
        d: dict = dict(cfg_toml)
        d.update(core_cfg)
        d.setdefault("top_k", topk_cfg.get("top_k"))
        d.setdefault("buffer_k", topk_cfg.get("buffer_k"))

        # limits
        if "use_sector_limits" in lim_cfg:
            d.setdefault("use_sector_limits", lim_cfg["use_sector_limits"])

        if "max_per_sector" in lim_cfg:
            d.setdefault("max_per_sector", lim_cfg["max_per_sector"])

        if "gap_filter" in lim_cfg:
            d.setdefault("gap_filter", lim_cfg["gap_filter"])

        if "min_price" in lim_cfg:
            d.setdefault("min_price", lim_cfg["min_price"])

        if "min_volume" in lim_cfg:
            d.setdefault("min_volume", lim_cfg["min_volume"])

        if "friction_eps" in lim_cfg:
            d.setdefault("friction_eps", lim_cfg["friction_eps"])

        if "friction_eps_pct" in lim_cfg:
            d.setdefault("friction_eps_pct", lim_cfg["friction_eps_pct"])

        if "weight_round_step" in lim_cfg:
            d.setdefault("weight_round_step", lim_cfg["weight_round_step"])

        if "max_turnover_cap" in lim_cfg:
            d.setdefault("max_turnover_cap", lim_cfg["max_turnover_cap"])

        if "cost_bps" in lim_cfg:
            d.setdefault("cost_bps", lim_cfg["cost_bps"])

        if "slippage_bps" in lim_cfg:
            d.setdefault("slippage_bps", lim_cfg["slippage_bps"])

        if "max_lookback_days" in lim_cfg:
            d.setdefault("max_lookback_days", lim_cfg["max_lookback_days"])

        # windows (kann der Runner später direkt aus win_cfg lesen)
        if "score_days" in win_cfg:
            d.setdefault("score_days", win_cfg["score_days"])
        if "vol_days" in win_cfg:
            d.setdefault("vol_days", win_cfg["vol_days"])

        # Kern-Parameter period/as_of/max_lookback
        period = args.period or core_cfg.get("period") or d.get("period") or cls.period
        as_of = args.as_of or core_cfg.get("as_of") or d.get("as_of")
        max_lookback_days = args.max_lookback_days or int(d.get("max_lookback_days", cls.max_lookback_days))

        # Helper, um Boolean-Flags nur bei explizitem CLI-Setzen zu übernehmen:
        def _bool_merge(cli_val, cfg_key, default):
            if cli_val is not None:
                return bool(cli_val)
            return bool(d.get(cfg_key, default))

        cfg = cls(
            tickers_file=Path(_coalesce(args.tickers_file,
                                        d.get("tickers_file"),
                                        cls.tickers_file)),
            sector_meta_file=Path(_coalesce(args.sector_meta_file,
                                            d.get("sector_meta_file", d.get("sector_meta")),
                                            cls.sector_meta_file)),
            save_dir=Path(_coalesce(args.save_dir,
                                    d.get("save_dir"),
                                    cls.save_dir)),

            top_k=_coalesce(args.top_k, d.get("top_k"), cls.top_k),
            buffer_k=_coalesce(args.buffer_k, d.get("buffer_k"), cls.buffer_k),
            rebalance_frequency=_coalesce(
                args.rebalance_frequency,
                reb_cfg.get("frequency", d.get("rebalance_frequency", d.get("rebalance"))),
                cls.rebalance_frequency,
            ),

            require_above_sma=require_above_sma,
            regime_sma_days=regime_sma_days,
            regime_below_action=regime_below_action,

            use_equal_weight=_coalesce(args.use_equal_weight,
                                       d.get("use_equal_weight"),
                                       cls.use_equal_weight),
            weight_round_step=_coalesce(args.weight_round_step,
                                        d.get("weight_round_step"),
                                        cls.weight_round_step),

            max_per_sector=_coalesce(args.max_per_sector,
                                     d.get("max_per_sector"),
                                     cls.max_per_sector),

            verbose=_bool_merge(args.verbose, "verbose", cls.verbose),
            lib_debug=_bool_merge(args.lib_debug, "lib_debug", cls.lib_debug),
            force_rebalance=_bool_merge(args.force_rebalance, "force_rebalance", cls.force_rebalance),
            as_of=as_of,
            period=period,
            max_lookback_days=max_lookback_days,

            dump_decision_bundles=_bool_merge(args.dump_decision_bundles,
                                              "dump_decision_bundles",
                                              cls.dump_decision_bundles),
            decisions_dir=Path(_coalesce(args.decisions_dir,
                                         d.get("decisions_dir"),
                                         cls.decisions_dir)),
            decision_prefix=_coalesce(args.prefix,
                                      d.get("decision_prefix", d.get("prefix")),
                                      cls.decision_prefix),
        )

        # nested Sections anhängen, damit der Runner sie sieht
        object.__setattr__(cfg, "core", core_cfg)
        object.__setattr__(cfg, "windows", win_cfg)
        object.__setattr__(cfg, "limits", lim_cfg)
        object.__setattr__(cfg, "rebalance", reb_cfg)
        object.__setattr__(cfg, "regime", regime_cfg)

        # sinnvolle Fallbacks für score_days/vol_days direkt am cfg
        if "score_days" in d:
            object.__setattr__(cfg, "score_days", int(d["score_days"]))
        if "vol_days" in d:
            object.__setattr__(cfg, "vol_days", int(d["vol_days"]))

        # Logging früh setzen
        setup_logging(cfg.verbose, cfg.lib_debug)
        logging.debug(
            "CFG: top_k=%s buffer_k=%s rebalance=%s max_per_sector=%s "
            "sector_limits=%s decisions_dir=%s decision_prefix=%s dump=%s",
            cfg.top_k, cfg.buffer_k, cfg.rebalance_frequency,
            cfg.max_per_sector, cfg.sector_limits,
            str(getattr(cfg, "decisions_dir", None)),
            getattr(cfg, "decision_prefix", None),
            getattr(cfg, "dump_decision_bundles", None),
        )

        return cfg

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
    p.add_argument("--show-plots", action="store_true", default=True)

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
        show_plots=a.show_plots or defaults.show_plots,

        # 🔽 Defaults bleiben, bis CLI explizit überschreibt
        sector_meta_file=Path(a.sector_meta) if a.sector_meta is not None else defaults.sector_meta_file,
        max_per_sector=_coerce_limit(a.max_per_sector) if a.max_per_sector is not None else defaults.max_per_sector,
        sector_limits=sector_limits if sector_limits is not None else defaults.sector_limits,
        as_of=a.as_of if hasattr(a, "as_of") else None,

    )
