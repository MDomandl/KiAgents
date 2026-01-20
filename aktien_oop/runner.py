# runner.py
from __future__ import annotations

import dataclasses
from typing import List, Dict
import itertools

import datetime
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from aktien_oop.core_calc import CalcParams, calculate_portfolio, slice_to_window

import logging
import json
from json import dumps
from dataclasses import asdict
from .config import Config, normalize_ticker, setup_logging
from .data_client import DataClient
from .engine import SignalEngine
from .store import PortfolioStore
from .rebalance import Rebalancer

def _json_default(obj):
    """
    Helper für json.dumps: wandelt Timestamp/Datum in Strings um.
    Alles andere lässt er bewusst knallen, damit wir Fehler sehen.
    """
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    return str(obj)  # fallback: notfalls als String

# --- Kleine Helper-Funktion für JSON-kompatible Dicts -----------------------
def _series_to_dict(s):
    """
    Wandelt eine pandas.Series (oder bereits ein dict) in ein
    JSON-freundliches Dict[str, float] um. None bleibt None.
    """
    if s is None:
        return None

    # schon ein dict -> direkt zurück
    if isinstance(s, dict):
        return s

    try:
        import pandas as pd  # zur Sicherheit, ist aber oben meist schon importiert
    except ImportError:
        # Falls pandas nicht verfügbar ist, einfach normal casten
        return {str(k): float(v) for k, v in dict(s).items()}

    if isinstance(s, pd.Series):
        s = s.dropna()
        return {str(k): float(v) for k, v in s.items()}

    # Fallback: alles, was sich wie ein Mapping/Iterable verhält
    return {str(k): float(v) for k, v in dict(s).items()}


@dataclasses.dataclass(frozen=True)
class RunCfgNormalized:
    # Kern-Infos
    as_of: pd.Timestamp
    period: str              # z.B. "400d"

    # Scoring / Volatilität
    score_days: int
    vol_days: int | None     # None = keine Vol-Berechnung

    # Selektion / Buffer
    top_k: int
    buffer_k: int

    # Sektor-Limits
    use_sector_limits: bool
    max_per_sector: int | None

    # Cash / Reibung / Rundung
    eps: float               # absolute Friction (z.B. 0.0)
    eps_pct: float           # prozentuale Friction (z.B. 0.0)
    cap: float               # Turnover-Cap (0.0 = aus)
    round_step: float        # Rundungsschritt für Gewichte
    cost_bps: float          # z.B. 0.0 oder 5.0
    slippage_bps: float      # z.B. 0.0 oder 0.0

    # Kurs-Daten
    adjusted: bool

    # Runner-Output
    decisions_dir: str | None
    decision_prefix: str | None
    dump_decision_bundles: bool

    # Sonstiges
    rebalance: str           # "monthly", "weekly", ...
    max_lookback_days: int | None

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = DataClient(cfg)
        self.engine = SignalEngine(cfg, self.data)
        self.store = PortfolioStore(cfg.save_dir)
        self.rebalancer = Rebalancer(self.store, cfg.top_k, cfg.buffer_k, cfg.force_rebalance)
        # Eindeutige ID pro Runner-Lauf (für Dateinamen)
        self.run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------
    # Helpers
    # ---------------------------
    # --- Decision-Bundle Helpers ---
    @staticmethod
    def _weights_from_positions(df: pd.DataFrame) -> dict[str, float]:
        if df is None or df.empty:
            return {}

        d: dict[str, float] = {}
        for _, r in df.iterrows():
            t = str(r.get("ticker") or "").strip()
            if not t:
                continue

            # Prefer true weights if present (0..1)
            w = r.get("weight", None)
            if w is not None:
                try:
                    v = float(w)
                    if v > 0:
                        d[t] = v
                    continue
                except Exception:
                    pass

            # Fallback for older files: allocation_pct (0..100)
            ap = r.get("allocation_pct", None)
            if ap is not None:
                try:
                    v = float(ap) / 100.0
                    if v > 0:
                        d[t] = v
                except Exception:
                    pass

        # Final clean
        return {k: v for k, v in d.items() if v > 0.0}

    def _write_decision_bundle(self, as_of_raw, old_w, new_w, *, regime=None, turnover_raw=None, turnover_eff=None, first_run_no_prev_state=False) -> None:

        """
        Schreibt das Runner-Decision-Bundle als JSON.
        Stellt sicher, dass:
          - as_of in der JSON = 'YYYY-MM-DD'
          - Dateiname keine ungültigen Zeichen (Windows) enthält.
        """
        prefix = getattr(self.cfg, "decision_prefix", "RUN")
        ddir = Path(getattr(self.cfg, "decisions_dir", "aktien_oop/decisions"))
        ddir.mkdir(parents=True, exist_ok=True)

        # 1) as_of normalisieren → immer YYYY-MM-DD
        try:
            as_of_ts = pd.Timestamp(as_of_raw)
            as_of_str = as_of_ts.strftime("%Y-%m-%d")
        except Exception:
            # Fallback, falls wirklich etwas Exotisches reinkommt
            as_of_str = str(as_of_raw)
            as_of_str = (
                as_of_str.replace(" ", "_")
                .replace(":", "-")
                .replace("/", "-")
            )

        # 2) Laufzeit-Stamp für eindeutige Dateinamen
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 3) Bundle aufbauen – as_of als Datum-String
        bundle = {
            "kind": "RUN",
            "run_id": f"{prefix}_{run_ts}",
            "as_of": as_of_str,
            "old_weights": _series_to_dict(old_w),
            "new_weights": _series_to_dict(new_w),
            "turnover_raw": float(turnover_raw) if turnover_raw is not None else None,
            "turnover_eff": float(turnover_eff) if turnover_eff is not None else None,
            "first_run_no_prev_state": bool(first_run_no_prev_state),
        }

        if isinstance(regime, dict):
            bundle["regime"] = regime
        # 4) Dateiname nur mit sauberem Datumsteil
        out = ddir / f"{prefix}_{run_ts}_{as_of_str}.json"

        out.write_text(
            dumps(bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("[DECISION] wrote %s", out)

    def load_tickers(self) -> List[str]:
        """Liest das Universum aus Datei und normalisiert Yahoo-kompatibel."""
        try:
            with open(self.cfg.tickers_file, "r", encoding="utf-8") as f:
                raw = [line.strip() for line in f if line.strip()]
            ticks = sorted(set(normalize_ticker(t) for t in raw))
            if not ticks:
                raise ValueError("Datei leer")
            return ticks
        except Exception as e:
            logging.warning(
                "Konnte '%s' nicht laden (%s). Fallback: AAPL, MSFT, NVDA",
                self.cfg.tickers_file, e
            )
            return ["AAPL", "MSFT", "NVDA"]

    def _load_sector_map(self) -> Dict[str, str]:
        """Lädt Ticker→Sektor aus cfg.sector_meta_file (falls vorhanden)."""
        try:
            df = pd.read_csv(self.cfg.sector_meta_file)
            if "ticker" not in df.columns or "sector" not in df.columns:
                return {}
            df["ticker"] = df["ticker"].astype(str).map(normalize_ticker)
            df = df.dropna(subset=["ticker", "sector"]).drop_duplicates(subset=["ticker"])
            return dict(zip(df["ticker"], df["sector"]))
        except Exception as e:
            logging.info("Kein/ungültiges Sektor-Meta gefunden (%s) – Sektor-Limits ggf. inaktiv.", e)
            return {}

    def _should_rebalance(self, last_dt) -> bool:
        if self.cfg.force_rebalance:  # <— Force hat Vorrang
            logging.info("Force-Rebalance aktiv – ignoriere Cadence.")
            return True
        if last_dt is None:
            return True
        # Monats-Cadence (Default)
        return (pd.Timestamp.now().strftime("%Y-%m")
                != pd.Timestamp(last_dt).strftime("%Y-%m"))

    @staticmethod
    def _same_period(a: pd.Timestamp, b: pd.Timestamp, freq: str) -> bool:
        """Vergleicht zwei Zeitpunkte auf gleiche Woche/Monat (abhängig von freq)."""
        if freq == "weekly":
            # Woche mit Montags-Anker ist stabiler über Zeitzonen
            return a.to_period("W-MON") == b.to_period("W-MON")
        # Default: monatlich
        return a.to_period("M") == b.to_period("M")

    def _print_existing_positions(self) -> None:
        pos = self.store.load_positions()
        if pos.empty:
            print("⚠️  Keine gespeicherten Positionen gefunden.")
            return

        # Sektor-Spalte anreichern, falls Meta vorhanden
        sector_map = self._load_sector_map()
        if sector_map:
            pos["sector"] = pos["ticker"].map(sector_map)

        # Letzte TopK-Metriken (volatility/stop_loss_pct) mergen, falls vorhanden
        try:
            topk = self.store.load_last_topk()
            if not topk.empty:
                pos = pos.merge(
                    topk[["ticker", "volatility", "stop_loss_pct"]],
                    on="ticker", how="left"
                )
        except Exception as e:
            logging.debug("Konnte TopK-Metriken nicht mergen: %s", e)

        cols = ["as_of", "ticker", "sector", "rank", "score",
                "volatility", "stop_loss_pct", "allocation_pct"]
        print("\nBestehende Positionen:\n")
        print(pos[[c for c in cols if c in pos.columns]])

    # -------------------------
    # Config / Params Helpers
    # -------------------------

    def _normalize_cfg(self, as_of_str: str | None = None) -> dict:
        """
        Liest runner_config.toml-/strategy.toml-Abschnitte zusammen und berechnet
        lokale, nicht-mutable Normalized-Parameter.
        Rückgabe ist ein dict, NICHT self.cfg mutieren.
        """
        cfg = self.cfg

        def _sec_get(section, key, default):
            """Hilfshelper: funktioniert sowohl für dict als auch für einfache Objekte."""
            if section is None:
                return default
            if isinstance(section, dict):
                return section.get(key, default)
            return getattr(section, key, default)

        core_cfg = getattr(cfg, "core", None)
        win_cfg = getattr(cfg, "windows", None)
        lim_cfg = getattr(cfg, "limits", None)
        reb_cfg = getattr(cfg, "rebalance", None)

        # Zeitraum
        period = _sec_get(core_cfg, "period", getattr(cfg, "period", "800d"))

        # as_of-Priorität: CLI → [core].as_of → cfg.as_of
        as_of = (
                as_of_str
                or _sec_get(core_cfg, "as_of", getattr(cfg, "as_of", None))
        )

        # Fenster
        score_days = int(_sec_get(win_cfg, "score_days", getattr(cfg, "score_days", 252)))
        vol_days = int(_sec_get(win_cfg, "vol_days", getattr(cfg, "vol_days", 63)))

        # Sektor-Limits
        use_sector_limits = bool(_sec_get(lim_cfg, "use_sector_limits", getattr(cfg, "use_sector_limits", True)))
        max_per_sector = _sec_get(lim_cfg, "max_per_sector", getattr(cfg, "max_per_sector", 3))
        raw_gap = _sec_get(lim_cfg, "gap_filter", None)
        gap_filter = float(0.12 if raw_gap is None else raw_gap)

        # Rebalance-Frequenz
        rebalance_frequency = _sec_get(reb_cfg, "frequency", getattr(cfg, "rebalance_frequency", "monthly"))

        return dict(
            period=period,
            as_of=as_of,
            score_days=score_days,
            vol_days=vol_days,
            use_sector_limits=use_sector_limits,
            max_per_sector=max_per_sector,
            gap_filter=gap_filter,
            rebalance=rebalance_frequency,
        )

    def _build_params(self, norm: dict) -> "CalcParams":
        """
        Baut CalcParams ausschließlich aus norm + cfg.
        norm kommt aus _normalize_cfg und enthält bereits
        period, as_of, score_days, vol_days, use_sector_limits, max_per_sector, rebalance, ...
        """
        cfg = self.cfg

        return CalcParams(
            # Basis
            as_of=str(norm["as_of"]),
            period=norm["period"],
            adjusted=bool(getattr(cfg, "adjusted", True)),

            score_days=int(norm["score_days"]),
            vol_days=int(norm["vol_days"]),

            # === Filter (müssen identisch zum Backtester sein!) ===
            use_under_sma=bool(getattr(cfg, "use_under_sma", False)),
            sma_days=int(getattr(cfg, "sma_days", 200)),
            gap_filter=norm["gap_filter"],
            min_price=float(getattr(cfg, "min_price", 0.0)),
            min_volume=float(getattr(cfg, "min_volume", 0.0)),

            # === Limits & Auswahl ===
            use_sector_limits=bool(norm["use_sector_limits"]),
            max_per_sector=(
                int(norm["max_per_sector"])
                if norm["max_per_sector"] is not None
                else None
            ),
            top_k=int(cfg.top_k),
            buffer_k=int(cfg.buffer_k),

            # === Finalisierung / Sizing ===
            include_cash=bool(getattr(cfg, "include_cash", False)),
            weight_round_step=float(getattr(cfg, "weight_round_step", 0.0)),
            max_turnover_cap=float(getattr(cfg, "max_turnover_cap", 1.0)),
            friction_eps=float(getattr(cfg, "friction_eps", 0.0)),
            friction_eps_pct=float(getattr(cfg, "friction_eps_pct", 0.0)),
            cost_bps=float(getattr(cfg, "cost_bps", 0.0)),
            slippage_bps=float(getattr(cfg, "slippage_bps", 0.0)),

            # === Meta ===
            rebalance=norm["rebalance"],
            max_lookback_days=getattr(cfg, "max_lookback_days", None),

            dump_scores=True,
            dump_tag="RUN",
        )

    # ---------------------------
    # Main
    # ---------------------------
    def run(self) -> None:

        # optional: run_id einmalig
        setup_logging(self.cfg.verbose, lib_debug=self.cfg.lib_debug, log_file=self.cfg.save_dir / "run.log")

        def _assign_attr(obj, name, value):
            """Hilfsfunktion, die auch mit (halb-)frozen Config-Objekten klarkommt."""
            try:
                object.__setattr__(obj, name, value)
            except Exception:
                setattr(obj, name, value)

        # === NORMALIZE (Runner) – nur noch über _normalize_cfg ===
        cfg = self.cfg
        #_assign_attr(cfg, "as_of", str((cfg.__getattribute__("core"))["as_of"]))
        # --- AS-OF zentral ---
        as_of_ts = (pd.Timestamp(self.cfg.as_of).normalize()
                    if getattr(self.cfg, "as_of", None)
                    else pd.Timestamp.today().normalize())
        as_of_str = as_of_ts.strftime("%Y-%m-%d")
        # NEU: Stichtag an DataClient/Engine durchreichen (Duck-Typing)
        for obj in (self.data, self.engine):
            if hasattr(obj, "set_as_of"):
                obj.set_as_of(as_of_ts)
            elif hasattr(obj, "as_of"):
                setattr(obj, "as_of", as_of_ts)

        norm = self._normalize_cfg(as_of_str=as_of_str)
        _assign_attr(cfg, "include_cash", bool((cfg.__getattribute__("regime"))["include_cash"]))
        # Relevante Felder zurück ins cfg spiegeln, damit Logs/Store konsistent sind
        _assign_attr(cfg, "score_days", int(norm["score_days"]))
        _assign_attr(cfg, "vol_days", int(norm["vol_days"]))
        _assign_attr(cfg, "use_sector_limits", bool(norm["use_sector_limits"]))
        _assign_attr(cfg, "max_per_sector", (
            int(norm["max_per_sector"]) if norm["max_per_sector"] is not None else None
        ))
        _assign_attr(cfg, "rebalance_frequency", norm["rebalance"])
        _assign_attr(cfg, "period", norm["period"])
        if norm["as_of"] is not None:
            _assign_attr(cfg, "as_of", norm["as_of"])

        logging.debug(
            "CFG(normalized/TOML): period=%s as_of=%s top_k=%s buffer_k=%s "
            "score_days=%s vol_days=%s use_sector_limits=%s max_per_sector=%s",
            norm["period"],
            norm["as_of"],
            cfg.top_k,
            cfg.buffer_k,
            norm["score_days"],
            norm["vol_days"],
            norm["use_sector_limits"],
            norm["max_per_sector"],
        )

        now = pd.Timestamp.now()
        last_dt = self.store.last_rebalance_time()
        logging.info("Force=%s, last_rebalance=%s", self.cfg.force_rebalance, last_dt)

        # EINHEITLICH: nur diese Funktion entscheidet über Cadence/Force
        if not self._should_rebalance(last_dt):
            self._print_existing_positions()
            logging.info(
                "Bereits rebalanced in dieser %s – (--force/--force-rebalance) für sofort",
                "Woche" if self.cfg.rebalance_frequency == "weekly" else "Monat"
            )
            return

        else:
            if self.cfg.force_rebalance:
                logging.info("Force-Rebalance aktiv – ignoriere Cadence.")

        tickers = self.load_tickers()
        sector_map = self._load_sector_map()
        logging.info("Starte Bewertung (%d Ticker)...", len(tickers))
        has_sector_meta = bool(sector_map)
        # Sichtbare Zusammenfassung der Sektor-Settings
        limits_active = has_sector_meta and (
                (self.cfg.max_per_sector is not None and self.cfg.max_per_sector > 0)
                or bool(self.cfg.sector_limits)
        )

        print(
            f"Sektor-Limits: {'AKTIV' if limits_active else 'inaktiv'} | "
            f"max_per_sector={self.cfg.max_per_sector} | "
            f"sector_limits={self.cfg.sector_limits or '-'} | "
            f"meta={self.cfg.sector_meta_file}"
        )

        # Markt-Regime-Filter (S&P 500 > 200DMA?)
        # --- Regime (zentral, as-of-korrekt) ---
        decision = self.data.regime_decision(self.cfg, as_of_str)

        if not decision["ok"]:
            logging.warning("Regime: %s | action=%s", decision["reason"], decision["action"])

            # Logging/Meta schreiben wie bisher (runs_log / runs_meta_jsonl),
            # aber als "aborted_reason" nimm decision["reason"].

            if decision["action"] == "SELL":
                # -> hier NICHT 'return' ohne Decision Bundle,
                # sondern: new_weights = {} (oder nur CASH, wenn include_cash)
                # und Turnover gegen old_w berechnen (wie normal).
                pass

            if decision["action"] == "HOLD":
                # -> keep old_w unverändert, Turnover=0
                pass

            # Danach: entweder return (wenn du "aborted run" willst)
            # oder normal weiter, aber mit erzwungenen weights.

        # --- /Regime ---

        # === Core-Calc Callbacks ===
        def _get_prices(universe, as_of, period, adjusted):
            px = self.data.load_prices(universe, as_of, period, adjusted)
            return slice_to_window(px, as_of, period)

        def _get_sectors(universe):
            # identische Meta wie oben geladen (sector_map)
            return {t: sector_map.get(t, "UNKNOWN") for t in universe}

        def _turnover(old_d: dict, new_d: dict) -> float:
            keys = set(old_d.keys()) | set(new_d.keys())
            s = 0.0
            for k in keys:
                s += abs(float(old_d.get(k, 0.0)) - float(new_d.get(k, 0.0)))
            return s / 2.0

        # === CalcParams aus normalisierter cfg ===
        cfg = self.cfg  # Kurzalias

        # --- 1) Normalize cfg (ohne cfg zu mutieren) ---
        norm = self._normalize_cfg(as_of_str=as_of_str)

        logging.debug(
            "CFG(normalized): period=%s as_of=%s top_k=%s buffer_k=%s score_days=%s vol_days=%s "
            "use_sector_limits=%s max_per_sector=%s",
            norm["period"], norm["as_of"], self.cfg.top_k, self.cfg.buffer_k,
            norm["score_days"], norm["vol_days"],
            norm["use_sector_limits"], norm["max_per_sector"],
        )

        # --- 2) CalcParams aus norm bauen ---
        params = self._build_params(norm)

        debug_dir = Path("aktien_oop/debug")
        debug_dir.mkdir(exist_ok=True, parents=True)

        cp_path = debug_dir / f"RUN_CalcParams_{as_of_str}.json"
        with cp_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(params), f, indent=2, sort_keys=True, default=str)
        logging.debug("RUN CalcParams dump: %s", cp_path)

        # === prev_holdings für Turnover-Buffer (aus Snapshot VOR as_of) ===
        prev_df = self.store.load_positions_before(as_of_str)  # BEFORE!
        old_w = self._weights_from_positions(prev_df) if prev_df is not None else {}
        prev_holdings = list(old_w.keys())

        is_first_run = (prev_df is None) or (len(prev_df) == 0)
        if is_first_run:
            logging.info(
                "[RUN] first_run_no_prev_state=True (no positions before as_of=%s)",
                as_of_str
            )

        logging.debug("prev_df rows=%s cols=%s",
                      0 if prev_df is None else len(prev_df),
                      [] if prev_df is None else list(prev_df.columns))
        logging.debug("old_w n=%d sample=%s", len(old_w), list(old_w.items())[:5])
        logging.debug("prev_holdings n=%d tickers=%s", len(prev_holdings), prev_holdings)

        # --- DEBUG: Preismatrix prüfen ---
        probe = _get_prices(tickers, params.as_of, params.period, params.adjusted)
        if probe is None or probe.empty:
            logging.error("Price matrix EMPTY. universe=%d as_of=%s period=%s adjusted=%s",
                          len(tickers), params.as_of, params.period, params.adjusted)
            return  # oder sauber weiterhandlen
        else:
            last_dt = probe.index.max()
            if pd.Timestamp(params.as_of) > last_dt:
                logging.warning("as_of=%s nicht im Price-Index. Verwende last_dt=%s",
                                params.as_of, last_dt.date())
                # >>> as_of auf letzten vorhandenen Börsentag setzen <<<
                params = dataclasses.replace(params,  as_of=last_dt)  # bei @dataclass(frozen=False/frozen=True mit replace)

            first_dt = probe.index.min()
            non_na_cols = int((probe.notna().any(axis=0)).sum())
            logging.debug("Price matrix shape=%s first=%s last=%s nonNaCols=%d",
                          tuple(probe.shape), str(first_dt), str(last_dt), non_na_cols)

        # --- REGIME (Runner) ---
        # Wichtig: params.as_of kann Timestamp sein -> als YYYY-MM-DD string normieren
        as_of_regime = pd.Timestamp(params.as_of).strftime("%Y-%m-%d")

        decision = self.data.regime_decision(self.cfg, as_of_regime)
        logging.info("[DBG][REGIME][RUN] as_of=%s decision=%s", as_of_regime, decision)

        action = str(decision.get("action", "PROCEED") or "PROCEED").upper()
        # ------------------------

        # Jetzt erst der Core-Call:
        if action == "PROCEED":
            weights, scores = calculate_portfolio(
                tickers, params, _get_prices, _get_sectors, prev_holdings=prev_holdings
            )
        else:
            # HOLD / SELL -> kein Core Call
            scores = None
            if action == "HOLD":
                # Portfolio unverändert weiterführen
                weights = dict(old_w)  # old_w ist dein Snapshot BEFORE as_of
            elif action == "SELL":
                if bool(getattr(self.cfg, "include_cash", False)):
                    weights = {"CASH": 1.0}
                else:
                    weights = {}
            else:
                # defensiver Fallback
                weights, scores = calculate_portfolio(
                    tickers, params, _get_prices, _get_sectors, prev_holdings=prev_holdings
                )

        debug_dir = Path("aktien_oop/debug")
        debug_dir.mkdir(exist_ok=True, parents=True)

        as_of_str = params.as_of  # "YYYY-MM-DD"
        safe_as_of = norm["as_of"]

        if isinstance(scores, pd.Series):
            df_scores = scores.to_frame(name="score_adj")
            df_scores.index.name = "ticker"
            df_scores = df_scores.reset_index().sort_values("score_adj", ascending=False)
            df_scores.to_csv(debug_dir / f"RUN_scores_{safe_as_of}.csv", index=False)

        # auch die finalen Gewichte dumpen
        if isinstance(weights, dict):
            df_w = pd.DataFrame(
                [{"ticker": k, "weight": v} for k, v in weights.items()]
            ).sort_values("weight", ascending=False)
            df_w.to_csv(debug_dir / f"RUN_weights_{safe_as_of}.csv", index=False)

        logging.debug(
            "[DEBUG/RUN] calculate_portfolio -> %d Namen, Beispiele: %s",
            len(weights),
            list(itertools.islice(weights.items(), 0, 5)),
        )

        if not weights or sum(abs(v) for v in weights.values()) == 0.0:
            logging.error("Core returned EMPTY weights. universe=%d  as_of=%s", len(tickers), params.as_of)

        # --- DEBUG: gezielte Probe für BT-Titel ---
        _bt_names = ["GEV", "PLTR", "STX"]
        probe3 = _get_prices(_bt_names, params.as_of, params.period, params.adjusted)
        if probe3 is None or probe3.empty:
            logging.error("Probe(GEV,PLTR,STX) EMPTY")
        else:
            logging.debug("Probe(GEV,PLTR,STX) shape=%s last=%s cols=%s last_row=%s",
                          tuple(probe3.shape),
                          str(probe3.index.max()),
                          list(probe3.columns),
                          probe3.tail(1).to_dict(orient="records"))

        # === 'sel' DataFrame aus den Ergebnissen bauen (für Logs/Output) ===
        sel_ticks = list(weights.keys())
        sel = pd.DataFrame({"ticker": list(weights.keys())})
        sel["weight"] = sel["ticker"].map(lambda t: float(weights.get(t, 0.0)))
        if isinstance(scores, (pd.Series, dict)):
            sel["score"] = sel["ticker"].map(lambda t: float(scores.get(t, np.nan)))
            sel["rank"] = sel["score"].rank(ascending=False, method="first").astype("Int64")
        else:
            sel["score"] = np.nan
            # bei HOLD/SELL ist rank nicht wirklich relevant -> 1 als Fallback oder NaN
            sel["rank"] = 1

        # Anzeige: Prozent
        sel["allocation_pct"] = (sel["weight"] * 100.0).round(2)

        # As-of Spalte Logs
        sel.insert(0, "as_of", as_of_str)

        # Sektoren für Ansicht
        if sector_map:
            sel["sector"] = sel["ticker"].map(sector_map).fillna("Unknown")

        # Ausgabe
        want = ["as_of", "ticker"]
        if "sector" in sel.columns: want.append("sector")
        want += ["rank", "score", "allocation_pct"]
        print("\nNeues Portfolio (Turnover-Puffer aktiv):\n")
        print(sel[want])

        df = sel.copy()
        # ===== /Gemeinsame Kernberechnung =====

        # --- Decision-Bundle (Runner) ---
        if getattr(self.cfg, "dump_decision_bundles", False):
            # alte Gewichte (aus letzter positions.csv)
            prev_df = self.store.load_positions_before(as_of_str)
            old_w = self._weights_from_positions(prev_df) if prev_df is not None else {}

            # neue Gewichte (aus aktueller Auswahl)
            new_w = {str(k): float(v) for k, v in weights.items() if float(v) > 0.0}

            # --- LOCKSTEP: RUN-Parameter/Universe-Check ---

            ticks = self.load_tickers()
            univ_sha = hashlib.sha1(("|".join(sorted(map(str, ticks)))).encode()).hexdigest()

            old_w_dict = old_w  # dict
            new_w_dict = weights  # dict
            turnover_raw = None
            turnover_eff = None
            if not is_first_run:
                turnover_raw = _turnover(old_w_dict, new_w_dict)
                cap = float(getattr(self.cfg, "max_turnover_cap", 0.0) or 0.0)
                turnover_eff = min(turnover_raw, cap) if cap > 0 else turnover_raw
            else:
                # First-Run: Turnover bewusst nicht definieren
                turnover_raw = None
                turnover_eff = None

            logging.info(
                "[LOCKSTEP][RUN] as_of=%s top_k=%d buffer_k=%d use_sector_limits=%s max_per_sector=%s "
                "score_days=%d vol_days=%d adjusted=%s universe_len=%d sha=%s",
                as_of_str,
                self.cfg.top_k,
                self.cfg.buffer_k,
                norm["use_sector_limits"],
                norm["max_per_sector"],
                norm["score_days"],
                norm["vol_days"],
                params.adjusted,
                len(tickers),
                univ_sha,
            )

            # --- /LOCKSTEP ---

            # Bundle schreiben – nutzt den oben berechneten as_of_str
            self._write_decision_bundle(
                as_of_str,  # besser als norm["as_of"], weil ggf. gepadded auf last_dt
                old_w,
                weights,  # direkt die dict-weights nehmen
                regime=decision,  # falls du decision schon erzeugst (HOLD/SELL/PROCEED)
                turnover_raw=turnover_raw,
                turnover_eff=turnover_eff,
                first_run_no_prev_state=is_first_run,
            )

        self.store.append_csv(self.store.topk_log, sel)
        run_row = pd.DataFrame([{
            "as_of": as_of_str,
            "adjusted": self.cfg.adjusted, "period": self.cfg.period, "days_win": self.cfg.days_win,
            "gap_th": self.cfg.gap_th, "adv_min": self.cfg.adv_min_dollars,
            "top_k": self.cfg.top_k, "buffer_k": self.cfg.buffer_k,
            "num_universe": len(tickers), "num_pass": len(df),
            "fail_counts": dict(self.engine.fail_counts),
            "sector_limits_active": limits_active,
        }])

        self.store.append_csv(self.store.runs_log, run_row)

        # JSONL-Meta mit Sektor-Counts der Auswahl (falls verfügbar)
        sector_counts = {}
        if sector_map:
            sector_counts = sel["ticker"].map(sector_map).fillna("Unknown").value_counts().to_dict()
        meta = {
            "as_of": as_of_str,
            "universe_size": len(tickers),
            "num_pass": int(len(df)),
            "adjusted": self.cfg.adjusted,
            "period": self.cfg.period,
            "days_win": self.cfg.days_win,
            "gap_th": self.cfg.gap_th,
            "adv_min_dollars": self.cfg.adv_min_dollars,
            "top_k": self.cfg.top_k,
            "buffer_k": self.cfg.buffer_k,
            "sector_limits_active": limits_active,
            "max_per_sector": self.cfg.max_per_sector,
            "sector_limits": self.cfg.sector_limits,
            "sector_meta_file": str(self.cfg.sector_meta_file),
            "selected_sector_counts": sector_counts,
            "fail_counts": dict(self.engine.fail_counts),
        }

        try:
            self.store.append_jsonl(self.store.runs_meta_jsonl, meta)  # type: ignore[attr-defined]
        except Exception:
            pass

        # aktuelles Portfolio kompakt speichern
        #self.store.write_positions(sel[["as_of", "ticker", "allocation_pct", "rank", "score"]])

        # ==== BEGIN DEBUG: Sichtbar machen, wer warum rausfliegt ====
        # Wir versuchen, eine DataFrame-Quelle zu finden, die die Filter-Flags enthält.
        # Typische Kandidaten-Variablen heißen bei dir z.B. df_all, universe_df, candidates_df.
        _df_candidates = None
        for _name in ("df_all", "universe_df", "candidates_df", "df"):
            if _name in locals():
                _df_candidates = locals()[_name]
                break
            if _name in globals():
                _df_candidates = globals()[_name]
                break

        if isinstance(_df_candidates, pd.DataFrame):
            dfu = _df_candidates.copy()

            # Hilfsfunktion: nimm den ersten existierenden Spaltennamen
            def _first_col(cands):
                for c in cands:
                    if c in dfu.columns:
                        return c
                return None

            # Mögliche Spaltennamen, je nachdem wie du die Flags nennst
            col_ticker = _first_col(("ticker", "symbol", "Ticker"))
            col_under = _first_col(("under_sma", "below_sma200", "under_200dma", "under200"))
            col_gap = _first_col(("gap", "gap_fail", "gapped"))

            removed_by = {}

            def _collect(mask_or_col, reason):
                if mask_or_col is None:
                    return []
                # bool-Maske oder Spaltenname erlauben
                if isinstance(mask_or_col, str):
                    if mask_or_col not in dfu.columns:
                        return []
                    m = dfu[mask_or_col].astype(bool)
                else:
                    m = mask_or_col.astype(bool)

                if col_ticker and col_ticker in dfu.columns:
                    tickers = dfu.loc[m, col_ticker].astype(str).tolist()
                else:
                    # Fallback: Indexnamen als "Ticker"
                    tickers = dfu.index[m].astype(str).tolist()

                for t in tickers:
                    removed_by.setdefault(t, []).append(reason)
                return tickers

            out_under = _collect(col_under, "under_sma") if col_under else []
            out_gap = _collect(col_gap, "gap") if col_gap else []

            # Schöne, gekürzte Ausgaben (nicht die ganze Wall of Text)
            def _preview(lst, n=30):
                return ", ".join(lst[:n]) + (" …" if len(lst) > n else "")

            if out_under:
                logging.debug(f"[FILTER/DBG] removed by under_sma: {len(out_under)} → {_preview(out_under)}")
            else:
                logging.debug("[FILTER/DBG] removed by under_sma: 0")

            if out_gap:
                logging.debug(f"[FILTER/DBG] removed by gap     : {len(out_gap)} → {_preview(out_gap)}")
            else:
                logging.debug("[FILTER/DBG] removed by gap     : 0")

            # Aggregierte Sicht: Ticker → Gründe
            if removed_by:
                # Nur bis zu 40 Beispiele drucken
                examples = list(removed_by.items())[:40]
                pretty = ", ".join([f"{t}({','.join(reasons)})" for t, reasons in examples])
                if len(removed_by) > 40:
                    pretty += " …"
                logging.debug(f"[FILTER/DBG] total removed: {len(removed_by)} → {pretty}")
            else:
                logging.debug("[FILTER/DBG] total removed: 0")
        else:
            logging.debug("[FILTER/DBG] Kein Kandidaten-DataFrame gefunden – Debug-Ausgabe übersprungen.")
        # ==== END DEBUG ====

        logging.info("Filter-Statistik: %s", dict(self.engine.fail_counts))

        self.store.save_positions(sel)

        filter_stats = locals().get("filter_stats", {})

        self.store.append_run(
            universe_size=len(df),  # Anzahl nach Scoring/Filter
            top_k=self.cfg.top_k,
            buffer_k=self.cfg.buffer_k,
            max_per_sector=self.cfg.max_per_sector,
            sector_limits_on=bool(sector_map),  # True, wenn Sektor-Map geladen war
            tickers_file=str(self.cfg.tickers_file),
            sector_meta_file=str(self.cfg.sector_meta_file),
            filters=filter_stats,  # landet in meta_json
        )

        print("Positions:", self.store.positions_path)
        print("Runs log:", self.store.runs_log)
        print("TopK log:", self.store.topk_log)
        print("Meta:", self.store.runs_meta_jsonl)

