# runner.py
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import logging
import hashlib
from pathlib import Path
from aktien_oop.core_calc import CalcParams, calculate_portfolio

from datetime import datetime
import json
from .config import Config, setup_logging, normalize_ticker, setup_logging
from .data_client import DataClient
from .engine import SignalEngine
from .store import PortfolioStore
from .rebalance import Rebalancer


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
        if df is None or df.empty: return {}
        d = {}
        for _, r in df.iterrows():
            t = str(r.get("ticker"));
            ap = r.get("allocation_pct")
            if t and ap is not None:
                try:
                    d[t] = float(ap) / 100.0
                except:
                    pass
        return {k: v for k, v in d.items() if v > 0}

    def _write_decision_bundle(self, as_of: str, old_w: dict[str, float], new_w: dict[str, float]) -> Path:
        def with_cash(w: dict[str, float]) -> dict[str, float]:
            s = float(sum(w.values()));
            out = dict(w)
            if s < 1.0:
                out["CASH"] = max(0.0, 1.0 - s)
            elif s > 1.0 and s > 0.0:
                out = {k: v / s for k, v in out.items()}
            return out

        ow, nw = with_cash(old_w), with_cash(new_w)
        keys = set(ow) | set(nw)
        turnover = 0.5 * sum(abs(nw.get(k, 0.0) - ow.get(k, 0.0)) for k in keys)
        bundle = {
            "as_of": as_of,
            "old_weights": {k: round(float(v), 6) for k, v in sorted(ow.items())},
            "new_weights": {k: round(float(v), 6) for k, v in sorted(nw.items())},
            "turnover": round(turnover, 6),
            "source": "RUN",
        }
        ddir = Path(self.cfg.decisions_dir);
        ddir.mkdir(parents=True, exist_ok=True)
        rid = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out = ddir / f"{self.cfg.decision_prefix}_{rid}_{as_of}.json"
        out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DECISION] wrote {out}")
        return out

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

    # ---------------------------
    # Main
    # ---------------------------
    def run(self) -> None:
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

        # optional: run_id einmalig
        setup_logging(self.cfg.verbose, lib_debug=self.cfg.lib_debug, log_file=self.cfg.save_dir / "run.log")

        # --- NORMALIZE (Runner) ---
        def _assign(obj, name, value):
            try:
                # für evtl. frozen dataclasses
                object.__setattr__(obj, name, value)
            except Exception:
                setattr(obj, name, value)

        cfg = self.cfg  # <— jetzt existiert es im Runner

        # Root-Keys bevorzugen; unterschiedliche TOML-Sektionen robust abholen
        top_k = getattr(cfg, "top_k", None) or getattr(getattr(cfg, "topk", object()), "top_k", None) or 12
        buffer_k = getattr(cfg, "buffer_k", None) or getattr(getattr(cfg, "topk", object()), "buffer_k", None) or 3
        mps = getattr(cfg, "max_per_sector", None) or getattr(getattr(cfg, "limits", object()), "max_per_sector",
                                                              None) or 3
        use_sl = getattr(cfg, "use_sector_limits", None)
        if use_sl is None:
            use_sl = bool(mps and int(mps) > 0)

        days_win = getattr(cfg, "days_win", None) or getattr(getattr(cfg, "windows", object()), "score_days",
                                                             None) or 252
        vol_win = getattr(cfg, "vol_win", None) or getattr(getattr(cfg, "windows", object()), "vol_days", None) or 63
        adjusted = getattr(cfg, "adjusted", None)
        if adjusted is None:
            adjusted = getattr(getattr(cfg, "data", object()), "adjusted", True)

        _assign(cfg, "top_k", int(top_k))
        _assign(cfg, "buffer_k", int(buffer_k))
        _assign(cfg, "max_per_sector", int(mps) if mps is not None else None)
        _assign(cfg, "use_sector_limits", bool(use_sl))
        _assign(cfg, "days_win", int(days_win))
        _assign(cfg, "vol_win", int(vol_win))
        _assign(cfg, "adjusted", bool(adjusted))
        # --- /NORMALIZE ---

        logging.debug(
            "CFG: period=%s as_of=%s max_lookback_days=%s top_k=%s buffer_k=%s rebalance=%s max_per_sector=%s sector_limits=%s decisions_dir=%s prefix=%s dump=%s",
            self.cfg.period, self.cfg.as_of, self.cfg.max_lookback_days, self.cfg.top_k, self.cfg.buffer_k, self.cfg.rebalance_frequency,
            self.cfg.max_per_sector, self.cfg.sector_limits,
            getattr(self.cfg, "decisions_dir", None),
            getattr(self.cfg, "decision_prefix", None),
            getattr(self.cfg, "dump_decision_bundles", None),
        )

        now = pd.Timestamp.now()
        last_dt = self.store.last_rebalance_time()
        logging.info("Force=%s, last_rebalance=%s", self.cfg.force_rebalance, last_dt)

        if (not self.cfg.force_rebalance) and last_dt is not None:
            if self._same_period(last_dt, now, self.cfg.rebalance_frequency):
                self._print_existing_positions()
                logging.info(
                    "Bereits rebalanced in dieser %s – (--force/--force-rebalance) für sofort",
                    "Woche" if self.cfg.rebalance_frequency == "weekly" else "Monat"
                )
                return

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
        if not self.data.sp500_above_200dma():
            logging.warning("Abbruch: S&P 500 unter 200DMA (kein Long-Markt).")
            # optional: minimalistischer Lauf-Eintrag
            run_row = pd.DataFrame([{
                "as_of": as_of_str,
                "adjusted": self.cfg.adjusted, "period": self.cfg.period, "days_win": self.cfg.days_win,
                "gap_th": self.cfg.gap_th, "adv_min": self.cfg.adv_min_dollars,
                "top_k": self.cfg.top_k, "buffer_k": self.cfg.buffer_k,
                "num_universe": len(tickers), "num_pass": 0,
                "fail_counts": dict(self.engine.fail_counts),
                "sector_limits_active": limits_active,
            }])

            self.store.append_csv(self.store.runs_log, run_row)
            # JSONL-Meta (falls verfügbar)
            meta = {
                "as_of": as_of_str,
                "universe_size": len(tickers),
                "num_pass": 0,
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
                "selected_sector_counts": {},
                "fail_counts": dict(self.engine.fail_counts),
                "aborted_reason": "sp500_below_200dma",
            }
            try:
                # nur wenn in store.py vorhanden
                self.store.append_jsonl(self.store.runs_meta_jsonl, meta)  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        # Signale berechnen
        # ===== Gemeinsame Kernberechnung (Core-Calc) =====

        # Universe & Sektoren liegen bereits vor:
        tickers = self.load_tickers()
        sector_map = self._load_sector_map()
        has_sector_meta = bool(sector_map)
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

        # Regime-Check beibehalten (wie vorher)
        if not self.data.sp500_above_200dma():
            logging.warning("Abbruch: S&P 500 unter 200DMA (kein Long-Markt).")
            run_row = pd.DataFrame([{
                "as_of": as_of_str,
                "adjusted": self.cfg.adjusted, "period": self.cfg.period, "days_win": self.cfg.days_win,
                "gap_th": self.cfg.gap_th, "adv_min": self.cfg.adv_min_dollars,
                "top_k": self.cfg.top_k, "buffer_k": self.cfg.buffer_k,
                "num_universe": len(tickers), "num_pass": 0,
                "fail_counts": dict(self.engine.fail_counts),
                "sector_limits_active": limits_active,
            }])
            self.store.append_csv(self.store.runs_log, run_row)
            meta = {
                "as_of": as_of_str,
                "universe_size": len(tickers),
                "num_pass": 0,
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
                "selected_sector_counts": {},
                "fail_counts": dict(self.engine.fail_counts),
                "aborted_reason": "sp500_below_200dma",
            }
            try:
                self.store.append_jsonl(self.store.runs_meta_jsonl, meta)  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        # === Core-Calc Callbacks ===
        def _get_prices(universe, as_of, period, adjusted):
            # Nimm den DataClient – gleiche Datenquelle wie im BT
            return self.data.load_prices(universe, as_of, period, adjusted)

        def _get_sectors(universe):
            # identische Meta wie oben geladen (sector_map)
            return {t: sector_map.get(t, "UNKNOWN") for t in universe}

        # === CalcParams aus normalisierter cfg ===
        params = CalcParams(
            as_of=as_of_str,
            period=getattr(self.cfg, "period", "800d"),
            adjusted=bool(self.cfg.adjusted),
            score_days=int(self.cfg.days_win),
            vol_days=int(vol_days),
            use_under_sma=bool(getattr(self.cfg, "use_under_sma", False)),
            sma_days=int(getattr(self.cfg, "sma_days", 200)),
            gap_filter=float(getattr(self.cfg, "gap_filter", 0.0)),
            min_price=float(getattr(self.cfg, "min_price", 0.0)),
            min_volume=float(getattr(self.cfg, "min_volume", 0.0)),
            use_sector_limits=bool(self.cfg.use_sector_limits),
            max_per_sector=getattr(self.cfg, "max_per_sector", 3),
            top_k=int(self.cfg.top_k),
            buffer_k=int(self.cfg.buffer_k),
            include_cash=bool(getattr(self.cfg, "include_cash", False)),
            weight_round_step=float(getattr(self.cfg, "weight_round_step", 0.0)),
            max_turnover_cap=float(getattr(self.cfg, "max_turnover_cap", 1.0)),
            friction_eps=float(getattr(self.cfg, "friction_eps", 0.0)),
            friction_eps_pct=float(getattr(self.cfg, "friction_eps_pct", 0.0)),
        )

        # === prev_holdings für Turnover-Buffer (aus letzter positions.csv) ===
        prev_df = self.store.load_positions()
        old_w = self._weights_from_positions(prev_df)
        prev_holdings = list(old_w.keys())

        # === Gemeinsame Berechnung (identisch zum BT) ===
        weights, scores = calculate_portfolio(
            tickers, params, _get_prices, _get_sectors, prev_holdings=prev_holdings
        )

        # === 'sel' DataFrame aus den Ergebnissen bauen (für deine Logs/Output) ===
        sel_ticks = list(weights.keys())
        sel = pd.DataFrame({"ticker": sel_ticks})
        sel["score"] = sel["ticker"].map(lambda t: float(scores.get(t, np.nan)))
        # Rank nur innerhalb der Auswahl:
        sel["rank"] = sel["score"].rank(ascending=False, method="first").astype("Int64")

        # Allocation in Prozent (weights sind 0..1)
        w_pct = {t: 100.0 * float(w) for t, w in weights.items()}
        sel["allocation_pct"] = sel["ticker"].map(lambda t: round(w_pct.get(t, 0.0), 2))

        # Optional: Rundungsschritt wie zuvor
        step = float(getattr(self.cfg, "weight_round_step", 0.0) or 0.0)
        if step > 0:
            sel["allocation_pct"] = (sel["allocation_pct"] / step).round() * step
            s = float(sel["allocation_pct"].sum())
            if s > 0:
                sel["allocation_pct"] = sel["allocation_pct"] * (100.0 / s)

        # As-of Spalte für deine Logs
        sel.insert(0, "as_of", as_of_str)

        # Sektoren für Ansicht
        if sector_map:
            sel["sector"] = sel["ticker"].map(sector_map).fillna("Unknown")

        # Ausgabe wie gehabt
        want = ["as_of", "ticker"]
        if "sector" in sel.columns: want.append("sector")
        want += ["rank", "score", "allocation_pct"]
        print("\nNeues Portfolio (Turnover-Puffer aktiv):\n")
        print(sel[want])

        # Für die Folge-Logs (unten) brauchst du auch 'df' nicht mehr;
        # wir setzen num_pass auf len(sel)
        df = sel.copy()
        # ===== /Gemeinsame Kernberechnung =====

        # --- Decision-Bundle (Runner) ---
        if getattr(self.cfg, "dump_decision_bundles", False):
            # alte Gewichte (aus letzter positions.csv)
            prev_df = self.store.load_positions()
            old_w = self._weights_from_positions(prev_df)

            # neue Gewichte (aus aktueller Auswahl)
            new_w = self._weights_from_positions(sel)

            # --- LOCKSTEP: RUN-Parameter/Universe-Check ---

            ticks = self.load_tickers()
            univ_sha = hashlib.sha1(("|".join(sorted(map(str, ticks)))).encode()).hexdigest()

            print(
                f"[LOCKSTEP][RUN] as_of={as_of_str} "
                f"top_k={self.cfg.top_k} buffer_k={self.cfg.buffer_k} "
                f"use_sector_limits={(self.cfg.max_per_sector is not None and self.cfg.max_per_sector > 0) or bool(self.cfg.sector_limits)} "
                f"max_per_sector={self.cfg.max_per_sector} "
                f"score_days={getattr(self.cfg, 'days_win', None)} vol_days={getattr(self.cfg, 'vol_win', None)} adjusted={self.cfg.adjusted} "
                f"universe_len={len(ticks)} sha={univ_sha}"
            )
            # --- /LOCKSTEP ---

            # Bundle schreiben – nutzt den oben berechneten as_of_str
            self._write_decision_bundle(as_of_str, old_w, new_w)

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

