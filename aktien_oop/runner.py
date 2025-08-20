# runner.py
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import logging

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

    # ---------------------------
    # Helpers
    # ---------------------------
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
        setup_logging(self.cfg.verbose, lib_debug=self.cfg.lib_debug, log_file=self.cfg.save_dir / "run.log")
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
                "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
                "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
        rows = []
        for i, t in enumerate(tickers, 1):
            logging.info("[%d/%d] %s ...", i, len(tickers), t)
            sig = self.engine.compute_for_ticker(t)
            if sig:
                rows.append(sig.__dict__)

        # Nichts durchgekommen?
        if not rows:
            logging.warning("Keine Ergebnisse nach Filtern/Berechnung.")
            run_row = pd.DataFrame([{
                "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "adjusted": self.cfg.adjusted, "period": self.cfg.period, "days_win": self.cfg.days_win,
                "gap_th": self.cfg.gap_th, "adv_min": self.cfg.adv_min_dollars,
                "top_k": self.cfg.top_k, "buffer_k": self.cfg.buffer_k,
                "num_universe": len(tickers), "num_pass": 0,
                "fail_counts": dict(self.engine.fail_counts),
                "sector_limits_active": limits_active,
            }])
            self.store.append_csv(self.store.runs_log, run_row)
            meta = {
                "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
            }
            try:
                self.store.append_jsonl(self.store.runs_meta_jsonl, meta)  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        # DataFrame + Scoring
        df = pd.DataFrame(rows)

        # Ranks & Kombi-Score (Ø der Perzentil-Ranks)
        df["rank_lin"] = df["score_lin"].rank(pct=True)
        df["rank_m121"] = df["mom_12_1"].rank(pct=True).fillna(df["rank_lin"])
        df["score"] = 0.5 * df["rank_lin"] + 0.5 * df["rank_m121"]

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)

        # vollständiges Ranking loggen
        full_rank_log = df.copy()
        full_rank_log.insert(0, "as_of", pd.Timestamp.now().strftime("%Y-%m-%d"))
        self.store.append_csv(self.store.rankings_log, full_rank_log)

        # Rebalance-Entscheid
        prev_positions = self.store.read_positions()
        if not self.rebalancer.should_rebalance():
            logging.info("Diesen Monat bereits rebalanced – bestehende Positionen bleiben. (--force für sofort)")
            print("\nBestehende Positionen:\n")
            print(prev_positions)
            # optional: auch diesen „No-Op“-Lauf protokollieren
            run_row = pd.DataFrame([{
                "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "adjusted": self.cfg.adjusted, "period": self.cfg.period, "days_win": self.cfg.days_win,
                "gap_th": self.cfg.gap_th, "adv_min": self.cfg.adv_min_dollars,
                "top_k": self.cfg.top_k, "buffer_k": self.cfg.buffer_k,
                "num_universe": len(tickers), "num_pass": len(df),
                "fail_counts": dict(self.engine.fail_counts),
                "sector_limits_active": limits_active,
            }])
            self.store.append_csv(self.store.runs_log, run_row)
            meta = {
                "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
                "selected_sector_counts": {},
                "fail_counts": dict(self.engine.fail_counts),
                "note": "no_rebalance_this_month",
            }
            try:
                self.store.append_jsonl(self.store.runs_meta_jsonl, meta)  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        # Auswahl mit Turnover-Puffer + Sektor-Limits
        sel = self.rebalancer.select_with_buffer(
            df[["ticker", "rank", "score", "volatility", "stop_loss_pct"]],
            prev_positions,
            self.cfg.top_k,
            self.cfg.buffer_k,
            sector_map=sector_map if limits_active else None,
            max_per_sector=self.cfg.max_per_sector if limits_active else None,
            sector_limits=self.cfg.sector_limits if limits_active else None,
        )

        # Allokation & Ausgabe
        sel["allocation_pct"] = self.rebalancer.inverse_vol_allocation(sel)
        sel.insert(0, "as_of", pd.Timestamp.now().strftime("%Y-%m-%d"))

        # optional Sektor in der Sicht anzeigen (ändert nicht die Logs/Speicherform)
        if sector_map:
            sel["sector"] = sel["ticker"].map(sector_map).fillna("Unknown")

        # schön sortierte Ansicht
        want = ["as_of", "ticker"]
        if "sector" in sel.columns:
            want.append("sector")
        want += ["rank", "score", "volatility", "stop_loss_pct", "allocation_pct"]
        for c in want:
            if c not in sel.columns:
                sel[c] = pd.NA

        print("\nNeues Portfolio (Turnover-Puffer aktiv):\n")
        print(sel[want])

        # Logs & Persistenz
        self.store.append_csv(self.store.topk_log, sel)
        run_row = pd.DataFrame([{
            "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
            "as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
        self.store.write_positions(sel[["as_of", "ticker", "allocation_pct", "rank", "score"]])

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
