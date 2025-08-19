# Momentum Screener (OOP) — README

> **Kurzfassung:** Systematischer Aktien‑Screener mit 12‑1‑Momentum, Trend‑Score (slope×R²), SMA‑Filtern, monatlichem Rebalancing (Turnover‑Puffer) und inverse‑Vol‑Gewichtung. Datenquelle: Yahoo Finance (`yfinance`).

## Features
- **Signale**
  - **Trend-Score:** lineare Regression auf Log‑Preisen (≈ 100 Handelstage), Score = *Steigung × R²*
  - **12‑1 Momentum:** Rendite 12 Monate ohne den letzten Monat
- **Filter**
  - **SMA100 (Aktie):** Kurs muss > 100‑Tage‑SMA sein
  - **Gap‑Filter:** keine zu großen Tages‑Gaps (bereinigte Serie, einstellbar)
  - **Liquidität:** Mindest‑Durchschnittsumsatz (Close×Volume, 20d)
- **Markt‑Regime:** S&P 500 muss über **SMA200** liegen (defensiver Long‑Filter)
- **Umsetzung**
  - **Monatliches Rebalancing** mit **Turnover‑Puffer** (z. B. Top‑K halten bis Rang ≤ Buffer‑K)
  - **Inverse‑Vol‑Gewichtung** (risikoärmere Titel erhalten höhere Gewichte)
- **Logging:** CSV‑Dateien mit Rankings, Läufen und aktuellem Portfolio

---

## Projektstruktur (Vorschlag)
```text
/dein-projekt
├─ main.py                      # Entry Point (ruft Runner auf)
├─ config.py                    # Config, CLI, Logging, Ticker-Normalisierung
├─ models.py                    # Dataclasses (TickerSignal, PortfolioPosition)
├─ utils.py                     # Hilfsfunktionen (as_series, ...)
├─ data_client.py               # yfinance-Zugriff + S&P-200DMA-Check
├─ indicators.py                # Indikatorlogik (Momentum, ATR, Vol, ...)
├─ engine.py                    # Filter & Scoring (SignalEngine)
├─ store.py                     # CSV-I/O (PortfolioStore)
├─ rebalance.py                 # Rebalancer (Buffer, Allocation, Timing)
├─ runner.py                    # Orchestrierung
├─ update_universe.py           # (optional) S&P-500-Liste aktualisieren
├─ sp500_tickers.txt            # Universum (eine Zeile pro Ticker)
├─ alias_map.json               # (optional) Alias-Map (BRK.B→BRK-B, FB→META, ...)
└─ __init__.py                  # nur bei Paket-Start (python -m ...)
```

> **Hinweis Startmodus**
> - **Einzel‑Skripte (absolute Imports):** `python main.py ...`
> - **Als Paket (relative Imports):** `python -m aktien_oop.main ...` und `__init__.py` im Ordner.

---

## Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install numpy pandas yfinance scikit-learn
# optional für update_universe.py
pip install lxml html5lib beautifulsoup4
```

---

## Quickstart
**Universum (empfohlen) aktualisieren:**
```bash
python update_universe.py
# schreibt: sp500_tickers.txt (ok) und sp500_invalid.txt (Ausfälle)
```

**Run (monolithisch oder modular):**
```bash
# modular
python main.py --tickers sp500_tickers.txt --verbose

# als Paket (bei relativen Imports)
python -m aktien_oop.main --tickers aktien_oop\sp500_tickers.txt --verbose

Nur globales Limit (Default ist bereits 2):
python -m aktien_oop.main --tickers aktien_oop\sp500_tickers.txt --max-per-sector 2 --verbose

Global 2, aber Industrials strikt 1:
python -m aktien_oop.main --tickers aktien_oop\sp500_tickers.txt \
  --max-per-sector 2 \
  --sector-limit "Industrials=1" \
  --verbose
  
Sektor-Limits komplett aus (keine Begrenzung):
python -m aktien_oop.main --tickers aktien_oop\sp500_tickers.txt --max-per-sector 0
```

**Ablauf (vereinfacht):**
1. **S&P‑Filter:** Index über SMA200? → sonst Abbruch.
2. **Aktien‑Filter:** Liquidität, **SMA100**, Gap, Datenfenster.
3. **Signale:** Trend‑Score (slope×R²), 12‑1 Momentum, Volatilität, ATR/Stop‑Loss.
4. **Kombi‑Score:** Ø der Perzentil‑Ranks von Trend‑Score und 12‑1 Momentum.
5. **Puffer:** bestehende Titel bleiben bis Rang ≤ *buffer_k*; Rest bis *top_k* auffüllen.
6. **Gewichtung:** inverse Volatilität; CSV‑Logs schreiben.

---

## Wichtige CLI‑Flags (Auszug)
| Flag | Bedeutung | Default |
|---|---|---|
| `--adjusted` / `--no-adjusted` | bereinigte Kurse (Splits/Dividenden) | `--adjusted` |
| `--period` | Downloadfenster (z. B. `400d`) | `400d` |
| `--days-win` | Fenster für Trend/ATR/Vol | `100` |
| `--gap` | max. Tages‑Gap (bereinigt) | `0.18` |
| `--adv-min` | Mindest‑Ø‑Umsatz (Close×Volume, 20d) | `2000000` |
| `--top-k` | Zielanzahl Titel | `8` |
| `--buffer-k` | Turnover‑Puffergrenze | `12` |
| `--force` | Rebalance erzwingen | `False` |
| `--save-dir` | Ausgabepfad für CSV | `.` |
| `--tickers` | Universumsdatei | `sp500_tickers.txt` |
| `--verbose` | ausführlichere Logs | `False` |
| `--lib-debug` | Bibliotheks‑Logs (yfinance/urllib3) anzeigen | `False` |

---

## SMA‑Filter & Markt‑Regime
- **SMA100 (Aktie):** Kurs muss über SMA100 liegen, sonst Ausschluss. Optional loggen: `sma100_val`, `sma100_dist_pct` (%‑Abstand).
- **SMA200 (Markt):** S&P 500 über SMA200? → sonst defensiv: kein Portfolio‑Build.

---

## CSV‑Ausgaben (Spalten‑Referenz)
**`portfolio_positions.csv`** – aktueller Bestand (**überschreibt** sich)
- `as_of`, `ticker`, `allocation_pct` (%), `rank`, `score`

**`topk_log.csv`** – Historie der Portfolios (**append**)
- `as_of`, `ticker`, `rank`, `score`, `volatility` (p.a.), `stop_loss_pct` (%), `allocation_pct` (%)
- optional: `sma100_val`, `sma100_dist_pct`

**`rankings_log.csv`** – vollständiges Ranking (**append**)
- Signale: `score_lin` (slope×R²), `mom_12_1`, `slope`, `r2`, `volatility`, `stop_loss_pct`
- Ranks: `rank_lin`, `rank_m121`; Gesamt: `score` (0..1), `rank` (1=top)
- optional: `sma100_val`, `sma100_dist_pct`

**`runs_log.csv`** – Lauf‑Metadaten (**append**)
- Parameter: `adjusted`, `period`, `days_win`, `gap_th`, `adv_min`, `top_k`, `buffer_k`
- Umfang: `num_universe`, `num_pass`; Diagnose: `fail_counts`

**Universums‑Files**
- `sp500_tickers.txt` (aktualisiert & normalisiert), `sp500_invalid.txt` (Ausfälle)

---

## Sektor-Limits (Diversifikation)

Dieses Projekt unterstützt **Sektor-Limits**, um Klumpenrisiken (z. B. 2 Cruise Lines) zu vermeiden.  
Die Limits greifen bei der **Auswahl** (Turnover-Puffer + Auffüllen).

**Defaults (ohne Flags):**
- `sector_meta_file` = `sp500_meta.csv`  
- `max_per_sector`  = **2** (globales Limit je Sektor)  
- `sector_limits`   = *keine speziellen Limits* (nur das globale Limit gilt)

> **Reihenfolge der Regeln:** Spezifische Limits (z. B. „Industrials=1“) haben **Vorrang** vor dem globalen Limit.

### Meta-Datei (Ticker → Sektor)
Die Datei `sp500_meta.csv` enthält mindestens die Spalten:


## Best Practices
- **Disziplin:** fixe Rebalance‑Zyklen, keine Ad‑hoc‑Entscheidungen.
- **Kosten/Steuern:** Wechsel minimieren (Puffer!), realistisch testen.
- **Datenqualität:** `--adjusted` nutzen; Universum regelmäßig updaten.
- **Diversifikation:** optional Sektor‑/Cluster‑Limits ergänzen.
- **Dokumentation:** CSV‑Logs für Nachvollziehbarkeit archivieren.

---

## Troubleshooting
- **„Keine Ergebnisse“:** Filter lockern (`--gap 0.22`, `--adv-min 1000000`, `--days-win 120`) und Universum updaten.
- **yfinance laut:** ohne `--verbose` starten oder `--lib-debug` weglassen.
- **Import‑Fehler:** Entweder *alle* Imports absolut + `python main.py`, **oder** relative + `python -m paket.main` mit `__init__.py`.

---

## Roadmap / Ideen
- HTML/PNG‑**Chart‑Report** (Preis + SMA100 + Regressionslinie) der Top‑Titel
- Sektor‑Caps / Cluster‑Kontrolle
- Dual Momentum (Index vs. Cash)
- Vol‑Targeting auf Portfolioebene
- Parallele Downloads (vorsichtig wg. Rate‑Limits)

---

## Haftung
Kein Anlage‑/Finanzberatung. Historische Signale sind **keine** Garantie für zukünftige Ergebnisse.
