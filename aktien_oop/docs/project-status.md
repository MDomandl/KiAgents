# Projektstatus `aktien_oop`

Stand: 12. Oktober 2026, Repository-Basis `59b2eb3`

## Einordnung

`aktien_oop` ist das Fachsystem für Aktienstrategie, Backtest, Runner und
Portfoliozustand. Das separate Projekt `automation-agents` nutzt diese
Funktionalität und orchestriert darauf den BT/RUN-Ablauf. Die Systemgrenze und
die belegten Verträge sind in [architecture.md](architecture.md) festgehalten.

## Aktuell abgesicherter Fix-Stand

Die jüngsten Änderungen beheben zusammenhängende Fehler an Konfiguration und
Portfolio-Snapshots:

- `Config.from_cli()` übernimmt `include_cash` und `cash_yield_annual` aus dem
  Abschnitt `[regime]` und reicht beide Werte in die Runner-Konfiguration weiter.
- Der Backtest schreibt auch für eine leere Zielauswahl einen Positions-Snapshot
  mit stabilem CSV-Schema (`as_of`, `ticker`, `weight`, `score`, `rank`, `sector`).
- `PortfolioStore.save_positions()` behandelt einen Stichtag als vollständigen
  Snapshot: erneutes Speichern ersetzt für dieses `as_of` alle alten Positionen,
  statt nur übereinstimmende Ticker zu aktualisieren.
- Ein ausdrücklich leerer Snapshot entfernt über `snapshot_as_of` die bisherigen
  Positionen dieses Stichtags. Ältere und neuere Stichtage bleiben erhalten.
- Der additive Index `portfolio_snapshots.csv` hält jeden ausdrücklich
  gespeicherten Stichtag fest, ohne das Schema von `portfolio_positions.csv` zu
  verändern. `load_positions_snapshot_before()` unterscheidet dadurch fehlende,
  leere und nicht leere Snapshots; `load_positions_before()` liefert für einen
  bekannten leeren Snapshot ein leeres DataFrame statt auf ältere Positionen
  zurückzufallen.
- Der Runner behandelt nur einen fehlenden Snapshot als First Run. Ein bekannter
  leerer Snapshot ist der autoritative Zustand „flat“.

Diese Semantik ist für `automation-agents` relevant, weil dessen Vergleiche,
Seed-Logik und Reports die Konfigurationswerte sowie Backtest- und Runner-Artefakte
von `aktien_oop` lesen.

## Reproduzierbare Testumgebung

Die erwartete Laufzeit ist Python 3.11. Vom Repository-Root aus wird die
Entwicklungsumgebung unter Windows so synchronisiert und geprüft:

```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\python.exe -m pip install -r aktien_oop/requirements-dev.txt
.venv311\Scripts\python.exe -m pytest aktien_oop/tests
```

Unter macOS/Linux werden entsprechend `.venv311/bin/python` und `python3.11`
verwendet. `requirements-dev.txt` bindet die Laufzeitabhängigkeiten aus
`requirements.txt` ein und ergänzt `pytest`. Die Abhängigkeiten sind vor jedem
Testlauf zu synchronisieren, nicht nur beim erstmaligen Erstellen der Umgebung.

Die fokussierten Regressionstests liegen insbesondere in
`test_config_cash.py`, `test_backtest_empty_positions.py` und
`test_store_positions.py`. Der vollständige Stand wird mit der gesamten Suite
unter `aktien_oop/tests` geprüft.

Der dokumentierte Stand wurde am 12. Oktober 2026 mit Python 3.11.9 und den
synchronisierten Entwicklungsabhängigkeiten geprüft: `26 passed`. Dabei traten
zwei bekannte NumPy-`RuntimeWarning`s im Test des leeren Backtest-Snapshots auf;
sie führten nicht zu Testfehlern.

## Noch offene Architekturarbeit

Die Kopplung zu `automation-agents` ist real und wird derzeit über CLI,
Konfigurationen, Dateien und teilweise direkte Python-Imports getragen. Sie ist
noch nicht als versionierte öffentliche API gekapselt. Außerdem ist die dortige
Zielregel „Agenten koordinieren, Side Effects liegen in Infrastructure“ noch
nicht überall technisch erzwungen. Beides ist bei späteren Refactorings als
Kompatibilitäts- beziehungsweise Architekturrisiko zu behandeln; dieser
Dokumentationsstand nimmt keine entsprechende Produktionsänderung vorweg.

Die Seed-Logik in `automation-agents` ist noch nicht auf die neue tri-state
Lesegrenze migriert. Sie kann anschließend
`PortfolioStore.load_positions_snapshot_before()` verwenden, ohne die
Snapshot-Semantik selbst aus CSV-Zeilen ableiten zu müssen.
