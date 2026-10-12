# Architektur und Systemgrenze

Stand: 12. Oktober 2026

## Verantwortlichkeiten

`aktien_oop` ist das Fachsystem. Es besitzt die fachlichen Regeln und Datenmodelle
für Konfiguration, Universum, Signal- und Portfolioermittlung, Backtest, Runner und
Portfolio-Persistenz. Änderungen an diesen Regeln gehören deshalb in dieses
Repository und dürfen nicht in einer Agentenschicht nachgebaut werden.

`automation-agents` ist ein separates, aufsetzendes Automatisierungssystem. Sein
BT/RUN-Agent startet Backtest und Runner von `aktien_oop`, vergleicht deren
Decision Bundles und erzeugt Berichte. Agenten koordinieren diesen Ablauf; sie
sollen keine Fachlogik von `aktien_oop` duplizieren.

Für `automation-agents` gilt die Zielrichtung
`Agent -> Tools -> Application -> Domain`, mit Ports an der Grenze zu technischen
Diensten. Side Effects wie Subprozesse, Dateisystem- und API-Zugriffe gehören in
Infrastructure-Adapter und werden im Bootstrap injiziert. Der derzeitige Code
erfüllt diese Trennung noch nicht durchgängig: Process-Tools importieren den
konkreten `SubprocessRunner`, und der BT/RUN-Agent enthält einzelne Dateizugriffe.
Das ist bestehende technische Schuld, kein alternatives Architekturziel.

## Belegte Integrationsverträge

Die Integration ist derzeit eine lokale Prozess- und Artefaktkopplung, keine
eigenständige HTTP- oder Paket-API. `automation-agents` setzt insbesondere auf:

- die Einstiegspunkte `python -m aktien_oop.backtest` und
  `python -m aktien_oop.main` einschließlich `--config` und beim Runner `--as-of`;
- TOML-Verträge wie `aktien_oop/backtest_config.toml` und
  `aktien_oop/configs/runner_config.toml`, darunter `as_of`, `save_dir`, Universe-,
  Top-k-, Rebalance-, Limit- und Regime/Cash-Werte;
- Decision Bundles unter dem konfigurierten `decisions_dir`, einschließlich
  `BT_*.json`, `RUN_*.json` und ihrer Identifikatoren, Stichtage und Portfolios;
- Positions-CSV-Dateien, besonders Backtest-Snapshots und
  `portfolio_positions.csv`, mit stabilen Spalten wie `as_of`, `ticker`, `weight`,
  `score` und `rank`;
- direkte Python-Imports in Analyse- und Vertragstests, unter anderem
  `aktien_oop.core_calc` und `aktien_oop.store.PortfolioStore`.

Die Anwendungsgrenze von `automation-agents` enthält mit `ProcessRunner` und
`DecisionBundleStorePort` bereits Ports für Prozessausführung und Artefaktzugriff.
Der konkrete `SubprocessRunner` und `FileDecisionBundleStore` sind
Infrastructure-Adapter. Spätere Umbauten sollen diese Richtung vervollständigen,
nicht Fachverantwortung in Agenten verlagern.

## Änderungsregel für die Projektgrenze

Die genannten CLI-, Konfigurations-, Import-, Dateinamen- und Schema-Verträge
dürfen bei Refactorings in `aktien_oop` nicht unbemerkt brechen. Vor einer
inkompatiblen Änderung sind die Aufrufer in `automation-agents` zu ermitteln und
beide Seiten gemeinsam zu migrieren. Mindestens die betroffenen Tests beider
Repositories müssen den alten Vertrag absichern oder die absichtlich geänderte
Version dokumentieren.

Interne Bereinigung allein ist kein Grund, einen beobachtbaren Vertrag zu ändern.
Wo eine stabilere Grenze eingeführt wird, bleibt die bisherige Schnittstelle bis
zur nachgewiesenen Migration von `automation-agents` kompatibel oder erhält einen
expliziten Migrationspfad.
