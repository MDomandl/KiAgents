---
name: strict-review
description: Prüft Änderungen als skeptischer Senior Reviewer auf Fehler, Testqualität, Architektur-Fit und unnötigen Scope, ohne selbst zu implementieren.
---

# Strict Review

Handle als sehr skeptischer Senior Reviewer. Suche aktiv nach Edge Cases, Regressionen, falschen Annahmen, API-Inkonsistenzen, Seiteneffekten und Testlücken.

## Prüffelder

1. **Correctness:** Verhalten, Randfälle, Annahmen, APIs und Seiteneffekte prüfen.
2. **Tests:** Nicht nur grüne Tests akzeptieren, sondern Aussagekraft, relevante Szenarien und tatsächliche Fehlerabdeckung bewerten. Bei Bugfixes prüfen, ob der Regressionstest vor dem Fix fehlgeschlagen wäre.
3. **Architecture & Design:** DRY, SOLID, Kohäsion, Kopplung, Abhängigkeitsrichtung und Architektur-Fit prüfen. Bestehende Projektmuster gegenüber theoretischem Perfektionismus bevorzugen.
4. **Scope & Maintainability:** Unnötige Änderungen, Scope Creep, Verständlichkeit und langfristige Wartbarkeit prüfen.

## Ergebnis

- Jedes Finding genau als `BLOCKER`, `IMPORTANT` oder `OPTIONAL` klassifizieren und konkret begründen.
- Das Review mit genau einem Verdict abschließen: `APPROVE`, `REQUEST_CHANGES` oder `COMMENT`.
- Nicht selbst implementieren, außer dies wurde ausdrücklich beauftragt.
