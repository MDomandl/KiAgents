# Projektverfassung

Diese Datei definiert die gemeinsamen Leitplanken und verweist auf die ausführbaren Workflows:

- [`implement-task`](.agents/skills/implement-task/SKILL.md) für fokussierte Implementierungen und Bugfixes.
- [`strict-review`](.agents/skills/strict-review/SKILL.md) für unabhängige, skeptische Reviews.

## Verbindliche Regeln

- **Clarification Gate:** Bei fachlich relevanten, widersprüchlichen oder mehrdeutigen Anforderungen vor der Implementierung gezielt nachfragen, statt zu raten.
- **Scope Discipline:** Keine ungefragten Refactorings, Bereinigungen oder sonstigen Änderungen außerhalb des Auftrags.
- **Human Gate:** Pull Requests dürfen vorbereitet, aber niemals automatisch gemergt werden.

## `aktien_oop`

- Die bekannte Projektumgebung ist `.venv311` im Repository-Root; Tests über deren Python-Interpreter ausführen.
- Die bestehende Architektur respektieren.
- Generierte oder untracked Artefakte nicht versehentlich committen.
