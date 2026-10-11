# Projektverfassung

Diese Datei definiert die gemeinsamen Leitplanken und verweist auf die ausführbaren Workflows:

- [`implement-task`](.agents/skills/implement-task/SKILL.md) für fokussierte Implementierungen und Bugfixes.
- [`strict-review`](.agents/skills/strict-review/SKILL.md) für unabhängige, skeptische Reviews.

## Verbindliche Regeln

- **Clarification Gate:** Bei fachlich relevanten, widersprüchlichen oder mehrdeutigen Anforderungen vor der Implementierung gezielt nachfragen, statt zu raten.
- **Scope Discipline:** Keine ungefragten Refactorings, Bereinigungen oder sonstigen Änderungen außerhalb des Auftrags.
- **Human Gate:** Pull Requests dürfen vorbereitet, aber niemals automatisch gemergt werden.

## `aktien_oop`

- Erwartete Python-Version ist 3.11.
- Falls `.venv311` im Repository-Root fehlt, mit Python 3.11 erstellen.
- Vor Tests die Entwicklungsabhängigkeiten stets aus `aktien_oop/requirements-dev.txt` installieren beziehungsweise aktualisieren; dafür den Interpreter der Umgebung mit `-m pip install -r aktien_oop/requirements-dev.txt` verwenden.
- Tests anschließend über den Interpreter dieser Umgebung ausführen: `.venv311/Scripts/python.exe -m pytest aktien_oop/tests` unter Windows beziehungsweise `.venv311/bin/python -m pytest aktien_oop/tests` unter macOS/Linux.
- Die bestehende Architektur respektieren.
- Generierte oder untracked Artefakte nicht versehentlich committen.
