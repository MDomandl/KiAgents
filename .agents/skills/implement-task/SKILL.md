---
name: implement-task
description: Implementiert klar abgegrenzte Änderungen und Bugfixes minimal, architekturtreu und mit angemessener Verifikation.
---

# Implement Task

1. Vor Änderungen den relevanten aktuellen Code, die Tests und die bestehende Architektur verstehen.
2. **Clarification Gate:** Bei fachlich relevanten, widersprüchlichen oder mehrdeutigen Anforderungen gezielt nachfragen, bevor implementiert wird.
3. Die kleinste vollständige Lösung umsetzen. Keine Scope-Creep-Refactorings oder unrelated Cleanups vornehmen.
4. Bestehende Projektmuster und Architektur respektieren; DRY und SOLID pragmatisch anwenden, nicht dogmatisch.
5. Bei Bugfixes einen Regressionstest ergänzen, der den Fehler zuverlässig abdeckt.
6. Zuerst gezielte Tests ausführen, danach angemessene breitere Tests. Nicht ausführbare Tests und den Grund transparent melden.
7. Vor Abschluss den finalen Diff auf Korrektheit, Scope und unbeabsichtigte Artefakte prüfen.
8. Wenn ein Review-Workflow vorhanden ist, Review anfordern. Pull Requests niemals automatisch mergen.
