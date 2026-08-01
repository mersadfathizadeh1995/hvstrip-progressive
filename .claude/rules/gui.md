# Rule: `gui/` — legacy PyQt5 (retiring) + the v2 house-style workbench

Loaded when you touch `HV_Strip_Progressive/gui/**`.

## The split (until the cutover — POSTPONED past GUI Round 2/3)
- **`gui/` (legacy, PyQt5)** — FROZEN except bug fixes, and since Round 2 it is the **PORT
  QUARRY**: the v2 mpl figures are PORTS of the legacy widgets (format_input_stack,
  layer_table_widget, profile_preview_widget, the interactive HV figure), significantly
  enhanced — never deleted before their port lands. It bypasses the api and owns the legacy
  dict-config — do NOT extend either; new capability goes into `api/` + the v2 tree.
  `matplotlib.use("Qt5Agg")` at import in `widgets/plot_widget.py` + `profile_preview_widget.py`
  poisons a PySide6 process — v2 code never IMPORTS legacy modules (porting = copy + strip the
  Qt5Agg/QT_API lines + rebase on `canvas/mpl_widget.MplFigureWidget`).
- **Porting rules (Round 2):** every empirical/physics derivation goes through
  `api.suggest_layer_fill` / `VelocityConverter` — the legacy table's LOCAL ν/density copies
  diverged from core and must never come back.
- **`gui/v2/` (the rebuild, PySide6-only)** — the house-style TOOL-COLLECTION workbench
  (Forward | Strip | Research switcher; amber accent on theme_core; per-tool status). Panels talk
  ONLY to the AppState over ONE `HVStripAnalysis`; long ops on the OpQueue (research on its OWN
  queue); role/dynamic-property QSS only (no `setStyleSheet` literals — the legacy 45-file
  diaspora is the anti-pattern); per-window `apply_theme` following `theme_authority`.
  Copy-not-import house widgets (from `invert_hvsr/gui` or the gui-house-style plugin templates).

## Conventions
- The co-design docs (`HV_Pro_docs/hvstrip_gui/`) + `specs/001-gui-house-style/` are the design
  law for v2; the AST layering guard + theme-conformance test become enforcing at the cutover.
- Never block the UI thread (>100 ms → the queue); progress via the ProgressBridge coalescer.
- v2 plotting: `backend_qtagg`/pyqtgraph — never a hard `matplotlib.use("Qt5Agg")`.

## Don't
- Mix PyQt5 and PySide6 in one process; add gui→core frozen-compute reach-ins (route via the
  api/AppState); extend the legacy dict-config; import legacy gui modules from v2.
