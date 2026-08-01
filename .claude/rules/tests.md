# Rule: `tests/` + `tests_v2/` — two suites, two bindings, NEVER one process

Loaded when you touch `tests/**` or `tests_v2/**`.

## The split (until the cutover)
- **`tests/` (legacy)** — the PyQt5 suite (~150 pass / 24 skip). Its `conftest.py` FORCES
  `QT_API=pyqt5`, `Qt5Agg`, and eagerly imports PyQt5 — never import PySide6 here. It guards the
  still-shipping legacy GUI + the shared core. Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests`
  from the distribution root.
- **`tests_v2/` (the rebuild)** — PySide6-offscreen conftest (manual `QApplication` fixture, NOT
  qtbot; assert with `isHidden()`), api + AppState + v2 GUI tests. Run as a SEPARATE pytest
  process: `python -m pytest tests_v2`. At the cutover it becomes `tests/`.

## The golden lock
- `tests/test_golden_compute.py` + `tests/golden/` = the frozen-compute proof. `sh_wave` runs
  everywhere; `diffuse_field` is `-m hvf` (needs the vendored exe). The digests must stay
  bit-identical through the uplift; regenerate ONLY for an intended numeric change
  (`python tests/test_golden_compute.py --regen`) with user sign-off.
- `tests/golden/legacy_gui_config.json` pins the legacy dict-config shape (the migration fixture).

## Conventions
- Environment-dependent engines (ellipticity/gpell) SKIP cleanly when binaries are absent — never
  fail on a missing machine path.
- The layering guard (`tests_v2/test_layering.py`) is report-only until the cutover, then law.
- Ship a test with every core/api change; numeric changes pin expected values.

## Don't
- Import PySide6 in `tests/` or PyQt5 in `tests_v2/`; run the two suites in one pytest process;
  or "fix" a golden mismatch by regenerating without understanding the drift.
