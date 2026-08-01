# Spec 001 — HV Strip house-style GUI (the invert_hvsr treatment)

*What & why. The approved design contract = `HV_Pro_docs/hvstrip_gui/{SKETCH.txt, DECISIONS.md}`
(user-approved 2026-07-10). The plan of record = `.context/ARCHITECTURE_ASSESSMENT.md` §2.*

## Why

The legacy PyQt5 GUI bypasses the package's own api facade, carries a divergent dict config, a
one-off colliding blue theme, and can only be launched from the retiring legacy HV Pro app.
HV Strip must join the family: the house-style PySide6 workbench over the ONE `HVStripAnalysis`
facade, opened from the HV Hub and a standalone `.bat`, guard-enforced.

## Functional requirements

- **FR-1 Tool-collection shell** — a tool switcher `Forward Model | HV Strip | Research` with
  PER-TOOL status (no pipeline chain); amber accent on theme_core; per-window theming.
- **FR-2 Forward tool** — Single | Multiple sub-stages; profile from file OR an editable layer
  table (auto vp/ρ derivation); engine/frequency/peaks cards; runs via the api on the OpQueue;
  HV∥Vs, overlay, all-profiles, summary views.
- **FR-3 Strip tool** — Single = the 3-sub-stage wizard (Model → Configure & Run → Review);
  Batch = multi-profile with the Progress-dock table (row per profile: status·step·f₀); views:
  waterfall overlay, per-step HV∥Vs, summary table, Figure Studio (matplotlib).
- **FR-4 Research tool** — phase sub-stages (Profiles→Comparison→Metrics→Field→Report), each
  runnable alone + Run-full-study + cancel-between-phases, on its OWN OpQueue.
- **FR-5 Steps-as-layers** — strip steps / forward-multi profiles are keyed, toggleable,
  stylable canvas items driven by the right-rail Layers | Properties (bedrock nesting; per-item
  color/style/width/opacity + an f₀/A₀/Vs30/n-layers info readout; no blank Properties).
- **FR-6 One backend** — panels talk ONLY to AppState over ONE `HVStripAnalysis`; long ops
  stream progress frames through the ProgressBridge; config is `HVStripConfig` only (v2
  payloads; legacy dicts migrate on read via `api.load_config_payload`).
- **FR-7 Settings split** — run knobs in tool cards; ONE File→Settings dialog for engine binary
  paths + output defaults; engine availability = status badges (`check_engines`), never a crash.
- **FR-8 Hub + launch** — package-root `app.py` `main_window_for_project(project, item_id)` →
  `Project.strip_dir`; an hv_hub `StripProvider` + real `StripView` replacing the
  PlaceholderView (rail description fixed); `hv_strip.bat` standalone. The legacy-app launch
  path is REMOVED (no shim).
- **FR-9 Guards** — the AST layering guard flips ENFORCING at cutover (PySide6-only, api-only
  compute, Qt-free package import) + a theme-conformance test (amber distinct, no hex outside
  theme.py).

## Design constraints (DC)

- **DC-1 Compute frozen** — `tests/golden/` digests stay bit-identical through every stage.
- **DC-2 Parallel tree** — `gui/v2/` beside the legacy tree; legacy + `hv_strip_old.bat` retire
  only at cutover; two pytest lanes (`tests/` PyQt5 · `tests_v2/` PySide6) until then.
- **DC-3 Copy-not-import** — house widgets copied from `invert_hvsr/gui` (the newest exemplar)
  / the gui-house-style plugin templates; theme_core imported from HV_Pro (family-live mode).
- **DC-4 v2 never imports legacy gui modules** (their `matplotlib.use("Qt5Agg")` poisons a
  PySide6 process); v2 plots via pyqtgraph + `backend_qtagg` (Figure Studio only).
