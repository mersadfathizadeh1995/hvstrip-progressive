# Architecture assessment — Core-API-Consumers readiness + the uplift plan of record

*Assessed 2026-07-09 against the real code (v2.2.0). This document is the plan of record for the
house-style rebuild; keep it current (docs-sync).*

## 0. VERDICT

**Core-API-Consumers = PARTIAL, with an unusual failure mode: the api layer EXISTS and is
well-shaped, but it is ORPHANED.** Unlike HV Invert (which had no api at all), HV Strip built
`HVStripAnalysis` + a complete config-dataclass tree and then never wired the GUI to it. The GUI
grew its own parallel universe: direct `core/` imports (~20 sites) and a hand-rolled nested dict
config. The migration is therefore a REWIRING + GUI rebuild, not a facade build.

| Aspect | State | Evidence |
|---|---|---|
| `core/` purity | ✅ genuinely pure (no Qt) | `core/*` import only numpy/scipy/pandas/matplotlib(lazy) |
| `api/` facade | ✅ exists, dict envelopes | `api/analysis.py:89` `HVStripAnalysis`; op modules |
| Config dataclasses | ✅ complete tree | `api/config.py` → root `HVStripConfig` (`:455`) |
| GUI → api | ❌ ZERO imports | grep `HVStripAnalysis|from ..api` in `gui/` = 0 hits |
| GUI → core reach-ins | ❌ ~20 sites | e.g. `gui/workers/workflow_worker.py:23` → `core.batch_workflow`; `gui/views/strip_wizard_view.py:822` → `core.peak_detection`; ~15 widgets → `core.soil_profile` |
| Config duplication | ❌ two universes | `gui/strip_window.py:64-177` `_get_default_config()` dict vs the dataclasses (fixture: `tests/golden/legacy_gui_config.json`) |
| Qt binding | ❌ PyQt5 direct (58 files) | house rule = PySide6-only |
| Theme | ❌ 45 files of scattered `setStyleSheet`, own blue `#2E86AB` (collides with gui_v2) | `gui/widgets/style_constants.py` |
| api consumers | ⚠️ only `research/` (2 fns) | `research/strip_comparison.py:97,140`; the "MCP consumer" docstrings are FALSE |
| api tests | ❌ none | `tests/` covers core + legacy GUI only |
| Entry points | ❌ none until 2026-07-09 | now: `gui/legacy_main.py` (temp) + the two bats |
| Hub | ❌ PlaceholderView | rail row `"hvstrip"` exists (`module_nav.py:90`, wrong description); no provider |

## 1. What's good (keep)

- `core/` layering is real: `soil_profile` (the shared model), `stripper`, `hv_forward`,
  `batch_workflow`, the **pluggable engine registry** (`core/engines/__init__.py` — diffuse_field/
  sh_wave/ellipticity), `peak_detection`, `hv_postprocess`, `dual_resonance/`, `report_generator`.
- `HVStripAnalysis` is already the invert-style session: stateful, config-carrying, and every
  method returns a JSON-serialisable dict envelope. It needs EXTENSION (progress streaming,
  engine probing, config funnel), not replacement.
- `Project.strip_dir()` + `project_manager.module_state.hvstrip_state_io` already exist on the
  HV Pro side — the project persistence seam is in place.

## 2. The gaps (evidence above) → the migration

Plan of record = the approved 7-phase plan (`~/.claude/plans/snuggly-crafting-spring.md`, mirrors
invert's handoffs):

- **P0 (DONE 2026-07-09):** mapping docs (this file + STRIP_MODEL + GLOSSARY + rules) · the
  GOLDEN COMPUTE LOCK (`tests/golden/` — sh_wave + diffuse_field digests, bit-stable ×2) · the
  legacy suite REPAIRED (was 100% broken by the package rename: `hvstrip_progressive` →
  `HV_Strip_Progressive` in tests; the moved `examples/different_files/example_model.txt`; a stale
  3-tab assertion; a gpell env-skip; + a REAL app bug fixed: `gui/views/all_profiles_view_modules/
  ui_builder.py` still used flat pre-rename imports so the all_profiles canvas silently failed) →
  **150 passed / 24 skipped** baseline · dead `gui/pages/` deleted (4 files, zero importers) ·
  `gui/legacy_main.py` + `hv_strip_old.bat` (side-by-side comparison) + `hv_strip.bat` (the new
  workbench target) · the legacy GUI config dict snapshotted (`tests/golden/legacy_gui_config.json`).
- **P1 (DONE 2026-07-09) — API foundation:** `progress_cb=None` kwargs on the long ops
  (`run_stripping` / `run_batch_stripping` / `compute_forward_batch` + the facade methods) —
  streaming via the `api/_progress.py` stdout TEE over `batch_workflow`'s existing narration
  (core frozen; frames = `phase`/`log`/`profile`/`op`, step-granular); `check_engines()`
  existence-only probe; `run_research_study` wrapping `ComparisonStudyRunner` (its own
  `set_progress_callback` → `study`/`study_phase` frames); config funnel —
  `HVStripConfig.to_dict()` now carries `config_version: 2`, `from_dict` auto-detects legacy
  shapes → `from_legacy_gui_dict()` (renames: `dual_resonance.enable→enabled`,
  `hv_postprocess.output→output_files`; per-engine `engine_settings` merged onto the one
  `EngineConfig`; active-engine freq → `FrequencyConfig`; unmapped keys logged), public
  `api.load_config_payload` = THE load funnel. **Two real api defects fixed:** `StepResult.n_layers`
  was ALWAYS 0 (`_parse_core_results` read a key core never provides → now derived from core's
  step name, header fallback) and `FrequencyConfig.n_samples` defaulted to 500 vs core's/legacy's
  **512** → the api path silently shifted the sh_wave grid + detected peak (caught by golden
  parity; aligned to 512). `tests_v2/` born (own PySide6-lane conftest, matplotlib Agg): 12 api
  tests — golden parity, frame ordering, `progress_cb=None` byte-identity, migration, funnel,
  probe, research pins.
- **P2 (DONE 2026-07-09) — Core-gap audit + guard:** the FULL gui→core audit (post `gui/pages`
  deletion): **frozen-compute reach-ins = 4 sites, all in `gui/workers/`**
  (`batch_worker`+`workflow_worker`→`batch_workflow`; `forward_worker`+`multi_forward_worker`→
  `hv_forward`) — every one has an api counterpart already (`run_stripping`/`run_batch_stripping`/
  `compute_forward(_batch)`), and the workers are REPLACED by the OpQueue+api in v2 → **zero new
  api ops needed**. Shared model importable: `soil_profile` (×10), `vs_average` (×10),
  `velocity_utils`; analysis importable: `report_generator` (×3), `peak_detection` (×2),
  `dual_resonance` (0 sites left). `tests_v2/test_layering.py` SHIPPED (adapted from invert's;
  `ENFORCING=False`): rules (a) core/api Qt-free, (b) api↛gui, (c) Qt-free package import are
  **enforced now** — (c) caught + fixed a real defect: the package `__init__` eagerly imported
  the PyQt5 window (now a lazy `__getattr__`; the HV Pro shim verified intact); rules (d)
  PySide6-only and (e) consumer↛frozen-compute are report-only for the legacy `gui/` and
  ALREADY-ENFORCED for `gui/v2/`+everything else. Suites: legacy 150/24skip · tests_v2 **17**.
- **P3 (DONE 2026-07-09) — GUI co-design GATE (with the user):** archetype **C tool-collection**
  (Forward | Strip | Research switcher; the strip wizard = a nested linear mini-flow), amber
  accent; sketch + question rounds → `HV_Pro_docs/hvstrip_gui/{DISCOVERY,ARCHETYPE,SKETCH.txt,
  DECISIONS}.md`; then spec-driven `specs/001-gui-house-style/`. User approval: "Approved — build it".
- **P4 (DONE 2026-07-09) — Scaffold + AppState:** `gui/v2/` parallel tree (house widgets
  copy-not-import from invert_hvsr / the gui-house-style plugin templates); AppState over ONE
  `HVStripAnalysis`; **two OpQueues** (main + research) + `_ensure_research_imports()` (concurrent
  first-imports of sklearn/matplotlib in two QThreads hard-abort Windows — pre-import on the GUI
  thread at submit); `tests_v2/` PySide6-offscreen (separate pytest process from the PyQt5
  `tests/`; `addopts = "-p no:pytest-qt"` — its hooks abort PySide6 suites).
- **P5 (DONE 2026-07-10) — Shell + all 3 tools:**
  - **Shell:** `StripMainWindow` (ToolSwitcher · 3-pane splitter w/ bedrock-nested right rail ·
    Log|Problems|Progress dock w/ `RunTable` + cancel-research · engine badges status bar) +
    `gui/v2/app.py` (the `hv_strip.bat` target, theme_authority-primed).
  - **Forward (S3):** Single|Multiple sub-stages; HV∥Vs + Overlay (`prof::` keyed items) +
    Summary; single f₀ 4.245 == golden.
  - **Strip (S4):** the 3-sub-stage GATED wizard (Model → Configure & Run → Review; Batch as a
    peer sub-tab) + StripCanvas (Waterfall Overlay `step::N` keyed items · Step HV∥Vs via
    `profile_layers_from_file` · Summary · **Figure Studio** = the ONE matplotlib view w/ the
    report HV-overlay + PNG/PDF/SVG export). Full sh_wave strip through the GUI == goldens
    (f₀/A₀ per step to 1e-6); steps-as-layers toggles/styles drive the canvas.
  - **Research (S5):** 5 phase sub-stages (Profiles|Comparison|Metrics|Field|Report), each
    runnable alone; Report page = Run-full-study + cooperative Cancel; ResearchCanvas =
    Comparison-Figures `fig::` gallery (rail-driven) · Metrics Tables · Report manifest.
    **api fixes shipped with it:** (a) the facade now holds ONE persistent
    `ComparisonStudyRunner` (`reset=` starts fresh) — per-call runners lost all state between
    phases, so phase sequencing/cancel could never work; (b) inner phase `{"error": ...}` dicts
    now surface as `success: False` (were masked); (c) `study_config["profiles_dir"]` LOADS an
    existing suite folder — the clean degrade path when SoilGen is absent (SoilGen-path +
    suite-folder fields on the Profiles page); (d) **research/metrics.py recursion bug fixed** —
    `compute_metrics` recursed unconditionally per category (a single-category subset re-entered
    itself forever): the metrics phase could NEVER complete on a non-empty dataset; now recurses
    only when >1 category (pinned in `tests_v2/test_api/test_research_study.py`).
  - **One-process test-stability rules (Windows/PySide6):** `api.preload_heavy_modules()` (the
    heavy compute/plot stack imported on the MAIN thread; AppState calls it before any op — a
    worker FIRST-importing native extensions can hard-abort); tests that create an AppState
    `shutdown()` it; op-running tests share ONE module-scoped AppState; ONE window test module
    (`tests_v2/test_gui/test_tools_window.py`) hosts ALL window-based tests. The product path
    itself is verified stable (3 strips + research + forward in one window). Suites: legacy
    150/24 skip · tests_v2 **38** (one process, ~43 s).
- **P6 — Hub + guards + retirement:** package-root `app.py` `main_window_for_project` →
  `strip_dir`; hv_hub `StripProvider` + real `StripView` (fix the rail description — it is NOT
  "time-frequency"); REMOVE the legacy-app launch path (no shim — user decision); retire the
  legacy `gui/` (~58 files) + `legacy_main.py` + `hv_strip_old.bat`; promote `gui/v2/`→`gui/`,
  `tests_v2/`→`tests/`; guards flip enforcing; PyQt5→PySide6 in pyproject; version 3.0.0
  (fixes the 2.1.0/2.2.0 drift).

### Round 2 — the user's design revision (2026-07-11; plan = the Round-2 Track-1 plan)

The user compared the new workbench against the legacy app and revised the design. **P6
(legacy retirement) is POSTPONED — the legacy `gui/` is the PORT QUARRY** until Tracks 2–3 land.
Locked decisions: figures go COMPLETELY matplotlib by PORTING + enhancing the legacy mpl
components (interactive peak picking etc.), NOT re-skinning the pyqtgraph views · a first-class
**Data Input stage** · a global **ProfilesPanel** (HV Pro gui_v2 pattern) · per-stage top Run
strips over the CHECKED profiles (dissolves Forward Single|Multiple + Strip Single|Batch —
Track 2) · the right rail becomes the figure-properties context (Track 2).

- **Track 1 (DONE 2026-07-11)** — suites: tests_v2 **50** (one process) · legacy 150/24 ·
  goldens bit-identical:
  - **api input unification**: `load_profile_dinver(vs,vp,rho)` (the 3-file mode the legacy GUI
    did by calling core directly) · `load_profile_dinver_prefix` · `load_profiles_from_directory`
    (per-file errors collected) · `suggest_layer_fill(vs, nu=None)` — the ONE derivation surface
    (the legacy LayerTableWidget carried DIVERGENT local ν/density tables) · facade
    `update_profile(name, layers)` (replace + INVALIDATE results) · `fmt="simple"` (from_txt_file).
  - **AppState profile-centric model**: `ProcessingStatus` (state/profile_status.py, gui_v2
    colors) · checked set (the RUN SET) + focus + per-tool per-profile settings clones +
    per-tool statuses auto-wired from the op lifecycle (QUEUED→RUNNING→DONE/FAILED; batch
    `profile` frames narrow RUNNING; cancel → NOT_STARTED; update/remove resets).
  - **mpl foundation**: `canvas/mpl_widget.py` (`MplFigureWidget` = FigureCanvasQTAgg +
    NavigationToolbar2QT + `apply_theme`) · `canvas/vs_profile_mpl.py` (the legacy
    ProfilePreviewWidget PORTED — Qt5Agg poison stripped — + Vs30 line + halfspace shading).
  - **ProfilesPanel** (`widgets/house/profiles_panel.py`, adapted from gui_v2's process
    stations_panel + `_BadgeDelegate`): leftmost collapsible rail in EVERY stage; checkable rows;
    **stage-aware badge columns** — tool stages show `[Set][Run]` for that tool, the Data stage
    shows the `[Fwd][Str][Res]` overview; tri-state select-all; Assign-settings-to-checked
    (main_window snapshots the active tool's config sections per profile).
  - **Data Input stage** (`tools/data/panel.py`, StripTool.DATA FIRST): the unified loader —
    HVf/CSV/Excel/Simple-TXT browse pages + the **Dinver 3-row page with `_auto_link` sibling
    discovery** + directory load — all through AppState→api; DataCanvas = mpl **Vs Profile**
    (follows ProfilesPanel focus) + the ported **LayerTable** (9 cols, Vp Mode Auto/Nu/Manual
    live re-derive, Auto-fill — all values from `suggest_layer_fill`; Apply →
    `update_profile` + badge reset; New-profile-from-table = the manual-editor path).
- **Review fixes (DONE 2026-08-15)** — the HV_Pro code-review findings repaired (suites:
  tests_v2 **65** · legacy 150/24 · goldens bit-identical):
  - `OpWorker.run` now catches every raise + non-dict return and synthesizes a
    `{success: False, error, traceback}` envelope — a raising op (e.g. `_resolve_profile`
    KeyError) used to wedge the OpQueue busy until app restart. The invert-inherited
    "nothing is caught here" contract does NOT hold for this api.
  - Stale cached profile names: `ForwardSinglePanel`/`StripModelPanel` drop a removed
    profile's name on refresh (Run gating was staying enabled → the wedge trigger);
    `AppState._resolve_op_profile` resolves `None` → first profile BEFORE submit.
  - `LayerTable` cell-widget handlers resolve their row at FIRE time (`_widget_row`) —
    creation-time captures went stale after `removeRow` and silently edited the wrong
    layer; `_swap_rows` now moves the Vp-mode with its layer; `_on_hs_changed`
    save/restores `_block` instead of clobbering it.
  - `HVStripConfig.from_dict` routing: legacy-only keys (`engine_settings`, `hv_forward`, …)
    → the legacy migrator; otherwise top-level-⊆-v2-fields (incl. PARTIAL dicts and the
    overlap keys `engine`/`dual_resonance`/`peak_detection`) applies as v2; `_apply_dict`
    collects unmapped keys for logging (the "never silently dropped" contract, both routes).
  - **Settings files split**: the legacy window now owns `~/.hvstrip/settings_legacy.yaml`
    (one-time seed from a legacy-shaped `settings.yaml`; v2 payloads never merged) —
    `settings.yaml` belongs to gui/v2 alone. The shared-file deep-merge used to clobber
    both sides' edits.
  - The stdout tee (`api/_progress.py`) parses only its installing thread's lines and is
    single-flight (a second concurrent install degrades to no-parse) — the "ONE OpQueue"
    safety assumption died when the research queue landed.
  - `research/metrics.compute_metrics` caps the per-category recursion with a flag, not the
    category COUNT — a single-category study gets its one `per_category` entry again.
  - `ComparisonPage` heals the "(unavailable)" engine label/check when the engine becomes
    available (only OUR forced un-checks are restored, not the user's).
  - The mpl Vs preview + both pyqtgraph canvases share ONE staircase builder
    (`canvas/constants.layers_to_staircase`, proportional halfspace `max(0.25·depth, 1 m)`
    — the legacy-preview rule); the duplicated copies had drifted (25 % vs fixed 50 m).
  - `os.startfile` → `QDesktopServices.openUrl` (platform-safe).
  - Regression pins: `tests_v2/test_api/test_review_fixes_api.py` +
    `tests_v2/test_gui/test_review_fixes_gui.py` (+ the single-category contract updated in
    `test_research_study.py`).
- **Track 2 (NEXT — spec `specs/002-gui-round2-track2/`)**: per-stage top Run strips +
  checked-set dispatch (the Single/Multiple/Batch merge) · port the legacy interactive mpl HV
  figure (f0/secondary PICKING) for Forward/Strip · right rail → figure properties · Research
  re-skin — plus the 2026-08-15 v2 audit's hardening backlog (refresh storm, first-click busy
  UX, AppState privates → api accessors, dead `tables.py` adoption, persistence gaps,
  cancel/progress, O1 blank Vs pane, inert peak-markers node).
- **Track 3**: Figure Studio / report overhaul. Then the deferred P6 cutover + Hub provider.

## 3. Performance invariants — DO NOT regress (user constraint)

- **HVf.exe subprocess** (`core/engines/diffuse_wave_field/`) — the hot path; invocation
  byte-identical.
- **Adaptive frequency rescanning** (`batch_workflow._compute_hv_curve_adaptive`) — re-runs the
  engine when a peak sits near a boundary; call pattern unchanged; progress frames are
  step-granular, never per-rescan-call.
- The engine registry contract (`BaseForwardEngine.compute/compute_from_profile`) unchanged.
- The proof: `tests/golden/` digests stay bit-identical through every phase.

## 4. Known quirks

- The hyphenated folder → `PYTHONPATH` must carry the distribution root; HV Pro's shim is
  `hvsr_pro/packages/hvstrip_progressive_pkg.py`.
- `local_config.py` (gpell/git-bash paths) is machine-local; the ellipticity engine + its test
  degrade to skip when absent.
- The legacy `tests/` conftest force-imports PyQt5 + `Qt5Agg` — never import PySide6 there; the
  rebuild's tests live in `tests_v2/` and run as a separate process.
- `matplotlib.use("Qt5Agg")` at import time in `gui/widgets/plot_widget.py` +
  `profile_preview_widget.py` — v2 code must never import legacy gui modules.
