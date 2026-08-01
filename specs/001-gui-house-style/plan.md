# Plan 001 — how (constitution-checked)

*Stages = the approved macro plan's P4–P6 (P0–P3 shipped; see
`.context/ARCHITECTURE_ASSESSMENT.md` §2 for the log). Handoffs in `handoffs/`.*

## Constitution check

- **Core-API-Consumers** ✓ — v2 panels → AppState → `HVStripAnalysis` only; the layering guard
  (already live report-only) flips enforcing at cutover. Frozen compute untouched (goldens).
- **API parity** ✓ — every GUI capability is an api method (P1/P2 closed the gaps); CLI/MCP
  become cheap later (out of scope this round, recorded).
- **Config-driven** ✓ — `HVStripConfig` only; the funnel migrates legacy payloads on read.
- **Reproducibility** ✓ — DC-1 goldens; the 512-grid defect class is what they catch.
- **Responsive GUI** ✓ — OpQueue ×2 (main + research) + ProgressBridge; no UI-thread compute.
- **Spec before code / verify before done** ✓ — this spec; per-stage verification below.

## Stage plan (each independently verifiable)

### S1 — Scaffold + AppState (macro P4)
`gui/v2/` skeleton: `theme.py` (AMBER palettes + full role-QSS, check assets),
`widgets/house/` (collapsible_group, sub_breadcrumb, tables, _qss, the tool-switcher variant
of stage_ribbon, layer_tree, color_swatch), `workbench/side_rail.py`, `stages/base.py`,
`workers/op_worker.py` — all copy-not-import from invert_hvsr. `state/`: `tool.py` (StripTool
enum + ToolStatus), `app_state.py` (ONE `HVStripAnalysis`; signals `profiles_changed`/
`forward_changed`/`strip_changed`/`research_changed`/`config_changed(str)`/op signals/`error`/
`dirty_changed`; `status_for(tool)` incl. `check_engines`; TWO OpQueues; persistence via
`api.load_config_payload` — settings.yaml v2 + project `hvstrip_state_io` v2), `layer_model.py`
(steps/profiles → keyed layer nodes).
**Verify:** import-hygiene test (no PyQt5/no legacy gui/no Qt5Agg on `import ...gui.v2`);
headless AppState tests (op → coalesced frames → envelope parity with a direct api call;
concurrent research+forward; legacy-payload load); theme test (amber, 3 modes, distinct).

### S2 — Shell (macro P5.1)
`gui/v2/main_window.py` + `gui/v2/app.py`: tool switcher, per-tool stacked left panels +
canvas stacks (placeholders), right Layers|Properties rail (bedrock nesting), Log|Problems|
Progress dock (+ the batch table widget), File/View/Help menus, engine badges in the status
bar, per-window `_retheme`.
**Verify:** offscreen renders light+dark; tool-switch state test; `hv_strip.bat` launches.

### S3 — Forward tool (macro P5.2a)
Panels (Single: profile file/table + engine/frequency/peaks cards + Run; Multiple: profiles
list + Run-all) + pyqtgraph views (HV∥Vs, overlay, all-profiles, summary) + forward-multi
profiles as rail layers.
**Verify:** GUI-path envelope == direct api envelope (sh_wave) == goldens; interaction tests;
renders.

### S4 — Strip tool (macro P5.2b)
The 3-sub-stage wizard + Batch; steps-as-layers (the invert scenario-tree pattern:
`step::N::hv|vs` keyed items, per-step Properties + info); waterfall/per-step/summary/
Figure Studio (matplotlib, house-styled) views; the Progress-dock batch table live.
**Verify:** full sh_wave strip through the GUI == goldens; step toggles/styles drive the
canvas; batch table updates from frames; renders.

### S5 — Research tool (macro P5.2c)
Phase sub-stages + per-phase run + full-study + cancel, on the research queue; figure/table/
report views.
**Verify:** a tiny stub study runs concurrently with a Forward op; cancel between phases lands.

### S6 — Hub + guards + retirement (macro P6)
Package-root `app.py`; hvsr_pro: `StripProvider` + `StripView` + rail-description fix +
REMOVE `submodule_manager.open_hvstrip_progressive`; retire the legacy `gui/` tree +
`legacy_main.py` + `hv_strip_old.bat`; promote `gui/v2/`→`gui/`, `tests_v2/`→`tests/`
(PySide6 conftest; port the meaningful legacy tests); guard `ENFORCING=True` + theme test;
pyproject PyQt5→PySide6, version 3.0.0; docs-sync.
**Verify:** ONE PySide6 `pytest tests` green; goldens bit-identical; Qt-free import; Hub
round-trip on a real project (`strip_dir` save→reopen); bat cold-launch; renders.

## Complexity tracking

- The tool-switcher = a re-labelled StageRibbon (per-tool `status`, no `[step]` chain) — a
  variant, not a fork.
- Research cancel = cooperative (between phases/iterations) — the api runner already loops
  per phase; no subprocess kill needed.
- `hvstrip_state_io` payload keeps its `{config, results, extra}` envelope — only `config`
  becomes v2 inside (best-effort read of old payloads, per the no-backcompat stance: read if
  trivial, else start fresh).
