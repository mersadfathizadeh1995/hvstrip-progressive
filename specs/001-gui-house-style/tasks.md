# Tasks 001 — layer-ordered

*S-stages from plan.md; every task carries its verification. [P] = parallelizable.*

## S1 — Scaffold + AppState  ✅ DONE 2026-07-09
- T01 `gui/v2/theme.py` — amber LIGHT/GRAY/DARK on theme_core + full role-QSS (clone invert's
  `gui/theme.py`, swap the violet family → `#C87A20` family; retint `check.svg`) + assets.
- T02 [P] copy house widgets → `gui/v2/widgets/house/` (collapsible_group · sub_breadcrumb ·
  tables · _qss · color_swatch · layer_tree) + `workbench/side_rail.py` + `stages/base.py` +
  `workers/op_worker.py` (verbatim from invert, imports renamed).
- T03 tool switcher widget (`widgets/house/tool_switcher.py` — the stage_ribbon variant:
  per-tool pills, `status` property, no chain).
- T04 `state/tool.py` (StripTool, ToolStatus) + `state/app_state.py` (the spine; TWO queues;
  `status_for`; persistence funnel) + `state/layer_model.py` (steps/profiles → nodes).
- T05 `tests_v2/test_gui/test_imports.py` (hygiene) · `test_theme.py` (amber conformance) ·
  `test_state/` (op→frames→envelope parity; concurrency; legacy payload).

## S2 — Shell  ✅ DONE 2026-07-10
- T06 `gui/v2/main_window.py` (switcher + stacks + rail + dock + menus + badges + `_retheme`)
  + `gui/v2/app.py` (standalone main; theme priming).
- T07 Progress-dock batch table widget (`widgets/house/run_table.py`).
- T08 shell tests + light/dark renders; `hv_strip.bat` smoke.

## S3 — Forward tool  ✅ DONE 2026-07-10
- T09 profile input card (file + editable layer table w/ vp/ρ auto-derive via the api).
- T10 engine/frequency/peaks cards (availability-aware) + Run wiring.
- T11 [P] pyqtgraph views: hv_curve, vs_profile, overlay, all_profiles, summary_table.
- T12 forward-multi profiles as rail layers; tests + parity + renders.

## S4 — Strip tool  ✅ DONE 2026-07-10 (GUI strip == goldens 1e-6)
- T13 the 3-sub-stage wizard container (Model → Configure & Run → Review).
- T14 batch panel + the live Progress table wiring.
- T15 steps-as-layers: keyed canvas items + LayerModel branch + Properties info (f₀/A₀/Vs30).
- T16 waterfall overlay view + per-step HV∥Vs view + summary view.
- T17 Figure Studio view (matplotlib `backend_qtagg`, house-styled).
- T18 tests: GUI==api==golden parity; toggles/styles; batch frames; renders.

## S5 — Research tool  ✅ DONE 2026-07-10 (+ research-api fixes: persistent runner · profiles_dir load · inner-error propagation · metrics recursion bug)
- T19 phase sub-stage container + per-phase config cards + run/full/cancel on the research
  queue; figure/table/report views; concurrency test.

## S6 — Hub + guards + retirement
- T20 package-root `app.py` (`main_window_for_project` → `strip_dir`; Qt-lazy) + boundary test.
- T21 hvsr_pro: `StripProvider` + `StripView` + rail fix + REMOVE the legacy launch path +
  hub tests updated.
- T22 retire legacy `gui/` + `legacy_main.py` + `hv_strip_old.bat`; promote `gui/v2/`→`gui/`;
  `tests_v2/`→`tests/` (port the meaningful legacy tests; PySide6 conftest).
- T23 guards ENFORCING; pyproject PyQt5→PySide6; version 3.0.0; CHANGELOG; docs-sync
  (CLAUDE.md/.context/rules).
- T24 final verification: one green PySide6 suite · goldens bit-identical · Qt-free import ·
  Hub round-trip · bat · light/dark renders of all three tools.

## Round 2 · Track 1 — Data Input stage + ProfilesPanel + mpl foundation  ✅ DONE 2026-07-11
- R2-T1.1 api input unification (dinver 3-file/prefix, directory, suggest_layer_fill(vs,nu),
  update_profile, fmt="simple") + tests_v2/test_api/test_profile_input.py.
- R2-T1.2 AppState profile model (ProcessingStatus · checked/focus · per-tool settings ·
  op-lifecycle status wiring).
- R2-T1.3 canvas/mpl_widget (MplFigureWidget + nav toolbar + apply_theme) + vs_profile_mpl
  (legacy ProfilePreviewWidget port, Qt5Agg poison stripped).
- R2-T1.4 widgets/house/profiles_panel (stage-aware [Set][Run] / [Fwd][Str][Res] badges,
  _BadgeDelegate, select-all, assign-to-checked) mounted LEFTMOST in every stage.
- R2-T1.5 tools/data (StripTool.DATA first): unified loader (6 formats + dinver auto-link +
  directory) + DataCanvas (mpl Vs Profile + ported LayerTable, Apply→update_profile).
- R2-T1.6 wiring + docs + verification (tests_v2 50 · legacy 150/24 · goldens bit-identical ·
  offscreen drive + light/dark renders).

## Round 2 · Track 2 — NEXT (plan to be written)
- Per-stage top Run strips + checked-set dispatch (merges Fwd Single|Multiple + Strip
  Single|Batch); port the legacy interactive mpl HV figure (f0/secondary PICKING) for
  Forward/Strip; right rail → figure-properties context; Research re-skin.
