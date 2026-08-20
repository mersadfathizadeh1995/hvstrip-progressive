# Tasks 002 — GUI Round 2 · Track 2: legacy parity + hardening

Layer order inside every stage: `api/config` → `api` facade → `gui/v2 state` → `gui/v2` widgets/
tools → tests. Paths are relative to the distribution root (`HV_Strip_Progressive/…` = source,
`tests_v2/…` = the PySide6 lane). `[P]` = parallelizable (different files, no dependency).
Parity note: CLI/MCP are deferred (plan Complexity row) — "headless" for each capability means
its facade op is script-callable and tested Qt-free.

## T2-S1 — Facade: accessors + peaks + rehydrate + dispatch (Qt-free)
### ✅ STAGE DONE 2026-08-15 — tests_v2 89 green (+24) · legacy 150/24 · goldens bit-identical

- [x] **T01** — Numerical guard FIRST: produce the SC-1 reference fixture. Run the legacy app's
  picking walkthrough on the sh_wave golden run (script `tests/golden/make_peaks_fixture.py`
  driving the legacy write-back, or a hand-verified capture) → commit
  `tests_v2/fixtures/legacy_picked_run/` (step summaries + `vs_results.json` + peak_info shape).
  The S4 walkthrough pins against it. File(s): `tests_v2/fixtures/legacy_picked_run/*`.
- [x] **T02 [P]** — `api/config.py`: typed auto-peak strategy config (range bands list w/ armed
  flags · preset name · advanced params), marker/label style defaults, research study section.
  Test: `tests_v2/test_api/test_config_roundtrip_track2.py` (funnel round-trip incl. new fields).
- [x] **T03** *(landed as extensions of the existing `api/peak_ops.py`/`forward_engine.py` + the facade — `set_exact_peaks` verbatim-store, step-picks store)* — `api/analysis.py` (+ new `api/peaks_ops.py`): public accessors
  `forward_results()/strip_results()/batch_result()/get_config()`; peak ops `detect_peaks`,
  `set_manual_peaks` (f0 + **secondaries**), `get_peaks`, `set_step_vs_context`; label-position
  state carried with peaks. Envelope-returning, Qt-free.
  Test: `tests_v2/test_api/test_peaks_ops.py` (round-trip incl. secondaries + label positions).
- [x] **T04** *(module named `api/persist_ops.py`; sidecar `picked_peaks.json`; summary/vs writers byte-parity-pinned vs the T01 fixture)* — `api/persist_ops.py` (or extend `session_io`): `persist_peaks` → step
  `*summary*.csv` cells (create-if-missing) + the **peaks sidecar** (f0/secondaries/label
  positions/Vs context); report regeneration entry + dual-resonance `peak_overrides`
  pass-through. Test: `tests_v2/test_api/test_peaks_persist.py` on a tmp results tree.
- [x] **T05** — `load_results_folder(path)` rehydrate: v2 trees AND legacy trees
  (`vs_results.json`, `peak_info.txt`, no-secondaries sidecars) → curves + auto + picked peaks,
  no recompute. Test: same file as T04 + a legacy-shaped fixture dir.
- [x] **T06** *(cancel = between profiles; a single workflow is atomic — frozen core has no hook)* — Checked-set dispatch ops: `compute_forward_for(names, settings_by_name)` /
  `run_stripping_for(names, settings_by_name, output_dir)` + cooperative `cancel_token` checked
  between items/steps; per-item progress frames. Frozen compute untouched (loops live in api).
  Test: `tests_v2/test_api/test_dispatch.py` (names honored · per-profile settings applied ·
  cancel between items · envelope partials).
- [x] **T07** — Gate: layering guard + full `tests_v2/test_api` green; goldens bit-identical
  (`python -m pytest tests -q` stays 150/24).

## T2-S2 — AppState rewire
### ✅ STAGE DONE 2026-08-15 — tests_v2 94 green; reach-in grep = 0; SC-3 routing pinned

- [x] **T08** — `gui/v2/state/app_state.py`: wrap every S1 op; retire the 12 `# noqa: SLF001`
  private reach-ins (config get/set through the facade funnel; results through accessors;
  session payload ops moved into the facade). Test: grep-based pin in
  `tests_v2/test_gui/test_appstate_track2.py` (`_analysis._` count == sanctioned set).
- [x] **T09** — Section-scoped refresh: panels register the config sections they own;
  `config_changed(section)` routes only to owners; `PhasePanel` gains `sections: tuple`;
  focus/edit guards for spins/edits (no mid-type writeback). Files: `gui/v2/stages/base.py`,
  panels. Test: SC-3 refresh-count pin (one keystroke ⇒ 1 owner refresh, 0 others).
- [x] **T10 [P]** — Busy scope + preload UX: `AppState.preload()` wrapped with busy
  cursor/status signal (`op_started("preparing")`-style) — doctrine unchanged (main-thread,
  serialized). Files: `state/app_state.py`, `main_window.py`. Test: signal emitted around first
  run (stub preload).
- [x] **T11 [P]** *(geometry/docks/splitter/rails via QSettings w/ offscreen guard; batch output → `batch.output.output_dir`; save-act bool-arg fixed; engine badges live [pulls the O20 slice of T25 forward]; research-widget binding stays T26)* — Persistence spine: QSettings for geometry/splitters/rail state
  (`main_window.py` save/restore); batch output + research study section through the config
  funnel; fix `save_settings` bool-arg binding. Test:
  `tests_v2/test_gui/test_persistence.py` (round-trip on a tmp settings org).
- [x] **T12** — Gate: scaffold + new state tests green; reach-in pin green.

## T2-S3 — The interactive figure + Vs mini-panel (widget-level) `[P with S5]`
### ✅ STAGE DONE 2026-08-15 — 11 widget tests green; light+dark renders verified
### (tests_v2 total 105; T16's layer-tree/properties WIRING moved into S4 — the widget
### exposes `set_markers_visible` / `set_marker_style`; facade gained `vs_context_for_layers`)

- [x] **T13** — Port the figure: `gui/v2/canvas/hv_interactive_mpl.py` — `MplFigureWidget`-based
  HV figure with markers + labels + nav toolbar; picking mechanics from the quarry
  (`gui/views/strip_wizard_view.py` press/motion/release, band `axvspan` preview, 2 %-span
  click-vs-drag threshold, band-argmax vs exact-click+interp) — copy + strip Qt5Agg, no legacy
  imports. Uniform verbs (FR-2): right-click delete-nearest, Undo-last-secondary, Clear-scope;
  toolbar-mode guard. Test: `tests_v2/test_gui/test_interactive_figure.py` (synthetic events →
  pick semantics, guard, verbs).
- [x] **T14** — Labels: draggable + place-on-release persisted positions + leader arrows
  (quarry: `profile_wizard_view` `_ann_positions` idiom); positions travel through
  `set_manual_peaks`. Same test file.
- [x] **T15 [P]** — Vs mini-panel: `gui/v2/canvas/vs_context_mpl.py` — Vs30/VsAvg toggles,
  bedrock-interface combo + click-on-plot selection, live readout via api `vs_average` ops;
  values pushed via `set_step_vs_context`. Test: `tests_v2/test_gui/test_vs_context.py`.
- [x] **T16** *(widget hooks done + tested; the LayerModel/PropertiesPanel wiring lands with the S4 mounting)* — Peaks as layers: markers node toggles real artists; `PropertiesPanel` style
  section (marker shape/size/annotation font) drives the figure (R5). Files:
  `state/layer_model.py`, `widgets/house/properties_panel.py`, the figure. Test: toggle +
  restyle round-trip.
- [x] **T17** — Gate: widget tests green; offscreen light+dark renders of the figure reviewed.

## T2-S4 — Forward + Strip wiring + write-back + re-open

- [ ] **T18** — Mount the figure in Strip (per-step, ✓ list, natural step sort) and Forward
  (per-profile via ProfilesPanel focus); auto-peak config surface (FR-8: 3 strategies + accept-
  defaults idiom) as a dialog/card bound to the T02 config. Files: `tools/strip/panel.py`,
  `tools/forward/panel.py`, `dialogs/auto_peak_dialog.py`.
- [ ] **T19** — The write-back chain (FR-4): Finish/apply → `persist_peaks` + report regeneration
  (when enabled) + dual-resonance overrides + tables/dock refresh; **secondaries persisted**
  (DC-8 gap fixed). Test: SC-2 pin on the sh_wave run tree.
- [ ] **T20** — Re-open & re-pick (FR-5): results-folder open action → T05 rehydrate → figure
  shows prior picks; O1 fixed (Forward Vs pane fed). Test: window-module extension.
- [ ] **T21** — Gate: SC-1 walkthrough vs the T01 legacy fixture (persisted values identical, +
  secondaries present); goldens bit-identical.

## T2-S5 — Checked-set dispatch (R4) `[P with S3]`

- [ ] **T22** — Run strips: one Run strip per tool acting on `checked_profiles()` +
  `profile_settings`; Forward Single|Multiple and Strip Single|Batch collapse (config sub-stages
  remain, multiplicity = the check set); zero-checked ⇒ disabled + reason. Files:
  `tools/forward/panel.py`, `tools/strip/panel.py`, `state/app_state.py` (submit via T06 ops).
- [ ] **T23** — Progress + cancel UX: per-item rows in the Progress table; determinate bar where
  totals known (O14); Cancel button for main-queue runs (between items/steps), truthful
  enabled-state (O15). Files: `main_window.py`, `widgets/house/run_table.py`.
- [ ] **T24** — Gate: SC-4 test (2-of-10 · cancel mid-run · badges truthful) in the window
  module.

## T2-S6 — Hardening sweep

- [ ] **T25 [P]** — Truthful chrome: engine badges re-render on engine config change (O20);
  failure envelopes → tool status ERROR + Problems + FAILED badge (FR-13); forward-single logs
  start/finish (O13). Test: pins in `tests_v2/test_gui/test_chrome_truth.py`.
- [ ] **T26 [P]** — Research (FR-15): phase gating on prerequisites (O27); matplotlib comparison
  gallery styling; `ResearchCanvas.set_palette` (O8); study settings persistence (T11 fields).
  Test: research window-module extensions.
- [ ] **T27 [P]** — Small repairs: `tables.py` `smart_columns` adopted by the 5 summary tables
  (DC-6); empty legends fixed (named items, O25); `str(dual)` → formatted readout (O26);
  `PhasePanel.session` landmine removed (O2); `_on_browse`/list-panel duplication collapsed into
  shared helpers (audit #3); Figure-Studio toolbar + theming (O9) *minimal* (full studio =
  Track 3).
- [ ] **T28** — Retire replaced pyqtgraph views + their dead APIs (DC-3): `hv_curve_canvas` pick
  scaffolding + unused methods, `vs_profile_canvas` unused methods — delete what the mpl figures
  replace, keep what Overlay/Waterfall still use (until Track 3 decides their fate). Test:
  layering + import hygiene stay green; no dead-API references.
- [ ] **T29** — Gate: full tests_v2 green; dark-mode offscreen render of all four tools.

## T2-S7 — Verify + docs + vault

- [ ] **T30** — Evidence bundle: full suites (tests_v2 · legacy `150/24` · goldens; `-m hvf`
  where the exe exists); light+dark renders; a 10-step `quickstart.md` manual smoke for the user
  (`hv_strip.bat`: load → run checked → pick → write-back → re-open). File:
  `specs/002-gui-round2-track2/quickstart.md`.
- [ ] **T31** — Docs-sync (same commit set): `CLAUDE.md` tree; `.context/ARCHITECTURE_ASSESSMENT.md`
  phase log (Track 2 entry); `.context/STRIP_MODEL.md` (peaks sidecar + summary-CSV write-back
  contract); `HV_Pro_docs/hvstrip_gui/DECISIONS.md` addendum for any moved choice (e.g. the
  focus-driven per-profile picking synthesis from the spec's Assumptions).
- [ ] **T32** — **Vault-sync (write)**: session note + package hub-note refresh covering
  P0→Track 2 + the 2026-08-15 review fixes; ADR for the peaks sidecar/write-back contract.
  Clears the write-protocol debt flagged in the plan.

## Coverage check

FR-1→T13 · FR-2→T13 · FR-3→T14 · FR-4→T19 · FR-5→T20 · FR-6→T16 · FR-7→T22/T06 · FR-8→T18/T02 ·
FR-9→T15 · FR-10→T20 · FR-11→T09 · FR-12→T10/T23/T25 · FR-13→T25 · FR-14→T11/T26 · FR-15→T26 ·
FR-16→T13 (toolbar save) + T19 (data exports).
DC-1→T07/T21/T30 · DC-2→every test task · DC-3→T13/T28 · DC-4→T03/T08 · DC-5→T08 · DC-6→T27 ·
DC-7→user commits · DC-8→T19/T22 (fixed behaviors) · DC-9→T10 · DC-10→T17/T29 renders.

*Next: `/spec-vault-sync read` is already satisfied (stale cluster, plan records it);
`/spec-handoff` can slice S1→S7 into staged prompts when implementation is delegated.*
