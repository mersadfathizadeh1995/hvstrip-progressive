# Spec 002 — GUI Round 2 · Track 2: legacy parity + hardening

**Package:** hvstrip-progressive (HV Strip Progressive) · **Status:** DRAFT for plan
**Date:** 2026-08-15 · **Follows:** `specs/001-gui-house-style/` (P0–P5 + Round 2 Track 1 shipped)
**No git branch** — the feature is this folder; the user commits.

**Grounding inputs (read before planning):**
- The 2026-08-15 **legacy-GUI capability inventory** (this conversation): every tab/control of the
  PyQt5 app, §7 = the exact manual peak-picking mechanics + the post-Finish write-back chain, and
  the 25 "capabilities most likely to be missed in a rebuild".
- The 2026-08-15 **v2 audit**: findings #1–#7 (refresh storm · first-click freeze · panel
  duplication · 12 private reach-ins · dead table helper · platform-locked open · misc) and
  O1–O29 (unwired pick-mode, no peak markers anywhere, RUN-SET checkboxes driving nothing,
  write-only assign-settings, persistence/cancel/progress/theming gaps).
- `HV_Pro_docs/hvstrip_gui/DECISIONS.md` **Round 2 (R1–R6)** — the agreed direction this spec
  executes: completely-matplotlib figures ported + enhanced from the legacy quarry (R1); Data
  Input first-class (R2, shipped); Files panel with checked run-set (R3, shipped); per-stage Run
  strips with checked-set dispatch dissolving Single|Multiple and Single|Batch (R4); right rail →
  figure-properties context (R5); legacy `gui/` = port quarry, retirement postponed (R6).
- `.context/ARCHITECTURE_ASSESSMENT.md` §2 (the phase log, incl. the 2026-08-15 review fixes
  already landed) and §3 (frozen-compute invariants).

## Domain & Vault Context

- **Vault:** `GeoVault/Projects/HV_Pro/packages/hvstrip-progressive/` — **stale (2026-05-21,
  pre-rebuild)**. The write protocol is owed for P0–Track 1 and for this feature; a vault-sync
  task is part of this spec's task list.
- **Contracts:** `.context/STRIP_MODEL.md` (Layer/SoilProfile invariants, the HVf text format,
  peel sequences, result envelopes, the output tree — the step `*summary*.csv` + sidecar files
  this feature writes back into). `.context/GLOSSARY.md` for terms.
- **Terms a reader needs:** *f0* (fundamental peak), *secondary peaks*, *strip step* (Step0 …
  StepN, one peeled model each), *peak migration*, *dual resonance* (f0/f1 pairing), *Vs30 /
  VsAvg-to-bedrock*, *bedrock interface*, *run set* (the checked profiles), *port quarry* (the
  legacy widgets that must be ported before retirement).

## What & Why

Track 1 rebuilt the shell: the amber tool-collection workbench (Data Input | Forward | Strip |
Research) over one session, with the Files rail, statuses, and the Data stage. But the rebuild is
**far below the legacy app's capability** — the user's verdict: *"it isn't yet like the previous
gui … we could do the manual peaking and … there were a lot of opportunities there."* Today v2
draws **no peak markers at all**, has **no picking**, ignores the **checked run set** it
advertises, never consumes assigned per-profile settings, and carries the audit's responsiveness
and truthfulness debts.

Track 2 closes that gap **on the modern chassis**: the legacy interactive figures (the app's
soul — pick, correct, annotate, and have the outputs follow) are ported and enhanced, the run
model becomes the checked-set dispatch the panels already promise, and the audit's hardening
backlog is paid down. Figure Studio / report overhaul and the cutover stay Track 3.

## User stories (prioritised)

### US-1 (P1) — Manual peak picking on any HV figure
Given a completed strip run (or forward run), when the user opens the step's HV figure, arms
**Select f0**, and clicks on the curve, then an f0 marker + label appears at the exact clicked
frequency (amplitude interpolated) and the arm auto-releases; when the user instead **drags a
band**, a live band highlight follows the cursor and on release the peak **snaps to the maximum
inside the band**; when the user arms **Select Secondary**, each click/drag adds one secondary
(the arm persists); when the user **right-clicks near a peak**, that peak is deleted; **Undo**
removes the last secondary; **Clear** empties the current step only. Picking is suppressed while
the toolbar's pan/zoom is armed. The step list shows a ✓ once the step has an f0.

### US-2 (P1) — Picked peaks change the outputs, not just the screen
Given manually picked peaks, when the user finishes the picking pass, then the step summaries on
disk, the in-memory results, the summary tables, the regenerated report, and the dual-resonance
figure all carry the picked values — **including secondary peaks** (the legacy app silently
dropped them from persistence; this spec fixes, not copies, that) — and re-opening the results
folder later restores curves *and* picked peaks for further correction **without recomputing**.

### US-3 (P1) — The checked set is the run set
Given 10 loaded profiles with 2 checked, when the user presses a tool's **Run**, then exactly
those 2 run (with their assigned per-profile settings if any), their badges advance
QUEUED→RUNNING→DONE/FAILED, the progress table shows per-item rows, and a long run can be
**cancelled** between items/steps. Zero checked ⇒ Run is disabled with a reason.

### US-4 (P2) — Vs context while picking
Given a step (or profile) HV figure, when the user shows the Vs mini-panel, then Vs30 and
VsAvg-to-bedrock render with the profile; the bedrock interface is selectable from a list **or by
clicking the Vs plot**, and the Vs30/VsAvg readout updates live and is persisted with the step's
results.

### US-5 (P2) — Honest, responsive chrome
Given any config edit, when the user types into a field, then the field is not reformatted
mid-type and unrelated panels do not rebuild; the first Run after launch shows a busy cursor +
status note during the one-time heavy preparation instead of appearing frozen; progress is
determinate wherever totals are known; engine badges reflect path changes immediately; a failed
run marks the tool/profile status FAILED and lists the error in Problems; dark mode themes every
canvas.

### US-6 (P3) — Research parity
Given the Research tool, when phases run, then phase pages gate on their prerequisites, the
comparison figures render in the agreed matplotlib style, and the study configuration persists
across launches.

## Functional requirements

- **FR-1 Interactive HV figure (Forward + Strip).** One matplotlib HV figure component with nav
  toolbar (pan/zoom/home/save), peak markers + frequency labels, and the **click-vs-drag picking
  duality**: short click = exact clicked frequency with interpolated amplitude; drag beyond ~2 %
  of the frequency span = argmax inside the dragged band, with a live translucent band preview
  (f0 red, secondary orange). Arm buttons for f0 (one-shot) and secondary (accumulating),
  mutually exclusive, visibly armed. Toolbar-mode guard on **every** picking surface.
- **FR-2 Peak editing verbs, uniform.** Right-click deletes the nearest peak; Undo pops the last
  secondary; Clear wipes the current step/profile only. (The legacy app's asymmetry — no
  right-click in the strip wizard, no undo in some views — is resolved to the uniform verb set.)
- **FR-3 Annotation labels.** Peak labels are draggable; a drag-to-place (or place-on-release)
  position is remembered per peak and re-applied on redraw with a leader arrow back to the peak.
- **FR-4 The write-back chain.** Picked peaks (f0 **and secondaries**) + per-step Vs30/VsAvg/
  bedrock flow to: the step summary files (created if absent), the sidecar Vs results, the
  in-memory results, the summary tables/dock, the regenerated comprehensive report (when the
  report option is on), and the dual-resonance figure as explicit overrides. Everything remains
  reachable headlessly through the session facade (API parity).
- **FR-5 Re-open & re-pick.** An existing results folder can be re-opened without recompute:
  curves, auto-detected AND previously picked peaks (incl. secondaries) rehydrate; picking then
  proceeds as in FR-1/FR-4.
- **FR-6 Peaks as first-class layers.** The layer tree's peak-markers node genuinely toggles
  marker visibility; the properties rail styles markers/labels (shape, size, annotation font) —
  the R5 "figure properties" context — and per-curve style continues to work on the mpl figures.
- **FR-7 Checked-set dispatch (R4).** Each tool's Run strip runs the **checked** profiles through
  that tool, consuming per-profile assigned settings; Forward Single|Multiple and Strip
  Single|Batch collapse into the one dispatch model (multiplicity = the check set; the sub-stage
  wizards that remain are for *configuration*, not multiplicity). Per-item progress rows; badge
  lifecycle; cooperative cancel between items/steps for main-queue runs.
- **FR-8 Auto-peak configuration parity.** The three-strategy auto-peak surface returns:
  range-constrained (N secondary bands, each with min/max + an arming toggle), preset-based (with
  live preset details), advanced (prominence/distance/width/rel-height/exclude-N/clarity). The
  auto→manual flow is: detect, review, hand-correct on the figure; an "accept defaults for the
  unpicked rest" idiom exists.
- **FR-9 Vs mini-panel (US-4).** Vs30/VsAvg toggles, bedrock-interface combo + click-to-choose on
  the Vs plot, live readout, persisted per step.
- **FR-10 Forward tool completeness.** The Forward "HV ∥ Vs" view actually populates its Vs pane;
  the overlay/summary stay consistent with the run set.
- **FR-11 Responsiveness.** A config edit refreshes only the panels that display that section;
  no widget being edited is written back into mid-type; profile-list refreshes do not recompute
  profile summaries redundantly. (Measured in SC-3.)
- **FR-12 Busy & progress truthfulness.** The one-time heavy preparation on first Run shows busy
  feedback; determinate progress bars wherever an item/step total exists; single forward runs
  report at least start/finish into the log; engine badges re-render on engine-config change.
- **FR-13 Failure surfacing.** A failed op sets the tool status and the profile badge to FAILED,
  appends to Problems, and shows the failure envelope's message (the 2026-08-15 worker guard
  makes every failure produce one).
- **FR-14 Persistence.** Research study settings, batch/output directories, window geometry,
  splitter sizes, and rail collapsed-state persist across launches; the v2 settings file remains
  the single v2 payload (`settings_legacy.yaml` stays the legacy window's — already split).
- **FR-15 Research re-skin (US-6).** Phase gating on prerequisites; matplotlib-styled comparison
  gallery; themed canvas (the missing set-palette path); study config in FR-14's persistence.
- **FR-16 View exports.** The interactive views keep the legacy per-view exports that users rely
  on daily (save figure via toolbar; save curve/peak data), with the PDF-twin convention where a
  figure export offers formats. (The full Figure-Studio/report overhaul stays Track 3.)

## Design constraints

- **DC-1 Frozen compute.** The HVf subprocess invocation, adaptive rescanning, engine registry,
  and stdout narration stay byte-identical; `tests/golden/` digests must not drift. Peak
  *picking* is presentation/session state — it never alters forward computation.
- **DC-2 Two suites, two bindings.** All new tests land in `tests_v2/` (PySide6-offscreen,
  separate process, needs the distribution root **and** HV_Pro on PYTHONPATH); the legacy
  `tests/` suite (PyQt5) stays at its 150/24 baseline until the Track-3 cutover.
- **DC-3 Matplotlib figures, ported not imported (R1).** Interactive figures are ports of the
  legacy quarry widgets (interactive HV figure, wizard views) rebased on the v2 mpl foundation —
  copy + strip the Qt5Agg/QT_API poison; **never import a legacy `gui/` module**; the pyqtgraph
  HV/Vs canvases retire where their mpl replacements land (dead code removed with them).
- **DC-4 One derivation + one data path.** Every physics/empirical derivation goes through the
  one api surface; panels talk only to the AppState; picked peaks and style state flow
  AppState → session facade, never widget-to-widget or to disk from a panel.
- **DC-5 Facade before privates.** New GUI needs are met by **public** session-facade accessors
  (results, config, peaks) — the audit's 12 private reach-ins are reduced by this work, never
  grown; config edits go through the facade's setters or its one sanctioned funnel.
- **DC-6 Shared helpers stay single.** The table-sizing helper is adopted for the summary tables
  (no per-table hand-rolled stretch); the one staircase builder remains the only staircase
  source.
- **DC-7 Submodule pairing.** hvstrip changes and any hvsr_pro-side wiring land as a coordinated
  pair; **the user commits**; no branches.
- **DC-8 Do not copy legacy defects.** Known-broken legacy behaviors are excluded by design: the
  batch worker's mis-passed engine argument + always-zero success count; the dead Strip-Batch
  canvas tabs; the orphaned dialogs (peak-picker dialog whose result is discarded, the second
  picker implementation, the unwired batch-settings/figure-wizard/multi-profile/output-viewer
  dialogs); the secondary-peaks-never-persisted gap (FR-4 fixes it).
- **DC-9 UI-thread discipline.** >100 ms work runs on the queues; the serialized main-thread
  preload doctrine stands (Windows native-import abort), but FR-12's busy feedback wraps it.
- **DC-10 House style.** Role-QSS only, per-window theming via the authority, copy-not-import
  house widgets, the amber accent registry entry — as locked in 001.

## Success criteria

- **SC-1 Picking parity.** On the reference dataset, a scripted walkthrough (pick f0 by click,
  f0 by band-drag, two secondaries, one right-click delete, one undo, bedrock re-select) produces
  persisted step values identical to the same actions performed in the legacy app — plus
  secondaries persisted (which legacy lost).
- **SC-2 Write-back proof.** After picking, the step summary files, sidecar Vs results, report
  regeneration inputs, and dual-resonance overrides all reflect the picks; re-open restores them.
- **SC-3 Responsiveness.** One config keystroke triggers exactly one owning-panel refresh and
  zero rebuilds of unrelated lists/tables (instrumented in a test); typing in a spin/edit is
  never reformatted mid-type.
- **SC-4 Run-set proof.** 2 checked of 10 ⇒ exactly 2 processed, 2 badge lifecycles, 2 progress
  rows; cancel stops between items and statuses read truthfully.
- **SC-5 Suites.** tests_v2 green and grown (picking, write-back, dispatch, refresh-count,
  persistence pins); legacy 150/24 unchanged; goldens bit-identical; light+dark offscreen renders
  of all four tools pass a visual review.

## Edge cases

- Click/drag outside the curve's frequency range; a drag band containing no samples; deleting the
  only picked peak; undo with nothing to undo.
- A profile without a half-space row; a single-layer (half-space-only) profile in the Vs panel.
- Zero checked profiles; a checked profile whose file was removed on disk; an engine that becomes
  unavailable between arming and Run.
- Re-opening a results folder produced by the LEGACY app (peak sidecars without secondaries).
- Picking while the other queue (research) is running; theme switch mid-pick; cancel during the
  step the user is currently viewing.

## Out of scope (Track 3+)

- Figure Studio / publication-figure + report overhaul (embedded and floating), the export-options
  dialogs beyond FR-16's per-view exports.
- The P6 cutover: Hub `StripProvider`/`StripView`, legacy `gui/` + bats retirement, promote
  `gui/v2/`→`gui/`, guards flip enforcing, version 3.0.0, CHANGELOG/README rewrite.
- The All-Profiles publication view's full option surface (palettes/legend modes/smart-Y) — its
  *picking* semantics are in FR-1/FR-2; the styling surface joins the Track-3 figure work.
- New compute or engine features of any kind.

## Assumptions

- The v2 audit's already-landed 2026-08-15 review fixes are the baseline (worker failure
  envelopes, stale-name drops, config routing, settings split, single staircase, metrics
  breakdown, tee guards).
- The legacy app remains runnable side-by-side as the behavioral reference until cutover.
- The reference dataset: `examples/different_files/example_model.txt` + an sh_wave strip run
  (golden-locked) suffice for SC-1/SC-2 without the HVf exe; diffuse-field manual smokes are
  machine-local.
- Per R4, per-profile *configuration* wizards remain, but multiplicity is the check set — the
  Profile-Wizard-style per-profile stepping is served by ProfilesPanel focus + the interactive
  figure (decision synthesized from R3/R4; flag at plan review if the user wants the legacy
  n/N stepper look retained).

*Next: `/spec-plan` (Constitution / Architecture-impact / API-parity gates), then `/spec-tasks`.*
