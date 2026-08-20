# Plan 002 — GUI Round 2 · Track 2: legacy parity + hardening

**From:** `spec.md` (this folder) · **Format:** as `specs/001-gui-house-style/plan.md` —
constitution gates, then independently-verifiable stages, each with a named proof.
**Baseline:** the 2026-08-15 review fixes (tests_v2 65 · legacy 150/24 · goldens bit-identical).

## Constitution check (P1–P8)

| # | Principle | Verdict | How this plan satisfies it |
|---|---|---|---|
| P1 | Core-API-Consumers | ✅ | No new compute in `core/`. All new capability lands as **session-facade ops first** (S1), consumed via AppState (S2+). The frozen-compute set is untouched (DC-1). |
| P2 | API parity | ✅* | Every GUI capability added here is a facade op usable headlessly (peaks set/get/persist, rehydrate, checked-set batch with cancel). *This package has no CLI/MCP consumer yet — parity = facade + `research/` + scriptability; recorded in Complexity Tracking (001 precedent).* |
| P3 | Config-driven | ✅ | New knobs (auto-peak strategies, marker style, research study settings) become `HVStripConfig` fields / documented session state — never widget-local truth (FR-14, DC-4). |
| P4 | Scientific reproducibility | ✅ | Picking is presentation/session state; **no f0-affecting compute changes**. The golden digests are the regression lock; SC-1's legacy-vs-v2 walkthrough pins the persisted values. |
| P5 | Responsive GUI | ✅ | Long ops stay on the two OpQueues; the preload doctrine stands with busy feedback (DC-9, FR-12); refresh routing by section kills the storm (FR-11). |
| P6 | Vault-first | ⚠️→✅ | The vault cluster is stale since 2026-05-21 (read done — nothing usable beyond method scope). The **write protocol debt** (P0–Track 1 + this feature) is an explicit S7 task. |
| P7 | Spec before code | ✅ | `spec.md` precedes this plan; tasks follow. |
| P8 | Verify before done | ✅ | S7 runs `/spec-verify`-style evidence: suites, goldens, renders, SC walkthroughs, docs-sync. |

**Complexity Tracking:** P2* — no `cli/`/`mcp/` exists in this package; building them is not in
Track 2's scope (the Hub/consumer story is Track 3's P6). Justification: the facade keeps every
new op headless-callable, so consumers can be added without rework.

## Architecture impact

| Layer | Change |
|---|---|
| `core/` | **None** (frozen). Read-only reuse: `peak_detection` presets/strategies, `vs_average`, report regeneration entry, dual-resonance figure entry. |
| `api/` (facade) | NEW public ops (S1): results/config **accessors** (`forward_results()`, `strip_results()`, `batch_result()`, config get — retiring the GUI's 12 `_private` reach-ins per DC-5); **peaks**: `detect_peaks(...)` (wraps the 3 strategies), `set_manual_peaks(scope, f0, secondaries)`, `get_peaks(scope)`, `persist_peaks(...)` (summary files + sidecar incl. **secondaries**), `set_step_vs_context(step, bedrock, vs30, vsavg)`; **rehydrate**: `load_results_folder(path)` (curves + auto + picked peaks, no recompute); **dispatch**: `compute_forward_for(names, ...)` / `run_stripping_for(names, ...)` honoring per-profile settings + a cooperative `cancel_token` checked between items/steps; report regeneration + dual-resonance `peak_overrides` pass-through. All envelope-returning, Qt-free. |
| `api/config.py` | Auto-peak strategy config (range bands / preset / advanced) as typed fields; marker/label style defaults; research study section (so FR-14 persistence rides the existing funnel). |
| `gui/v2 state` | AppState wrappers for every S1 op; **section-scoped refresh** (`config_changed(section)` consumed, not discarded); busy scope for the preload; run-set dispatch (checked + assigned settings); cancel; persistence (geometry/splitters/rails via QSettings; config sections via the funnel); engine-badge refresh hook. |
| `gui/v2 widgets/canvas` | The ported **interactive mpl HV figure** (from the quarry: `strip_wizard_view` + `hv_curve_view` mechanics on `MplFigureWidget`) + the **Vs mini-panel** (bedrock combo + click-to-choose); peak markers/labels as layer-tree + properties citizens; `tables.py` helper adopted; pyqtgraph HV/Vs views retired where replaced (with their dead APIs). |
| `gui/v2 tools` | Forward/Strip rewired to the figure + write-back + re-open; per-stage Run strips over the checked set (Single|Multiple, Single|Batch dissolve); Research re-skin + gating + persistence + theming. |
| Legacy `gui/` | **Untouched** (port quarry, R6). |
| Contracts | The step `*summary*.csv` peak columns (existing shape, now also written by v2); a **peaks sidecar** next to step folders carrying f0 + secondaries + label positions + Vs context (supersedes legacy `vs_results.json`, which rehydrate still READS for legacy folders); `STRIP_MODEL.md` gains both. |

## API-parity plan

GUI = AppState → facade. Headless = the same facade ops from a script/notebook (documented in the
quickstart snippet inside `tasks.md`'s verify task). `research/` continues to consume the facade.
CLI/MCP: deferred to Track 3 (Complexity row above).

## Scientific / numerical invariants

- The golden workflow digests (`tests/golden/`, sh_wave everywhere; `-m hvf` locally) stay
  bit-identical — run in S4 and S7.
- Vs30/VsAvg values shown while picking come from the same `vs_average` calls as the legacy app;
  SC-1's walkthrough compares persisted numbers against a legacy-produced fixture.
- Peak *detection* (auto) reuses `core.peak_detection` presets verbatim; manual picks are stored
  as picked (never re-derived).

## Vault impact

- **Read (done):** the hvstrip cluster is pre-rebuild stale; nothing constrains this design.
- **Write (S7):** session note + updated package hub note covering P0→Track 2 (shell, amber,
  Track 1, the 2026-08-15 review fixes, this feature), plus an ADR for the peaks
  sidecar/write-back contract. This clears the accumulated write-protocol debt.

## Stages (each independently verifiable)

**T2-S1 — Facade: accessors + peaks + rehydrate + dispatch (Qt-free).**
Build the api surface in the Architecture table; migrate `update_config` onto the facade funnel;
add the cancel token. *Proof:* new `tests_v2/test_api` tests (peaks round-trip incl. secondaries;
persist/rehydrate on a tmp results tree + a legacy-shaped tree; dispatch honors names + settings +
cancel); layering guard green; goldens untouched.

**T2-S2 — AppState rewire.**
Wrap S1; retire the 12 private reach-ins; section-scoped refresh plumbing; busy scope; run-set
dispatch; persistence spine. *Proof:* refresh-count test (one keystroke ⇒ one owning-panel
refresh, SC-3); reach-in grep = 0 outside sanctioned funnels; scaffold suite green.

**T2-S3 — The interactive figure + Vs mini-panel (widget-level).**
Port + enhance per FR-1/2/3/9: click-vs-drag with live band, uniform verbs, draggable/persisted
labels, markers/labels styleable, toolbar guard, Vs panel with bedrock click. *Proof:* widget
tests (pick semantics incl. band-argmax vs exact-click, right-click delete, undo, label position
persistence, toolbar guard); offscreen light+dark renders reviewed.

**T2-S4 — Forward + Strip wiring + write-back + re-open.**
Mount the figure in both tools; step list ✓s; the FR-4 chain (summaries, sidecar, report regen,
dual-resonance overrides); FR-5 re-open; O1 Vs pane fixed. *Proof:* SC-1 + SC-2 walkthrough
tests against the sh_wave golden run + a legacy-produced fixture; goldens bit-identical.

**T2-S5 — Checked-set dispatch (R4).**
Run strips per tool; Single/Multiple + Single/Batch collapse; per-item progress rows; badge
lifecycle; cancel UX. *Proof:* SC-4 test (2-of-10; cancel mid-run; statuses truthful);
window-module test extended.

**T2-S6 — Hardening sweep.**
FR-11..FR-15 remainder: busy UX + determinate progress + badge refresh + failure surfacing;
persistence (FR-14); Research gating/re-skin/theming; `tables.py` adoption; empty-legend +
`str(dual)` + `PhasePanel.session` landmine + duplication collapse; retire replaced pyqtgraph
views + dead canvas APIs. *Proof:* targeted pins per item; full tests_v2 green; dark-mode render
of all four tools.

**T2-S7 — Verify + docs + vault.**
Full suites (tests_v2 · legacy 150/24 · goldens; `-m hvf` where the exe exists); light+dark
renders; `hv_strip.bat` manual smoke script for the user; docs-sync (`CLAUDE.md` tree,
`ARCHITECTURE_ASSESSMENT.md` phase log, `STRIP_MODEL.md` contracts, `DECISIONS.md` addendum if
any choice moved); **GeoVault write protocol** (the P6 debt). *Proof:* the evidence bundle in
`PROJECT_STATUS`-style notes; user smoke sign-off.

Stage order is strict S1→S2→(S3∥S5 may overlap)→S4→S6→S7; S3 and S5 share no files.

*Next: `/spec-tasks` (layer-ordered tasks with [P] parallel markers), then handoffs if the
implementation is delegated.*
