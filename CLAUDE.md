# hvstrip-progressive — agent orientation

**HV Strip Progressive** — progressive HVSR **layer stripping**: take a layered Vs model, peel
layers one at a time, forward-model the H/V curve at every step (pluggable engines), track peak
migration + dual resonance, and report (Rahimi et al. — the bundled paper). A **git submodule**
inside HV Pro (`hvsr_pro/packages/hvstrip-progressive`) with its own repo/README/CHANGELOG.

> Import root is **`HV_Strip_Progressive`** (capitalised; the hyphenated folder itself is NOT
> importable — put `…/packages/hvstrip-progressive` on `PYTHONPATH`). HV Pro reaches it through
> the shim `hvsr_pro/packages/hvstrip_progressive_pkg.py`. The stale repo-level
> `.context/HVSTRIP_PROGRESSIVE_CONTEXT.md` is now just a pointer here.

## Architecture (Core-API-Consumers — API EXISTS but is ORPHANED; uplift in progress)

```
core/  (pure compute)  ──►  api/  (HVStripAnalysis — the intended ONE facade)  ─X─► gui/ (PyQt5, BYPASSES api)
   the frozen engines           config dataclasses (HVStripConfig) · op modules      └─► research/ (uses 2 api fns)
   + shared data model          dict-envelope methods                                 visualization/ (plotters)
```

**The central finding (see `.context/ARCHITECTURE_ASSESSMENT.md`):** a real `api/` facade +
full config dataclasses exist, but the 20k-LOC PyQt5 GUI **never imports them** — it reaches into
`core/` directly (~20 sites) and carries its OWN hand-rolled nested dict config
(`gui/strip_window.py:_get_default_config`), divergent from `api/config.py`. The uplift routes the
GUI through `HVStripAnalysis`, unifies config on the dataclasses, and rebuilds the GUI as a
PySide6 house-style workbench (plan of record in the assessment doc).

## Where things live

```
HV_Strip_Progressive/
  core/          PURE compute: soil_profile (Layer/SoilProfile — THE shared data model) ·
                 stripper (HVf model text + peel sequences) · hv_forward · batch_workflow
                 (run_complete_workflow: strip→forward→postprocess→report, ADAPTIVE freq
                 rescanning) · engines/ (registry; diffuse_field→vendored HVf.exe SUBPROCESS =
                 the hot path · sh_wave pure-python · ellipticity→Geopsy gpell via git-bash,
                 machine paths in local_config.py) · peak_detection · hv_postprocess ·
                 dual_resonance/ · report_generator · advanced_analysis · velocity_utils · vs_average
  api/           HVStripAnalysis (analysis.py — stateful facade, dict-envelope methods) over op
                 modules (profile_io/forward_engine/strip_engine/batch_engine/peak_ops/
                 dual_resonance_ops/report_ops/export/session_io/validation) · config.py (the
                 full dataclass tree, root HVStripConfig). Currently consumed ONLY by research/.
  gui/           LEGACY PyQt5 v3 — now the Round-2 PORT QUARRY (retirement POSTPONED):
                 strip_window.HVStripWindow — tabs Forward Model (Single|Multiple) · HV Strip
                 (Single|Batch) · Research. legacy_main.py = the standalone entry
                 (hv_strip_old.bat).
  gui/v2/        the PySide6 house-style workbench (LIVE; hv_strip.bat): ToolSwitcher
                 Data Input | Forward | Strip | Research over AppState (two OpQueues,
                 per-profile checked/focus/settings/ProcessingStatus) · the global
                 widgets/house/profiles_panel (stage-aware Set/Run badges) · canvas/
                 mpl_widget+vs_profile_mpl (matplotlib, Round 2 — pyqtgraph views retire in
                 Track 2) · tools/data (unified loader + the ported LayerTable).
  research/      the paper's comparison-study suite (runner + metrics + figures).
  visualization/ HVSRPlotter + resonance plots (matplotlib).
  local_config.py    machine-local engine paths (gpell/git-bash) — the ONLY machine-path surface.
tests/           legacy suite (~150 pass/24 skip; PyQt5-forced conftest) + tests/golden/ (the
                 COMPUTE LOCK: workflow digests per engine + the legacy GUI config fixture).
                 tests_v2/ = the PySide6-offscreen rebuild suite (65 green; SEPARATE process;
                 needs BOTH the distribution root and HV_Pro on PYTHONPATH for theme_core).
```

Data contract (Layer/SoilProfile invariants, the HVf text format, peel sequences, result
envelopes, the output tree): **`.context/STRIP_MODEL.md`**. Vocabulary: **`.context/GLOSSARY.md`**.

## The frozen compute (performance is load-bearing — DO NOT regress)

`stripper` · `hv_forward` · `batch_workflow` · `engines/` (all three) · `hv_postprocess` are
FROZEN: the HVf.exe subprocess invocation, the adaptive frequency rescanning, and the engine
registry stay byte-identical. The proof is **`tests/golden/`** (`test_golden_compute.py`) — per-step
curve checksums for sh_wave (any machine) + diffuse_field (`-m hvf`, needs the exe). Regenerate
ONLY for an intended compute change (`python tests/test_golden_compute.py --regen`).
`soil_profile`/`velocity_utils` (the shared model) and `peak_detection`/`report_generator`/
`dual_resonance` (analysis) stay directly importable — final classification in the assessment §audit.

## Adding a feature

Order: **core → api → consumers.** New compute goes in `core/` (+ a test), exposed via an
`HVStripAnalysis` method (dict envelope), then wired to a GUI panel (through the coming AppState —
never `core.*` from panels) / research script. Never add a new GUI→core reach-in.

## Run / test

- **Legacy GUI:** `HV_Pro/HV_Analyze_Pro/hv_strip_old.bat` (PyQt5) or
  `python -m HV_Strip_Progressive.gui.legacy_main` with the distribution root on PYTHONPATH.
- **New workbench (from the GUI-rebuild phase):** `HV_Pro/HV_Analyze_Pro/hv_strip.bat`
  (PySide6; PYTHONPATH carries BOTH the distribution root and `HV_Pro` for theme_core).
- **Tests:** `QT_QPA_PLATFORM=offscreen python -m pytest tests` from the distribution root
  (~150 pass / 24 skip; PyQt5-forced conftest — do NOT import PySide6 in this suite).
  The golden hvf tier: `pytest tests/test_golden_compute.py -m hvf` (local machines with the exe).
- **Engines:** DiffuseField needs the vendored `HVf.exe` (in-tree); Ellipticity needs Geopsy's
  `gpell` + Git Bash via `local_config.py` (copy `local_config.example.py`) — tests skip it when absent.

## Standing rules

- **Frozen compute** (above) — GUI/API wiring changes only; goldens must stay bit-identical.
- The legacy `tests/` conftest forces PyQt5; the rebuild's tests live in `tests_v2/` (PySide6,
  separate pytest process). Never mix bindings in one process.
- `local_config.py` is the only place machine paths live; code must degrade gracefully
  (skip/disable, never crash) when an engine's binary is absent.
- Per-area conventions → `.claude/rules/`. Structure/surface changes update `CLAUDE.md` +
  `.context/` in the same commit (`.claude/rules/docs-sync.md`).
- This is a **submodule**: its changes and hvsr_pro-side wiring (hub provider, bats) land as a
  coordinated pair; the user commits.
