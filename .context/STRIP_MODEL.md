# STRIP_MODEL.md — the central data contract

*The objects, file formats and result envelopes every layer shares. Grounded in the real code
(v2.2.0); update in the same commit as any change to them (docs-sync).*

## 1. The layered model — `core/soil_profile.py`

- **`Layer`**: `thickness` (m; **0.0 = the half-space**), `vp`, `vs` (m/s), `density` (g/cc),
  `is_halfspace`. Missing vp/density are AUTO-DERIVED from vs via
  `core/velocity_utils.VelocityConverter` (empirical relations) — a profile built from Vs alone is
  valid.
- **`SoilProfile`**: an ordered list of Layers, LAST layer must be the half-space.
  `to_hvf_format()` is the serialisation every engine consumes.
- Invariants: thickness > 0 for all but the last layer; exactly one half-space, last; vs > 0.

## 2. The HVf model text format (read/written by `core/stripper.py`)

```
N                       ← number of layers INCLUDING the half-space
thk  vp   vs   rho      ← one row per finite layer
...
0    vp   vs   rho      ← the half-space row (thickness 0)
```
Parsed by `stripper._read_hvf_model` / `_parse_rows`; emitted by `_to_model_lines`. Also accepted
at the api/profile_io level: Dinver-style Vp/Vs/Density triplet files and layer-table/median CSVs
(see `examples/different_files/`).

## 3. Peel sequences — `core/stripper.py`

`generate_peel_sequence(model)` peels the TOP layer repeatedly: an N-layer model yields N-1
stripped models (down to 2 layers: one finite + half-space)… plus the original = the step set.
`write_peel_sequence` materialises them as `strip/Step{i}_{n}-layer/model_Step{i}_{n}-layer.txt`.

## 4. Forward engines — `core/engines/`

`EngineRegistry` (`registry`) with three registered engines, all subclassing `BaseForwardEngine`
(`compute()`, `compute_from_profile()`, `get_default_config()`, `format_model()`):
- **`diffuse_field`** (default) — subprocess to the vendored `HVf.exe` (Win) / `HVf`/`HVf_Serial`
  (Linux) under `core/engines/diffuse_wave_field/`. THE hot path.
- **`sh_wave`** — pure-Python SH transfer-matrix (Kramer 1996 + Darendeli).
- **`ellipticity`** — subprocess to Geopsy `gpell` via Git Bash; paths from `local_config.py`.
Result: **`EngineResult`** (`core/engines/base.py:23`) = `frequencies`, `amplitudes`, `metadata`.

## 5. The workflow + its on-disk output tree — `core/batch_workflow.py`

`run_complete_workflow(initial_model_path, output_base_dir, workflow_config=None, engine_name=None)`
runs: **[1] strip → [2] forward per step (with ADAPTIVE freq rescanning when a peak hugs a
boundary) → [3] post-process/visualise → [4] report → [5] optional dual-resonance**, narrating
per-step progress on stdout (`[1/3] Layer Stripping`, `[OK] … in 1.75s`, `[2/8] Creating …`) —
the api's progress tee parses exactly this narration.

```
<output_base_dir>/
├── strip/Step{i}_{n}-layer/            ← per step
│   ├── model_Step{i}_{n}-layer.txt     ← the HVf model
│   ├── hv_curve.csv                    ← Frequency_Hz, HVSR_Amplitude (the full curve)
│   ├── hv_curve.png · vs_profile.png · step_summary.csv
└── reports/                            ← overlay/waterfall/evolution figures + CSV/TXT/JSON/PDF
```

**The workflow result dict** (returned; also what the golden lock digests):
`{"success": bool, "step_results": {step_name: {model_file, hv_csv, hv_curve_png, vs_profile_png,
summary_csv, n_frequencies, peak_frequency, peak_amplitude, peak_index, vs30,
vs30_extrapolated}}, ...}`.

## 6. The api envelopes — `api/*.py`

`HVStripAnalysis` methods return JSON-serialisable dicts built from typed results:
`ForwardResult`/`MultiForwardResult`/`PeakInfo` (`api/forward_engine.py`), `StripResult`/`StepResult`
(`api/strip_engine.py`), `BatchStripResult`/`ProfileStripResult` (`api/batch_engine.py`) — all
`.to_dict()`-able. Config = the `api/config.py` dataclass tree rooted at **`HVStripConfig`**
(`to_dict()` / `copy()` / `_apply_dict` recursive merge). From the uplift: saved payloads carry
`config_version: 2`; anything else goes through `HVStripConfig.from_legacy_gui_dict()` (the legacy
GUI dict shape is pinned at `tests/golden/legacy_gui_config.json`).

## 7. Persistence surfaces

- Standalone settings: `~/.hvstrip/settings.yaml`.
- HV Pro project mode: `project_manager.module_state.hvstrip_state_io`
  (`has/load/save_hvstrip_state`) storing `{config, results, extra}` under
  `project.ensure_module_dir('hv_strip', profile_id)`; `Project.strip_dir(profile_id)` =
  `<project>/hv_strip/<profile_id>`.
All loads funnel through `api/session_io.py` from the uplift's P1 on.

## 8. The compute freeze (proof)

`tests/golden/workflow_{sh_wave,diffuse_field}.json` — per-step digests (curve checksum
`amp_sum`/`amp_max`/`f_at_max` over the 512-pt `hv_curve.csv` + peak/vs30 scalars). Bit-identical
across runs and across the entire uplift; regenerate only for an INTENDED compute change.
