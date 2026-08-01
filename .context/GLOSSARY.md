# GLOSSARY.md — HV Strip vocabulary

| Term | Meaning |
|---|---|
| **Layer stripping / peeling** | Removing the TOP layer of a layered Vs model and re-forward-modelling the H/V response of what remains; done progressively to attribute H/V peaks to specific interfaces. |
| **Step** | One entry in the peel sequence: `Step{i}_{n}-layer` = the model after *i* peels, with *n* layers left (incl. half-space). Step0 = the original model. |
| **Peel sequence** | The ordered set of stripped models an N-layer profile generates (N-1 peels + the original). |
| **HVf** | The diffuse-field forward-model executable (vendored `HVf.exe`) computing an H/V spectral-ratio curve for a layered model — the default engine, subprocess-invoked. |
| **HVf model format** | The N-line text format (`N`, then `thk vp vs rho` rows, half-space row `0 vp vs rho`) all engines consume — see STRIP_MODEL §2. |
| **Engine** | A pluggable forward modeller registered in `core/engines`: `diffuse_field` (HVf.exe), `sh_wave` (pure-Python SH transfer function), `ellipticity` (Geopsy `gpell` Rayleigh ellipticity). |
| **Adaptive (frequency) rescanning** | `batch_workflow`'s loop that re-runs the engine with an expanded fmax / shrunk fmin when a detected peak sits too close to the frequency-band edge. |
| **Peak migration** | How the dominant H/V peak's frequency/amplitude moves step-to-step as layers are peeled — the core scientific readout (evolution plots). |
| **Dual resonance** | Detecting/characterising TWO co-existing resonance peaks (shallow + deep impedance contrasts) — `core/dual_resonance/` + its api op. |
| **Vs30** | The time-averaged shear-wave velocity of the top 30 m (`core/vs_average`); `vs30_extrapolated` marks profiles shallower than 30 m. |
| **Half-space** | The terminating infinite layer (thickness 0) every profile must end with. |
| **Waterfall / overlay plots** | The report figures stacking every step's H/V curve (offset stack vs same-axes overlay). |
| **Research suite** | `research/` — the batch comparison studies behind the Rahimi et al. paper (engine/field-data comparisons), driven today by the GUI's Research tab. |
| **Session (api)** | An `HVStripAnalysis` instance: config + loaded profiles + forward/strip/batch results, all methods returning dict envelopes. |
| **The legacy dict-config** | The GUI's hand-rolled nested config (`strip_window._get_default_config`) being retired in favour of `HVStripConfig`; shape pinned at `tests/golden/legacy_gui_config.json`. |
| **Golden lock** | `tests/golden/` + `test_golden_compute.py` — the frozen-compute proof (per-step curve digests per engine). |
| **hvstrip (hub key)** | The HV Hub module key/rail row for this package (`Project.strip_dir`, `hvstrip_state_io`). NOT "time-frequency" — the rail description is being fixed. |
