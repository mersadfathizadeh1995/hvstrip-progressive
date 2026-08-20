# legacy_picked_run — the SC-1 parity fixture (spec 002, T01)

A committed, deterministic strip-run tree with a **manual-picking walkthrough
persisted by the REAL legacy code** (`HVStripWindow._update_summary_csv` /
`_create_minimal_summary_csv` static methods + the `_persist_vs_data` json
shape), so the new api's write-back can be compared value-for-value and
byte-for-byte against what the legacy app produced.

- `run/strip/` — the sh_wave strip of `examples/different_files/
  example_model.txt` (report off, figures stripped): 6 step folders
  `Step0_6-layer` … `Step5_1-layer`, each `hv_curve.csv` + `model_*.txt` +
  `step_summary.csv`; steps 0–2 carry the legacy-persisted picks.
- `run/strip/vs_results.json` — the legacy Vs-context sidecar.
- `picks.json` — the walkthrough (sidecar-shaped) for replay through the
  new api. **Note the captured legacy truth:** picked SECONDARIES appear in
  `picks.json` but NOT in any `run/` file — the legacy app never persisted
  them (spec 002 DC-8 fixes that via `picked_peaks.json`).
- `_generate.py` — regenerates everything (own process; imports the legacy
  PyQt5 module — never import it from tests_v2). Regenerate only with user
  sign-off; the committed tree is the pin.
