# Rule: `core/` — pure compute + the FROZEN hot path

Loaded when you touch `HV_Strip_Progressive/core/**`.

## Conventions
- **No Qt anywhere in core.** matplotlib is tolerated (report_generator/visualization) but lazy.
- **FROZEN COMPUTE — do not regress (user constraint):** `stripper`, `hv_forward`,
  `batch_workflow`, `engines/` (all three), `hv_postprocess`. The `HVf.exe` subprocess invocation,
  the adaptive frequency rescanning (`_compute_hv_curve_adaptive`), and the
  `BaseForwardEngine`/registry contract stay byte-identical. Proof = `tests/golden/`
  (`test_golden_compute.py`); goldens must stay bit-identical. Regenerate ONLY for an intended
  numeric change, with the user's sign-off.
- The engine binaries: `HVf.exe` is vendored in-tree; `gpell`/git-bash paths come from
  `local_config.py` ONLY, and everything must degrade to a clean skip/disable when absent.
- Data-model invariants (`.context/STRIP_MODEL.md`): last layer = the half-space (thickness 0);
  vp/density auto-derive from vs; `to_hvf_format()` is the engine serialisation.
- `batch_workflow`'s stdout narration (`[i/n] …`, `[OK] …`) is a de-facto INTERFACE — the api's
  progress tee parses it. Don't reword those lines casually.
- Ship a test with every change; numeric changes pin expected values.

## Don't
- Import Qt/gui/api; change the parallel/subprocess model; touch engine argv; reword the
  narration; or bypass `soil_profile` with ad-hoc layer dicts.
