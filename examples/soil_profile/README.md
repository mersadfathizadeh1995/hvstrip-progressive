# Soil Profile Example

This example demonstrates a complete progressive layer stripping workflow using a simple soil profile model.

Files
- `model.txt`: HVf‑format model (N followed by N rows: thickness, Vp, Vs, density). Last row is the half‑space (thickness 0).

Run (CLI)
```bash
# Windows (pass your HVf.exe path)
hvstrip-progressive workflow examples/soil_profile/model.txt examples/soil_profile/output --exe-path C:\path\to\HVf.exe

# Linux/macOS (pass your HVf path)
hvstrip-progressive workflow examples/soil_profile/model.txt examples/soil_profile/output --exe-path /path/to/HVf
```

Outputs (created under `examples/soil_profile/output/`)
- `strip/StepX_Y-layer/`: per‑step model + HV results
- `reports/`: comprehensive figures and summaries (after running `report`)

Generate report
```bash
hvstrip-progressive report examples/soil_profile/output/strip --output-dir examples/soil_profile/output/reports
```

Notes
- HVf is an external dependency from HV‑INV (GPL‑3.0). It is not bundled with this package. See the project README for installation details.
- The workflow uses local peak detection and selects the lowest‑frequency prominent peak (fundamental) by default.
