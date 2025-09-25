# HVSR Progressive Layer Stripping Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python package for progressive layer stripping analysis of Horizontal-to-Vertical Spectral Ratio (HVSR) data using diffuse-field theory.

Repository: https://github.com/mersadfathizadeh1995/hvstrip-progressive

Author: Mersad Fathizadeh — Ph.D. Candidate, University of Arkansas (email: `mersadf@uark.edu`; GitHub: `mersadfathizadeh1995`)

Collaborator: Clinton Wood — Associate Professor, Ph.D., P.E., University of Arkansas (GitHub: `cmwood10`; email: `mycmwood@uark.edu`)

---

## Maintainers & Collaborators

- Maintainer: Mersad Fathizadeh (`@mersadfathizadeh1995`)
- Collaborator: Clinton Wood (`@cmwood10`)
---

## Overview

This package helps identify which subsurface interfaces control HVSR peaks through systematic layer removal analysis. By progressively “stripping” layers from a velocity model and computing the resulting HVSR curves, you can assess peak evolution, impedance contrasts, and controlling depths with publication‑quality plots and a complete report.

## Key Features

- Layer Stripping: Progressive removal of deepest finite layers
- HV Forward Modeling: Uses the external HVf solver (HV‑INV)
- Post‑Processing: Publication‑ready plots and summaries
- Batch Workflow: Turnkey, end‑to‑end analysis
- Report Generation: Multi‑panel figures, CSV summaries, metadata, and text report

## Installation

```bash
# Clone the repository
git clone https://github.com/mersadfathizadeh1995/hvstrip-progressive.git
cd hvstrip-progressive

# Install the package
pip install -e .

# Or install from PyPI (when available)
pip install hvstrip-progressive
```

## External Dependency: HVf (HV‑INV)

This workflow calls the external “HVf” solver from HV‑INV for forward HVSR computation. It is not bundled with this package. You must install/download it separately.

- Project: HV‑Inv — HV‑Inv: A MATLAB-based graphical tool for the direct and inverse problems of the horizontal-to-vertical spectral ratio under the diffuse field theory
- Repository: https://github.com/agarcia-jerez/HV-INV
- License: GPL‑3.0

Where to get HVf:
- Windows: download HVf.exe from the HV‑INV repo (see the `exe/` folder). Place it in a known path (e.g., `C:\tools\HVf.exe`).
- Linux/macOS: use the provided HVf binary for your OS if available; otherwise, compile from source via HV‑DFA: https://github.com/agarcia-jerez/HV-DFA

How to point hvstrip‑progressive to HVf:
- CLI: pass `--exe-path /path/to/HVf` (or `HVf.exe` on Windows) to the `workflow` command.
- Python API: pass `{"hv_forward": {"exe_path": "/path/to/HVf"}}` to `run_complete_workflow`.
- YAML: set `solver.exe_path` in `hvstrip_progressive/config/default_config.yaml`.

## Usage

### Complete workflow (CLI)

```bash
# IMPORTANT: pass the path to your HVf executable
hvstrip-progressive workflow model.txt output_dir/ --exe-path C:\path\to\HVf.exe
```

### Individual steps (CLI)

```bash
hvstrip-progressive strip model.txt output_dir/
hvstrip-progressive forward model.txt --exe-path /path/to/HVf --output hv_curve.csv
hvstrip-progressive postprocess hv_curve.csv model.txt output_dir/
hvstrip-progressive report output_dir/strip --output-dir output_dir/reports
```

### Python API

```python
from hvstrip_progressive.core import batch_workflow, report_generator

# Run complete analysis
results = batch_workflow.run_complete_workflow(
    initial_model_path="model.txt",
    output_base_dir="analysis_output/",
    workflow_config={"hv_forward": {"exe_path": "HVf.exe"}}
)

# Generate comprehensive reports
reporter = report_generator.ProgressiveStrippingReporter(
    strip_directory="analysis_output/strip/",
    output_dir="analysis_output/reports/"
)
report_files = reporter.generate_comprehensive_report()

## Included Example (Soil Profile)

An end‑to‑end example is included under `examples/soil_profile/`:

- Input model: `examples/soil_profile/model.txt`
- Run (Windows):
  `hvstrip-progressive workflow examples/soil_profile/model.txt examples/soil_profile/output --exe-path C:\\path\\to\\HVf.exe`
- Run (Linux/macOS):
  `hvstrip-progressive workflow examples/soil_profile/model.txt examples/soil_profile/output --exe-path /path/to/HVf`
- Generate report:
  `hvstrip-progressive report examples/soil_profile/output/strip --output-dir examples/soil_profile/output/reports`

Outputs are not committed to the repository (see `.gitignore`); you can regenerate them locally.
```

## Peak Selection (Fundamental by Default)

The workflow defaults to local peak detection (`scipy.signal.find_peaks`) and selects the lowest‑frequency prominent peak (the fundamental) with `prominence=0.2` and `distance=3`. Adjust these in:

- `hvstrip_progressive/core/batch_workflow.py` (DEFAULT_WORKFLOW_CONFIG)
- or in your custom config passed to the workflow/postprocess functions

## Adaptive Frequency Scanning

Fundamental frequency can fall outside a static band (e.g., >20 Hz for very shallow cases or <0.5 Hz for very soft sites). To avoid missing true peaks at band edges, the workflow can adapt the HVf frequency range when the global maximum lies near the lower/upper bound.

- Default behavior: up to 2 adaptive passes; expands `fmax` or shrinks `fmin` within safe limits.
- Configure in code: `DEFAULT_WORKFLOW_CONFIG['hv_forward']['adaptive']`.
- Configure in YAML: `solver.adaptive` (see `hvstrip_progressive/config/default_config.yaml`).

## Plotting Defaults

Publication plots are tuned for clarity:

- X-axis (frequency) is log-scaled and limited by default to ≥1 Hz with integer tick labels (1–10, then 20, 30, …). If any step has a peak <1 Hz, the axis expands just enough to include it.
- Colors use the cividis colormap (no yellow), improving readability.
- Interface analysis uses connected lines with markers vs depth to better convey trends.

## Outputs

Per step (e.g., `StepX_Y-layer/`):
- `model_*.txt` — HVf‑format velocity model
- `hv_curve.csv` — Theoretical HVSR curve (Frequency_Hz, HVSR_Amplitude)
- `hv_curve.png` — Publication‑ready HV curve plot
- `vs_profile.png` — Velocity profile visualization
- `step_summary.csv` — Peak + model data summary

Reports (in `reports/`):
- `hv_curves_overlay.png`, `peak_evolution_analysis.png`, `interface_analysis.png`, `waterfall_plot.png`
- `comprehensive_analysis.png`, `publication_figure.png` (and `.pdf`)
- `progressive_stripping_summary.csv`, `analysis_report.txt`, `analysis_metadata.json`

## Configuration Reference (excerpt)

- `peak_detection.method`: `find_peaks` (default), `max`, `manual`
- `peak_detection.select`: `leftmost` (default), `max`
- `peak_detection.find_peaks_params`: `prominence` (0.2), `distance` (3)
- `peak_detection.freq_min`: 0.5 Hz (default), `freq_max`: null
- `peak_detection.min_rel_height`: 0.25 (ignore weak local peaks)
- `peak_detection.exclude_first_n`: 1 (ignore the first frequency bin)
- `solver.adaptive`: `enable` (true), `max_passes` (2), `edge_margin_frac` (0.05),
  `fmax_expand_factor` (2.0), `fmin_shrink_factor` (0.5), `fmax_limit` (60.0), `fmin_limit` (0.05)

## Citation

If you use this software, please cite the following paper:

Rahimi, M., Wood, C., Fathizadeh, M., & Rahimi, S. (2025). A Multi-method Geophysical Approach for Complex Shallow Landslide Characterization. Annals of Geophysics, 68(3), NS336. https://doi.org/10.4401/ag-9203

You may also acknowledge the HV‑Inv project when referencing the solver used for forward HVSR calculations.

## License

This project is licensed under the GNU General Public License v3.0 (GPL‑3.0‑only). See the [LICENSE](LICENSE) file for details.

Third‑party dependency acknowledgment:
- HV‑Inv (HVf) is licensed under GPL‑3.0 (see their repository). This package does not distribute HVf. Users must download/install HVf separately and agree to the HV‑Inv license.

## Support

- Issues: https://github.com/mersadfathizadeh1995/hvstrip-progressive/issues
- Repository: https://github.com/mersadfathizadeh1995/hvstrip-progressive
- Email: mersadf@uark.edu
