# Changelog

All notable changes to the HVSR Progressive Layer Stripping Analysis package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-09-25

### Added
- **Core Modules**:
  - `stripper.py`: Progressive layer stripping algorithm with proper half-space promotion
  - `hv_forward.py`: HVf.exe interface for HVSR curve computation
  - `hv_postprocess.py`: Publication-ready plotting and analysis
  - `batch_workflow.py`: Complete workflow orchestration
  - `report_generator.py`: Comprehensive scientific reporting

- **Command-Line Interface**:
  - `hvstrip-progressive workflow`: Complete end-to-end analysis
  - `hvstrip-progressive strip`: Layer stripping only
  - `hvstrip-progressive forward`: HV forward modeling only
  - `hvstrip-progressive postprocess`: Post-processing only
  - `hvstrip-progressive report`: Report generation only

- **Visualization Features**:
  - Smart annotation placement for HV curve plots
  - Log/linear axis scaling options
  - Configurable curve smoothing (Savitzky-Golay)
  - Y-axis compression for better peak visualization
  - Frequency band highlighting around peaks
  - Velocity profile plots with layer annotations
  - Multi-panel comprehensive analysis figures
  - Waterfall plots showing progressive stripping effects

- **Analysis Capabilities**:
  - Multiple peak detection methods (max, find_peaks, manual)
  - Interface impedance and Vs contrast analysis
  - Peak frequency evolution tracking
  - Controlling interface identification
  - Comprehensive data summaries in CSV format

- **Scientific Reporting**:
  - Publication-ready figures (PNG + PDF)
  - Detailed text reports with conclusions
  - Analysis metadata in JSON format
  - Multi-panel comparison plots
  - Statistical summaries and trends

- **Configuration System**:
  - YAML/JSON configuration file support
  - Nested configuration merging
  - Flexible parameter customization
  - Default configurations for common use cases

- **Package Infrastructure**:
  - Professional package structure
  - Comprehensive documentation
  - Example data and scripts
  - MIT license
  - Development dependencies and tools

### Technical Details
- **Python Support**: 3.8+
- **Dependencies**: numpy, matplotlib, scipy, pandas, click, pyyaml
- **External Requirements**: HVf.exe for HVSR computation
- **Input Formats**: HVf-format velocity models, CSV data files
- **Output Formats**: PNG, PDF figures; CSV, JSON data files

### Scientific Background
- **Method**: Progressive layer stripping for HVSR analysis
- **Theory**: Diffuse-field theory via HVf.exe solver
- **Applications**: Site characterization, interface identification, model validation
- **Validation**: Proper layer removal with half-space promotion algorithm

## [Unreleased]

### Planned Features
- Interactive HTML reports
- Batch processing of multiple models
- Advanced statistical analysis
- Integration with other HVSR software
- Performance optimizations
- Extended visualization options

---

**Note**: This is the initial release of the package, representing a complete rewrite and enhancement of previous prototype scripts.
# Changelog

## 1.0.0 — 2025-09-25

- License change to GPL-3.0-only; added THIRD_PARTY.md and CITATION.cff.
- Peak detection improvements: default to `find_peaks` + `select=leftmost` with guards
  (`freq_min`, `min_rel_height`, `exclude_first_n`) to avoid edge/artefact picks.
- Adaptive HVf scanning: automatically expands `fmax` or shrinks `fmin` when peaks lie near band edges
  (bounded passes/limits); configurable in workflow and YAML.
- Reporting/plots:
  - X-axis: log-scale with default lower bound 1 Hz (auto-include sub-1 Hz peaks when needed);
    integer ticks for readability.
  - Colormap: switched to cividis (no yellow) for accessibility.
  - Interface analysis: connected lines with markers vs depth for improved interpretation.
- README: added example, HVf install/run instructions, collaborator/maintainer info,
  adaptive scanning and plotting defaults, and configuration reference.
- Packaging: updated `setup.py` metadata (author, email, URLs), `MANIFEST.in` to include docs.
