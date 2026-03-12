# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] — 2025-06-01

### Added
- Vs30/VsAvg output in CSV, text, PDF, and publication figure reports
- 3-strategy auto peak detection dialog (Range-Constrained, Preset-Based, Advanced)
- `width` parameter in core peak detection engine
- Diagonal staircase annotation layout for HV overlay and publication figures
- Persistent wizard peak selections written to disk on Finish
- Configurable legend settings in All Profiles View
- Dock defaults, auto-peak panel, drag selection, and data export in strip UI
- Figure studio palette support, annotations, and publication keyword args
- Click-and-release peak selection in HV strip wizard

### Changed
- Redesigned Strip Single: 8 tabs collapsed to 3 tabs + summary dock
- Redesigned HV Curve View with collapsible Plot Settings and fixed Data Input
- Upgraded Forward Single to feature parity with Multiple
- Modularised `all_profiles_view.py` into 9 sub-modules

### Fixed
- Fix 'cannot remove artist' errors in drag selection
- Fix step ordering, seaborn import, and annotation overlap
- Fix HVf exe path resolution and secondary peak detection
- Fix VsAvg keyword bug, save workflow overhaul, publication figures
- Fix Vs panel singular matrix crash on toggle
- Fix auto-detect peak: apply primary range, frequency-based exclusion
- Correct `style_constants` import path in `ui_builder`

## [2.0.0] — 2025-03-01

### Added
- Complete GUI reconstruction with faithful 4-page layout (PyQt5)
- MultiProfileDialog for overlay forward modeling
- Internalized `hvstrip_progressive` core into `HV_Strip_Progressive`

## [1.0.0] — 2025-01-01

### Added
- Initial release with CLI and Python API
- Progressive layer stripping engine
- HV forward modeling via HVf (HV-INV) solver
- Adaptive frequency scanning
- Comprehensive report generation (CSV, text, PDF, publication figures)
- Batch workflow support
