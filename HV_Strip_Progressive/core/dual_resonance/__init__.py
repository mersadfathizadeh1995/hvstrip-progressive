"""
Dual-resonance (two-resonance) analysis for HVSR progressive stripping.

Extracts deep (f0) and shallow (f1) resonance frequencies from stripping
step results.  Works for both single-profile and batch workflows.

This package replaces the former ``core/dual_resonance.py`` module.
All public names are re-exported here for backward compatibility.

Typical usage::

    from HV_Strip_Progressive.core.dual_resonance import (
        extract_dual_resonance,
        compute_batch_statistics,
    )

    dr = extract_dual_resonance(strip_dir)
    stats = compute_batch_statistics([dr1, dr2, ...])
"""

from .data_structures import (
    DualResonanceResult,
    BatchDualResonanceStats,
)
from .extraction import (
    theoretical_frequency,
    extract_dual_resonance,
    discover_step_folders,
    SEPARATION_RATIO_THRESHOLD,
    SEPARATION_SHIFT_THRESHOLD,
)
from .statistics import compute_batch_statistics
from .io_helpers import save_results_csv, save_statistics_json


__all__ = [
    "DualResonanceResult",
    "BatchDualResonanceStats",
    "theoretical_frequency",
    "extract_dual_resonance",
    "discover_step_folders",
    "compute_batch_statistics",
    "save_results_csv",
    "save_statistics_json",
    "SEPARATION_RATIO_THRESHOLD",
    "SEPARATION_SHIFT_THRESHOLD",
]
