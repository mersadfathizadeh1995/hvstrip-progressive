"""
Core modules for HVSR progressive layer stripping analysis.

This package contains the main analysis modules:
- stripper: Progressive layer removal
- hv_forward: HVSR curve computation
- peak_detection: Peak detection algorithms and presets
- hv_postprocess: Visualization and analysis
- batch_workflow: Complete workflow orchestration
- report_generator: Comprehensive scientific reports
- advanced_analysis: Statistical analysis and controlling interface detection
- velocity_utils: Vp/Vs/nu conversion utilities
- soil_profile: Soil profile data structures and I/O
- dual_resonance: Two-resonance (f0/f1) extraction and batch statistics
"""

from . import (
    stripper,
    hv_forward,
    peak_detection,
    hv_postprocess,
    batch_workflow,
    dual_resonance,
    report_generator,
    advanced_analysis,
    velocity_utils,
    soil_profile,
)

from .velocity_utils import VelocityConverter
from .soil_profile import Layer, SoilProfile

__all__ = [
    "stripper",
    "hv_forward",
    "peak_detection",
    "hv_postprocess",
    "batch_workflow",
    "dual_resonance",
    "report_generator",
    "advanced_analysis",
    "velocity_utils",
    "soil_profile",
    "VelocityConverter",
    "Layer",
    "SoilProfile",
]
