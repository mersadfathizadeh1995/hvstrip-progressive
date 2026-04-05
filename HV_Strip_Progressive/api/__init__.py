"""
HV Strip Progressive — API Layer
=================================

Config-driven, headless API wrapping the core computation modules.
Consumed by both the PyQt5 GUI and MCP server.

Main entry point: :class:`HVStripAnalysis` in :mod:`.analysis`.
"""

from .config import (
    EngineConfig,
    FrequencyConfig,
    PeakDetectionConfig,
    AutoPeakConfig,
    StripConfig,
    DualResonanceConfig,
    ReportConfig,
    OutputConfig,
    BatchConfig,
    HVStripConfig,
    PostProcessConfig,
    HVPlotConfig,
    VsPlotConfig,
    AdaptiveConfig,
    OutputFileConfig,
    SmoothingConfig,
)
from .analysis import HVStripAnalysis
from .forward_engine import ForwardResult, MultiForwardResult, PeakInfo
from .strip_engine import StripResult, StepResult
from .batch_engine import BatchStripResult, ProfileStripResult

__all__ = [
    # Orchestrator
    "HVStripAnalysis",
    # Configs
    "EngineConfig",
    "FrequencyConfig",
    "PeakDetectionConfig",
    "AutoPeakConfig",
    "StripConfig",
    "DualResonanceConfig",
    "ReportConfig",
    "OutputConfig",
    "OutputFileConfig",
    "BatchConfig",
    "HVStripConfig",
    "PostProcessConfig",
    "HVPlotConfig",
    "VsPlotConfig",
    "AdaptiveConfig",
    "SmoothingConfig",
    # Result types
    "ForwardResult",
    "MultiForwardResult",
    "PeakInfo",
    "StripResult",
    "StepResult",
    "BatchStripResult",
    "ProfileStripResult",
]
