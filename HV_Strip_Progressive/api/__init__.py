"""
HV Strip Progressive — API Layer
=================================

Config-driven, headless API wrapping the core computation modules — THE one
facade every consumer (the GUI, research/, future CLI/MCP) talks to.

Main entry point: :class:`HVStripAnalysis` in :mod:`.analysis`.
"""

from .config import (
    CONFIG_VERSION,
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


def preload_heavy_modules() -> None:
    """Import the heavy compute/plot stack (matplotlib + the core
    postprocess/report modules) NOW, on the calling thread.

    GUI consumers call this once from the MAIN thread before submitting
    long ops to worker threads: a worker thread FIRST-importing heavy
    native extensions while another worker touches the scipy/sklearn stack
    hard-aborts the process on Windows.  Idempotent and cheap after the
    first call.
    """
    import matplotlib.pyplot  # noqa: F401

    from ..core import hv_postprocess, report_generator  # noqa: F401
from .forward_engine import ForwardResult, MultiForwardResult, PeakInfo
from .strip_engine import StripResult, StepResult
from .batch_engine import BatchStripResult, ProfileStripResult
from .session_io import load_config_payload

__all__ = [
    # Orchestrator
    "HVStripAnalysis",
    "preload_heavy_modules",
    # Config funnel
    "CONFIG_VERSION",
    "load_config_payload",
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
