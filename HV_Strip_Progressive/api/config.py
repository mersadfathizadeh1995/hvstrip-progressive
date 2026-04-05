"""
Configuration dataclasses for HV Strip Progressive API.

Every GUI-scattered parameter is captured here as a typed, discoverable
dataclass with sensible defaults drawn from the core modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import copy
import json


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Forward-modelling engine selection and parameters.

    ``name`` selects which engine to use.  Engine-specific fields only
    take effect when the matching engine is active.
    """

    name: str = "diffuse_field"
    """Engine identifier: ``"diffuse_field"`` | ``"ellipticity"`` | ``"sh_wave"``."""

    # --- DiffuseField (HVf.exe) ---
    exe_path: str = ""
    """Path to HVf.exe.  Auto-detected when empty."""
    nmr: int = 10
    nml: int = 10
    nks: int = 10

    # --- Ellipticity (gpell.exe via Git Bash) ---
    gpell_path: str = ""
    """Path to gpell.exe (Geopsy).  Auto-detected when empty."""
    git_bash_path: str = ""
    """Path to git-bash.exe.  Auto-detected when empty."""
    n_modes: int = 1
    """Number of Rayleigh modes to compute."""
    sampling: str = "log"
    """Frequency sampling: ``"log"`` | ``"linear"`` | ``"period"``."""
    absolute: bool = True
    """Return absolute ellipticity values."""
    peak_refinement: bool = False
    """Use peak-corrected (``-pc``) curves."""
    love_alpha: float = 0.0
    """Love-wave mixing coefficient (0 – 0.99)."""
    auto_q: bool = False
    """Auto-generate Q (quality) factors from Vs."""
    q_formula: str = "default"
    """Q formula: ``"default"`` | ``"brocher"`` | ``"constant"``."""
    clip_factor: float = 50.0
    """Amplitude clipping threshold for ellipticity."""
    timeout: int = 30
    """Subprocess timeout in seconds for gpell."""

    # --- SH Wave (pure Python) ---
    Dsoil: Optional[float] = None
    """Soil damping ratio [%].  ``None`` → auto via Darendeli (2001)."""
    Drock: float = 0.5
    """Half-space damping ratio [%]."""
    d_tf: Any = 0
    """Reference depth: ``0`` = bedrock outcrop, ``'within'`` = top of
    half-space, or a float depth in metres."""
    darendeli_curvetype: int = 1
    """Darendeli curve: 1 = mean, 2 = mean+1σ, 3 = mean−1σ."""
    gamma_max: float = 23.0
    """Maximum unit weight [kN/m³] for SH wave computation."""
    f0_search_fmin: Optional[float] = None
    """Override lower bound for peak search in SH engine."""
    f0_search_fmax: Optional[float] = None
    """Override upper bound for peak search in SH engine."""
    clip_tf: float = 0.0
    """Transfer-function amplitude clipping (0 = disabled)."""

    def to_core_config(self) -> Dict[str, Any]:
        """Convert to the dict expected by the active core engine."""
        if self.name == "diffuse_field":
            cfg: Dict[str, Any] = {
                "fmin": None,   # filled from FrequencyConfig
                "fmax": None,
                "nf": None,
                "nmr": self.nmr,
                "nml": self.nml,
                "nks": self.nks,
            }
            if self.exe_path:
                cfg["exe_path"] = self.exe_path
            return cfg

        if self.name == "ellipticity":
            cfg = {
                "fmin": None,
                "fmax": None,
                "n_samples": None,
                "n_modes": self.n_modes,
                "sampling": self.sampling,
                "absolute": self.absolute,
                "peak_refinement": self.peak_refinement,
                "love_alpha": self.love_alpha,
                "auto_q": self.auto_q,
                "q_formula": self.q_formula,
                "clip_factor": self.clip_factor,
                "timeout": self.timeout,
            }
            if self.gpell_path:
                cfg["gpell_path"] = self.gpell_path
            if self.git_bash_path:
                cfg["git_bash_path"] = self.git_bash_path
            return cfg

        if self.name == "sh_wave":
            return {
                "fmin": None,
                "fmax": None,
                "n_samples": None,
                "sampling": self.sampling,
                "Dsoil": self.Dsoil,
                "Drock": self.Drock,
                "d_tf": self.d_tf,
                "darendeli_curvetype": self.darendeli_curvetype,
                "gamma_max": self.gamma_max,
                "f0_search_fmin": self.f0_search_fmin,
                "f0_search_fmax": self.f0_search_fmax,
                "clip_tf": self.clip_tf,
            }

        return {}


# ---------------------------------------------------------------------------
# Frequency configuration
# ---------------------------------------------------------------------------

@dataclass
class FrequencyConfig:
    """Frequency-axis settings shared across engines."""

    fmin: float = 0.2
    """Minimum frequency [Hz]."""
    fmax: float = 20.0
    """Maximum frequency [Hz]."""
    nf: int = 71
    """Number of frequency points (used by DiffuseField)."""
    n_samples: int = 500
    """Number of frequency samples (used by Ellipticity / SHWave)."""
    sampling: str = "log"
    """Frequency spacing: ``"log"`` | ``"linear"`` | ``"period"``."""


@dataclass
class AdaptiveConfig:
    """Adaptive frequency scanning (auto-expand if peak near boundary)."""

    enable: bool = True
    max_passes: int = 2
    edge_margin_frac: float = 0.05
    fmax_expand_factor: float = 2.0
    fmin_shrink_factor: float = 0.5
    fmax_limit: float = 60.0
    fmin_limit: float = 0.05


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

PEAK_PRESET_NAMES = [
    "default",
    "forward_modeling",
    "conservative",
    "forward_modeling_sharp",
    "custom",
]


@dataclass
class PeakDetectionConfig:
    """Peak detection parameters.

    When ``preset`` is set to a known preset name the remaining fields
    are overridden by :func:`core.peak_detection.get_peak_detection_preset`.
    Use ``"custom"`` to supply your own values.
    """

    preset: str = "forward_modeling"
    """Preset name (see :data:`PEAK_PRESET_NAMES`)."""
    method: str = "find_peaks"
    """Detection method: ``"find_peaks"`` | ``"max"`` | ``"manual"``."""
    select: str = "leftmost"
    """Selection strategy: ``"leftmost"`` | ``"sharpest"`` |
    ``"leftmost_sharpest"`` | ``"max"``."""
    prominence: float = 0.1
    """Minimum peak prominence for ``find_peaks``."""
    distance: int = 2
    """Minimum sample distance between peaks."""
    freq_min: Optional[float] = 0.3
    """Ignore peaks below this frequency [Hz]."""
    freq_max: Optional[float] = None
    """Ignore peaks above this frequency [Hz]."""
    min_amplitude: Optional[float] = 1.5
    """Minimum H/V amplitude to accept a peak."""
    min_rel_height: float = 0.15
    """Minimum relative height (fraction of global max)."""
    exclude_first_n: int = 1
    """Skip the first *n* frequency bins."""
    check_clarity_ratio: bool = True
    """Verify peak clarity (amplitude at f0/2 and 2·f0)."""
    clarity_ratio_threshold: float = 1.5
    """Clarity ratio threshold."""

    def to_core_config(self) -> Dict[str, Any]:
        """Build the dict expected by ``core.peak_detection.detect_peak``."""
        return {
            "preset": self.preset,
            "method": self.method,
            "select": self.select,
            "find_peaks_params": {
                "prominence": self.prominence,
                "distance": self.distance,
            },
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "min_amplitude": self.min_amplitude,
            "min_rel_height": self.min_rel_height,
            "exclude_first_n": self.exclude_first_n,
            "check_clarity_ratio": self.check_clarity_ratio,
            "clarity_ratio_threshold": self.clarity_ratio_threshold,
        }


@dataclass
class AutoPeakConfig:
    """Auto-peak settings for multi-peak detection in forward-multiple mode."""

    enabled: bool = True
    n_secondary: int = 2
    """Number of secondary peaks to detect beyond f0."""
    f0_range: tuple = (0.1, 50.0)
    f1_range: tuple = (0.1, 50.0)
    f2_range: tuple = (0.1, 50.0)
    min_prominence: float = 0.1
    min_amplitude: float = 1.5


# ---------------------------------------------------------------------------
# Post-processing & plotting
# ---------------------------------------------------------------------------

@dataclass
class SmoothingConfig:
    """Savitzky–Golay smoothing for HV curves."""

    enable: bool = True
    window_length: int = 7
    """Must be odd."""
    poly_order: int = 3


@dataclass
class HVPlotConfig:
    """HV-curve plot style."""

    x_axis_scale: str = "log"
    y_axis_scale: str = "log"
    y_compression: float = 1.5
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    show_bands: bool = True
    freq_window_mode: str = "relative"
    freq_window_left: float = 0.3
    freq_window_right: float = 3.0
    abs_freq_min: float = 0.1
    abs_freq_max: float = 50.0
    figure_width: int = 12
    figure_height: int = 6
    dpi: int = 200


@dataclass
class VsPlotConfig:
    """Velocity-profile plot style."""

    show: bool = True
    annotate_deepest: bool = True
    annotate_max_vs: bool = True
    annotate_f0: bool = True
    figure_width: int = 6
    figure_height: int = 8
    dpi: int = 200


@dataclass
class OutputFileConfig:
    """Per-step output file naming."""

    save_separate: bool = True
    save_combined: bool = True
    hv_filename: str = "hv_curve.png"
    vs_filename: str = "vs_profile.png"
    combined_filename: str = "combined_figure.png"
    summary_filename: str = "step_summary.csv"


@dataclass
class PostProcessConfig:
    """Post-processing settings (wraps hv_postprocess config)."""

    peak_detection: PeakDetectionConfig = field(
        default_factory=PeakDetectionConfig
    )
    hv_plot: HVPlotConfig = field(default_factory=HVPlotConfig)
    vs_plot: VsPlotConfig = field(default_factory=VsPlotConfig)
    output_files: OutputFileConfig = field(default_factory=OutputFileConfig)

    def to_core_config(self) -> Dict[str, Any]:
        """Build the nested dict expected by ``core.hv_postprocess.process``."""
        return {
            "peak_detection": self.peak_detection.to_core_config(),
            "hv_plot": {
                "x_axis_scale": self.hv_plot.x_axis_scale,
                "y_axis_scale": self.hv_plot.y_axis_scale,
                "y_compression": self.hv_plot.y_compression,
                "smoothing": {
                    "enable": self.hv_plot.smoothing.enable,
                    "window_length": self.hv_plot.smoothing.window_length,
                    "poly_order": self.hv_plot.smoothing.poly_order,
                },
                "show_bands": self.hv_plot.show_bands,
                "freq_window_mode": self.hv_plot.freq_window_mode,
                "freq_window_left": self.hv_plot.freq_window_left,
                "freq_window_right": self.hv_plot.freq_window_right,
                "abs_freq_min": self.hv_plot.abs_freq_min,
                "abs_freq_max": self.hv_plot.abs_freq_max,
                "figure_width": self.hv_plot.figure_width,
                "figure_height": self.hv_plot.figure_height,
                "dpi": self.hv_plot.dpi,
            },
            "vs_plot": {
                "show": self.vs_plot.show,
                "annotate_deepest": self.vs_plot.annotate_deepest,
                "annotate_max_vs": self.vs_plot.annotate_max_vs,
                "annotate_f0": self.vs_plot.annotate_f0,
                "figure_width": self.vs_plot.figure_width,
                "figure_height": self.vs_plot.figure_height,
                "dpi": self.vs_plot.dpi,
            },
            "output": {
                "save_separate": self.output_files.save_separate,
                "save_combined": self.output_files.save_combined,
                "hv_filename": self.output_files.hv_filename,
                "vs_filename": self.output_files.vs_filename,
                "combined_filename": self.output_files.combined_filename,
                "summary_filename": self.output_files.summary_filename,
            },
        }


# ---------------------------------------------------------------------------
# Strip / Dual-resonance / Report
# ---------------------------------------------------------------------------

@dataclass
class StripConfig:
    """Progressive-stripping workflow configuration."""

    output_folder_name: str = "strip"
    """Sub-folder name for stripped models."""
    engine: EngineConfig = field(default_factory=EngineConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    generate_report: bool = True
    interactive_mode: bool = False


@dataclass
class DualResonanceConfig:
    """Settings for dual-resonance (f0 / f1) extraction."""

    enabled: bool = True
    separation_ratio_threshold: float = 1.2
    """Minimum f1/f0 ratio to accept separation."""
    separation_shift_threshold: float = 0.3
    """Minimum absolute frequency shift [Hz] between steps."""


@dataclass
class ReportConfig:
    """Report / figure generation settings."""

    figure_types: List[str] = field(default_factory=lambda: [
        "hv_overlay",
        "peak_evolution",
        "interface_analysis",
        "waterfall",
        "publication",
    ])
    """Figure types to generate.  Available: ``hv_overlay``,
    ``peak_evolution``, ``interface_analysis``, ``waterfall``,
    ``comprehensive``, ``publication``, ``dual_resonance``."""
    dpi: int = 300
    format: str = "png"
    """Image format: ``"png"`` | ``"pdf"`` | ``"svg"``."""
    generate_text_report: bool = True
    generate_pdf_report: bool = False
    generate_metadata: bool = True


# ---------------------------------------------------------------------------
# Output / Export
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    """Top-level output settings."""

    output_dir: str = ""
    save_csv: bool = True
    save_json: bool = True
    save_mat: bool = False
    save_excel: bool = False
    save_png: bool = True
    save_pdf: bool = False


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """Batch-stripping configuration for multiple profiles."""

    strip: StripConfig = field(default_factory=StripConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    site_name: str = ""
    profiles: List[str] = field(default_factory=list)
    """List of absolute profile file paths."""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class HVStripConfig:
    """Master configuration combining all sub-configs.

    This is the single object that :class:`~.analysis.HVStripAnalysis`
    holds and exposes for modification.
    """

    engine: EngineConfig = field(default_factory=EngineConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    peak_detection: PeakDetectionConfig = field(
        default_factory=PeakDetectionConfig
    )
    auto_peak: AutoPeakConfig = field(default_factory=AutoPeakConfig)
    strip: StripConfig = field(default_factory=StripConfig)
    dual_resonance: DualResonanceConfig = field(
        default_factory=DualResonanceConfig
    )
    report: ReportConfig = field(default_factory=ReportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HVStripConfig":
        """Construct from a (possibly partial) dict.

        Missing keys keep their defaults.
        """
        cfg = cls()
        _apply_dict(cfg, d)
        return cfg

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "HVStripConfig":
        return cls.from_dict(json.loads(json_str))

    def copy(self) -> "HVStripConfig":
        return copy.deepcopy(self)

    def build_workflow_config(self) -> Dict[str, Any]:
        """Build the nested dict expected by
        ``core.batch_workflow.run_complete_workflow``.
        """
        engine_cfg = self.engine.to_core_config()
        # Inject frequency params
        engine_cfg["fmin"] = self.frequency.fmin
        engine_cfg["fmax"] = self.frequency.fmax
        if self.engine.name == "diffuse_field":
            engine_cfg["nf"] = self.frequency.nf
        else:
            engine_cfg["n_samples"] = self.frequency.n_samples

        return {
            "stripper": {
                "output_folder_name": self.strip.output_folder_name,
            },
            "hv_forward": engine_cfg,
            "hv_postprocess": self.postprocess.to_core_config(),
            "engine_name": self.engine.name,
            "dual_resonance": {
                "enable": self.dual_resonance.enabled,
                "separation_ratio_threshold": (
                    self.dual_resonance.separation_ratio_threshold
                ),
                "separation_shift_threshold": (
                    self.dual_resonance.separation_shift_threshold
                ),
            },
            "generate_report": self.strip.generate_report,
            "interactive_mode": self.strip.interactive_mode,
        }

    def build_engine_config(self) -> Dict[str, Any]:
        """Build just the engine dict with frequency params injected."""
        cfg = self.engine.to_core_config()
        cfg["fmin"] = self.frequency.fmin
        cfg["fmax"] = self.frequency.fmax
        if self.engine.name == "diffuse_field":
            cfg["nf"] = self.frequency.nf
        else:
            cfg["n_samples"] = self.frequency.n_samples
        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_dict(obj: Any, d: Dict[str, Any]) -> None:
    """Recursively apply *d* onto dataclass *obj*, keeping defaults for
    missing keys.
    """
    if not isinstance(d, dict):
        return
    for key, value in d.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(obj, key, value)
