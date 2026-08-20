"""
Configuration dataclasses for HV Strip Progressive API.

Every GUI-scattered parameter is captured here as a typed, discoverable
dataclass with sensible defaults drawn from the core modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields as dc_fields
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
    n_samples: int = 512
    """Number of frequency samples (used by Ellipticity / SHWave).

    512 matches core's ``SHTFConfig`` default AND the legacy GUI — the api
    previously said 500, which silently shifted the frequency grid (and the
    detected peak) whenever the api path was used.  Aligned 2026-07-09.
    """
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
    width: Optional[float] = None
    """Minimum peak width in samples for ``find_peaks`` (``None`` = off —
    the legacy Advanced dialog's Width knob)."""
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
        params: Dict[str, Any] = {
            "prominence": self.prominence,
            "distance": self.distance,
        }
        if self.width is not None:      # only when set — behavior-preserving
            params["width"] = self.width
        return {
            "preset": self.preset,
            "method": self.method,
            "select": self.select,
            "find_peaks_params": params,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "min_amplitude": self.min_amplitude,
            "min_rel_height": self.min_rel_height,
            "exclude_first_n": self.exclude_first_n,
            "check_clarity_ratio": self.check_clarity_ratio,
            "clarity_ratio_threshold": self.clarity_ratio_threshold,
        }


#: The three auto-peak strategies (the legacy dialog's strategy combo).
AUTO_PEAK_STRATEGIES = ("range_constrained", "preset", "advanced")


@dataclass
class AutoPeakConfig:
    """Auto-peak settings for multi-peak detection in forward-multiple mode.

    Track 2 adds the legacy dialog's full three-strategy surface:

    * ``"range_constrained"`` — one peak per armed frequency band
      (``ranges``; the FIRST band is f0's);
    * ``"preset"`` — delegate to a :data:`PEAK_PRESET_NAMES` preset;
    * ``"advanced"`` — the fully-custom :class:`PeakDetectionConfig` params.
    """

    enabled: bool = True
    strategy: str = "preset"
    """One of :data:`AUTO_PEAK_STRATEGIES`."""
    n_secondary: int = 2
    """Number of secondary peaks to detect beyond f0."""
    ranges: List[Dict[str, Any]] = field(default_factory=list)
    """Per-peak bands for ``range_constrained``: plain dicts
    ``{"fmin": float, "fmax": float, "use": bool}`` (JSON-shaped so the
    config funnel round-trips them exactly).  Index 0 = the f0 band."""
    f0_range: tuple = (0.1, 50.0)
    f1_range: tuple = (0.1, 50.0)
    f2_range: tuple = (0.1, 50.0)
    min_prominence: float = 0.1
    min_amplitude: float = 1.5

    def effective_ranges(self) -> List[tuple]:
        """The ARMED (fmin, fmax) bands, f0's first.

        ``ranges`` wins when non-empty; otherwise the legacy triple
        ``f0_range``/``f1_range``/``f2_range`` (capped by ``n_secondary``).
        """
        if self.ranges:
            return [(float(r["fmin"]), float(r["fmax"]))
                    for r in self.ranges if r.get("use", True)]
        legacy = [self.f0_range, self.f1_range, self.f2_range]
        return [tuple(r) for r in legacy[: 1 + max(0, self.n_secondary)]]


@dataclass
class MarkerStyleConfig:
    """Peak marker + annotation style for the interactive HV figures.

    Drives the mpl figure and the Properties rail (spec 002 FR-6);
    presentation-only — never touches detection or compute.
    """

    show_markers: bool = True
    show_annotations: bool = True
    f0_shape: str = "*"
    """Matplotlib marker for the primary peak."""
    f0_size: float = 14.0
    secondary_shape: str = "*"
    secondary_size: float = 10.0
    annotation_fontsize: int = 9


@dataclass
class ResearchStudyConfig:
    """Persisted Research-tool study settings (spec 002 FR-14 — these
    previously lived only in widgets and reset every launch)."""

    soilgen_path: str = ""
    profiles_dir: str = ""
    n_random: int = 0
    n_per_scenario: int = 2
    seed: int = 42
    engines: List[str] = field(default_factory=lambda: [
        "diffuse_field", "sh_wave", "ellipticity"])
    fmin: float = 0.1
    fmax: float = 30.0
    nf: int = 500
    output_dir: str = ""


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
    step_pair: Optional[tuple] = None
    """Which step indices to compare as (deep, shallow). None → (0, 1)."""
    user_peaks: Optional[Dict[str, tuple]] = None
    """Step name → (freq_Hz, amplitude) overrides from the wizard."""


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
    markers: MarkerStyleConfig = field(default_factory=MarkerStyleConfig)
    research: ResearchStudyConfig = field(
        default_factory=ResearchStudyConfig)

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to a plain dict (JSON-safe).

        Carries ``config_version`` so loaders can distinguish dataclass
        payloads (v2) from the retired legacy GUI dict shape.
        """
        d = asdict(self)
        d["config_version"] = CONFIG_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HVStripConfig":
        """Construct from a (possibly partial) dict.

        Missing keys keep their defaults.  Accepts BOTH shapes and routes:

        * ``config_version >= 2`` → the v2 dataclass payload, applied
          directly (explicit version always wins);
        * any legacy-only top-level key (``engine_settings``,
          ``hv_forward``, …) → :meth:`from_legacy_gui_dict` (best-effort,
          unmapped leaves logged);
        * otherwise a dict whose top-level keys all belong to the v2
          field set — including PARTIAL dicts like ``{"engine": …,
          "strip": …}`` — applies directly as v2 (the overlap keys
          ``engine``/``dual_resonance``/``peak_detection`` alone are
          treated as v2; real legacy payloads always carry a
          legacy-only key);
        * anything else falls back to the legacy migrator, whose
          unmapped-leaf log is the "never silently dropped" contract.
        """
        if not isinstance(d, dict):
            return cls()
        keys = set(d) - {"config_version"}
        v2_fields = {f.name for f in dc_fields(cls)}
        is_v2 = (
            d.get("config_version", 0) >= CONFIG_VERSION
            or (not (keys & _LEGACY_ONLY_KEYS)
                and (_looks_v2(d) or keys <= v2_fields))
        )
        if is_v2:
            cfg = cls()
            unmapped: List[str] = []
            _apply_dict(cfg, {k: v for k, v in d.items()
                              if k != "config_version"}, unmapped=unmapped)
            if unmapped:
                import logging

                logging.getLogger(__name__).warning(
                    "HVStripConfig.from_dict: %d unmapped v2 key(s) "
                    "ignored: %s", len(unmapped), unmapped)
            return cfg
        return cls.from_legacy_gui_dict(d)

    @classmethod
    def from_legacy_gui_dict(cls, d: Dict[str, Any]) -> "HVStripConfig":
        """Best-effort migration of the RETIRED legacy GUI nested dict
        (``gui/strip_window._get_default_config`` shape; fixture:
        ``tests/golden/legacy_gui_config.json``) onto the dataclasses.

        Legacy values win over defaults.  Unknown leaves are logged (never
        silently dropped) via the module logger.
        """
        import logging

        log = logging.getLogger(__name__)
        cfg = cls()
        if not isinstance(d, dict):
            return cfg
        unmapped: list = []

        # -- engine selection + per-engine settings (fields are disjoint) --
        name = (
            (d.get("engine") or {}).get("name")
            or d.get("engine_name")
            or cfg.engine.name
        )
        cfg.engine.name = str(name)
        freq_sources: Dict[str, Any] = {}
        for eng_name, eng_cfg in (d.get("engine_settings") or {}).items():
            if not isinstance(eng_cfg, dict):
                continue
            for key, value in eng_cfg.items():
                if key in ("fmin", "fmax", "nf", "n_samples"):
                    if eng_name == cfg.engine.name:
                        freq_sources[key] = value
                elif key == "sampling":
                    # `sampling` is both an engine field and a frequency
                    # field; keep the engine's own value.
                    if eng_name == cfg.engine.name:
                        cfg.engine.sampling = value
                        cfg.frequency.sampling = value
                elif hasattr(cfg.engine, key):
                    setattr(cfg.engine, key, value)
                else:
                    unmapped.append(f"engine_settings.{eng_name}.{key}")

        # -- the active-engine core config (hv_forward) -----------------
        hv_fwd = dict(d.get("hv_forward") or {})
        adaptive = hv_fwd.pop("adaptive", None)
        if isinstance(adaptive, dict):
            _apply_dict(cfg.adaptive, adaptive)
        for key, value in hv_fwd.items():
            if key in ("fmin", "fmax", "nf", "n_samples"):
                freq_sources.setdefault(key, value)
            elif hasattr(cfg.engine, key):
                setattr(cfg.engine, key, value)
            else:
                unmapped.append(f"hv_forward.{key}")
        for key, value in freq_sources.items():
            setattr(cfg.frequency, key, value)

        # -- post-processing (renames: output → output_files) -----------
        post = dict(d.get("hv_postprocess") or {})
        out_files = post.pop("output", None)
        if isinstance(out_files, dict):
            _apply_dict(cfg.postprocess.output_files, out_files)
        _apply_dict(cfg.postprocess, post)

        # -- peak detection (top level mirrors into postprocess too) ----
        peaks = d.get("peak_detection") or {}
        if isinstance(peaks, dict):
            _apply_dict(cfg.peak_detection, peaks)
            _apply_dict(cfg.postprocess.peak_detection, peaks)

        # -- dual resonance (rename: enable → enabled) -------------------
        dual = dict(d.get("dual_resonance") or {})
        if "enable" in dual:
            cfg.dual_resonance.enabled = bool(dual.pop("enable"))
        _apply_dict(cfg.dual_resonance, dual)

        # -- workflow flags ----------------------------------------------
        if "generate_report" in d:
            cfg.strip.generate_report = bool(d["generate_report"])
        if "interactive_mode" in d:
            cfg.strip.interactive_mode = bool(d["interactive_mode"])

        # -- global plot hints (best-effort onto the HV plot) ------------
        plot = d.get("plot") or {}
        if isinstance(plot, dict):
            _apply_dict(cfg.postprocess.hv_plot, plot)

        handled = {
            "engine", "engine_name", "engine_settings", "hv_forward",
            "hv_postprocess", "peak_detection", "dual_resonance",
            "generate_report", "interactive_mode", "plot", "config_version",
        }
        unmapped.extend(k for k in d.keys() if k not in handled)
        if unmapped:
            log.info("Legacy config migration: unmapped keys %s", unmapped)
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

#: Serialised-config schema version.  v2 = the dataclass payload
#: (``HVStripConfig.to_dict()``); anything without the key is the legacy GUI
#: dict and goes through ``from_legacy_gui_dict``.
CONFIG_VERSION = 2

#: Top-level keys unique to the v2 dataclass payload (recognition heuristic
#: for payloads saved by ``to_dict()`` before the version key existed).
_V2_MARKER_KEYS = {"frequency", "auto_peak", "postprocess", "adaptive"}

#: Top-level keys that exist ONLY in the legacy GUI dict shape
#: (``tests/golden/legacy_gui_config.json``).  Any of these routes the
#: payload to ``from_legacy_gui_dict`` — the overlap keys
#: (``engine``/``dual_resonance``/``peak_detection``) deliberately do NOT.
_LEGACY_ONLY_KEYS = {
    "engine_name", "engine_settings", "generate_report", "hv_forward",
    "hv_postprocess", "interactive_mode", "plot",
}


def _looks_v2(d: Dict[str, Any]) -> bool:
    return bool(_V2_MARKER_KEYS & set(d.keys()))


def _apply_dict(obj: Any, d: Dict[str, Any],
                unmapped: Optional[List[str]] = None,
                _prefix: str = "") -> None:
    """Recursively apply *d* onto dataclass *obj*, keeping defaults for
    missing keys.  Keys with no matching attribute are collected into
    *unmapped* (dotted paths) when a list is passed, so callers can log
    them — never silently dropped without a trace.
    """
    if not isinstance(d, dict):
        return
    for key, value in d.items():
        if not hasattr(obj, key):
            if unmapped is not None:
                unmapped.append(f"{_prefix}{key}")
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value, unmapped, f"{_prefix}{key}.")
        else:
            setattr(obj, key, value)
