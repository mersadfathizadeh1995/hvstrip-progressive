"""
Research config — ComparisonStudyConfig dataclass.

Central configuration for the comparative forward modeling study.
All parameters controlling profile generation, engine runs,
metrics computation, and report output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProfileSuiteConfig:
    """Configuration for synthetic profile generation."""

    # SoilGen scenarios to include
    scenarios: List[str] = field(default_factory=lambda: [
        "gradual_increase",
        "sharp_contrast",
        "velocity_inversion",
        "shallow_bedrock",
        "thick_soft_deposit",
        "thick_stiff_layer",
    ])
    n_random: int = 20
    n_per_scenario: int = 15
    seed: int = 42

    # Depth/Vs ranges for random profiles
    min_depth: float = 5.0
    max_depth: float = 100.0
    min_vs: float = 80.0
    max_vs: float = 800.0
    min_layers: int = 3
    max_layers: int = 12

    # SoilGen package path (set to None to use installed package)
    soilgen_path: Optional[str] = None


@dataclass
class EngineRunConfig:
    """Configuration for engine comparison runs."""

    engines: List[str] = field(default_factory=lambda: [
        "diffuse_field",
        "sh_wave",
        "ellipticity",
    ])

    # Frequency settings (shared across engines)
    fmin: float = 0.1
    fmax: float = 30.0
    n_frequencies: int = 500

    # Per-engine overrides (engine_name → dict of params)
    engine_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Peak detection
    peak_prominence: float = 0.1
    peak_min_amplitude: float = 1.5
    n_peaks: int = 5


@dataclass
class MetricsConfig:
    """Configuration for comparison metrics."""

    # Peak frequency comparison
    freq_tolerance_hz: float = 0.5
    freq_tolerance_ratio: float = 0.15

    # Curve shape metrics
    freq_range_for_rmse: tuple = (0.2, 20.0)
    normalize_curves: bool = True

    # Statistical thresholds
    agreement_threshold: float = 0.85
    strong_agreement_threshold: float = 0.95


@dataclass
class VisualizationConfig:
    """Configuration for figure generation."""

    dpi: int = 300
    figure_format: str = "png"
    style: str = "publication"
    figsize_single: tuple = (8, 6)
    figsize_comparison: tuple = (14, 10)
    figsize_panel: tuple = (18, 14)
    colormap: str = "Set2"
    engine_colors: Dict[str, str] = field(default_factory=lambda: {
        "diffuse_field": "#2196F3",
        "sh_wave": "#4CAF50",
        "ellipticity": "#FF9800",
    })
    engine_labels: Dict[str, str] = field(default_factory=lambda: {
        "diffuse_field": "Diffuse Field (DFA)",
        "sh_wave": "SH Transfer Function",
        "ellipticity": "Rayleigh Ellipticity",
    })


@dataclass
class OutputConfig:
    """Configuration for report output."""

    output_dir: str = "research_output"
    save_csv: bool = True
    save_json: bool = True
    save_latex: bool = True
    save_figures: bool = True


@dataclass
class FieldSiteConfig:
    """Configuration for a field validation site."""

    name: str = ""
    profile_path: str = ""
    description: str = ""
    known_f0: Optional[float] = None
    known_f1: Optional[float] = None
    measured_hvsr_path: Optional[str] = None
    site_class: Optional[str] = None


@dataclass
class ComparisonStudyConfig:
    """Top-level configuration for the comparison study.

    Combines all sub-configs and provides serialization.
    """

    study_name: str = "HVSR Forward Modeling Comparison"
    profiles: ProfileSuiteConfig = field(default_factory=ProfileSuiteConfig)
    engines: EngineRunConfig = field(default_factory=EngineRunConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    field_sites: List[FieldSiteConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        def _dc_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _dc_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [_dc_to_dict(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _dc_to_dict(v) for k, v in obj.items()}
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        return _dc_to_dict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComparisonStudyConfig":
        cfg = cls()
        for key, val in d.items():
            if key == "profiles" and isinstance(val, dict):
                cfg.profiles = ProfileSuiteConfig(**{
                    k: v for k, v in val.items()
                    if k in ProfileSuiteConfig.__dataclass_fields__
                })
            elif key == "engines" and isinstance(val, dict):
                cfg.engines = EngineRunConfig(**{
                    k: (tuple(v) if k == "freq_range_for_rmse" and isinstance(v, list) else v)
                    for k, v in val.items()
                    if k in EngineRunConfig.__dataclass_fields__
                })
            elif key == "metrics" and isinstance(val, dict):
                cfg.metrics = MetricsConfig(**{
                    k: (tuple(v) if isinstance(v, list) and k.endswith("for_rmse") else v)
                    for k, v in val.items()
                    if k in MetricsConfig.__dataclass_fields__
                })
            elif key == "visualization" and isinstance(val, dict):
                cfg.visualization = VisualizationConfig(**{
                    k: (tuple(v) if isinstance(v, list) and k.startswith("figsize") else v)
                    for k, v in val.items()
                    if k in VisualizationConfig.__dataclass_fields__
                })
            elif key == "output" and isinstance(val, dict):
                cfg.output = OutputConfig(**{
                    k: v for k, v in val.items()
                    if k in OutputConfig.__dataclass_fields__
                })
            elif key == "field_sites" and isinstance(val, list):
                cfg.field_sites = [
                    FieldSiteConfig(**{
                        k: v for k, v in site.items()
                        if k in FieldSiteConfig.__dataclass_fields__
                    }) for site in val
                ]
            elif key == "study_name":
                cfg.study_name = val
        return cfg

    @classmethod
    def from_json(cls, json_str: str) -> "ComparisonStudyConfig":
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ComparisonStudyConfig":
        with open(path) as f:
            return cls.from_json(f.read())
