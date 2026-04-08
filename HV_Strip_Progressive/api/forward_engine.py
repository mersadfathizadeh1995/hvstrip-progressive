"""
Forward Engine — single and multi-profile HV curve computation.

Wraps :func:`core.hv_forward.compute_hv_curve` and the engine registry
with typed result dataclasses and auto-peak detection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.engines import registry as engine_registry
from ..core.hv_forward import compute_hv_curve
from ..core.peak_detection import (
    detect_peak as core_detect_peak,
    find_all_peaks as core_find_all_peaks,
)
from ..core.vs_average import vs_average_from_profile

from .config import (
    HVStripConfig,
    EngineConfig,
    FrequencyConfig,
    PeakDetectionConfig,
)
from .profile_io import load_profile, get_profile_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PeakInfo:
    """A single detected peak."""

    frequency: float = 0.0
    amplitude: float = 0.0
    index: int = 0
    label: str = ""
    """Human-readable label, e.g. ``"f0"``, ``"f1"``."""
    source: str = "auto"
    """``"auto"`` or ``"manual"``."""


@dataclass
class ForwardResult:
    """Result of a single forward HV computation."""

    profile_name: str = ""
    profile_path: str = ""
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    peaks: List[PeakInfo] = field(default_factory=list)
    profile_summary: Dict[str, Any] = field(default_factory=dict)
    engine_name: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "profile_path": self.profile_path,
            "frequencies": self.frequencies.tolist() if len(self.frequencies) else [],
            "amplitudes": self.amplitudes.tolist() if len(self.amplitudes) else [],
            "peaks": [
                {
                    "frequency": p.frequency,
                    "amplitude": p.amplitude,
                    "index": p.index,
                    "label": p.label,
                    "source": p.source,
                }
                for p in self.peaks
            ],
            "profile_summary": self.profile_summary,
            "engine_name": self.engine_name,
            "elapsed_seconds": self.elapsed_seconds,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class MultiForwardResult:
    """Result of computing HV curves for multiple profiles."""

    results: List[ForwardResult] = field(default_factory=list)
    n_profiles: int = 0
    n_success: int = 0
    n_failed: int = 0
    combined_peak_statistics: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_profiles": self.n_profiles,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "results": [r.to_dict() for r in self.results],
            "combined_peak_statistics": self.combined_peak_statistics,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------


def _build_engine_config(config: HVStripConfig) -> Dict[str, Any]:
    """Merge engine + frequency configs into the dict the core expects."""
    return config.build_engine_config()


def _run_engine(
    model_path: str,
    engine_name: str,
    engine_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run a single engine and return (freqs, amps, metadata)."""
    freqs_list, amps_list = compute_hv_curve(
        model_path, config=engine_config, engine_name=engine_name
    )
    freqs = np.array(freqs_list, dtype=float)
    amps = np.array(amps_list, dtype=float)
    return freqs, amps, {}


def _detect_peaks(
    freqs: np.ndarray,
    amps: np.ndarray,
    peak_cfg: PeakDetectionConfig,
) -> List[PeakInfo]:
    """Detect peaks on a curve and return :class:`PeakInfo` list."""
    core_cfg = peak_cfg.to_core_config()

    # Primary peak
    try:
        f0, a0, idx0 = core_detect_peak(freqs, amps, core_cfg)
    except Exception:
        return []

    peaks = [PeakInfo(
        frequency=float(f0),
        amplitude=float(a0),
        index=int(idx0),
        label="f0",
        source="auto",
    )]

    # Secondary peaks
    try:
        all_peaks = core_find_all_peaks(
            freqs,
            amps,
            prominence=peak_cfg.prominence,
            distance=peak_cfg.distance,
            freq_min=peak_cfg.freq_min,
            freq_max=peak_cfg.freq_max,
            min_amplitude=peak_cfg.min_amplitude,
        )
    except Exception:
        all_peaks = []

    for i, (f, a, idx) in enumerate(all_peaks):
        # Skip if same as f0
        if abs(f - f0) / max(f0, 1e-9) < 0.05:
            continue
        peaks.append(PeakInfo(
            frequency=float(f),
            amplitude=float(a),
            index=int(idx),
            label=f"f{len(peaks)}",
            source="auto",
        ))

    return peaks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_forward(
    profile_or_path: Any,
    config: Optional[HVStripConfig] = None,
    engine_name: Optional[str] = None,
    detect_peaks: bool = True,
) -> ForwardResult:
    """Compute a forward HV curve for a single soil profile.

    Parameters
    ----------
    profile_or_path
        A :class:`SoilProfile` instance or a path string to a model file.
    config : HVStripConfig, optional
        Full configuration.  Defaults are used when ``None``.
    engine_name : str, optional
        Override the engine in *config*.
    detect_peaks : bool
        Auto-detect peaks on the computed curve.

    Returns
    -------
    ForwardResult
    """
    if config is None:
        config = HVStripConfig()

    if engine_name:
        config = config.copy()
        config.engine.name = engine_name

    # Resolve profile path
    from ..core.soil_profile import SoilProfile

    import tempfile

    if isinstance(profile_or_path, SoilProfile):
        profile = profile_or_path
        profile_name = profile.name or "unnamed"
        profile_path = ""
    else:
        # Load from file — supports CSV, Excel, HVf, Dinver, etc.
        file_path = str(profile_or_path)
        profile_name = Path(file_path).stem
        profile_path = file_path
        try:
            profile = load_profile(file_path)
        except Exception as exc:
            logger.error("Failed to load profile from %s: %s", file_path, exc)
            return ForwardResult(
                profile_name=profile_name,
                profile_path=profile_path,
                engine_name=config.engine.name if config else "",
                error=f"Failed to load profile: {exc}",
                success=False,
            )

    # Validate profile before running engine
    valid, errors = profile.validate()
    if not valid:
        err_msg = "; ".join(errors)
        logger.error("Profile validation failed for %s: %s", profile_name, err_msg)
        return ForwardResult(
            profile_name=profile_name,
            profile_path=profile_path,
            engine_name=config.engine.name,
            error=f"Profile validation failed: {err_msg}",
            success=False,
        )

    # Write temp HVf file for the engine (all engines expect HVf format)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w"
    )
    tmp.write(profile.to_hvf_format())
    tmp.close()
    model_path = tmp.name

    engine_cfg = _build_engine_config(config)
    t0 = time.perf_counter()

    try:
        freqs, amps, meta = _run_engine(
            model_path, config.engine.name, engine_cfg
        )
        elapsed = time.perf_counter() - t0

        peaks: List[PeakInfo] = []
        if detect_peaks and len(freqs) > 0:
            peaks = _detect_peaks(freqs, amps, config.peak_detection)

        profile_summary = {}
        if profile is not None:
            try:
                summary = get_profile_summary(profile)
                profile_summary = {
                    "n_layers": summary.n_layers,
                    "total_depth": summary.total_depth,
                    "vs30": summary.vs30,
                    "f0_estimate": summary.f0_estimate,
                }
            except Exception:
                pass

        return ForwardResult(
            profile_name=profile_name,
            profile_path=profile_path,
            frequencies=freqs,
            amplitudes=amps,
            peaks=peaks,
            profile_summary=profile_summary,
            engine_name=config.engine.name,
            config_snapshot=engine_cfg,
            elapsed_seconds=round(elapsed, 3),
            success=True,
            metadata=meta,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("Forward computation failed for %s: %s", profile_name, exc)
        return ForwardResult(
            profile_name=profile_name,
            profile_path=profile_path,
            engine_name=config.engine.name,
            config_snapshot=engine_cfg,
            elapsed_seconds=round(elapsed, 3),
            success=False,
            error=str(exc),
        )


def compute_forward_batch(
    profiles: List[Any],
    config: Optional[HVStripConfig] = None,
    detect_peaks: bool = True,
) -> MultiForwardResult:
    """Compute forward HV curves for multiple profiles.

    Parameters
    ----------
    profiles : list
        List of file paths (str) or :class:`SoilProfile` instances.
    config : HVStripConfig, optional
    detect_peaks : bool

    Returns
    -------
    MultiForwardResult
    """
    if config is None:
        config = HVStripConfig()

    t0 = time.perf_counter()
    results: List[ForwardResult] = []

    for prof in profiles:
        res = compute_forward(
            prof, config=config, detect_peaks=detect_peaks
        )
        results.append(res)

    elapsed = time.perf_counter() - t0
    n_success = sum(1 for r in results if r.success)

    # Compute peak statistics across profiles
    peak_stats = _compute_peak_statistics(results)

    return MultiForwardResult(
        results=results,
        n_profiles=len(profiles),
        n_success=n_success,
        n_failed=len(profiles) - n_success,
        combined_peak_statistics=peak_stats,
        elapsed_seconds=round(elapsed, 3),
    )


def detect_peaks_on_curve(
    freqs: np.ndarray,
    amps: np.ndarray,
    config: Optional[PeakDetectionConfig] = None,
) -> List[PeakInfo]:
    """Detect peaks on an existing HV curve.

    Parameters
    ----------
    freqs, amps : array-like
    config : PeakDetectionConfig, optional

    Returns
    -------
    list of PeakInfo
    """
    if config is None:
        config = PeakDetectionConfig()
    return _detect_peaks(
        np.asarray(freqs, dtype=float),
        np.asarray(amps, dtype=float),
        config,
    )


def set_manual_peaks(
    result: ForwardResult,
    peaks: List[Dict[str, float]],
) -> ForwardResult:
    """Override auto-detected peaks with manual selections.

    Parameters
    ----------
    result : ForwardResult
        Existing result with frequency/amplitude arrays.
    peaks : list of dict
        Each dict has ``frequency`` (and optionally ``label``).
        The closest point on the curve is snapped to.

    Returns
    -------
    ForwardResult
        A new result with updated peaks.
    """
    result = ForwardResult(**{
        k: getattr(result, k) for k in result.__dataclass_fields__
    })
    manual_peaks: List[PeakInfo] = []
    for i, p in enumerate(peaks):
        freq = p["frequency"]
        idx = int(np.argmin(np.abs(result.frequencies - freq)))
        manual_peaks.append(PeakInfo(
            frequency=float(result.frequencies[idx]),
            amplitude=float(result.amplitudes[idx]),
            index=idx,
            label=p.get("label", f"f{i}"),
            source="manual",
        ))
    result.peaks = manual_peaks
    return result


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------


def list_engines() -> List[Dict[str, Any]]:
    """Return metadata for all registered forward engines."""
    infos = engine_registry.get_engine_info()
    return [
        {
            "name": info.get("name", ""),
            "description": info.get("description", ""),
            "implemented": info.get("implemented", False),
        }
        for info in infos
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_peak_statistics(results: List[ForwardResult]) -> Dict[str, Any]:
    """Compute cross-profile peak statistics."""
    f0_values = []
    a0_values = []
    for r in results:
        if r.success and r.peaks:
            f0_values.append(r.peaks[0].frequency)
            a0_values.append(r.peaks[0].amplitude)

    if not f0_values:
        return {}

    return {
        "n_profiles_with_peaks": len(f0_values),
        "f0_mean": float(np.mean(f0_values)),
        "f0_std": float(np.std(f0_values)),
        "f0_min": float(np.min(f0_values)),
        "f0_max": float(np.max(f0_values)),
        "a0_mean": float(np.mean(a0_values)),
        "a0_std": float(np.std(a0_values)),
    }
