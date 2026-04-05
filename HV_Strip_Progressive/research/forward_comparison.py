"""
Forward Comparison — run all engines on all profiles and collect results.

Core module for the comparison study: dispatches profiles to each engine,
collects HV curves, and detects peaks for cross-engine analysis.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import ComparisonStudyConfig
from .profile_suite import ProfileEntry

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Result of running one engine on one profile."""

    profile_name: str
    engine_name: str
    frequencies: List[float] = field(default_factory=list)
    amplitudes: List[float] = field(default_factory=list)
    peaks: List[Dict[str, float]] = field(default_factory=list)
    success: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class ProfileComparison:
    """All engine results for a single profile."""

    profile_name: str
    category: str
    engine_results: Dict[str, EngineResult] = field(default_factory=dict)

    @property
    def all_successful(self) -> bool:
        return all(r.success for r in self.engine_results.values())

    @property
    def successful_engines(self) -> List[str]:
        return [n for n, r in self.engine_results.items() if r.success]


@dataclass
class ComparisonDataset:
    """Complete dataset of all profile × engine combinations."""

    comparisons: List[ProfileComparison] = field(default_factory=list)
    engine_names: List[str] = field(default_factory=list)
    total_runs: int = 0
    successful_runs: int = 0
    elapsed_seconds: float = 0.0

    def get_comparison(self, profile_name: str) -> Optional[ProfileComparison]:
        for c in self.comparisons:
            if c.profile_name == profile_name:
                return c
        return None

    def get_engine_results(self, engine_name: str) -> List[EngineResult]:
        return [
            c.engine_results[engine_name]
            for c in self.comparisons
            if engine_name in c.engine_results
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_names": self.engine_names,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "elapsed_seconds": self.elapsed_seconds,
            "n_profiles": len(self.comparisons),
            "comparisons": [
                {
                    "profile_name": c.profile_name,
                    "category": c.category,
                    "engines": {
                        name: {
                            "success": r.success,
                            "n_peaks": len(r.peaks),
                            "peaks": r.peaks,
                            "error": r.error,
                            "elapsed_seconds": r.elapsed_seconds,
                        }
                        for name, r in c.engine_results.items()
                    },
                }
                for c in self.comparisons
            ],
        }


def run_forward_comparison(
    profiles: List[ProfileEntry],
    config: ComparisonStudyConfig,
    progress_callback: Any = None,
) -> ComparisonDataset:
    """Run all engines on all profiles.

    Parameters
    ----------
    profiles : list of ProfileEntry
    config : ComparisonStudyConfig
    progress_callback : callable, optional
        Called with (current, total, message) for progress tracking.

    Returns
    -------
    ComparisonDataset
    """
    from ..api.forward_engine import compute_forward
    from ..api.config import HVStripConfig, EngineConfig, FrequencyConfig, PeakDetectionConfig

    ecfg = config.engines
    dataset = ComparisonDataset(engine_names=ecfg.engines)
    t0 = time.time()

    total = len(profiles) * len(ecfg.engines)
    current = 0

    for profile in profiles:
        comparison = ProfileComparison(
            profile_name=profile.name,
            category=profile.category,
        )

        for engine_name in ecfg.engines:
            current += 1
            if progress_callback:
                progress_callback(
                    current, total,
                    f"{profile.name} × {engine_name}",
                )

            result = _run_single(
                profile.profile_path,
                engine_name,
                ecfg,
                config,
            )
            comparison.engine_results[engine_name] = result
            dataset.total_runs += 1
            if result.success:
                dataset.successful_runs += 1

        dataset.comparisons.append(comparison)

    dataset.elapsed_seconds = time.time() - t0
    logger.info(
        "Forward comparison complete: %d/%d successful in %.1fs",
        dataset.successful_runs,
        dataset.total_runs,
        dataset.elapsed_seconds,
    )
    return dataset


def _run_single(
    profile_path: str,
    engine_name: str,
    ecfg: Any,
    config: ComparisonStudyConfig,
) -> EngineResult:
    """Run a single engine on a single profile."""
    from ..api.forward_engine import compute_forward
    from ..api.config import HVStripConfig

    hvstrip_config = HVStripConfig()
    hvstrip_config.engine.name = engine_name
    hvstrip_config.frequency.fmin = ecfg.fmin
    hvstrip_config.frequency.fmax = ecfg.fmax
    hvstrip_config.frequency.nf = ecfg.n_frequencies
    hvstrip_config.frequency.n_samples = ecfg.n_frequencies
    hvstrip_config.peak_detection.prominence = ecfg.peak_prominence
    hvstrip_config.peak_detection.min_amplitude = ecfg.peak_min_amplitude

    # Apply engine-specific overrides
    overrides = ecfg.engine_overrides.get(engine_name, {})
    for k, v in overrides.items():
        if hasattr(hvstrip_config.engine, k):
            setattr(hvstrip_config.engine, k, v)

    t0 = time.time()
    try:
        from ..api.profile_io import load_profile
        profile = load_profile(profile_path)

        fwd = compute_forward(profile, config=hvstrip_config, engine_name=engine_name)
        elapsed = time.time() - t0

        return EngineResult(
            profile_name=os.path.basename(profile_path),
            engine_name=engine_name,
            frequencies=fwd.frequencies if hasattr(fwd, "frequencies") else [],
            amplitudes=fwd.amplitudes if hasattr(fwd, "amplitudes") else [],
            peaks=[p.__dict__ for p in fwd.peaks] if hasattr(fwd, "peaks") else [],
            success=fwd.success if hasattr(fwd, "success") else True,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        return EngineResult(
            profile_name=os.path.basename(profile_path),
            engine_name=engine_name,
            success=False,
            error=str(exc),
            elapsed_seconds=time.time() - t0,
        )
