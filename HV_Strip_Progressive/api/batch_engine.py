"""
Batch Engine — batch stripping of multiple soil profiles.

Orchestrates :func:`strip_engine.run_stripping` across a list of profiles
and collects cross-profile statistics.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import HVStripConfig, BatchConfig
from .strip_engine import run_stripping, StripResult
from .dual_resonance_ops import compute_batch_dual_statistics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProfileStripResult:
    """Result for a single profile within a batch."""

    profile_name: str = ""
    profile_path: str = ""
    strip_result: Optional[StripResult] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "profile_path": self.profile_path,
            "strip_result": (
                self.strip_result.to_dict() if self.strip_result else None
            ),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class BatchStripResult:
    """Aggregated result of batch stripping."""

    results: List[ProfileStripResult] = field(default_factory=list)
    n_profiles: int = 0
    n_success: int = 0
    n_failed: int = 0
    combined_statistics: Dict[str, Any] = field(default_factory=dict)
    output_directory: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_profiles": self.n_profiles,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "results": [r.to_dict() for r in self.results],
            "combined_statistics": self.combined_statistics,
            "output_directory": self.output_directory,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_batch_stripping(
    profiles: List[str],
    output_dir: str,
    config: Optional[HVStripConfig] = None,
    generate_report: bool = True,
) -> BatchStripResult:
    """Run progressive stripping on multiple profiles.

    Each profile gets its own sub-directory under *output_dir*.

    Parameters
    ----------
    profiles : list of str
        Absolute paths to soil-profile files.
    output_dir : str
        Base output directory.
    config : HVStripConfig, optional
    generate_report : bool

    Returns
    -------
    BatchStripResult
    """
    if config is None:
        config = HVStripConfig()

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.perf_counter()

    results: List[ProfileStripResult] = []

    for profile_path in profiles:
        name = Path(profile_path).stem
        profile_out = os.path.join(output_dir, name)

        logger.info("Batch stripping: %s → %s", name, profile_out)

        try:
            strip_res = run_stripping(
                profile_path,
                output_dir=profile_out,
                config=config,
                generate_report=generate_report,
            )
            results.append(ProfileStripResult(
                profile_name=name,
                profile_path=profile_path,
                strip_result=strip_res,
                success=strip_res.success,
                error=strip_res.error,
            ))
        except Exception as exc:
            logger.error("Batch: %s failed — %s", name, exc)
            results.append(ProfileStripResult(
                profile_name=name,
                profile_path=profile_path,
                success=False,
                error=str(exc),
            ))

    elapsed = time.perf_counter() - t0
    n_success = sum(1 for r in results if r.success)

    # Cross-profile statistics
    stats = get_batch_statistics(results)

    return BatchStripResult(
        results=results,
        n_profiles=len(profiles),
        n_success=n_success,
        n_failed=len(profiles) - n_success,
        combined_statistics=stats,
        output_directory=output_dir,
        elapsed_seconds=round(elapsed, 3),
    )


def get_batch_statistics(
    results: List[ProfileStripResult],
) -> Dict[str, Any]:
    """Compute cross-profile peak statistics.

    Parameters
    ----------
    results : list of ProfileStripResult

    Returns
    -------
    dict
        Mean/std/min/max of f0, f1, amplitude, n_steps across profiles.
    """
    f0_values: List[float] = []
    a0_values: List[float] = []
    n_steps_values: List[int] = []

    for r in results:
        if not r.success or r.strip_result is None:
            continue
        steps = [s for s in r.strip_result.steps if s.success]
        if not steps:
            continue

        # f0 from the first (full-model) step
        f0_values.append(steps[0].peak_frequency)
        a0_values.append(steps[0].peak_amplitude)
        n_steps_values.append(len(steps))

    if not f0_values:
        return {}

    return {
        "n_profiles_successful": len(f0_values),
        "f0_mean": float(np.mean(f0_values)),
        "f0_std": float(np.std(f0_values)),
        "f0_min": float(np.min(f0_values)),
        "f0_max": float(np.max(f0_values)),
        "a0_mean": float(np.mean(a0_values)),
        "a0_std": float(np.std(a0_values)),
        "n_steps_mean": float(np.mean(n_steps_values)),
        "n_steps_range": [int(np.min(n_steps_values)), int(np.max(n_steps_values))],
    }
