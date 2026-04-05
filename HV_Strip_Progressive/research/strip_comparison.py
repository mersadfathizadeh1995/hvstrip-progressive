"""
Strip Comparison — optional module for comparing stripping results
across engines.

Extends the forward comparison by running progressive stripping
on selected profiles and comparing the extracted f0/f1 evolution.
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
class StripComparisonEntry:
    """Stripping comparison result for one profile × one engine."""

    profile_name: str
    engine_name: str
    n_steps: int = 0
    f0_initial: Optional[float] = None
    f0_final: Optional[float] = None
    f1_detected: bool = False
    f1_value: Optional[float] = None
    peak_evolution: List[Dict[str, float]] = field(default_factory=list)
    success: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class StripComparisonDataset:
    """Complete stripping comparison results."""

    entries: List[StripComparisonEntry] = field(default_factory=list)
    engine_names: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def get_for_profile(self, profile_name: str) -> List[StripComparisonEntry]:
        return [e for e in self.entries if e.profile_name == profile_name]

    def get_for_engine(self, engine_name: str) -> List[StripComparisonEntry]:
        return [e for e in self.entries if e.engine_name == engine_name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_names": self.engine_names,
            "n_entries": len(self.entries),
            "elapsed_seconds": self.elapsed_seconds,
            "entries": [
                {
                    "profile_name": e.profile_name,
                    "engine_name": e.engine_name,
                    "n_steps": e.n_steps,
                    "f0_initial": e.f0_initial,
                    "f0_final": e.f0_final,
                    "f1_detected": e.f1_detected,
                    "f1_value": e.f1_value,
                    "success": e.success,
                    "error": e.error,
                    "elapsed_seconds": e.elapsed_seconds,
                }
                for e in self.entries
            ],
        }


def run_strip_comparison(
    profiles: List[ProfileEntry],
    config: ComparisonStudyConfig,
    output_dir: Optional[str] = None,
    progress_callback: Any = None,
) -> StripComparisonDataset:
    """Run stripping on selected profiles with each engine.

    Parameters
    ----------
    profiles : list of ProfileEntry
        Profiles to strip (typically a subset of the full suite).
    config : ComparisonStudyConfig
    output_dir : str, optional
    progress_callback : callable, optional

    Returns
    -------
    StripComparisonDataset
    """
    from ..api.strip_engine import run_stripping
    from ..api.config import HVStripConfig

    if output_dir is None:
        output_dir = os.path.join(config.output.output_dir, "strip_comparison")

    ecfg = config.engines
    dataset = StripComparisonDataset(engine_names=ecfg.engines)
    t0 = time.time()

    total = len(profiles) * len(ecfg.engines)
    current = 0

    for profile in profiles:
        for engine_name in ecfg.engines:
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Strip {profile.name} × {engine_name}")

            entry = _strip_single(
                profile,
                engine_name,
                config,
                output_dir,
            )
            dataset.entries.append(entry)

    dataset.elapsed_seconds = time.time() - t0
    n_ok = sum(1 for e in dataset.entries if e.success)
    logger.info(
        "Strip comparison: %d/%d successful in %.1fs",
        n_ok, len(dataset.entries), dataset.elapsed_seconds,
    )
    return dataset


def _strip_single(
    profile: ProfileEntry,
    engine_name: str,
    config: ComparisonStudyConfig,
    output_dir: str,
) -> StripComparisonEntry:
    """Run stripping for one profile × engine combination."""
    from ..api.strip_engine import run_stripping
    from ..api.config import HVStripConfig

    hvstrip_config = HVStripConfig()
    hvstrip_config.engine.name = engine_name
    hvstrip_config.frequency.fmin = config.engines.fmin
    hvstrip_config.frequency.fmax = config.engines.fmax

    strip_dir = os.path.join(output_dir, engine_name, profile.name)
    os.makedirs(strip_dir, exist_ok=True)

    t0 = time.time()
    try:
        result = run_stripping(
            profile.profile_path,
            output_dir=strip_dir,
            config=hvstrip_config,
            generate_report=False,
        )
        elapsed = time.time() - t0

        entry = StripComparisonEntry(
            profile_name=profile.name,
            engine_name=engine_name,
            n_steps=result.n_steps,
            peak_evolution=result.peak_evolution,
            success=result.success,
            elapsed_seconds=elapsed,
        )

        # Extract f0 from first and last steps
        if result.peak_evolution:
            entry.f0_initial = result.peak_evolution[0].get("frequency")
            entry.f0_final = result.peak_evolution[-1].get("frequency")

        return entry

    except Exception as exc:
        return StripComparisonEntry(
            profile_name=profile.name,
            engine_name=engine_name,
            success=False,
            error=str(exc),
            elapsed_seconds=time.time() - t0,
        )
