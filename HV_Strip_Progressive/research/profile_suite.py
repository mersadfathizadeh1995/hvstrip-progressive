"""
Profile Suite — synthetic profile generation for the comparison study.

Uses SoilGen to generate a diverse set of soil profiles covering
multiple geological scenarios plus random variations.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import ComparisonStudyConfig, ProfileSuiteConfig

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """A single profile in the study suite."""

    name: str
    category: str  # scenario name or "random"
    index: int
    n_layers: int
    total_depth: float
    vs30: Optional[float]
    f0_estimate: Optional[float]
    profile_path: str  # path to exported file
    profile_obj: Any = None  # SoilGen SoilProfile (kept in memory)


def _ensure_soilgen(soilgen_path: Optional[str] = None):
    """Import SoilGen, adding its path if needed."""
    if soilgen_path and soilgen_path not in sys.path:
        sys.path.insert(0, soilgen_path)
    try:
        import soilgen
        return soilgen
    except ImportError:
        raise ImportError(
            "SoilGen package not found. Either install it or set "
            "profiles.soilgen_path in the study config."
        )


def generate_profile_suite(
    config: ComparisonStudyConfig,
    output_dir: Optional[str] = None,
) -> List[ProfileEntry]:
    """Generate the full suite of synthetic profiles.

    Parameters
    ----------
    config : ComparisonStudyConfig
    output_dir : str, optional
        Where to write profile files. If None, uses config.output.output_dir/profiles.

    Returns
    -------
    list of ProfileEntry
    """
    if output_dir is None:
        output_dir = os.path.join(config.output.output_dir, "profiles")
    os.makedirs(output_dir, exist_ok=True)

    pcfg = config.profiles
    sg = _ensure_soilgen(pcfg.soilgen_path)

    entries: List[ProfileEntry] = []
    idx = 0

    # Scenario-based profiles
    for scenario in pcfg.scenarios:
        scenario_dir = os.path.join(output_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)

        for i in range(pcfg.n_per_scenario):
            try:
                gen = sg.ScenarioGenerator(
                    config={"scenario": scenario},
                    seed=pcfg.seed + idx,
                )
                profile = gen.generate()
                name = f"{scenario}_{i + 1:03d}"
                path = os.path.join(scenario_dir, f"{name}.txt")

                # Export in HVf text format
                exporter = sg.TXTExporter()
                exporter.export(profile, path)

                entry = ProfileEntry(
                    name=name,
                    category=scenario,
                    index=idx,
                    n_layers=len(profile.layers),
                    total_depth=profile.total_depth,
                    vs30=_safe_vs30(profile),
                    f0_estimate=_safe_f0(profile),
                    profile_path=path,
                    profile_obj=profile,
                )
                entries.append(entry)
                idx += 1
            except Exception as exc:
                logger.warning("Failed to generate %s #%d: %s", scenario, i, exc)

    # Random profiles
    random_dir = os.path.join(output_dir, "random")
    os.makedirs(random_dir, exist_ok=True)

    for i in range(pcfg.n_random):
        try:
            gen = sg.RandomGenerator(
                config={
                    "min_depth": pcfg.min_depth,
                    "max_depth": pcfg.max_depth,
                    "min_vs": pcfg.min_vs,
                    "max_vs": pcfg.max_vs,
                    "min_layers": pcfg.min_layers,
                    "max_layers": pcfg.max_layers,
                },
                seed=pcfg.seed + idx,
            )
            profile = gen.generate()
            name = f"random_{i + 1:03d}"
            path = os.path.join(random_dir, f"{name}.txt")

            exporter = sg.TXTExporter()
            exporter.export(profile, path)

            entry = ProfileEntry(
                name=name,
                category="random",
                index=idx,
                n_layers=len(profile.layers),
                total_depth=profile.total_depth,
                vs30=_safe_vs30(profile),
                f0_estimate=_safe_f0(profile),
                profile_path=path,
                profile_obj=profile,
            )
            entries.append(entry)
            idx += 1
        except Exception as exc:
            logger.warning("Failed to generate random #%d: %s", i, exc)

    logger.info(
        "Generated %d profiles (%d scenarios × %d + %d random)",
        len(entries),
        len(pcfg.scenarios),
        pcfg.n_per_scenario,
        pcfg.n_random,
    )
    return entries


def load_profile_suite(
    profiles_dir: str,
) -> List[ProfileEntry]:
    """Load a previously exported profile suite from disk.

    Parameters
    ----------
    profiles_dir : str
        Directory containing scenario subfolders and/or .txt files.

    Returns
    -------
    list of ProfileEntry
    """
    entries: List[ProfileEntry] = []
    idx = 0

    for dirpath, dirs, files in os.walk(profiles_dir):
        category = os.path.basename(dirpath)
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(dirpath, fname)
            name = os.path.splitext(fname)[0]
            entries.append(ProfileEntry(
                name=name,
                category=category,
                index=idx,
                n_layers=0,
                total_depth=0.0,
                vs30=None,
                f0_estimate=None,
                profile_path=path,
            ))
            idx += 1

    return entries


def get_suite_summary(entries: List[ProfileEntry]) -> Dict[str, Any]:
    """Return summary statistics for a profile suite."""
    categories: Dict[str, int] = {}
    for e in entries:
        categories[e.category] = categories.get(e.category, 0) + 1

    vs30_values = [e.vs30 for e in entries if e.vs30 is not None]
    f0_values = [e.f0_estimate for e in entries if e.f0_estimate is not None]

    return {
        "n_profiles": len(entries),
        "categories": categories,
        "vs30_range": (min(vs30_values), max(vs30_values)) if vs30_values else None,
        "f0_range": (min(f0_values), max(f0_values)) if f0_values else None,
    }


# --- Helpers ---

def _safe_vs30(profile: Any) -> Optional[float]:
    try:
        return float(profile.vs30)
    except Exception:
        return None


def _safe_f0(profile: Any) -> Optional[float]:
    try:
        return float(profile.fundamental_frequency)
    except Exception:
        return None
