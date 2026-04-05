"""
Session I/O — JSON-based session persistence.

Saves and restores :class:`HVStripAnalysis` state without pickle,
ensuring MCP and cross-platform compatibility.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import HVStripAnalysis

from .config import HVStripConfig
from .profile_io import profile_to_dict, profile_from_dict, save_profile

logger = logging.getLogger(__name__)


def save_session(
    analysis: "HVStripAnalysis",
    session_dir: str,
) -> Dict[str, Any]:
    """Save the analysis session to a directory.

    Creates:
    - ``config.json`` — full configuration
    - ``profiles/`` — one HVf file per profile
    - ``session.json`` — profile list, result summaries

    Parameters
    ----------
    analysis : HVStripAnalysis
    session_dir : str

    Returns
    -------
    dict
        ``{"saved": True, "session_dir": str, "n_profiles": int}``
    """
    os.makedirs(session_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(session_dir, "config.json")
    with open(config_path, "w") as f:
        f.write(analysis._config.to_json(indent=2))

    # Save profiles
    profiles_dir = os.path.join(session_dir, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    profile_manifest = {}
    for name, profile in analysis._profiles.items():
        profile_path = os.path.join(profiles_dir, f"{name}.txt")
        save_profile(profile, profile_path, fmt="hvf")
        profile_manifest[name] = {
            "file": f"profiles/{name}.txt",
            "dict": profile_to_dict(profile),
        }

    # Save forward result summaries (no large arrays)
    forward_summaries = {}
    for name, result in analysis._forward_results.items():
        forward_summaries[name] = {
            "profile_name": result.profile_name,
            "engine_name": result.engine_name,
            "n_peaks": len(result.peaks),
            "peaks": [
                {"frequency": p.frequency, "amplitude": p.amplitude, "label": p.label}
                for p in result.peaks
            ],
            "success": result.success,
            "elapsed_seconds": result.elapsed_seconds,
        }

    # Save strip result summaries
    strip_summaries = {}
    for name, result in analysis._strip_results.items():
        strip_summaries[name] = {
            "output_directory": result.output_directory,
            "strip_directory": result.strip_directory,
            "n_steps": result.n_steps,
            "peak_evolution": result.peak_evolution,
            "success": result.success,
            "elapsed_seconds": result.elapsed_seconds,
        }

    # Master session file
    session_data = {
        "session_id": analysis._session_id,
        "profiles": profile_manifest,
        "forward_results": forward_summaries,
        "strip_results": strip_summaries,
        "has_batch_result": analysis._batch_result is not None,
    }

    session_path = os.path.join(session_dir, "session.json")
    with open(session_path, "w") as f:
        json.dump(session_data, f, indent=2, default=str)

    logger.info(
        "Session saved to %s (%d profiles)", session_dir, len(profile_manifest)
    )

    return {
        "saved": True,
        "session_dir": session_dir,
        "n_profiles": len(profile_manifest),
        "files": ["config.json", "session.json"]
        + [f"profiles/{n}.txt" for n in profile_manifest],
    }


def load_session(
    analysis: "HVStripAnalysis",
    session_dir: str,
) -> Dict[str, Any]:
    """Restore a session from a saved directory.

    Loads config and profiles.  Forward/strip results are *not*
    restored (they can be recomputed).

    Parameters
    ----------
    analysis : HVStripAnalysis
    session_dir : str

    Returns
    -------
    dict
        ``{"loaded": True, "n_profiles": int, "session_id": str}``
    """
    # Load config
    config_path = os.path.join(session_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            analysis._config = HVStripConfig.from_json(f.read())

    # Load session manifest
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.isfile(session_path):
        return {"loaded": False, "error": "session.json not found"}

    with open(session_path) as f:
        session_data = json.load(f)

    analysis._session_id = session_data.get("session_id", "default")

    # Load profiles
    analysis._profiles.clear()
    analysis._forward_results.clear()
    analysis._strip_results.clear()
    analysis._batch_result = None

    for name, info in session_data.get("profiles", {}).items():
        # Try loading from file first
        profile_file = os.path.join(session_dir, info.get("file", ""))
        if os.path.isfile(profile_file):
            from .profile_io import load_profile

            profile = load_profile(profile_file, name=name)
            analysis._profiles[name] = profile
        elif "dict" in info:
            profile = profile_from_dict(info["dict"])
            analysis._profiles[name] = profile

    logger.info(
        "Session loaded from %s (%d profiles)",
        session_dir,
        len(analysis._profiles),
    )

    return {
        "loaded": True,
        "session_dir": session_dir,
        "session_id": analysis._session_id,
        "n_profiles": len(analysis._profiles),
        "profile_names": list(analysis._profiles.keys()),
    }
