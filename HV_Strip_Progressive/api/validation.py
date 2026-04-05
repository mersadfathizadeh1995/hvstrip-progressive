"""
Validation — config, profile, and engine availability checks.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .config import HVStripConfig, EngineConfig

logger = logging.getLogger(__name__)


def validate_config(config: HVStripConfig) -> Dict[str, Any]:
    """Validate the full configuration.

    Returns
    -------
    dict
        ``{"valid": bool, "errors": [...], "warnings": [...]}``
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Frequency
    if config.frequency.fmin >= config.frequency.fmax:
        errors.append(
            f"fmin ({config.frequency.fmin}) must be < fmax ({config.frequency.fmax})"
        )
    if config.frequency.nf < 2:
        errors.append(f"nf must be ≥ 2 (got {config.frequency.nf})")
    if config.frequency.n_samples < 2:
        errors.append(f"n_samples must be ≥ 2 (got {config.frequency.n_samples})")

    # Engine
    engine = config.engine
    if engine.name not in ("diffuse_field", "ellipticity", "sh_wave"):
        errors.append(f"Unknown engine: '{engine.name}'")

    if engine.name == "diffuse_field" and engine.exe_path:
        if not os.path.isfile(engine.exe_path):
            warnings.append(f"HVf.exe not found at: {engine.exe_path}")

    if engine.name == "ellipticity":
        if engine.gpell_path and not os.path.isfile(engine.gpell_path):
            warnings.append(f"gpell.exe not found at: {engine.gpell_path}")
        if engine.git_bash_path and not os.path.isfile(engine.git_bash_path):
            warnings.append(f"git-bash.exe not found at: {engine.git_bash_path}")

    if engine.name == "sh_wave":
        if engine.Drock < 0:
            errors.append(f"Drock must be ≥ 0 (got {engine.Drock})")
        if engine.Dsoil is not None and engine.Dsoil < 0:
            errors.append(f"Dsoil must be ≥ 0 (got {engine.Dsoil})")

    # Peak detection
    pd = config.peak_detection
    if pd.prominence < 0:
        errors.append(f"Peak prominence must be ≥ 0 (got {pd.prominence})")
    if pd.distance < 1:
        errors.append(f"Peak distance must be ≥ 1 (got {pd.distance})")

    # Dual resonance
    dr = config.dual_resonance
    if dr.separation_ratio_threshold <= 1.0:
        warnings.append(
            f"Dual resonance ratio threshold ({dr.separation_ratio_threshold}) "
            "should be > 1.0"
        )

    # Output
    if config.output.output_dir and not os.path.isdir(config.output.output_dir):
        warnings.append(
            f"Output directory does not exist: {config.output.output_dir} "
            "(will be created)"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_engine_availability(engine_name: str) -> Dict[str, Any]:
    """Check whether an engine's external dependencies are available.

    Returns
    -------
    dict
        ``{"available": bool, "engine": str, "message": str}``
    """
    if engine_name == "sh_wave":
        return {
            "available": True,
            "engine": engine_name,
            "message": "SH Wave engine is pure Python — always available",
        }

    if engine_name == "diffuse_field":
        try:
            from ..core.engines.diffuse_field import DiffuseFieldEngine

            eng = DiffuseFieldEngine()
            default_cfg = eng.get_default_config()
            exe = default_cfg.get("exe_path", "")
            if exe and os.path.isfile(exe):
                return {
                    "available": True,
                    "engine": engine_name,
                    "message": f"HVf.exe found at {exe}",
                }
            return {
                "available": False,
                "engine": engine_name,
                "message": f"HVf.exe not found (expected at {exe})",
            }
        except Exception as exc:
            return {
                "available": False,
                "engine": engine_name,
                "message": str(exc),
            }

    if engine_name == "ellipticity":
        try:
            from ..core.engines.ellipticity import EllipticityEngine

            eng = EllipticityEngine()
            default_cfg = eng.get_default_config()
            gpell = default_cfg.get("gpell_path", "")
            bash = default_cfg.get("git_bash_path", "")
            issues = []
            if not gpell or not os.path.isfile(gpell):
                issues.append(f"gpell.exe not found ({gpell})")
            if not bash or not os.path.isfile(bash):
                issues.append(f"git-bash.exe not found ({bash})")
            if issues:
                return {
                    "available": False,
                    "engine": engine_name,
                    "message": "; ".join(issues),
                }
            return {
                "available": True,
                "engine": engine_name,
                "message": "Ellipticity engine dependencies found",
            }
        except Exception as exc:
            return {
                "available": False,
                "engine": engine_name,
                "message": str(exc),
            }

    return {
        "available": False,
        "engine": engine_name,
        "message": f"Unknown engine: {engine_name}",
    }


def validate_profile_for_engine(
    profile_dict: Dict[str, Any],
    engine_name: str,
) -> Dict[str, Any]:
    """Validate a profile dict is suitable for a specific engine.

    Returns
    -------
    dict
        ``{"valid": bool, "errors": [...], "warnings": [...]}``
    """
    errors: List[str] = []
    warnings: List[str] = []

    layers = profile_dict.get("layers", [])
    if not layers:
        errors.append("Profile has no layers")
        return {"valid": False, "errors": errors, "warnings": warnings}

    for i, l in enumerate(layers):
        tag = f"Layer {i + 1}"
        vs = l.get("vs", 0)
        vp = l.get("vp")
        density = l.get("density")

        if vs <= 0:
            errors.append(f"{tag}: Vs must be > 0")

        if engine_name == "diffuse_field":
            # Needs density in g/cm³ (will be converted from kg/m³)
            if density is None:
                warnings.append(f"{tag}: density missing — will use 2000 kg/m³")

        if engine_name == "ellipticity":
            if vp is None:
                warnings.append(f"{tag}: Vp missing — will be computed from Vs")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_strip_directory(strip_dir: str) -> Dict[str, Any]:
    """Validate a stripping output directory has the expected structure.

    Returns
    -------
    dict
        ``{"valid": bool, "n_steps": int, "errors": [...]}``
    """
    errors: List[str] = []

    if not os.path.isdir(strip_dir):
        return {"valid": False, "n_steps": 0, "errors": [f"Not a directory: {strip_dir}"]}

    # Look for step folders
    from ..core.batch_workflow import find_step_folders

    try:
        step_folders = find_step_folders(strip_dir)
    except Exception:
        step_folders = []

    if not step_folders:
        errors.append("No step folders found in strip directory")

    # Check each step has a model file
    for folder in step_folders:
        models = list(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("model_"))
        if not models:
            errors.append(f"No model file in {os.path.basename(folder)}")

    return {
        "valid": len(errors) == 0,
        "n_steps": len(step_folders),
        "errors": errors,
    }
