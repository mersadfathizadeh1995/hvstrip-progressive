"""
Profile I/O — load, create, convert, and validate soil profiles.

Wraps :mod:`core.soil_profile` with a unified interface that auto-detects
file formats and provides JSON-safe serialisation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.soil_profile import Layer, SoilProfile
from ..core.velocity_utils import VelocityConverter
from ..core.vs_average import (
    compute_vs_average,
    vs_average_from_profile,
    VsAverageResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProfileSummary:
    """Summary information for a soil profile."""

    name: str = ""
    n_layers: int = 0
    total_depth: float = 0.0
    vs_range: Tuple[float, float] = (0.0, 0.0)
    vp_range: Tuple[float, float] = (0.0, 0.0)
    density_range: Tuple[float, float] = (0.0, 0.0)
    vs30: float = 0.0
    vs30_extrapolated: bool = False
    has_halfspace: bool = False
    halfspace_vs: float = 0.0
    f0_estimate: float = 0.0
    """Quarter-wavelength fundamental frequency estimate [Hz]."""


@dataclass
class ValidationResult:
    """Outcome of a profile validation check."""

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_FORMAT_MAP = {
    ".txt": "hvf",
    ".hvf": "hvf",
    ".csv": "csv",
    ".xml": "dinver",
    ".xlsx": "excel",
    ".xls": "excel",
    ".mat": "matlab",
}


def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    return _FORMAT_MAP.get(ext, "hvf")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_profile(
    path: str,
    fmt: str = "auto",
    name: Optional[str] = None,
) -> SoilProfile:
    """Load a soil profile from file.

    Parameters
    ----------
    path : str
        Absolute path to the profile file.
    fmt : str
        File format: ``"auto"`` (detect from extension), ``"hvf"``,
        ``"csv"``, ``"dinver"``, ``"excel"``, ``"txt"``.
    name : str, optional
        Override the profile name.  Defaults to the filename stem.

    Returns
    -------
    SoilProfile
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Profile file not found: {path}")

    if fmt == "auto":
        fmt = _detect_format(path)

    if fmt in ("hvf", "txt"):
        profile = SoilProfile.from_hvf_file(path)
    elif fmt == "csv":
        profile = SoilProfile.from_csv_file(path)
    elif fmt == "dinver":
        profile = SoilProfile.from_dinver_files(path)
    elif fmt in ("excel", "xlsx"):
        profile = SoilProfile.from_excel_file(path)
    else:
        # Fallback: try auto-detection in core
        try:
            profile = SoilProfile.from_auto(path)
        except Exception:
            profile = SoilProfile.from_hvf_file(path)

    if name is not None:
        profile.name = name
    elif not profile.name:
        profile.name = Path(path).stem

    logger.info(
        "Loaded profile '%s' (%d layers) from %s",
        profile.name,
        len(profile.layers),
        path,
    )
    return profile


def create_profile(
    layers: List[Dict[str, Any]],
    name: str = "",
    auto_compute: bool = True,
) -> SoilProfile:
    """Create a soil profile from a list of layer dicts.

    Parameters
    ----------
    layers : list of dict
        Each dict must have ``thickness`` and ``vs``.  Optional:
        ``vp``, ``density``, ``nu`` (Poisson's ratio), ``name``,
        ``is_halfspace``.  If ``vp`` or ``density`` are missing and
        *auto_compute* is True they are derived from Vs.
    name : str
        Profile identifier.
    auto_compute : bool
        Compute missing Vp and density from Vs.

    Returns
    -------
    SoilProfile
    """
    profile = SoilProfile(name=name)

    for i, ld in enumerate(layers):
        thickness = float(ld.get("thickness", 0.0))
        vs = float(ld["vs"])
        vp = ld.get("vp")
        density = ld.get("density")
        nu = ld.get("nu")
        is_hs = ld.get("is_halfspace", thickness == 0)

        if auto_compute:
            if nu is None:
                nu = VelocityConverter.suggest_nu(vs)
            if vp is None:
                vp = VelocityConverter.vp_from_vs_nu(vs, nu)
            if density is None:
                density = VelocityConverter.suggest_density(vs)

        layer = Layer(
            thickness=thickness,
            vs=vs,
            vp=float(vp) if vp is not None else None,
            nu=float(nu) if nu is not None else None,
            density=float(density) if density is not None else 2000.0,
            is_halfspace=is_hs,
        )
        profile.add_layer(layer)

    logger.info("Created profile '%s' with %d layers", name, len(layers))
    return profile


def save_profile(
    profile: SoilProfile,
    path: str,
    fmt: str = "hvf",
) -> str:
    """Save a soil profile to file.

    Parameters
    ----------
    profile : SoilProfile
    path : str
        Output path.
    fmt : str
        ``"hvf"`` | ``"csv"``

    Returns
    -------
    str
        The written file path.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fmt in ("hvf", "txt"):
        profile.save_hvf(path)
    elif fmt == "csv":
        profile.save_csv(path)
    else:
        raise ValueError(f"Unsupported save format: {fmt}")

    logger.info("Saved profile '%s' to %s", profile.name, path)
    return path


# ---------------------------------------------------------------------------
# Serialisation (JSON-safe)
# ---------------------------------------------------------------------------


def profile_to_dict(profile: SoilProfile) -> Dict[str, Any]:
    """Convert a :class:`SoilProfile` to a JSON-serialisable dict."""
    layers_list = []
    for layer in profile.layers:
        layers_list.append({
            "thickness": layer.thickness,
            "vs": layer.vs,
            "vp": layer.vp,
            "nu": layer.nu,
            "density": layer.density,
            "is_halfspace": layer.is_halfspace,
        })
    return {
        "name": profile.name,
        "description": getattr(profile, "description", ""),
        "n_layers": len(profile.layers),
        "layers": layers_list,
    }


def profile_from_dict(d: Dict[str, Any]) -> SoilProfile:
    """Reconstruct a :class:`SoilProfile` from a dict."""
    profile = SoilProfile(
        name=d.get("name", ""),
        description=d.get("description", ""),
    )
    for ld in d.get("layers", []):
        profile.add_layer(Layer(
            thickness=ld["thickness"],
            vs=ld["vs"],
            vp=ld.get("vp"),
            nu=ld.get("nu"),
            density=ld.get("density", 2000.0),
            is_halfspace=ld.get("is_halfspace", False),
        ))
    return profile


# ---------------------------------------------------------------------------
# Summary & validation
# ---------------------------------------------------------------------------


def get_profile_summary(profile: SoilProfile) -> ProfileSummary:
    """Compute a summary of profile properties."""
    if not profile.layers:
        return ProfileSummary(name=profile.name)

    finite_layers = [l for l in profile.layers if not l.is_halfspace]
    all_vs = [l.vs for l in profile.layers if l.vs]
    all_vp = [l.vp for l in profile.layers if l.vp]
    all_rho = [l.density for l in profile.layers if l.density]

    total_depth = sum(l.thickness for l in finite_layers)

    # Vs30
    vs30_result = vs_average_from_profile(profile)

    # F0 estimate (quarter-wavelength)
    f0 = 0.0
    if total_depth > 0 and all_vs:
        avg_vs = np.mean(all_vs[:len(finite_layers)]) if finite_layers else all_vs[0]
        f0 = avg_vs / (4.0 * total_depth)

    hs_layers = [l for l in profile.layers if l.is_halfspace]

    return ProfileSummary(
        name=profile.name,
        n_layers=len(profile.layers),
        total_depth=round(total_depth, 2),
        vs_range=(min(all_vs), max(all_vs)) if all_vs else (0.0, 0.0),
        vp_range=(min(all_vp), max(all_vp)) if all_vp else (0.0, 0.0),
        density_range=(min(all_rho), max(all_rho)) if all_rho else (0.0, 0.0),
        vs30=round(vs30_result.vs_avg, 2),
        vs30_extrapolated=vs30_result.extrapolated,
        has_halfspace=bool(hs_layers),
        halfspace_vs=hs_layers[0].vs if hs_layers else 0.0,
        f0_estimate=round(f0, 4),
    )


def validate_profile(profile: SoilProfile) -> ValidationResult:
    """Validate a soil profile for use with forward engines.

    Checks physical plausibility, layer ordering, and half-space presence.
    """
    result = ValidationResult()

    if not profile.layers:
        result.valid = False
        result.errors.append("Profile has no layers.")
        return result

    # Check half-space
    has_hs = any(l.is_halfspace for l in profile.layers)
    if not has_hs:
        last = profile.layers[-1]
        if last.thickness == 0:
            pass  # treated as half-space by engines
        else:
            result.warnings.append(
                "No explicit half-space layer; last layer may be treated as "
                "half-space by some engines."
            )

    for i, layer in enumerate(profile.layers):
        tag = f"Layer {i + 1}"

        # Vs > 0
        if layer.vs <= 0:
            result.valid = False
            result.errors.append(f"{tag}: Vs must be > 0 (got {layer.vs})")

        # Vp > Vs
        if layer.vp is not None and layer.vp <= layer.vs:
            result.warnings.append(
                f"{tag}: Vp ({layer.vp}) ≤ Vs ({layer.vs}) — unusual"
            )

        # Density
        if layer.density is not None:
            if layer.density < 500 or layer.density > 5000:
                result.warnings.append(
                    f"{tag}: density {layer.density} kg/m³ outside typical "
                    "range [500, 5000]"
                )

        # Thickness
        if not layer.is_halfspace and layer.thickness <= 0:
            result.valid = False
            result.errors.append(
                f"{tag}: finite layer thickness must be > 0"
            )

    return result
