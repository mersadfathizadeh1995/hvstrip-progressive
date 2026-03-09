"""Shared save/export utilities used by multiple save modules.

Consolidates patterns that were duplicated across the monolithic
all_profiles_view:

* ``write_peak_info``  — peak CSV writing (was duplicated 3×)
* ``build_depth_vs``   — depth/Vs array construction (was duplicated 3×)
* ``save_figure_pair`` — save PNG+PDF figure pattern (was duplicated 8×)
* ``get_median_f0``    — auto-select highest median peak (was duplicated 4×)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Peak info I/O ──────────────────────────────────────────────

def write_peak_info(
    path: Path,
    f0: Optional[Tuple[float, float, int]],
    secondary: Sequence[Tuple[float, float, int]] = (),
) -> None:
    """Write peak_info.txt for a single profile.

    Parameters
    ----------
    path : Path
        Target file (e.g. ``prof_dir / "peak_info.txt"``).
    f0 : tuple or None
        ``(frequency, amplitude, index)`` for the primary peak.
    secondary : sequence of tuples
        Each element is ``(frequency, amplitude, index)``.
    """
    if f0 is None:
        return
    with open(path, "w") as fh:
        fh.write(f"f0_Frequency_Hz,{f0[0]:.6f}\n")
        fh.write(f"f0_Amplitude,{f0[1]:.6f}\n")
        fh.write(f"f0_Index,{f0[2]}\n")
        for j, s in enumerate(secondary):
            fh.write(f"Secondary_{j+1}_Frequency_Hz,{s[0]:.6f}\n")
            fh.write(f"Secondary_{j+1}_Amplitude,{s[1]:.6f}\n")
            fh.write(f"Secondary_{j+1}_Index,{s[2]}\n")


# ── Depth / Vs array builder ──────────────────────────────────

def build_depth_vs(profile) -> Tuple[List[float], List[float],
                                      list, list]:
    """Build step-wise depth and Vs arrays from a SoilProfile.

    Returns
    -------
    depths : list[float]
        Depth values (two entries per finite layer for step plot).
    vs : list[float]
        Corresponding Vs values.
    finite : list
        Finite layer objects.
    halfspace : list
        Half-space layer objects (0 or 1 element).
    """
    depths: List[float] = []
    vs: List[float] = []
    z = 0.0
    finite = [L for L in profile.layers if not L.is_halfspace]
    hs = [L for L in profile.layers if L.is_halfspace]
    for L in finite:
        depths.append(z)
        vs.append(L.vs)
        z += L.thickness
        depths.append(z)
        vs.append(L.vs)
    return depths, vs, finite, hs


# ── Figure save pair ──────────────────────────────────────────

def save_figure_pair(
    fig,
    out_dir: Path,
    name: str,
    dpi: int = 300,
    fmt: str = "png",
    *,
    close: bool = True,
) -> None:
    """Save a matplotlib figure as *fmt* **and** PDF (if fmt ≠ pdf).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : Path
        Directory to write into.
    name : str
        Base filename without extension.
    dpi : int
    fmt : str
        Primary format (``"png"``, ``"pdf"``, ``"svg"``).
    close : bool
        If ``True``, close the figure after saving.
    """
    import matplotlib.pyplot as plt

    fig.savefig(out_dir / f"{name}.{fmt}", dpi=dpi)
    if fmt != "pdf":
        fig.savefig(out_dir / f"{name}.pdf", dpi=dpi)
    if close:
        plt.close(fig)


# ── Median f0 retrieval ───────────────────────────────────────

def get_median_f0(
    median_peaks: Dict[str, Any],
    med_f: np.ndarray,
    med_a: np.ndarray,
) -> Optional[Tuple[float, float, int]]:
    """Return the user-selected median f0 or auto-pick the highest peak.

    Parameters
    ----------
    median_peaks : dict
        ``{"f0": tuple|None, "secondary": [...]}``.
    med_f, med_a : ndarray
        Median frequency and amplitude arrays.

    Returns
    -------
    tuple or None
        ``(frequency, amplitude, index)`` of the primary median peak.
    """
    f0m = median_peaks.get("f0")
    if f0m is not None:
        return f0m
    if med_a is not None and len(med_a) > 0:
        idx = int(np.argmax(med_a))
        return (med_f[idx], med_a[idx], idx)
    return None


# ── Resolve per-profile peak data ─────────────────────────────

def resolve_peak(peak_data: dict, r) -> Tuple[
    Optional[Tuple[float, float, int]],
    List[Tuple[float, float, int]],
]:
    """Get (f0, secondary_list) for a profile result, preferring peak_data.

    Parameters
    ----------
    peak_data : dict
        Keyed by ``r.name`` → ``{"f0": ..., "secondary": [...]}``.
    r : ProfileResult
        The profile result object.

    Returns
    -------
    f0 : tuple or None
    secondary : list of tuples
    """
    pk = peak_data.get(r.name, {})
    f0 = pk.get("f0") or r.f0
    secondary = pk.get("secondary", list(r.secondary_peaks or []))
    return f0, secondary
