"""
Export — save analysis results in various formats.

Provides CSV, JSON, MAT, and Excel export for forward results,
stripping results, and batch results.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_forward_result(
    result: Dict[str, Any],
    output_dir: str,
    formats: Optional[List[str]] = None,
    base_name: str = "forward",
) -> Dict[str, str]:
    """Export a forward computation result.

    Parameters
    ----------
    result : dict
        From ``ForwardResult.to_dict()``.
    output_dir : str
    formats : list of str, optional
        Subset of ``["csv", "json", "mat"]``.  Default: all.
    base_name : str

    Returns
    -------
    dict
        Format name → file path.
    """
    if formats is None:
        formats = ["csv", "json"]

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    if "csv" in formats:
        p = os.path.join(output_dir, f"{base_name}_hv_curve.csv")
        _write_hv_csv(
            p,
            result.get("frequencies", []),
            result.get("amplitudes", []),
        )
        paths["csv"] = p

    if "json" in formats:
        p = os.path.join(output_dir, f"{base_name}_result.json")
        _write_json(p, result)
        paths["json"] = p

    if "mat" in formats:
        p = os.path.join(output_dir, f"{base_name}_result.mat")
        _write_mat(p, {
            "frequencies": np.array(result.get("frequencies", [])),
            "amplitudes": np.array(result.get("amplitudes", [])),
            "peak_frequency": result.get("peaks", [{}])[0].get("frequency", 0)
            if result.get("peaks") else 0,
        })
        paths["mat"] = p

    return paths


def export_strip_result(
    result: Dict[str, Any],
    output_dir: str,
    formats: Optional[List[str]] = None,
    base_name: str = "strip",
) -> Dict[str, str]:
    """Export a stripping result.

    Parameters
    ----------
    result : dict
        From ``StripResult.to_dict()``.
    output_dir : str
    formats : list of str, optional
    base_name : str

    Returns
    -------
    dict
        Format name → file path.
    """
    if formats is None:
        formats = ["csv", "json"]

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    if "csv" in formats:
        # Peak evolution CSV
        p = os.path.join(output_dir, f"{base_name}_peak_evolution.csv")
        evolution = result.get("peak_evolution", [])
        if evolution:
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "n_layers", "frequency", "amplitude"]
                )
                writer.writeheader()
                writer.writerows(evolution)
            paths["peak_evolution_csv"] = p

    if "json" in formats:
        p = os.path.join(output_dir, f"{base_name}_result.json")
        _write_json(p, result)
        paths["json"] = p

    return paths


def export_batch_result(
    result: Dict[str, Any],
    output_dir: str,
    formats: Optional[List[str]] = None,
    base_name: str = "batch",
) -> Dict[str, str]:
    """Export batch stripping results.

    Parameters
    ----------
    result : dict
        From ``BatchStripResult.to_dict()``.
    output_dir : str
    formats : list of str, optional
    base_name : str

    Returns
    -------
    dict
    """
    if formats is None:
        formats = ["csv", "json"]

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    if "csv" in formats:
        # Summary CSV
        p = os.path.join(output_dir, f"{base_name}_summary.csv")
        rows = []
        for r in result.get("results", []):
            sr = r.get("strip_result") or {}
            steps = sr.get("steps", [])
            f0 = steps[0].get("peak_frequency", 0) if steps else 0
            a0 = steps[0].get("peak_amplitude", 0) if steps else 0
            rows.append({
                "profile": r.get("profile_name", ""),
                "success": r.get("success", False),
                "n_steps": sr.get("n_steps", 0),
                "f0": f0,
                "a0": a0,
                "error": r.get("error", ""),
            })
        if rows:
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["profile", "success", "n_steps", "f0", "a0", "error"],
                )
                writer.writeheader()
                writer.writerows(rows)
            paths["summary_csv"] = p

    if "json" in formats:
        p = os.path.join(output_dir, f"{base_name}_result.json")
        _write_json(p, result)
        paths["json"] = p

    return paths


def export_profile_csv(
    profile_dict: Dict[str, Any],
    path: str,
) -> str:
    """Export a soil profile to CSV.

    Parameters
    ----------
    profile_dict : dict
        From ``profile_to_dict()``.
    path : str

    Returns
    -------
    str
        Written file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    layers = profile_dict.get("layers", [])
    if not layers:
        return ""

    with open(path, "w", newline="") as f:
        fields = ["thickness", "vs", "vp", "density", "nu", "is_halfspace"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for l in layers:
            writer.writerow({k: l.get(k, "") for k in fields})

    return path


def export_hv_curve_csv(
    frequencies: Any,
    amplitudes: Any,
    path: str,
) -> str:
    """Export an HV curve to a two-column CSV.

    Returns the written file path.
    """
    _write_hv_csv(path, frequencies, amplitudes)
    return path


def export_peak_summary(
    peaks: List[Dict[str, Any]],
    path: str,
) -> str:
    """Export detected peaks to CSV.

    Parameters
    ----------
    peaks : list of dict
        Each with ``frequency``, ``amplitude``, ``label``, ``source``.
    path : str

    Returns
    -------
    str
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        fields = ["label", "frequency", "amplitude", "source"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in peaks:
            writer.writerow({k: p.get(k, "") for k in fields})
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_hv_csv(path: str, freqs: Any, amps: Any) -> None:
    """Write a frequency/amplitude CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    freqs = np.asarray(freqs).ravel()
    amps = np.asarray(amps).ravel()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "hv_amplitude"])
        for freq, amp in zip(freqs, amps):
            writer.writerow([f"{freq:.6f}", f"{amp:.6f}"])


def _write_json(path: str, data: Any) -> None:
    """Write JSON, converting numpy types."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)


def _write_mat(path: str, data: Dict[str, Any]) -> None:
    """Write MAT file (requires scipy)."""
    try:
        from scipy.io import savemat
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        savemat(path, data)
    except ImportError:
        logger.warning("scipy not available — skipping MAT export")
