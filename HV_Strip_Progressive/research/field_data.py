"""
Field Data — field validation sites with known soil models.

Provides functions to load and compare forward-modeled HVSR
against measured field HVSR data.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ComparisonStudyConfig, FieldSiteConfig

logger = logging.getLogger(__name__)


@dataclass
class FieldValidation:
    """Validation results for one field site."""

    site_name: str
    measured_f0: Optional[float] = None
    measured_f1: Optional[float] = None
    measured_frequencies: List[float] = field(default_factory=list)
    measured_amplitudes: List[float] = field(default_factory=list)
    engine_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    best_engine: str = ""
    best_rmse: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_name": self.site_name,
            "measured_f0": self.measured_f0,
            "measured_f1": self.measured_f1,
            "engine_results": self.engine_results,
            "best_engine": self.best_engine,
            "best_rmse": self.best_rmse,
        }


def load_measured_hvsr(
    path: str,
) -> Dict[str, Any]:
    """Load measured HVSR curve from a CSV file.

    Expected format: two columns (frequency, amplitude) or
    three columns (frequency, amplitude, std).

    Returns
    -------
    dict
        ``{"frequencies": list, "amplitudes": list, "std": list or None}``
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    result: Dict[str, Any] = {
        "frequencies": data[:, 0].tolist(),
        "amplitudes": data[:, 1].tolist(),
    }
    if data.shape[1] >= 3:
        result["std"] = data[:, 2].tolist()
    else:
        result["std"] = None
    return result


def run_field_validation(
    config: ComparisonStudyConfig,
    progress_callback: Any = None,
) -> List[FieldValidation]:
    """Run forward modeling and compare with field HVSR for all sites.

    Parameters
    ----------
    config : ComparisonStudyConfig
    progress_callback : callable, optional

    Returns
    -------
    list of FieldValidation
    """
    from ..api.forward_engine import compute_forward
    from ..api.profile_io import load_profile
    from ..api.config import HVStripConfig

    validations: List[FieldValidation] = []

    for i, site in enumerate(config.field_sites):
        if progress_callback:
            progress_callback(i + 1, len(config.field_sites), site.name)

        val = FieldValidation(
            site_name=site.name,
            measured_f0=site.known_f0,
            measured_f1=site.known_f1,
        )

        # Load measured HVSR if available
        if site.measured_hvsr_path and os.path.isfile(site.measured_hvsr_path):
            measured = load_measured_hvsr(site.measured_hvsr_path)
            val.measured_frequencies = measured["frequencies"]
            val.measured_amplitudes = measured["amplitudes"]

        # Load soil profile
        if not os.path.isfile(site.profile_path):
            logger.warning("Profile not found for site %s: %s", site.name, site.profile_path)
            validations.append(val)
            continue

        profile = load_profile(site.profile_path)

        # Run each engine
        for engine_name in config.engines.engines:
            hvstrip_config = HVStripConfig()
            hvstrip_config.engine.name = engine_name
            hvstrip_config.frequency.fmin = config.engines.fmin
            hvstrip_config.frequency.fmax = config.engines.fmax
            hvstrip_config.frequency.nf = config.engines.n_frequencies
            hvstrip_config.frequency.n_samples = config.engines.n_frequencies

            try:
                fwd = compute_forward(profile, config=hvstrip_config, engine_name=engine_name)

                engine_result: Dict[str, Any] = {
                    "success": fwd.success,
                    "frequencies": fwd.frequencies,
                    "amplitudes": fwd.amplitudes,
                    "peaks": [p.__dict__ for p in fwd.peaks],
                    "f0": fwd.peaks[0].frequency if fwd.peaks else None,
                }

                # Compute RMSE against measured
                if val.measured_frequencies and fwd.frequencies:
                    rmse = _curve_rmse(
                        val.measured_frequencies,
                        val.measured_amplitudes,
                        fwd.frequencies,
                        fwd.amplitudes,
                    )
                    engine_result["rmse_vs_measured"] = rmse

                    if rmse < val.best_rmse:
                        val.best_rmse = rmse
                        val.best_engine = engine_name

                # f0 error
                if site.known_f0 and fwd.peaks:
                    f0_pred = fwd.peaks[0].frequency
                    engine_result["f0_error"] = abs(f0_pred - site.known_f0)
                    engine_result["f0_error_pct"] = (
                        abs(f0_pred - site.known_f0) / site.known_f0 * 100
                    )

                val.engine_results[engine_name] = engine_result

            except Exception as exc:
                val.engine_results[engine_name] = {
                    "success": False,
                    "error": str(exc),
                }

        validations.append(val)

    return validations


def _curve_rmse(
    freq_meas: List[float],
    amp_meas: List[float],
    freq_pred: List[float],
    amp_pred: List[float],
) -> float:
    """Compute RMSE between measured and predicted curves."""
    fa = np.array(freq_meas)
    aa = np.array(amp_meas)
    fp = np.array(freq_pred)
    ap = np.array(amp_pred)

    # Interpolate predicted onto measured frequency grid
    fmin = max(fa.min(), fp.min())
    fmax = min(fa.max(), fp.max())
    mask = (fa >= fmin) & (fa <= fmax)

    if mask.sum() < 5:
        return float("inf")

    interp_pred = np.interp(fa[mask], fp, ap)
    return float(np.sqrt(np.mean((aa[mask] - interp_pred) ** 2)))
