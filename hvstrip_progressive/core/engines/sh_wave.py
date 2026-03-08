"""
SH Wave Propagation engine.

Computes theoretical H/V spectral ratios using the propagator-matrix
method (Kramer 1996) for vertically-propagating SH waves through a
1-D layered medium.  Uses Mayne (2001) unit weights and Darendeli
(2001) small-strain damping.

Wraps the ``sh_transfer_function`` package located at
``core/engines/sh_transfer_function/``.
"""

import time
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseForwardEngine, EngineResult
from .sh_transfer_function.config import SHTFConfig
from .sh_transfer_function.runner import compute_sh_tf


class SHWaveEngine(BaseForwardEngine):
    """SH wave propagation H/V computation (propagator-matrix)."""

    name = "sh_wave"
    description = (
        "SH wave transfer-matrix propagation (Kramer 1996). Computes "
        "theoretical transfer function by propagating vertically-incident "
        "SH waves through a 1-D layered medium using Darendeli (2001) "
        "damping. Pure Python — no external executables."
    )

    def get_default_config(self) -> Dict:
        return {
            "fmin": 0.1,
            "fmax": 30.0,
            "n_samples": 512,
            "sampling": "log",
            "Dsoil": None,
            "Drock": 0.5,
            "d_tf": 0,
            "darendeli_curvetype": 1,
            "gamma_max": 23.0,
            "f0_search_fmin": None,
            "f0_search_fmax": None,
            "clip_tf": 0.0,
        }

    def get_required_params(self) -> List[str]:
        return ["fmin", "fmax", "n_samples"]

    def validate_config(self, config: Dict) -> Tuple[bool, str]:
        cfg = {**self.get_default_config(), **(config or {})}
        if cfg["fmin"] >= cfg["fmax"]:
            return False, "fmin must be less than fmax"
        if cfg["n_samples"] < 2:
            return False, "n_samples must be >= 2"
        if cfg["Drock"] < 0:
            return False, "Drock must be >= 0"
        if cfg["Dsoil"] is not None and cfg["Dsoil"] < 0:
            return False, "Dsoil must be >= 0"
        return True, ""

    def format_model(self, profile) -> str:
        """Convert SoilProfile to SH-wave input format.

        The SH engine reads the same HVf text format:
        ``thickness  Vp  Vs  density`` (one row per layer,
        last row has thickness = 0 for the half-space).
        """
        return profile.to_hvf_format()

    def compute(self, model_path: str, config: Dict = None) -> EngineResult:
        """Run SH transfer function on a model file.

        Parameters
        ----------
        model_path : str
            Path to model file (HVf format: thickness Vp Vs density).
        config : Dict, optional
            Engine configuration overrides.

        Returns
        -------
        EngineResult
        """
        cfg = {**self.get_default_config(), **(config or {})}
        t0 = time.perf_counter()

        sh_config = SHTFConfig(
            fmin=cfg["fmin"],
            fmax=cfg["fmax"],
            n_samples=cfg["n_samples"],
            sampling=cfg["sampling"],
            Dsoil=cfg["Dsoil"],
            Drock=cfg["Drock"],
            d_tf=cfg["d_tf"],
            darendeli_curvetype=cfg["darendeli_curvetype"],
            gamma_max=cfg["gamma_max"],
            f0_search_fmin=cfg.get("f0_search_fmin"),
            f0_search_fmax=cfg.get("f0_search_fmax"),
            clip_tf=cfg.get("clip_tf", 0.0),
        )

        result = compute_sh_tf(model_path, sh_config)
        elapsed = time.perf_counter() - t0

        # SHTFResult.amplitudes is (F, M); flatten to 1-D for single profile
        amps = np.asarray(result.amplitudes)
        if amps.ndim == 2:
            amps = amps[:, 0]

        return EngineResult(
            frequencies=result.frequencies,
            amplitudes=amps,
            metadata={
                "engine": self.name,
                "elapsed_seconds": elapsed,
                "f0_hz": result.f0,
                "peaks_hz": result.peaks,
                "n_layers": result.model.n_layers,
                "Dsoil": cfg["Dsoil"],
                "Drock": cfg["Drock"],
            },
        )


__all__ = ["SHWaveEngine"]
