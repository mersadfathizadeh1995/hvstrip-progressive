"""
Rayleigh Wave Ellipticity engine.

Computes theoretical H/V spectral ratios from the ellipticity of
Rayleigh waves in a horizontally layered medium using Geopsy's
``gpell.exe`` via Git Bash.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseForwardEngine, EngineResult
from .ellipticity_engine.config import EllipticityConfig
from .ellipticity_engine.runner import compute_ellipticity


class EllipticityEngine(BaseForwardEngine):
    """Rayleigh wave ellipticity H/V computation (gpell)."""

    name = "ellipticity"
    description = (
        "Rayleigh wave ellipticity theory (gpell). Computes H/V from "
        "the ellipticity of Rayleigh waves in a 1-D layered medium. "
        "Requires Vs, Vp, density, thickness, and Geopsy gpell.exe."
    )

    def get_default_config(self) -> Dict:
        return {
            "gpell_path": EllipticityConfig.gpell_path,
            "git_bash_path": EllipticityConfig.git_bash_path,
            "fmin": 0.5,
            "fmax": 20.0,
            "n_samples": 500,
            "n_modes": 1,
            "sampling": "log",
            "absolute": True,
            "peak_refinement": False,
            "love_alpha": 0.0,
            "auto_q": False,
            "q_formula": "default",
            "clip_factor": 50.0,
            "timeout": 30,
        }

    def get_required_params(self) -> List[str]:
        return ["gpell_path", "git_bash_path", "fmin", "fmax", "n_samples"]

    def validate_config(self, config: Dict) -> Tuple[bool, str]:
        cfg = {**self.get_default_config(), **(config or {})}
        gpell = Path(cfg["gpell_path"])
        if not gpell.exists():
            return False, f"gpell.exe not found: {gpell}"
        git_bash = Path(cfg["git_bash_path"])
        if not git_bash.exists():
            bash_exe = git_bash.parent / "usr" / "bin" / "bash.exe"
            if not bash_exe.exists():
                return False, f"Git Bash not found: {git_bash}"
        if cfg["fmin"] >= cfg["fmax"]:
            return False, "fmin must be less than fmax"
        if cfg["n_samples"] < 2:
            return False, "n_samples must be >= 2"
        return True, ""

    def format_model(self, profile) -> str:
        """Convert SoilProfile to gpell input format.

        Format per layer: thickness Vp Vs density_kg_m3
        Last row has thickness = 0 (half-space).
        """
        lines = [str(len(profile.layers))]
        for layer in profile.layers:
            thickness = 0.0 if layer.is_halfspace else layer.thickness
            vp = layer.vp
            vs = layer.vs
            density = layer.density / 1000.0  # SoilProfile stores kg/m³*1000
            lines.append(
                f"{thickness:.2f} {vp:.1f} {vs:.1f} {density:.1f}"
            )
        return "\n".join(lines) + "\n"

    def compute(self, model_path: str, config: Dict = None) -> EngineResult:
        """Run gpell on a model file and return EngineResult.

        Parameters
        ----------
        model_path : str
            Path to model file (HVf format: thickness Vp Vs density).
            Density may be in g/cm³ (HVf convention) — the underlying
            runner handles the format as-is since gpell expects the same
            column layout with density in kg/m³.
        config : Dict, optional
            Engine configuration overrides.

        Returns
        -------
        EngineResult
        """
        cfg = {**self.get_default_config(), **(config or {})}
        t0 = time.perf_counter()

        ell_config = EllipticityConfig(
            gpell_path=cfg["gpell_path"],
            git_bash_path=cfg["git_bash_path"],
            fmin=cfg["fmin"],
            fmax=cfg["fmax"],
            n_samples=cfg["n_samples"],
            n_modes=cfg["n_modes"],
            sampling=cfg["sampling"],
            absolute=cfg["absolute"],
            peak_refinement=cfg.get("peak_refinement", False),
            love_alpha=cfg.get("love_alpha", 0.0),
            auto_q=cfg.get("auto_q", False),
            q_formula=cfg.get("q_formula", "default"),
            clip_factor=cfg.get("clip_factor", 50.0),
            timeout=cfg.get("timeout", 30),
        )

        result = compute_ellipticity(model_path, ell_config)
        elapsed = time.perf_counter() - t0

        return EngineResult(
            frequencies=result.frequencies,
            amplitudes=result.amplitudes,
            metadata={
                "engine": self.name,
                "gpell_path": cfg["gpell_path"],
                "elapsed_seconds": elapsed,
                "peaks_hz": result.peaks,
                "mode": result.mode,
                "auto_q": cfg.get("auto_q", False),
            },
        )


__all__ = ["EllipticityEngine"]
