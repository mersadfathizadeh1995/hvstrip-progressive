"""
Rayleigh Ellipticity Engine
===========================

Portable package for computing theoretical H/V curves via Rayleigh wave
ellipticity using Geopsy's ``gpell`` command-line tool (invoked through
Git Bash).

Usage::

    from ellipticity_engine import EllipticityConfig, compute_ellipticity

    config = EllipticityConfig(fmin=0.5, fmax=20.0, n_samples=500)
    result = compute_ellipticity("path/to/model.txt", config)

    result.frequencies    # np.ndarray
    result.amplitudes     # np.ndarray (clipped)
    result.peaks          # list of peak frequencies in Hz
    result.raw_amplitudes # np.ndarray (original, with singularities)
"""

from .config import EllipticityConfig
from .postprocess import EllipticityResult
from .runner import compute_ellipticity, run_gpell

__all__ = [
    "EllipticityConfig",
    "EllipticityResult",
    "compute_ellipticity",
    "run_gpell",
]
