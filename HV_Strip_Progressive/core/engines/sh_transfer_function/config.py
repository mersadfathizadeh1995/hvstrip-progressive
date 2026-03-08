"""
config.py
=========
Configuration dataclass for the SH Transfer Function engine.

All tunables for frequency range, damping, reference depth, and
post-processing are consolidated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class SHTFConfig:
    """Settings for a single SH transfer function computation.

    Parameters
    ----------
    fmin : float
        Minimum frequency [Hz]. Default 0.1.
    fmax : float
        Maximum frequency [Hz]. Default 10.0.
    n_samples : int
        Number of frequency samples. Default 512.
    sampling : str
        Frequency sampling: ``"log"`` (log-spaced) or ``"linear"``.
    Dsoil : float or None
        Soil damping ratio [%].
        ``None`` → computed automatically via Darendeli (2001).
        Scalar → same value applied to all soil layers.
    Drock : float
        Half-space damping ratio [%]. Default 0.5.
    d_tf : float or str
        Reference depth of input ground motion.
        ``0``        → bedrock outcrop (default).
        ``'within'`` → top of half-space (within bedrock).
        ``float > 0``→ arbitrary depth [m].
    darendeli_curvetype : int
        Darendeli (2001) curve type:
        1 = mean (default), 2 = mean + 1σ, 3 = mean − 1σ.
    gamma_max : float
        Maximum allowable unit weight [kN/m³]. Default 23.
    f0_search_fmin : float or None
        Lower bound [Hz] for F0 peak search. Defaults to ``fmin``.
    f0_search_fmax : float or None
        Upper bound [Hz] for F0 peak search. Defaults to ``fmax``.
    clip_tf : float
        Clip TF amplitudes above this value (0 = disabled).
    """

    fmin: float = 0.1
    fmax: float = 10.0
    n_samples: int = 512
    sampling: str = "log"

    Dsoil: Optional[float] = None
    Drock: float = 0.5
    d_tf: Union[float, str] = 0

    darendeli_curvetype: int = 1
    gamma_max: float = 23.0

    f0_search_fmin: Optional[float] = None
    f0_search_fmax: Optional[float] = None

    clip_tf: float = 0.0

    def validate(self) -> None:
        """Validate parameter ranges.

        Raises
        ------
        ValueError
            If any parameter is out of range or inconsistent.
        """
        if self.fmin >= self.fmax:
            raise ValueError(f"fmin ({self.fmin}) must be less than fmax ({self.fmax})")
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {self.n_samples}")
        if self.sampling not in ("log", "linear"):
            raise ValueError(f"sampling must be 'log' or 'linear', got {self.sampling!r}")
        if self.Dsoil is not None and self.Dsoil < 0:
            raise ValueError(f"Dsoil must be >= 0, got {self.Dsoil}")
        if self.Drock < 0:
            raise ValueError(f"Drock must be >= 0, got {self.Drock}")
        if self.darendeli_curvetype not in (1, 2, 3):
            raise ValueError(f"darendeli_curvetype must be 1, 2 or 3, got {self.darendeli_curvetype}")
        if self.gamma_max <= 0:
            raise ValueError(f"gamma_max must be positive, got {self.gamma_max}")
        f0_lo = self.f0_search_fmin if self.f0_search_fmin is not None else self.fmin
        f0_hi = self.f0_search_fmax if self.f0_search_fmax is not None else self.fmax
        if f0_lo >= f0_hi:
            raise ValueError(
                f"f0_search_fmin ({f0_lo}) must be less than f0_search_fmax ({f0_hi})"
            )

    def freq_vector(self) -> np.ndarray:
        """Return the frequency array according to current settings.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Frequency vector [Hz].
        """
        if self.sampling == "log":
            return np.logspace(np.log10(self.fmin), np.log10(self.fmax), self.n_samples)
        return np.linspace(self.fmin, self.fmax, self.n_samples)


__all__ = ["SHTFConfig"]
