"""
Configuration dataclass for the Rayleigh ellipticity engine.

All tunables for gpell execution, model formatting, and post-processing
are consolidated here.
"""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_GPELL_PATH = r"C:\Geopsy.org\bin\gpell.exe"
DEFAULT_GIT_BASH = r"C:\Users\mersadf\AppData\Local\Programs\Git\git-bash.exe"


@dataclass
class EllipticityConfig:
    """Settings for a single gpell run.

    Parameters
    ----------
    gpell_path : str
        Absolute path to ``gpell.exe``.
    git_bash_path : str
        Absolute path to ``git-bash.exe`` used to invoke gpell.
    fmin : float
        Minimum frequency in Hz.
    fmax : float
        Maximum frequency in Hz.
    n_samples : int
        Number of frequency samples.
    n_modes : int
        Number of Rayleigh modes (1 = fundamental only).
    sampling : str
        Frequency sampling type: ``"log"``, ``"frequency"``, or ``"period"``.
    absolute : bool
        Output absolute ellipticity (True for H/V proxy).
    peak_refinement : bool
        Use ``-pc`` instead of ``-c`` for peak-refined curves.
    love_alpha : float
        Love wave mixing coefficient (0.0-0.99). 0 = pure Rayleigh.
    auto_q : bool
        Whether to auto-generate Qp/Qs when missing from model.
    q_formula : str
        Q estimation strategy: ``"default"``, ``"brocher"``, ``"constant"``.
    clip_factor : float
        Clip amplitudes at ``clip_factor * median(abs(amplitudes))``.
        Set to 0 to disable clipping.
    timeout : int
        Subprocess timeout in seconds.
    """

    gpell_path: str = DEFAULT_GPELL_PATH
    git_bash_path: str = DEFAULT_GIT_BASH
    fmin: float = 0.5
    fmax: float = 20.0
    n_samples: int = 500
    n_modes: int = 1
    sampling: str = "log"
    absolute: bool = True
    peak_refinement: bool = False
    love_alpha: float = 0.0
    auto_q: bool = False
    q_formula: str = "default"
    clip_factor: float = 50.0
    timeout: int = 30

    def validate(self):
        """Check that paths exist and parameters are sane.

        Raises
        ------
        FileNotFoundError
            If gpell or git-bash executables are missing.
        ValueError
            If frequency or sampling parameters are invalid.
        """
        if not Path(self.gpell_path).exists():
            raise FileNotFoundError(
                f"gpell.exe not found: {self.gpell_path}"
            )
        if not Path(self.git_bash_path).exists():
            raise FileNotFoundError(
                f"git-bash.exe not found: {self.git_bash_path}"
            )
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be less than fmax")
        if self.n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if self.sampling not in ("log", "frequency", "period"):
            raise ValueError(
                f"sampling must be 'log', 'frequency', or 'period', "
                f"got {self.sampling!r}"
            )
        if not (0.0 <= self.love_alpha < 1.0):
            raise ValueError("love_alpha must be in [0, 1)")


__all__ = ["EllipticityConfig"]
