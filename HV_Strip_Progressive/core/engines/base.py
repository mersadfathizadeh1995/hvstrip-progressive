"""
Abstract base class for HVSR forward modeling engines.

Each engine wraps a specific theoretical method for computing H/V spectral
ratios from a layered soil profile.  Engines must implement:

- ``compute``   – run the forward model and return (freqs, amps)
- ``format_model`` – convert a SoilProfile to the engine's native input string
- ``parse_output`` – normalise raw engine output into (freqs, amps)

The base class also exposes metadata (``name``, ``description``) and a
default configuration dict that the registry and GUI can introspect.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EngineResult:
    """Normalised output returned by every engine.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency array in Hz.
    amplitudes : np.ndarray
        H/V amplitude array (same length as *frequencies*).
    metadata : Dict
        Engine-specific metadata (execution time, version, warnings …).
    """

    frequencies: np.ndarray
    amplitudes: np.ndarray
    metadata: Dict = field(default_factory=dict)


class BaseForwardEngine(ABC):
    """Interface that every forward-modeling engine must satisfy."""

    # ------------------------------------------------------------------
    # Metadata — subclasses override as class attributes
    # ------------------------------------------------------------------
    name: str = "base"
    description: str = "Abstract forward engine"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_default_config(self) -> Dict:
        """Return the engine's default configuration dictionary.

        The keys are whatever the engine needs (exe path, frequency grid,
        solver parameters, …).  The GUI and YAML config will merge
        user overrides on top of this.
        """

    @abstractmethod
    def format_model(self, profile: "SoilProfile") -> str:
        """Convert a ``SoilProfile`` to the engine's native model string.

        Parameters
        ----------
        profile : SoilProfile
            Layered soil profile with layers, Vs, Vp, density, etc.

        Returns
        -------
        str
            Model text that the engine can consume.
        """

    @abstractmethod
    def compute(
        self,
        model_path: str,
        config: Dict = None,
    ) -> EngineResult:
        """Run the forward model.

        Parameters
        ----------
        model_path : str
            Path to a model file **already written in the engine's format**.
        config : Dict, optional
            Merged configuration (engine defaults + user overrides).

        Returns
        -------
        EngineResult
            Normalised frequencies and amplitudes.
        """

    # ------------------------------------------------------------------
    # Optional hooks — subclasses may override
    # ------------------------------------------------------------------

    def validate_config(self, config: Dict) -> Tuple[bool, str]:
        """Check whether *config* is valid for this engine.

        Returns ``(True, "")`` when valid, ``(False, reason)`` otherwise.
        """
        return True, ""

    def get_required_params(self) -> List[str]:
        """Return list of required config keys for this engine."""
        return []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_from_profile(
        self,
        profile: "SoilProfile",
        config: Dict = None,
    ) -> EngineResult:
        """Format *profile*, write to a temp file, and call ``compute``.

        This is a helper so callers don't have to manage temp files
        themselves.  Engines that don't need an on-disk file can override.
        """
        import tempfile
        from pathlib import Path

        model_text = self.format_model(profile)
        with tempfile.TemporaryDirectory() as tdir:
            model_path = Path(tdir) / "model.txt"
            model_path.write_text(model_text, encoding="utf-8")
            return self.compute(str(model_path), config)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"


__all__ = [
    "BaseForwardEngine",
    "EngineResult",
]
