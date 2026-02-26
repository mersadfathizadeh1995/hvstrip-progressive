"""
HV forward computation facade.

Thin wrapper that delegates to the pluggable engine registry while
preserving full backward compatibility.  Existing callers can keep
using ``compute_hv_curve(model_path, config)`` unchanged — internally
it routes to the ``diffuse_field`` engine by default.

To use a different engine::

    freqs, amps = compute_hv_curve(model_path, config, engine_name="sh_wave")
"""

from typing import Dict, List, Tuple

from .engines import registry
from .engines.diffuse_field import DiffuseFieldEngine


# Backward-compatible default config (delegates to diffuse-field engine)
DEFAULT_CONFIG = DiffuseFieldEngine().get_default_config()


def compute_hv_curve(
    model_path: str,
    config: Dict = None,
    engine_name: str = "diffuse_field",
) -> Tuple[List[float], List[float]]:
    """Run forward modeling and return (freqs, amps).

    Parameters
    ----------
    model_path : str
        Path to a model file in the target engine's format.
    config : Dict, optional
        Engine configuration overrides (merged with engine defaults).
    engine_name : str
        Name of the registered engine to use (default ``"diffuse_field"``).

    Returns
    -------
    Tuple[List[float], List[float]]
        (frequencies, amplitudes)
    """
    engine = registry.get(engine_name)
    result = engine.compute(model_path, config)
    return result.frequencies.tolist(), result.amplitudes.tolist()


__all__ = [
    "compute_hv_curve",
    "DEFAULT_CONFIG",
]
