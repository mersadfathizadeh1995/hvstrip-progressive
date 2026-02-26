"""
Model I/O utilities for the Rayleigh ellipticity engine.

Handles reading HVf-style model files and converting them to
gpell-compatible format.  Q-factor estimation is delegated to
the ``q_estimation`` module.
"""

from pathlib import Path
from typing import List

from .q_estimation import (
    add_q_to_layers,
    layers_to_gpell_text,
    parse_model_layers,
)


def format_gpell_model(
    model_path: str,
    auto_q: bool = False,
    q_formula: str = "default",
) -> str:
    """Convert an HVf model file to gpell-compatible text.

    Parameters
    ----------
    model_path : str
        Path to the source model file.
    auto_q : bool
        If True and model lacks Qp/Qs, auto-generate them.
    q_formula : str
        Quality-factor estimation strategy (see ``q_estimation``).

    Returns
    -------
    str
        Model text ready for gpell stdin / temp file.
    """
    layers = parse_model_layers(model_path)

    if auto_q:
        layers = add_q_to_layers(layers, formula=q_formula)

    return layers_to_gpell_text(layers, include_q=True)


__all__ = ["format_gpell_model"]
