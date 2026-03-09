"""
Quality factor (Qp, Qs) estimation module.

Provides empirical formulas to auto-compute Qp and Qs for soil layers
that lack attenuation data.  The module can process raw model files
(4-column: thickness Vp Vs density) and produce 6-column files
(thickness Vp Vs density Qp Qs) ready for gpell.

Supported estimation strategies
-------------------------------
- ``"default"``  — Qp = 0.05*(Vp - 400) + 50, clamped at 50.  Qs = Qp/2.
- ``"brocher"``  — Qs = 0.04*Vs - 1, clamped at 10.  Qp = 2*Qs.
- ``"constant"`` — User-supplied constant Qp/Qs for all layers.

References
----------
Brocher (2008) "Key elements of regional seismic velocity models …"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Estimation formulas
# ---------------------------------------------------------------------------

def estimate_default(vp: float, vs: float) -> Tuple[float, float]:
    """Default empirical formula.

    Parameters
    ----------
    vp : float
        P-wave velocity in m/s.
    vs : float
        S-wave velocity in m/s (unused in this formula).

    Returns
    -------
    Tuple[float, float]
        (Qp, Qs).
    """
    qp = max(0.05 * (vp - 400.0) + 50.0, 50.0)
    qs = qp / 2.0
    return qp, qs


def estimate_brocher(vp: float, vs: float) -> Tuple[float, float]:
    """Brocher-style empirical formula (Qs from Vs).

    Parameters
    ----------
    vp : float
        P-wave velocity in m/s (unused in this formula).
    vs : float
        S-wave velocity in m/s.

    Returns
    -------
    Tuple[float, float]
        (Qp, Qs).
    """
    qs = max(0.04 * vs - 1.0, 10.0)
    qp = 2.0 * qs
    return qp, qs


def estimate_constant(
    vp: float,
    vs: float,
    qp_value: float = 100.0,
    qs_value: float = 50.0,
) -> Tuple[float, float]:
    """Constant Q for all layers.

    Parameters
    ----------
    vp, vs : float
        Velocities (unused).
    qp_value : float
        Constant Qp.
    qs_value : float
        Constant Qs.

    Returns
    -------
    Tuple[float, float]
        (Qp, Qs).
    """
    return qp_value, qs_value


# Registry of available formulas
_FORMULAS = {
    "default": estimate_default,
    "brocher": estimate_brocher,
    "constant": estimate_constant,
}


def compute_qp_qs(
    vp: float,
    vs: float,
    formula: str = "default",
    **kwargs,
) -> Tuple[float, float]:
    """Compute Qp and Qs for a single layer.

    Parameters
    ----------
    vp : float
        P-wave velocity in m/s.
    vs : float
        S-wave velocity in m/s.
    formula : str
        Estimation strategy name.
    **kwargs
        Extra parameters forwarded to the formula function
        (e.g. ``qp_value``, ``qs_value`` for ``"constant"``).

    Returns
    -------
    Tuple[float, float]
        (Qp, Qs).

    Raises
    ------
    ValueError
        If the formula name is unknown.
    """
    if formula not in _FORMULAS:
        available = ", ".join(_FORMULAS.keys())
        raise ValueError(
            f"Unknown Q formula {formula!r}. Available: {available}"
        )
    fn = _FORMULAS[formula]
    if formula == "constant":
        return fn(vp, vs, **kwargs)
    return fn(vp, vs)


# ---------------------------------------------------------------------------
# Layer dataclass
# ---------------------------------------------------------------------------

@dataclass
class LayerQ:
    """A single layer with quality factors.

    Attributes
    ----------
    thickness : float
        Layer thickness in metres (0 for half-space).
    vp : float
        P-wave velocity in m/s.
    vs : float
        S-wave velocity in m/s.
    density : float
        Density in kg/m^3.
    qp : float
        P-wave quality factor.
    qs : float
        S-wave quality factor.
    """

    thickness: float
    vp: float
    vs: float
    density: float
    qp: float
    qs: float


# ---------------------------------------------------------------------------
# File-level operations
# ---------------------------------------------------------------------------

def parse_model_layers(model_path: str) -> List[LayerQ]:
    """Read a model file and return layers (auto-detect 4 or 6 columns).

    Parameters
    ----------
    model_path : str
        Path to model file (HVf format).

    Returns
    -------
    List[LayerQ]
        Parsed layers. If the file has only 4 columns, Qp and Qs are
        set to 0.0 (indicating "not provided").
    """
    text = Path(model_path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    n_layers = int(lines[0])
    layers: List[LayerQ] = []

    for i in range(1, min(n_layers + 1, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue

        thickness = float(parts[0])
        vp = float(parts[1])
        vs = float(parts[2])
        density = float(parts[3])

        qp = float(parts[4]) if len(parts) >= 5 else 0.0
        qs = float(parts[5]) if len(parts) >= 6 else 0.0

        layers.append(LayerQ(thickness, vp, vs, density, qp, qs))

    return layers


def layers_have_q(layers: List[LayerQ]) -> bool:
    """Check whether any layer already has non-zero Q values."""
    return any(l.qp > 0 and l.qs > 0 for l in layers)


def add_q_to_layers(
    layers: List[LayerQ],
    formula: str = "default",
    overwrite: bool = False,
    **kwargs,
) -> List[LayerQ]:
    """Auto-compute Qp/Qs for layers that lack them.

    Parameters
    ----------
    layers : List[LayerQ]
        Input layers.
    formula : str
        Estimation strategy.
    overwrite : bool
        If True, overwrite existing Q values. If False, only fill
        layers where Qp == 0 or Qs == 0.
    **kwargs
        Extra parameters for the formula.

    Returns
    -------
    List[LayerQ]
        New list with Q values filled in.
    """
    result = []
    for layer in layers:
        needs_q = overwrite or layer.qp <= 0 or layer.qs <= 0
        if needs_q:
            qp, qs = compute_qp_qs(
                layer.vp, layer.vs, formula=formula, **kwargs
            )
            result.append(LayerQ(
                layer.thickness, layer.vp, layer.vs,
                layer.density, qp, qs,
            ))
        else:
            result.append(layer)
    return result


def layers_to_gpell_text(layers: List[LayerQ], include_q: bool = True) -> str:
    """Format layers as gpell model text.

    Parameters
    ----------
    layers : List[LayerQ]
        Layer list.
    include_q : bool
        Whether to include Qp/Qs columns.

    Returns
    -------
    str
        Model text ready for gpell.
    """
    lines = [str(len(layers))]
    for layer in layers:
        base = (
            f"{layer.thickness:.6g} {layer.vp:.6g} "
            f"{layer.vs:.6g} {layer.density:.6g}"
        )
        if include_q and (layer.qp > 0 or layer.qs > 0):
            base += f" {layer.qp:.0f} {layer.qs:.0f}"
        lines.append(base)
    return "\n".join(lines) + "\n"


def process_model_file(
    input_path: str,
    output_path: Optional[str] = None,
    formula: str = "default",
    overwrite: bool = False,
    **kwargs,
) -> str:
    """Read a model file, add Q if needed, write result.

    Parameters
    ----------
    input_path : str
        Source model file.
    output_path : str, optional
        Destination. If None, returns text without writing.
    formula : str
        Q estimation strategy.
    overwrite : bool
        Overwrite existing Q values.
    **kwargs
        Extra parameters for the formula.

    Returns
    -------
    str
        The processed model text.
    """
    layers = parse_model_layers(input_path)
    layers = add_q_to_layers(
        layers, formula=formula, overwrite=overwrite, **kwargs
    )
    text = layers_to_gpell_text(layers, include_q=True)

    if output_path is not None:
        Path(output_path).write_text(text, encoding="utf-8")

    return text


__all__ = [
    "compute_qp_qs",
    "estimate_default",
    "estimate_brocher",
    "estimate_constant",
    "parse_model_layers",
    "layers_have_q",
    "add_q_to_layers",
    "layers_to_gpell_text",
    "process_model_file",
    "LayerQ",
]
