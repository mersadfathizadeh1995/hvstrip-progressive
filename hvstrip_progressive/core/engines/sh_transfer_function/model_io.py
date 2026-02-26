"""
model_io.py
===========
Model I/O utilities for the SH Transfer Function engine.

Handles reading Vs profiles from:
  * NumPy arrays directly
  * Text files  (thickness | Vp | Vs | density per row)
  * MATLAB .mat files (depth, Vs, Vp variables)

All loaders return a validated ``SHTFModel`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# SHTFModel — the canonical input container
# ---------------------------------------------------------------------------

@dataclass
class SHTFModel:
    """Layered earth model for SH transfer function computation.

    Attributes
    ----------
    depth : np.ndarray, shape (N,)
        Depth to TOP of each layer [m]. ``depth[0]`` must be 0.
        The last entry is the depth to the top of the half-space.
    Vs : np.ndarray, shape (N,)
        Shear-wave velocity [m/s]. Last entry = half-space Vs.
    Vp : np.ndarray, shape (N,)
        Compression-wave velocity [m/s]. Last entry = half-space Vp.
    source : str
        Origin of the data ('arrays', file path, etc.).
    """

    depth: np.ndarray
    Vs:    np.ndarray
    Vp:    np.ndarray
    source: str = "arrays"

    def __post_init__(self):
        self.depth = np.asarray(self.depth, dtype=float)
        self.Vs    = np.asarray(self.Vs,    dtype=float)
        self.Vp    = np.asarray(self.Vp,    dtype=float)

    @property
    def n_layers(self) -> int:
        """Total number of rows (soil layers + half-space)."""
        return len(self.depth)

    @property
    def n_soil(self) -> int:
        """Number of finite-thickness soil layers (excludes half-space)."""
        return self.n_layers - 1

    def validate(self) -> None:
        """Check internal consistency of the model.

        Raises
        ------
        ValueError
            On shape mismatch, non-zero first depth, non-positive velocities,
            or Vp < Vs in any layer.
        """
        if not (self.depth.shape == self.Vs.shape == self.Vp.shape):
            raise ValueError(
                f"depth, Vs, Vp must have the same shape; "
                f"got {self.depth.shape}, {self.Vs.shape}, {self.Vp.shape}"
            )
        if self.n_layers < 2:
            raise ValueError("Model must have at least 2 rows (1 layer + half-space)")
        if self.depth[0] != 0.0:
            raise ValueError(f"depth[0] must be 0.0, got {self.depth[0]}")
        if not np.all(self.depth[1:] > self.depth[:-1]):
            raise ValueError("depth must be strictly increasing")
        if np.any(self.Vs <= 0):
            raise ValueError("All Vs values must be positive")
        if np.any(self.Vp <= 0):
            raise ValueError("All Vp values must be positive")
        if np.any(self.Vp < self.Vs):
            raise ValueError("Vp must be >= Vs for all layers")

    def summary(self) -> str:
        """One-line human-readable model summary."""
        return (
            f"SHTFModel({self.n_layers} layers, "
            f"depth 0–{self.depth[-1]:.1f} m, "
            f"Vs {self.Vs.min():.0f}–{self.Vs.max():.0f} m/s, "
            f"source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_arrays(
    depth: Union[np.ndarray, Sequence],
    Vs:    Union[np.ndarray, Sequence],
    Vp:    Union[np.ndarray, Sequence],
    validate: bool = True,
) -> SHTFModel:
    """Create an ``SHTFModel`` from numpy arrays or sequences.

    Parameters
    ----------
    depth : array_like, shape (N,)
        Depth to top of each layer [m].  Must start at 0.
    Vs : array_like, shape (N,)
        Shear-wave velocity [m/s].
    Vp : array_like, shape (N,)
        Compression-wave velocity [m/s].
    validate : bool
        If True (default), run model validation.

    Returns
    -------
    SHTFModel
    """
    model = SHTFModel(
        depth=np.asarray(depth, dtype=float),
        Vs=np.asarray(Vs, dtype=float),
        Vp=np.asarray(Vp, dtype=float),
        source="arrays",
    )
    if validate:
        model.validate()
    return model


def load_from_txt(
    path: Union[str, Path],
    col_thickness: int = 0,
    col_vp: int = 1,
    col_vs: int = 2,
    validate: bool = True,
) -> SHTFModel:
    """Load a model from a plain-text file.

    Expected file format::

        <N>                              ← number of rows (layers + half-space)
        <thickness> <Vp> <Vs> <density>  ← one row per layer
        ...
        <0>         <Vp> <Vs> <density>  ← last row: thickness=0 flags half-space

    Column indices are configurable via ``col_*`` arguments (0-based).
    Density is read from the file but **not used** — unit weights are
    computed internally via Mayne (2001).

    Parameters
    ----------
    path : str or Path
        Path to the text file.
    col_thickness : int
        Column index for layer thickness [m]. Default 0.
    col_vp : int
        Column index for Vp [m/s]. Default 1.
    col_vs : int
        Column index for Vs [m/s]. Default 2.
    validate : bool
        Run model validation after loading.

    Returns
    -------
    SHTFModel
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    # Strip comments and empty lines
    data_lines = [l for l in lines[1:] if l.strip() and not l.strip().startswith("#")]

    n = int(lines[0].split()[0])
    if len(data_lines) < n:
        raise ValueError(
            f"File declares {n} rows but only {len(data_lines)} data lines found"
        )

    raw = np.array(
        [[float(v) for v in ln.split()] for ln in data_lines[:n]]
    )

    thickness = raw[:, col_thickness]
    Vp        = raw[:, col_vp]
    Vs        = raw[:, col_vs]

    # Convert thickness → cumulative depth (first layer always starts at 0)
    depth = np.concatenate([[0.0], np.cumsum(thickness[:-1])])

    model = SHTFModel(depth=depth, Vs=Vs, Vp=Vp, source=str(path))
    if validate:
        model.validate()
    return model


def load_from_mat(
    path: Union[str, Path],
    key_depth: str = "depth",
    key_vs: str = "Vs",
    key_vp: str = "Vp",
    validate: bool = True,
) -> SHTFModel:
    """Load a model from a MATLAB ``.mat`` file.

    The file must contain variables named ``depth``, ``Vs``, and ``Vp``
    (column vectors of shape ``(N, 1)`` or flat ``(N,)``).

    Parameters
    ----------
    path : str or Path
        Path to the ``.mat`` file.
    key_depth, key_vs, key_vp : str
        Variable names inside the ``.mat`` file.
    validate : bool
        Run model validation after loading.

    Returns
    -------
    SHTFModel
    """
    try:
        import scipy.io
    except ImportError as e:
        raise ImportError("scipy is required to load .mat files: pip install scipy") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    data = scipy.io.loadmat(str(path))

    def _get(key):
        if key not in data:
            raise KeyError(f"Variable '{key}' not found in {path}. Available: {list(data.keys())}")
        return data[key].ravel().astype(float)

    model = SHTFModel(
        depth=_get(key_depth),
        Vs=_get(key_vs),
        Vp=_get(key_vp),
        source=str(path),
    )
    if validate:
        model.validate()
    return model


def load_model(
    source: Union[str, Path, SHTFModel],
    **kwargs,
) -> SHTFModel:
    """Auto-detecting model loader.

    Dispatches to the appropriate loader based on file extension or type:

    * ``SHTFModel``  → returned as-is (with optional validation)
    * ``.mat``       → :func:`load_from_mat`
    * ``.txt`` / anything else → :func:`load_from_txt`

    Parameters
    ----------
    source : str, Path, or SHTFModel
        Model source.
    **kwargs
        Forwarded to the underlying loader.

    Returns
    -------
    SHTFModel
    """
    if isinstance(source, SHTFModel):
        if kwargs.get("validate", True):
            source.validate()
        return source

    path = Path(source)
    if path.suffix.lower() == ".mat":
        return load_from_mat(path, **kwargs)
    return load_from_txt(path, **kwargs)


__all__ = [
    "SHTFModel",
    "load_model",
    "load_from_arrays",
    "load_from_txt",
    "load_from_mat",
]
