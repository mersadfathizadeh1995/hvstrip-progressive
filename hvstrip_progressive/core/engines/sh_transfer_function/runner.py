"""
runner.py
=========
Top-level runner for the SH Transfer Function engine.

Provides the single user-facing entry point ``compute_sh_tf()``, which
mirrors ``compute_ellipticity()`` in the Rayleigh ellipticity engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config     import SHTFConfig
from .model_io   import SHTFModel, load_model
from .postprocess import SHTFResult, postprocess_tf
from .tf_for_ground_models import tf_for_ground_models


def compute_sh_tf(
    model:  Union[str, Path, SHTFModel],
    config: Optional[SHTFConfig] = None,
) -> SHTFResult:
    """Compute the SH-wave transfer function for a layered earth model.

    Full pipeline:

    1. Load / validate the model (from file or ``SHTFModel``).
    2. Validate the configuration.
    3. Build frequency vector.
    4. Run the SH transfer function physics
       (``assign_unit_wt → mean_eff_stress → darendeli_calc → calc_wave_coeff``).
    5. Detect F0 and package results into ``SHTFResult``.

    Parameters
    ----------
    model : str, Path, or SHTFModel
        Layered velocity profile. Accepts:

        * ``str`` / ``Path`` ending in ``.mat``  → MATLAB file
        * ``str`` / ``Path`` (any other)         → plain-text thickness file
        * ``SHTFModel``                          → pre-built model object
    config : SHTFConfig, optional
        Computation settings. Uses defaults (0.1–10 Hz, 512 samples,
        Darendeli damping) if not provided.

    Returns
    -------
    SHTFResult
        Post-processed result with:

        * ``.frequencies``   — frequency vector [Hz]
        * ``.amplitudes``    — TF amplitude array (F × M)
        * ``.f0``            — fundamental frequency [Hz] per profile
        * ``.peaks``         — all detected peaks per profile
        * ``.model``         — the input ``SHTFModel``
        * ``.config``        — the ``SHTFConfig`` used
        * ``.metadata``      — auxiliary run information

    Examples
    --------
    **Path-based** (mirrors ellipticity_engine)::

        from sh_transfer_function import SHTFConfig, compute_sh_tf

        config = SHTFConfig(fmin=1.0, fmax=30.0, n_samples=1024)
        result = compute_sh_tf("exampl_model2.txt", config)

        print(result.f0)         # [9.30]  Hz
        print(result.amplitudes) # shape (1024, 1)

    **Array-based**::

        import numpy as np
        from sh_transfer_function import SHTFModel, compute_sh_tf

        model = SHTFModel(
            depth=np.array([0, 2, 5, 9, 14, 20, 28, 38], dtype=float),
            Vs   =np.array([186, 297, 489, 646, 973, 1226, 1335, 1413], dtype=float),
            Vp   =np.array([369, 590, 970, 1282, 1686, 2124, 2312, 2447], dtype=float),
        )
        result = compute_sh_tf(model)

    Raises
    ------
    FileNotFoundError
        If a path is given but the file does not exist.
    ValueError
        If the model or config parameters are invalid.
    """
    # ── 1. Config ────────────────────────────────────────────────────────── #
    if config is None:
        config = SHTFConfig()
    config.validate()

    # ── 2. Model ─────────────────────────────────────────────────────────── #
    m = load_model(model)

    # ── 3. Frequency vector ──────────────────────────────────────────────── #
    freq = config.freq_vector()

    # ── 4. Physics ───────────────────────────────────────────────────────── #
    # Resolve Dsoil argument for tf_for_ground_models
    Dsoil_arg = config.Dsoil  # None → Darendeli auto; float → constant %

    tf_amp, freq_out = tf_for_ground_models(
        depth = m.depth,
        Vs    = m.Vs,
        Vp    = m.Vp,
        Dsoil = Dsoil_arg,
        Drock = config.Drock,
        d_tf  = config.d_tf,
        freq  = freq,
    )

    # ── 5. Post-process ──────────────────────────────────────────────────── #
    return postprocess_tf(
        freq   = freq_out,
        tf     = tf_amp,
        model  = m,
        config = config,
    )


__all__ = ["compute_sh_tf"]
