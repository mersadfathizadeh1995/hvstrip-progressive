"""
SH Transfer Function Engine
============================

Computes the SH-wave (horizontally-polarized shear wave) transfer function
for vertically-propagating seismic waves through a 1-D layered earth model.

Based on the propagator-matrix method (Kramer, 1996), using:
  - Mayne (2001) for unit weights
  - Darendeli (2001) for small-strain damping
  - Standard complex impedance recurrence (Equations 7.34 a/b, Kramer 1996)

Quick start::

    from sh_transfer_function import SHTFConfig, compute_sh_tf

    config = SHTFConfig(fmin=1.0, fmax=30.0, n_samples=1024)
    result = compute_sh_tf("model.txt", config)

    result.f0            # [F0 in Hz]  per profile
    result.amplitudes    # TF curve  (F × M)
    result.frequencies   # freq vector (F,)

Multi-method Convergence Index::

    from sh_transfer_function import compute_convergence_index

    CI = compute_convergence_index(f0_sh=9.30, f0_ell=9.15, f0_dft=9.40)
    # CI = 0.97  →  very good agreement
"""

# ── High-level API ────────────────────────────────────────────────────────── #
from .config      import SHTFConfig
from .model_io    import SHTFModel, load_model, load_from_arrays, load_from_txt, load_from_mat
from .postprocess import SHTFResult, detect_f0, detect_peaks, compute_convergence_index
from .runner      import compute_sh_tf

# ── Low-level / advanced access ───────────────────────────────────────────── #
from .tf_for_ground_models import tf_for_ground_models
from .assign_unit_wt       import assign_unit_wt
from .mean_eff_stress      import mean_eff_stress
from .darendeli_calc       import darendeli_calc
from .calc_wave_coeff      import calc_wave_coeff

__all__ = [
    # ── High-level ──────────────────────── #
    "SHTFConfig",
    "SHTFModel",
    "SHTFResult",
    "compute_sh_tf",
    # ── I/O ─────────────────────────────── #
    "load_model",
    "load_from_arrays",
    "load_from_txt",
    "load_from_mat",
    # ── Post-processing ──────────────────── #
    "detect_f0",
    "detect_peaks",
    "compute_convergence_index",
    # ── Physics (advanced / legacy) ──────── #
    "tf_for_ground_models",
    "assign_unit_wt",
    "mean_eff_stress",
    "darendeli_calc",
    "calc_wave_coeff",
]
