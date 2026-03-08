"""
tf_for_ground_models.py
=======================
Python port of TFforGroundModels.m (David Teague, UT Austin, 2016).

Computes the SH-wave transfer function for one or more layered earth models.
This is the main orchestrator that calls:
    assign_unit_wt  →  mean_eff_stress  →  darendeli_calc  →  calc_wave_coeff

Reference:
    Kramer, S.L. (1996). Geotechnical Earthquake Engineering.
    Prentice Hall: New Jersey, pp. 257–270.
"""

import numpy as np

from .assign_unit_wt import assign_unit_wt
from .mean_eff_stress import mean_eff_stress
from .darendeli_calc import darendeli_calc
from .calc_wave_coeff import calc_wave_coeff


def tf_for_ground_models(
    depth: np.ndarray,
    Vs: np.ndarray,
    Vp: np.ndarray,
    Dsoil=None,
    Drock: float = 0.5,
    d_tf=0,
    freq: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the SH-wave transfer function for vertically-propagating,
    horizontally-polarized shear waves.

    Parameters
    ----------
    depth : array_like, shape (N,) or (N, M)
        Depth to TOP of each layer [m].  One column per profile (M profiles).
    Vs : array_like, shape (N,) or (N, M)
        Shear-wave velocity [m/s].
    Vp : array_like, shape (N,) or (N, M)
        Compression-wave velocity [m/s].
    Dsoil : None | scalar | array_like, optional
        Damping for soil layers.
        • None (default): use Darendeli (2001) Dmin.
        • scalar float: same value (%) for all soil layers.
        • 1-D array: per-layer values (%).
    Drock : float, optional
        Damping ratio of the half-space [%].  Default = 0.5.
    d_tf : int | float | str, optional
        Depth of the recorded ground motion.
        • 0 (default) : surface/outcrop transfer function.
        • 'within'    : top of rock (last layer depth).
        • float > 0   : arbitrary depth [m].
    freq : array_like, shape (F,) or None, optional
        Frequencies [Hz] at which to evaluate the TF.
        Default: 512 log-spaced points from 0.1 to 10 Hz.

    Returns
    -------
    surfaceTF : np.ndarray, shape (F, M)
        SH-wave transfer function amplitude (|TF|).
    freq : np.ndarray, shape (F,)
        Frequency vector [Hz].

    Notes
    -----
    For the surface/outcrop case:
        TF = |A[0] + B[0]| / |2 · A[-1]|
    For the surface/within case:
        TF = |A[0] + B[0]| / |A[-1] + B[-1]|
    """
    # ── Input handling ─────────────────────────────────────────────── #
    depth = np.atleast_2d(np.asarray(depth, dtype=float))
    Vs    = np.atleast_2d(np.asarray(Vs,    dtype=float))
    Vp    = np.atleast_2d(np.asarray(Vp,    dtype=float))

    # Ensure column vectors → 2-D (N, M)
    if depth.shape[0] == 1:
        depth = depth.T
        Vs    = Vs.T
        Vp    = Vp.T

    N, M = depth.shape

    # Default frequency vector
    if freq is None:
        freq = np.logspace(np.log10(0.1), np.log10(10.0), 512)
    else:
        freq = np.asarray(freq, dtype=float).ravel()

    # ── Unit weights ───────────────────────────────────────────────── #
    gamma = assign_unit_wt(depth, Vs, Vp)          # (N,) or (N, M)
    gamma = np.atleast_2d(gamma)
    if gamma.shape[0] == 1:
        gamma = gamma.T

    # ── Shear modulus ──────────────────────────────────────────────── #
    G = Vs**2 * (gamma / 9.80665)                  # small-strain G [kPa*m/kN/m3 → kPa]
    # Note: gamma [kN/m³], g=9.80665 [m/s²] → density [kN·s²/m⁴] × Vs² = G [kPa]

    # ── Damping matrix D  (N × M, in fraction)  ────────────────────── #
    n_soil = N - 1         # number of finite-thickness layers

    if Dsoil is None:
        # Use Darendeli (2001) Dmin for each profile
        sigma_eff = mean_eff_stress(depth, gamma, Vp)   # (n_soil,) or (n_soil, M)
        sigma_eff = np.atleast_2d(sigma_eff)
        if sigma_eff.shape[0] == 1:
            sigma_eff = sigma_eff.T                     # (n_soil, M)

        D = np.zeros((N, M))                            # includes half-space row
        for k in range(M):
            _, damping_k, _ = darendeli_calc(sigma_eff[:, k])
            # Column 0 → smallest strain → Dmin [%] / 100 → fraction
            D[:n_soil, k] = damping_k[:, 0] / 100.0

    elif np.isscalar(Dsoil):
        D = np.full((N, M), Dsoil / 100.0)

    else:
        Dsoil = np.asarray(Dsoil, dtype=float).ravel()
        D = np.zeros((N, M))
        for k in range(M):
            D[:n_soil, k] = Dsoil / 100.0          # broadcast same values per profile

    # Half-space damping (last row)
    D[-1, :] = Drock / 100.0

    # ── Transfer function computation ──────────────────────────────── #
    surfaceTF = np.zeros((M, len(freq)))

    for k in range(M):
        c_depth = depth[:, k].copy()
        c_G     = G[:, k].copy()
        c_D     = D[:, k].copy()
        c_gamma = gamma[:, k].copy()

        # ── Handle d_tf (depth of input motion) ─────────────────── #
        use_within = False

        if isinstance(d_tf, str):
            # 'within' → input at top of half-space
            use_within = True

        elif d_tf != 0:
            use_within = True
            rock_top = c_depth[-1]

            if d_tf < rock_top:
                # Input depth is within the soil column
                idx = np.searchsorted(c_depth, d_tf, side='right')
                c_depth = c_depth[:idx].copy()
                c_depth[-1] = d_tf
                c_G     = c_G[:idx].copy()
                c_D     = c_D[:idx].copy()
                c_gamma = c_gamma[:idx].copy()

            elif d_tf > rock_top:
                # Input depth is below the soil-rock interface
                c_depth = np.append(c_depth, d_tf)
                c_G     = np.append(c_G,     c_G[-1])
                c_D     = np.append(c_D,     c_D[-1])
                c_gamma = np.append(c_gamma, c_gamma[-1])

        waveA, waveB = calc_wave_coeff(freq, c_depth, c_G, c_D, c_gamma)

        if not use_within:
            # Surface/Outcrop:  TF = |A[0]+B[0]| / |2·A[-1]|
            surfaceTF[k, :] = np.abs(
                (waveA[0, :] + waveB[0, :]) / (2.0 * waveA[-1, :])
            )
        else:
            # Surface/Within:  TF = |A[0]+B[0]| / |A[-1]+B[-1]|
            surfaceTF[k, :] = np.abs(
                (waveA[0, :] + waveB[0, :]) / (waveA[-1, :] + waveB[-1, :])
            )

    # Return (F × M) and freq as column vector — same as MATLAB
    return surfaceTF.T, freq
