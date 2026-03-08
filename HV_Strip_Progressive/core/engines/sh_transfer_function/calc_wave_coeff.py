"""
calc_wave_coeff.py
==================
Python port of calcWaveCoeff.m (David Teague, UT Austin, 2016).

Computes upgoing (A) and downgoing (B) wave amplitude coefficients for
vertically-propagating SH waves in a layered Kelvin-Voigt solid system.

Equations follow:
  * Kottke & Rathje (2009) Eq. 2.4 — complex shear modulus
  * Kramer (1996) §7.3 Eqs. 7.9, 7.10, 7.34a/b, 7.35

MATLAB quirk preserved:
  comp_k[:, 0] = 1  before the propagation loop.  This avoids a DC
  singularity (ω=0) where the impedance ratio would be 0/0.
"""

import numpy as np


def calc_wave_coeff(
    freq: np.ndarray,
    depth: np.ndarray,
    G: np.ndarray,
    D: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute upgoing and downgoing SH-wave amplitude coefficients.

    Parameters
    ----------
    freq : array_like, shape (F,)
        Frequencies [Hz].
    depth : array_like, shape (N,)
        Depth to TOP of each layer [m].  Last entry = top of half-space.
    G : array_like, shape (N,)
        Small-strain shear modulus for each layer [kPa].
    D : array_like, shape (N,)
        Damping ratio for each layer [-] (e.g., 0.05 for 5 %).
    gamma : array_like, shape (N,)
        Unit weight for each layer [kN/m³].

    Returns
    -------
    waveA : np.ndarray, shape (N, F)   complex
        Upgoing wave amplitude coefficients — A[0] is at the surface.
    waveB : np.ndarray, shape (N, F)   complex
        Downgoing wave amplitude coefficients — B[0] is at the surface.

    Notes
    -----
    The recursion starts with A[0] = B[0] = 1 (arbitrary normalisation;
    only the *ratio* of coefficients matters for the transfer function).
    """
    freq  = np.asarray(freq,  dtype=float).ravel()
    depth = np.asarray(depth, dtype=float).ravel()
    G     = np.asarray(G,     dtype=float).ravel()
    D     = np.asarray(D,     dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()

    N = len(depth)   # number of layers (including half-space)
    F = len(freq)

    # ── Angular frequency ──────────────────────────────────────────── #
    omega = 2.0 * np.pi * freq                                  # (F,)

    # ── Layer thicknesses (half-space gets 0) ─────────────────────── #
    h = np.zeros(N)
    h[:-1] = np.diff(depth)                                     # (N,)

    # ── Mass density ─────────────────────────────────────────────── #
    rho = gamma / 9.80665                                        # (N,) [kN·s²/m⁴]

    # ── Complex shear modulus  (Kottke & Rathje Eq. 2.4) ─────────── #
    compG = G * (1.0 - 2.0 * D**2
                 + 2j * D * np.sqrt(1.0 - D**2))               # (N,)

    # ── Complex shear-wave velocity  (Kramer Eq. 7.9) ────────────── #
    compVs = np.sqrt(compG / rho)                               # (N,)

    # ── Complex wavenumber  (Kramer Eq. 7.10)                        #
    #    k* = ω / Vs*  →  shape (N, F)                               #
    comp_k = omega[np.newaxis, :] / compVs[:, np.newaxis]       # (N, F)

    # ── Impedance ratio  (Kramer Eq. 7.35)                           #
    #    α_m = (k*_m · G_m) / (k*_{m+1} · G_{m+1})                  #
    # MATLAB quirk: set DC column to 1 to avoid 0/0 at ω=0
    comp_k_dc = comp_k.copy()
    comp_k_dc[:, 0] = 1.0 + 0j

    comp_Imp = np.zeros((N, F), dtype=complex)
    for m in range(N - 1):
        comp_Imp[m, :] = (comp_k_dc[m, :] * compG[m]) / (comp_k_dc[m + 1, :] * compG[m + 1])

    # ── Phase propagation term  i·k*·h ────────────────────────────── #
    expTerm = 1j * comp_k * h[:, np.newaxis]                    # (N, F)

    # ── Wave coefficient recursion  (Kramer Eqs. 7.34a/b) ─────────── #
    waveA = np.ones((N, F), dtype=complex)
    waveB = np.ones((N, F), dtype=complex)

    for m in range(N - 1):
        exp_pos = np.exp( expTerm[m, :])
        exp_neg = np.exp(-expTerm[m, :])
        alpha   = comp_Imp[m, :]

        waveA[m + 1, :] = (0.5 * waveA[m, :] * (1.0 + alpha) * exp_pos
                          + 0.5 * waveB[m, :] * (1.0 - alpha) * exp_neg)

        waveB[m + 1, :] = (0.5 * waveA[m, :] * (1.0 - alpha) * exp_pos
                          + 0.5 * waveB[m, :] * (1.0 + alpha) * exp_neg)

    return waveA, waveB
