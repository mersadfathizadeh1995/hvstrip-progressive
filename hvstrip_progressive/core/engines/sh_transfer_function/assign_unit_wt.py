"""
assign_unit_wt.py
=================
Python port of assign_unit_wt.m (David Teague, UT Austin, 2016).

Assigns total unit weights to a Vs profile using Equation 32 of Mayne (2001):
    gamma_sat = 8.32 * log10(Vs) - 1.61 * log10(mid_depth)

Unsaturated layers (Vp < 1400 m/s OR above user-specified GWT depth) are
assigned 90 % of the saturated unit weight.

Reference:
    Mayne, P. (2001). "Stress-strength-flow parameters from enhanced in-situ
    tests." Proc. In-Situ 2001, Bali, Indonesia, pp. 27-48.
"""

import numpy as np


def assign_unit_wt(
    depth,
    Vs,
    VpOrGWT,
    gamma_max: float = 23.0,
):
    """
    Assign unit weights to one or more Vs profiles.

    Parameters
    ----------
    depth : array_like, shape (N,) or (N, M)
        Depth to the TOP of each layer [m].  One column per profile.
    Vs : array_like, shape (N,) or (N, M)
        Shear-wave velocity for each layer [m/s].
    VpOrGWT : array_like (N,) or (N, M)  OR  scalar float
        * If scalar: depth to ground water table [m].
        * If array: Vp for each layer [m/s].
          Layers with Vp < 1400 m/s are treated as unsaturated.
    gamma_max : float, optional
        Maximum allowable unit weight [kN/m3].  Default = 23.

    Returns
    -------
    gamma : np.ndarray, shape (N,) or (N, M)
        Unit weight for each layer [kN/m3].
    """
    depth = np.asarray(depth, dtype=float)
    Vs    = np.asarray(Vs,    dtype=float)

    single_profile = (depth.ndim == 1)

    # Ensure 2-D column-oriented: (N, M)
    if depth.ndim == 1:
        depth = depth[:, np.newaxis]
        Vs    = Vs[:, np.newaxis]

    N, M = depth.shape

    # ------------------------------------------------------------------ #
    # Mid-depth of each layer                                              #
    # ------------------------------------------------------------------ #
    mid_depth = np.empty((N, M))
    mid_depth[:-1, :] = (depth[:-1, :] + depth[1:, :]) / 2.0
    # Last layer is a half-space: mid-point at +20 m below its top
    mid_depth[-1, :]  = depth[-1, :] + 20.0

    # ------------------------------------------------------------------ #
    # Saturated unit weight  (Eq. 32, Mayne 2001)                         #
    # ------------------------------------------------------------------ #
    gamma = 8.32 * np.log10(Vs) - 1.61 * np.log10(mid_depth)

    # ------------------------------------------------------------------ #
    # Reduce unsaturated layers to 90% of saturated value                 #
    # ------------------------------------------------------------------ #
    VpOrGWT = np.asarray(VpOrGWT, dtype=float)

    if VpOrGWT.ndim == 0:
        # Scalar: GWT depth provided directly
        gwt = float(VpOrGWT)
        unsat_mask = depth < gwt
    else:
        # Array: Vp; unsaturated where Vp < 1400 m/s
        if VpOrGWT.ndim == 1:
            VpOrGWT = VpOrGWT[:, np.newaxis]
        unsat_mask = VpOrGWT[:N, :M] < 1400.0

    gamma[unsat_mask] *= 0.9

    # ------------------------------------------------------------------ #
    # Cap at gamma_max                                                     #
    # ------------------------------------------------------------------ #
    gamma = np.minimum(gamma, gamma_max)

    if single_profile:
        return gamma[:, 0]
    return gamma
