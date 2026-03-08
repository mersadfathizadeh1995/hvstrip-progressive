"""
mean_eff_stress.py
==================
Python port of mean_eff_stress.m (David Teague, UT Austin, 2016).

Computes mean effective stress at the mid-point of each layer assuming:
  * Hydrostatic pore-water pressure conditions
  * K0 = 0.5 (at-rest earth pressure coefficient, default)
  * Water table determined from Vp > 1400 m/s  OR  user-supplied GWT depth

Result is returned in atmospheres [atm].
"""

import numpy as np


def mean_eff_stress(
    depth,
    gamma,
    VpOrGWT,
    K=None,
):
    """
    Calculate mean effective stress at the mid-point of each *soil* layer.
    (The half-space / last entry is excluded, matching MATLAB which returns
    N-1 values for N-layer input.)

    Parameters
    ----------
    depth : array_like, shape (N,) or (N, M)
        Depth to TOP of each layer [m].
    gamma : array_like, shape (N,) or (N, M)
        Unit weight for each layer [kN/m3].
    VpOrGWT : array_like (N,) or (N, M)  OR  scalar float
        * Scalar: GWT depth [m].
        * Array: Vp [m/s]; first layer where Vp > 1400 m/s gives GWT depth.
    K : array_like or float, optional
        At-rest earth pressure coefficient K0 for each layer.
        Scalar applies to all layers.  Default = 0.5.

    Returns
    -------
    mid_mean_sigma_eff : np.ndarray, shape (N-1,) or (N-1, M)
        Mean effective stress at mid-point of each layer [atm].
    """
    depth = np.asarray(depth, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    single_profile = (depth.ndim == 1)

    # Ensure 2-D column-oriented: (N, M)
    if depth.ndim == 1:
        depth = depth[:, np.newaxis]
        gamma = gamma[:, np.newaxis]

    N, M = depth.shape
    n_layers = N - 1          # finite-thickness layers (excludes half-space)

    # Default K0 = 0.5
    if K is None:
        K = 0.5 * np.ones((n_layers, M))
    else:
        K = np.asarray(K, dtype=float)
        K = np.broadcast_to(K, (n_layers, M)).copy()

    # ------------------------------------------------------------------ #
    # Geometry                                                             #
    # ------------------------------------------------------------------ #
    mid_depth = (depth[:n_layers, :] + depth[1:, :]) / 2.0      # (n_layers, M)
    thickness  = depth[1:, :] - depth[:n_layers, :]              # (n_layers, M)

    # ------------------------------------------------------------------ #
    # Vertical total stress at mid-point (vectorised cumsum)              #
    # ------------------------------------------------------------------ #
    d_sigma  = thickness * gamma[:n_layers, :]                   # (n_layers, M)
    cum_sigma = np.cumsum(d_sigma, axis=0)                       # (n_layers, M)
    # Stress at mid = sum of full layers above + half the current layer
    mid_sigma = cum_sigma - d_sigma * 0.5                        # (n_layers, M)

    # ------------------------------------------------------------------ #
    # Pore pressure at mid-point (hydrostatic)                            #
    # ------------------------------------------------------------------ #
    VpOrGWT = np.asarray(VpOrGWT, dtype=float)
    mid_u    = np.zeros((n_layers, M))

    for m in range(M):
        if VpOrGWT.ndim == 0:
            gwt_depth = float(VpOrGWT)
        else:
            Vp_col = VpOrGWT if VpOrGWT.ndim == 1 else VpOrGWT[:, m]
            sat_ids = np.where(Vp_col > 1400.0)[0]
            if len(sat_ids) > 0:
                gwt_depth = float(depth[sat_ids[0], m])
            else:
                gwt_depth = float(depth[-1, m])   # no saturated layer → no GWT

        below_gwt = mid_depth[:, m] > gwt_depth
        mid_u[below_gwt, m] = (mid_depth[below_gwt, m] - gwt_depth) * 9.81

    # ------------------------------------------------------------------ #
    # Effective and mean stresses                                          #
    # ------------------------------------------------------------------ #
    mid_sigma_eff     = mid_sigma - mid_u
    mid_mean_sigma_eff = (1.0 + 2.0 * K) / 3.0 * mid_sigma_eff / 101.325

    if single_profile:
        return mid_mean_sigma_eff[:, 0]
    return mid_mean_sigma_eff
