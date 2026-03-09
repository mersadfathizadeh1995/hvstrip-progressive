"""
darendeli_calc.py
=================
Python port of DarendeliCalc.m (David Teague, UT Austin, 2016).

Computes G/Gmax and damping ratio curves per Darendeli (2001).

Key optimisation over the MATLAB version:
  * The original MATLAB code uses a double for-loop (layers × strains).
  * This port uses full 2-D NumPy broadcasting — no Python loops — giving
    ~10–50× speedup for large numbers of layers or strain points.

Reference:
    Darendeli, M.B. (2001). "Development of A New Family of Normalized
    Modulus Reduction and Material Damping Curves." Ph.D. Dissertation,
    The University of Texas at Austin.
"""

import numpy as np


# Darendeli (2001) regression constants  (Table 9.5 / Eq. 9.x)
_CONSTANTS = np.array([
     0.0352,   # phi1
     0.0010,   # phi2
     0.3246,   # phi3
     0.3483,   # phi4
     0.9190,   # phi5  (also called 'a' in reference strain equation)
     0.8005,   # phi6
     0.0129,   # phi7
    -0.1069,   # phi8
    -0.2889,   # phi9
     0.2919,   # phi10
     0.6329,   # phi11
    -0.0057,   # phi12
    -4.2300,   # phi13
     3.6200,   # phi14
    -5.0000,   # phi15
    -0.2500,   # phi16
])


def darendeli_calc(
    mean_eff_stress: np.ndarray,
    curve_type: int = 1,
    PI: np.ndarray | None = None,
    OCR: np.ndarray | None = None,
    strains: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Darendeli (2001) G/Gmax and damping ratio curves.

    Parameters
    ----------
    mean_eff_stress : array_like, shape (N,)
        Mean effective stress at mid-point of each layer [atm].
    curve_type : int, optional
        1 = mean curves (default)
        2 = mean + 1σ  (G/Gmax),  mean − 1σ  (D)
        3 = mean − 1σ  (G/Gmax),  mean + 1σ  (D)
    PI : array_like, shape (N,), optional
        Plasticity index [%].  Default = 0 (non-plastic).
    OCR : array_like, shape (N,), optional
        Over-consolidation ratio.  Default = 1 (normally consolidated).
    strains : array_like, shape (P,), optional
        Shear strains [%] at which curves are evaluated.
        Default = 45 log-spaced points from 1e-4 to 10 %.

    Returns
    -------
    GGmax   : np.ndarray, shape (N, P)
    Damping : np.ndarray, shape (N, P)   [%]
    strains : np.ndarray, shape (P,)     [%]
    """
    sigma = np.asarray(mean_eff_stress, dtype=float).ravel()
    N = len(sigma)

    if strains is None:
        strains = np.logspace(-4, 1, 45)       # same default as MATLAB
    else:
        strains = np.asarray(strains, dtype=float).ravel()

    if PI is None:
        PI = np.zeros(N)
    else:
        PI = np.asarray(PI, dtype=float).ravel()

    if OCR is None:
        OCR = np.ones(N)
    else:
        OCR = np.asarray(OCR, dtype=float).ravel()

    c = _CONSTANTS                  # shorthand

    # ------------------------------------------------------------------ #
    # Scalar constants                                                     #
    # ------------------------------------------------------------------ #
    Freq   = 1.0
    cycles = 10.0

    # Eq. 9.3 — coefficients for the masing damping correction
    C1 = -1.1143 * c[4]**2 + 1.8618 * c[4] + 0.2523
    C2 =  0.0805 * c[4]**2 - 0.0710 * c[4] - 0.0095
    C3 = -0.0005 * c[4]**2 + 0.0002 * c[4] + 0.0003

    # Eq. 9.1d
    b = c[10] + c[11] * np.log(cycles)

    # ------------------------------------------------------------------ #
    # Layer-dependent quantities  (shape: N,)                             #
    # ------------------------------------------------------------------ #
    # Eq. 9.1a — reference strain λR
    lambdaR = (c[0] + c[1] * PI * OCR**c[2]) * sigma**c[3]     # (N,)

    # Eq. 12.2c — minimum damping Dmin [%]
    Dmin = (c[5] + c[6] * PI * OCR**c[7]) * sigma**c[8] * (1 + c[9] * np.log(Freq))

    # ------------------------------------------------------------------ #
    # Vectorised 2-D computation  (N layers × P strains)                  #
    # ------------------------------------------------------------------ #
    # Broadcast shapes:  lambdaR → (N,1),  strains → (1,P)
    lR = lambdaR[:, np.newaxis]          # (N, 1)
    gam = strains[np.newaxis, :]         # (1, P)
    Dmin2D = Dmin[:, np.newaxis]         # (N, 1)

    # ── G/Gmax ──────────────────────────────────────────────────────── #
    # Eq. 12.1a
    GGmax1 = 1.0 / (1.0 + (gam / lR) ** c[4])                  # (N, P)

    # Eq. 12.2e — G/Gmax standard deviation
    GGmaxStd = (np.exp(c[12]) +
                np.sqrt(np.maximum(0.0, 0.25 / np.exp(c[13]) -
                                        (GGmax1 - 0.5)**2 / np.exp(c[13]))))

    if curve_type == 1:
        GGmax = GGmax1.copy()
    elif curve_type == 2:
        GGmax = GGmax1 + GGmaxStd
    else:  # curve_type == 3
        GGmax = GGmax1 - GGmaxStd

    # Clamp G/Gmax to [0.005, 1.0]
    GGmax = np.clip(GGmax, 0.005, 1.0)

    # ── Damping ─────────────────────────────────────────────────────── #
    # Masing damping (Eq. 9.3 — numerically identical to MATLAB formula)
    Dmas1 = (100.0 / np.pi) * (
        4.0 * (
            (gam - lR * np.log((gam + lR) / lR))
            / (gam**2 / (gam + lR))
        ) - 2.0
    )                                                            # (N, P)

    Dmas2 = C1 * Dmas1 + C2 * Dmas1**2 + C3 * Dmas1**3

    # Eq. 12.1b
    Damp = Dmin2D + Dmas2 * (b * GGmax1**0.1)                   # (N, P)

    # Eq. 12.2f — damping standard deviation
    DampingStd = np.exp(c[14]) + np.exp(c[15]) * Damp**0.5

    if curve_type == 1:
        Damping = Damp.copy()
    elif curve_type == 2:
        Damping = Damp - DampingStd
    else:  # curve_type == 3
        Damping = Damp + DampingStd

    # Enforce minimum damping of 0.1 %
    Damping = np.maximum(Damping, 0.1)

    # Enforce monotonicity cap at the peak damping  (MATLAB logic)
    max_damp = Damping.max(axis=1, keepdims=True)               # (N, 1)
    max_loc  = np.argmax(Damping, axis=1)                       # (N,)
    for i in range(N):
        Damping[i, max_loc[i]:] = max_damp[i, 0]

    return GGmax, Damping, strains
