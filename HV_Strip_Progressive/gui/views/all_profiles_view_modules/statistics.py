"""Statistics and color utilities for the All Profiles View."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .constants import BUILTIN_COLORS


def compute_stats(
    results,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute median and standard-deviation of all computed HV curves.

    All curves are interpolated onto the first profile's frequency grid.

    Parameters
    ----------
    results : list[ProfileResult]
        Only results with ``freqs is not None`` are used.

    Returns
    -------
    ref_freqs : ndarray or None
    median_amps : ndarray or None
    std_amps : ndarray or None
    """
    computed = [r for r in results if r.freqs is not None]
    if len(computed) < 2:
        return None, None, None
    ref = computed[0].freqs
    interps = [np.interp(ref, r.freqs, r.amps) for r in computed]
    arr = np.array(interps)
    return ref, np.median(arr, axis=0), np.std(arr, axis=0)


def get_colors(palette_name: str, n: int) -> List:
    """Return *n* colours from the named palette.

    Looks up custom palettes first, then falls back to a matplotlib
    colormap.  If the colormap doesn't exist, ``tab10`` is used.

    Parameters
    ----------
    palette_name : str
        Case-insensitive palette name.
    n : int
        Number of colours needed.
    """
    import matplotlib.pyplot as plt

    key = palette_name.lower()
    if key in BUILTIN_COLORS:
        base = BUILTIN_COLORS[key]
        return [base[i % len(base)] for i in range(n)]
    try:
        cmap = plt.get_cmap(key, max(n, 2))
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    except ValueError:
        cmap = plt.get_cmap("tab10", max(n, 2))
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
