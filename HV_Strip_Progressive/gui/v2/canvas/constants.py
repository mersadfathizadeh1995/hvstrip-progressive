"""Shared constants for the HVSR plotting canvas."""

from __future__ import annotations

# ----------------------------------------------------------------------
# Multi-model overlay palette (matches the old matplotlib widgets so
# colors stay familiar during the migration).
# ----------------------------------------------------------------------
MULTI_COLORS = (
    "#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#F44336", "#3F51B5",
)

# Semantic colors — FALLBACK defaults only.
#
# The canvases are themed at runtime via ``apply_theme(palette)`` (T028): the
# observed curve is repainted in ``palette.fg`` (so it stays visible on the dark
# card — the old hard-coded ``#111111`` was invisible on dark) and the
# best/synthetic overlays in the HV Invert ``palette.accent`` violet family (the
# old ``#1976D2`` blue clashed with the identity hue).  These constants are the
# pre-theme fallbacks used only until the first ``apply_theme`` call.
OBSERVED_COLOR = "#1E1E1E"       # observed HV curve (fg; re-themed at runtime)
OBSERVED_BAND = (100, 150, 220)  # ±1σ fill RGB (blue tint for visibility)
SYNTHETIC_COLOR = "#6A4CAF"      # single synthetic overlay (accent violet)
PEAK_COLOR = "#E63946"           # peak markers (danger family — high contrast)
BEST_MODEL_COLOR = "#6A4CAF"     # best Vs profile line (accent violet)

# Layer/line widths
OBS_LINE_WIDTH = 2.0
SYN_LINE_WIDTH = 1.5
BEST_LINE_WIDTH = 2.2
ENSEMBLE_LINE_WIDTH = 1.0

# Fill alphas (0-255)
BAND_ALPHA = 120
ENSEMBLE_FILL_ALPHA = 50

# PyQtGraph legend anchors
LEGEND_ANCHORS = {
    "top-right":    ((1, 0), (1, 0)),
    "top-left":     ((0, 0), (0, 0)),
    "bottom-right": ((1, 1), (1, 1)),
    "bottom-left":  ((0, 1), (0, 1)),
}

# Fixed halfspace draw extension (metres) — pass explicitly when a fixed
# depth is wanted; the shared default is the proportional legacy rule below.
HALFSPACE_EXTENSION_M = 50.0


def layers_to_staircase(layers, halfspace_extension=None):
    """Layer dicts → staircase ``(vs_vals, depths)`` lists.

    The ONE staircase builder — the pyqtgraph canvases and the mpl previews
    all draw from here, so the same profile renders the same bottom geometry
    in every tool (the two prior copies had drifted on this).

    Each dict needs ``vs`` and one of ``thickness`` / ``h``; the LAST layer
    is the half-space, drawn ``halfspace_extension`` metres deep — or, when
    ``None`` (the default), the legacy-preview rule
    ``max(0.25 × finite depth, 1 m)``.
    """
    n = len(layers)
    finite_depth = sum(
        float(lay.get("thickness", lay.get("h", 0.0)))
        for lay in layers[:-1])
    if halfspace_extension is None:
        halfspace_extension = max(finite_depth * 0.25, 1.0)
    depths, vs_vals = [], []
    z = 0.0
    for i, lay in enumerate(layers):
        vs = float(lay["vs"])
        h = float(lay.get("thickness", lay.get("h", 0.0)))
        if i == n - 1:
            h = halfspace_extension
        depths.extend([z, z + h])
        vs_vals.extend([vs, vs])
        z += h
    return vs_vals, depths
