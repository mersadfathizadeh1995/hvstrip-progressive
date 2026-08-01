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

# Default halfspace draw extension (metres) below the last real layer
HALFSPACE_EXTENSION_M = 50.0
