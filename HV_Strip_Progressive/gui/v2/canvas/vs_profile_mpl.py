"""``VsProfileMpl`` — the Vs-vs-depth step preview on matplotlib.

Ported (copy-not-import) from the legacy
``gui/widgets/profile_preview_widget.py`` — the Qt5Agg/QT_API poison
stripped, rebased on :class:`MplFigureWidget` (PySide6 ``backend_qtagg``),
and enhanced: theme-aware colors, the Vs30 marker line (parity with the
legacy app's Vs Profile view), layer-count title, half-space shading.
Consumes plain profile DICTS (``api.profile_to_dict`` shape) — GUI-only,
no core imports.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from HV_Strip_Progressive.gui.v2.canvas.constants import layers_to_staircase
from HV_Strip_Progressive.gui.v2.canvas.mpl_widget import MplFigureWidget


class VsProfileMpl(MplFigureWidget):
    """Compact Vs step-function preview of ONE profile."""

    def __init__(self, parent=None, *, toolbar: bool = True) -> None:
        super().__init__(parent, figsize=(3.6, 4.8), toolbar=toolbar)
        self._profile: Optional[Dict[str, Any]] = None
        self._vs30: Optional[float] = None
        self.draw_placeholder("No profile loaded")

    # ------------------------------------------------------------------
    def set_profile(
        self,
        profile: Optional[Dict[str, Any]],
        vs30: Optional[float] = None,
    ) -> None:
        """Show *profile* (an ``api`` profile dict with ``layers``)."""
        self._profile = profile
        self._vs30 = vs30
        self._redraw()

    def clear(self) -> None:
        self.set_profile(None)

    def apply_theme(self, palette) -> None:  # redraw with the new colors
        super().apply_theme(palette)
        self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        layers = (self._profile or {}).get("layers") or []
        if not layers:
            self.draw_placeholder("No profile loaded")
            return

        p = self._palette
        line_color = p.accent if p else "teal"
        hs_color = p.danger if p else "red"
        vs30_color = p.success if p else "blue"

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # ONE staircase source (canvas.constants) — the last layer is the
        # half-space, extended by the shared proportional rule so this
        # preview matches the Forward/Strip canvases exactly.
        vs_vals, depths = layers_to_staircase(layers)
        n_finite = max(len(layers) - 1, 0)
        total_finite = sum(
            float(ly.get("thickness", ly.get("h", 0.0)))
            for ly in layers[:-1])
        z = depths[-1] if depths else 0.0

        ax.plot(vs_vals, depths, color=line_color, linewidth=1.8)
        if len(layers) > 1:
            ax.axhline(total_finite, color=hs_color, linewidth=0.8,
                       linestyle="--", alpha=0.6)
            ax.axhspan(total_finite, z, color=hs_color, alpha=0.05)
        if self._vs30 and z >= 30.0:
            ax.axhline(30.0, color=vs30_color, linewidth=0.9,
                       linestyle="-.", alpha=0.8)
            ax.annotate(f"Vs30={self._vs30:.0f}",
                        xy=(max(vs_vals), 30.0), xytext=(-4, -4),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=8, color=vs30_color)

        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        name = (self._profile or {}).get("name", "")
        ax.set_title(f"{name} · {n_finite}L" if name else f"{n_finite}L",
                     fontsize=10)
        ax.tick_params(labelsize=8)
        self.style_axes(ax)
        self.canvas.draw_idle()


__all__ = ["VsProfileMpl"]
