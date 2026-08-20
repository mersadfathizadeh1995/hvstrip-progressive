"""``VsContextPanel`` — the Vs mini-panel beside the interactive HV
figure (spec 002 FR-9; the legacy strip wizard's right-hand Vs pane).

Shows the step's Vs staircase (through the ONE shared staircase builder),
Vs30 / VsAvg-to-bedrock toggles, a bedrock-interface combo **and**
click-on-the-plot bedrock selection (the legacy ``VsProfileView`` idiom),
plus a live readout.  Numbers come from the AppState's ``vs_context``
passthrough (Vs30 = 30 m with half-space extrapolation; VsAvg = to the
bedrock without it — the legacy rules).  Emits ``context_changed`` with
``{"vs30", "vsavg", "bedrock_depth"}`` so the owner can persist the
values with the step's picks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.canvas.constants import layers_to_staircase
from HV_Strip_Progressive.gui.v2.canvas.mpl_widget import MplFigureWidget

_AUTO_LABEL = "(auto — bottom of finite layers)"


class VsContextPanel(QWidget):
    """Vs staircase + bedrock selection + Vs30/VsAvg readout."""

    #: {"vs30": float|None, "vsavg": float|None, "bedrock_depth": float|None}
    context_changed = Signal(dict)

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._layers: List[Dict[str, Any]] = []
        self._interfaces: List[float] = []
        self._bedrock: Optional[float] = None
        self._context: Dict[str, Any] = {}
        self._loading = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        self._chk_vs30 = QCheckBox("Vs30")
        self._chk_vs30.setChecked(True)
        self._chk_vs30.toggled.connect(lambda _o: self._redraw())
        self._chk_vsavg = QCheckBox("VsAvg")
        self._chk_vsavg.setChecked(True)
        self._chk_vsavg.toggled.connect(lambda _o: self._redraw())
        row.addWidget(self._chk_vs30)
        row.addWidget(self._chk_vsavg)
        row.addStretch(1)
        lay.addLayout(row)

        brow = QHBoxLayout()
        brow.addWidget(QLabel("Bedrock:"))
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_combo)
        brow.addWidget(self._combo, 1)
        lay.addLayout(brow)

        self._fig = MplFigureWidget(self, figsize=(3.0, 4.6), toolbar=False)
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        lay.addWidget(self._fig, 1)

        self._readout = QLabel("")
        self._readout.setProperty("role", "caption")
        self._readout.setWordWrap(True)
        lay.addWidget(self._readout)

    # ------------------------------------------------------------------
    #  Public surface
    # ------------------------------------------------------------------
    def set_profile(self, layers: List[Dict[str, Any]],
                    bedrock_depth: Optional[float] = None) -> None:
        """Show a layer stack; optionally restore a persisted bedrock."""
        self._layers = [dict(ly) for ly in (layers or [])]
        self._bedrock = bedrock_depth
        self._recompute(emit=False)
        self._rebuild_combo()
        self._redraw()

    def context(self) -> Dict[str, Any]:
        """The current ``{"vs30", "vsavg", "bedrock_depth"}``."""
        return {
            "vs30": self._context.get("vs30"),
            "vsavg": self._context.get("vsavg"),
            "bedrock_depth": self._bedrock,
        }

    def apply_theme(self, palette) -> None:
        self._fig.apply_theme(palette)
        self._redraw()

    # ------------------------------------------------------------------
    #  Bedrock selection (combo + click-on-plot, the legacy idiom)
    # ------------------------------------------------------------------
    def _rebuild_combo(self) -> None:
        self._loading = True
        try:
            self._combo.clear()
            self._combo.addItem(_AUTO_LABEL)
            for i, z in enumerate(self._interfaces):
                self._combo.addItem(f"Interface {i + 1}: {z:g} m")
            if self._bedrock is not None and self._bedrock in \
                    self._interfaces:
                self._combo.setCurrentIndex(
                    1 + self._interfaces.index(self._bedrock))
        finally:
            self._loading = False

    def _on_combo(self, index: int) -> None:
        if self._loading:
            return
        self._bedrock = (None if index <= 0
                         else self._interfaces[index - 1])
        self._recompute(emit=True)
        self._redraw()

    def _on_click(self, event) -> None:
        """Click on the Vs plot → the nearest interface becomes bedrock."""
        if event.inaxes is None or event.ydata is None:
            return
        if not self._interfaces:
            return
        depth = float(event.ydata)
        nearest = min(self._interfaces, key=lambda z: abs(z - depth))
        self._loading = True
        try:
            self._combo.setCurrentIndex(
                1 + self._interfaces.index(nearest))
        finally:
            self._loading = False
        self._bedrock = nearest
        self._recompute(emit=True)
        self._redraw()

    # ------------------------------------------------------------------
    def _recompute(self, emit: bool) -> None:
        env = self._app_state.vs_context(self._layers,
                                         bedrock_depth=self._bedrock)
        self._context = env if env.get("success") else {}
        self._interfaces = list(self._context.get("interfaces") or [])
        if emit:
            self.context_changed.emit(self.context())

    def _redraw(self) -> None:
        if not self._layers:
            self._fig.draw_placeholder("No model")
            self._readout.setText("")
            return
        p = self._fig.palette_
        line = p.accent if p else "teal"
        vs30_c = p.success if p else "green"
        bed_c = p.danger if p else "darkred"

        fig = self._fig.figure
        fig.clear()
        ax = fig.add_subplot(111)
        vs_vals, depths = layers_to_staircase(self._layers)
        ax.plot(vs_vals, depths, color=line, lw=1.8)
        z_end = depths[-1] if depths else 0.0

        bedrock = self._bedrock or (self._interfaces[-1]
                                    if self._interfaces else None)
        if bedrock is not None:
            explicit = self._bedrock is not None
            ax.axhline(bedrock, color=bed_c,
                       lw=1.6 if explicit else 0.8,
                       ls="-" if explicit else "--", alpha=0.8)
        if self._chk_vs30.isChecked() and z_end >= 30.0 >= 0:
            ax.axhline(30.0, color=vs30_c, lw=0.9, ls="-.", alpha=0.8)

        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=8)
        ax.set_ylabel("Depth (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        self._fig.style_axes(ax)
        self._fig.canvas.draw_idle()

        parts = []
        if self._chk_vs30.isChecked() and self._context.get("vs30"):
            star = "*" if self._context.get("vs30_extrapolated") else ""
            parts.append(f"Vs30 = {self._context['vs30']:.1f} m/s{star}")
        if self._chk_vsavg.isChecked() and self._context.get("vsavg"):
            parts.append(f"VsAvg = {self._context['vsavg']:.1f} m/s")
        if bedrock is not None:
            parts.append(f"Bedrock @ {bedrock:g} m")
        self._readout.setText("  |  ".join(parts))


__all__ = ["VsContextPanel"]
