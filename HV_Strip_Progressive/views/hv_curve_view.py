"""HV Curve View — Interactive HV forward-model display with peak picking.

Right-panel canvas tab showing the computed H/V spectral ratio curve
with interactive peak selection (f0 and secondary peaks).
"""
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QDoubleSpinBox, QSizePolicy,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.style_constants import EMOJI


class HVCurveView(QWidget):
    """Canvas view for a single HV forward-model curve."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._freqs = None
        self._amps = None
        self._profile = None
        self._f0 = None          # (freq, amp, idx)
        self._secondary = []     # [(freq, amp, idx), ...]
        self._pick_f0 = False
        self._pick_sec = False
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        # Plot — fills available space
        self._plot = MatplotlibWidget(figsize=(14, 5))
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot.canvas.mpl_connect("button_press_event", self._on_click)
        lay.addWidget(self._plot, 1)

        # Options row
        opts = QHBoxLayout()
        self._log_x = QCheckBox("Log X"); self._log_x.setChecked(True)
        self._log_y = QCheckBox("Log Y")
        self._grid = QCheckBox("Grid"); self._grid.setChecked(True)
        self._show_vs = QCheckBox("Show Vs"); self._show_vs.setChecked(True)
        self._hs_pct = QDoubleSpinBox()
        self._hs_pct.setRange(10, 100); self._hs_pct.setValue(25)
        self._hs_pct.setSuffix(" %")
        for w in [self._log_x, self._log_y, self._grid, self._show_vs]:
            w.stateChanged.connect(lambda _: self._redraw())
            opts.addWidget(w)
        opts.addWidget(QLabel("HS:"))
        opts.addWidget(self._hs_pct)
        self._hs_pct.valueChanged.connect(lambda _: self._redraw())
        opts.addStretch()
        lay.addLayout(opts)

        # Peak selection row
        peak = QHBoxLayout()
        self._btn_f0 = QPushButton("Select f0")
        self._btn_f0.setCheckable(True)
        self._btn_f0.setStyleSheet(
            "QPushButton:checked { background-color: #4CAF50; color: white; }")
        self._btn_f0.toggled.connect(self._toggle_f0)
        peak.addWidget(self._btn_f0)

        self._btn_sec = QPushButton("Select Secondary")
        self._btn_sec.setCheckable(True)
        self._btn_sec.setStyleSheet(
            "QPushButton:checked { background-color: #FF9800; color: white; }")
        self._btn_sec.toggled.connect(self._toggle_sec)
        peak.addWidget(self._btn_sec)

        btn_clear = QPushButton("Clear Secondary")
        btn_clear.clicked.connect(self._clear_sec)
        peak.addWidget(btn_clear)

        self._sel_label = QLabel("")
        self._sel_label.setStyleSheet("font-size: 11px; color: #333;")
        peak.addWidget(self._sel_label, 1)
        lay.addLayout(peak)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def set_data(self, freqs, amps, profile=None):
        self._freqs = np.asarray(freqs)
        self._amps = np.asarray(amps)
        self._profile = profile
        # Auto-detect f0
        idx = int(np.argmax(self._amps))
        self._f0 = (self._freqs[idx], self._amps[idx], idx)
        self._secondary = []
        self._redraw()
        self._update_label()

    # ══════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════
    def _redraw(self):
        if self._freqs is None:
            return
        fig = self._plot.figure
        fig.clear()

        show_vs = (self._show_vs.isChecked() and self._profile is not None)
        if show_vs:
            import matplotlib.gridspec as gs
            spec = gs.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05, figure=fig)
            ax = fig.add_subplot(spec[0])
            ax_vs = fig.add_subplot(spec[1])
        else:
            ax = fig.add_subplot(111)
            ax_vs = None

        ax.plot(self._freqs, self._amps, color="royalblue", lw=1.5, label="H/V")

        if self._f0:
            f, a, _ = self._f0
            ax.plot(f, a, "*", color="red", ms=14, zorder=5)
            ax.axvline(f, color="red", ls="--", alpha=0.5, lw=0.8)
            ax.annotate(f"f0={f:.3f}", xy=(f, a), xytext=(10, 10),
                        textcoords="offset points", fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        for i, (sf, sa, _) in enumerate(self._secondary):
            ax.plot(sf, sa, "*", color="black", ms=10, zorder=5)
            ax.axvline(sf, color="gray", ls=":", alpha=0.5, lw=0.8)
            # Alternate annotation offset to avoid overlaps
            y_off = -14 if i % 2 == 0 else 12
            ax.annotate(
                f"{sf:.3f} Hz ({sa:.2f})", xy=(sf, sa),
                xytext=(8, y_off), textcoords="offset points",
                fontsize=7, color="#333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="gray", alpha=0.85, lw=0.5))

        if self._log_x.isChecked(): ax.set_xscale("log")
        if self._log_y.isChecked(): ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Ratio")
        ax.set_title("HV Forward Model")
        if self._grid.isChecked(): ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

        if show_vs and ax_vs:
            self._draw_vs(ax_vs)

        self._plot.refresh()

    def _draw_vs(self, ax):
        prof = self._profile
        if not prof:
            return
        depths, vs = [], []
        z = 0.0
        finite = [L for L in prof.layers if not L.is_halfspace]
        hs = [L for L in prof.layers if L.is_halfspace]
        for L in finite:
            depths.append(z); vs.append(L.vs)
            z += L.thickness
            depths.append(z); vs.append(L.vs)
        total = z
        if hs:
            hd = total * (self._hs_pct.value() / 100.0)
            depths.append(z); vs.append(hs[0].vs)
            z += hd
            depths.append(z); vs.append(hs[0].vs)

        ax.plot(vs, depths, color="teal", lw=1.5)
        ax.axhline(total, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=7)
        ax.set_title(f"{len(finite)}L", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.yaxis.tick_right()
        ax.grid(True, alpha=0.2)

        # Vs30 annotation
        try:
            from hvstrip_progressive.core.vs_average import vs_average_from_profile
            result = vs_average_from_profile(prof, target_depth=30.0)
            ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.5)
            ax.annotate(f"Vs30={result.vs_avg:.0f}",
                        xy=(max(vs) * 0.5, 30.0), xytext=(0, -8),
                        textcoords="offset points", fontsize=6,
                        color="blue", fontweight="bold")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  PEAK PICKING
    # ══════════════════════════════════════════════════════════════
    def _toggle_f0(self, on):
        self._pick_f0 = on
        if on: self._btn_sec.setChecked(False)

    def _toggle_sec(self, on):
        self._pick_sec = on
        if on: self._btn_f0.setChecked(False)

    def _clear_sec(self):
        self._secondary = []
        self._redraw()
        self._update_label()

    def _on_click(self, event):
        if event.inaxes is None or self._freqs is None:
            return
        if not (self._pick_f0 or self._pick_sec):
            return
        if self._plot.toolbar.mode:
            return

        cx = event.xdata  # clicked frequency
        # Interpolate amplitude at the exact clicked frequency
        amp_interp = float(np.interp(cx, self._freqs, self._amps))
        # Find nearest data-point index for storage
        idx = int(np.argmin(np.abs(self._freqs - cx)))
        f = float(cx)
        a = amp_interp

        if self._pick_f0:
            self._f0 = (f, a, idx)
            self._btn_f0.setChecked(False)
        elif self._pick_sec:
            self._secondary.append((f, a, idx))

        self._redraw()
        self._update_label()

    def _update_label(self):
        parts = []
        if self._f0:
            f, a, _ = self._f0
            parts.append(f"f0 = {f:.3f} Hz ({a:.2f})")
        for i, (sf, sa, _) in enumerate(self._secondary):
            parts.append(f"Sec.{i+1} = {sf:.3f} Hz ({sa:.2f})")
        self._sel_label.setText("  |  ".join(parts))
