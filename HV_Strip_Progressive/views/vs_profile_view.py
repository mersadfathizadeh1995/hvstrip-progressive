"""Vs Profile View — Step-function velocity profile display.

Right-panel canvas tab showing the current soil profile as a
Vs-depth step function with layer annotations, Vs30, and
user-selectable bedrock interface for VsAvg computation.
"""
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QDoubleSpinBox,
    QComboBox, QSizePolicy,
)

from ..widgets.plot_widget import MatplotlibWidget


class VsProfileView(QWidget):
    """Canvas view for Vs depth profile with Vs30 and VsAvg annotations."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._profile = None
        self._bedrock_idx = None  # user-selected bedrock interface index
        self._vs30 = None
        self._vs_avg = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self._plot = MatplotlibWidget(figsize=(6, 8))
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot.canvas.mpl_connect("button_press_event", self._on_click)
        lay.addWidget(self._plot, 1)

        # Options row 1
        opts = QHBoxLayout()
        self._grid = QCheckBox("Grid"); self._grid.setChecked(True)
        self._grid.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._grid)

        self._annotate = QCheckBox("Annotate"); self._annotate.setChecked(True)
        self._annotate.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._annotate)

        self._chk_vs30 = QCheckBox("Vs30"); self._chk_vs30.setChecked(True)
        self._chk_vs30.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._chk_vs30)

        self._chk_vsavg = QCheckBox("VsAvg to bedrock")
        self._chk_vsavg.setChecked(False)
        self._chk_vsavg.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._chk_vsavg)

        opts.addWidget(QLabel("HS %:"))
        self._hs_pct = QDoubleSpinBox()
        self._hs_pct.setRange(10, 100); self._hs_pct.setValue(25)
        self._hs_pct.setSuffix("%")
        self._hs_pct.valueChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._hs_pct)
        opts.addStretch()
        lay.addLayout(opts)

        # Options row 2 — bedrock interface selector
        bed_row = QHBoxLayout()
        bed_row.addWidget(QLabel("Bedrock interface:"))
        self._bedrock_combo = QComboBox()
        self._bedrock_combo.setToolTip(
            "Select which layer interface is the bedrock.\n"
            "VsAvg will be computed down to this depth.\n"
            "You can also click on the Vs profile plot to select.")
        self._bedrock_combo.currentIndexChanged.connect(self._on_bedrock_changed)
        bed_row.addWidget(self._bedrock_combo, 1)
        self._vsavg_label = QLabel("")
        self._vsavg_label.setStyleSheet("font-size: 11px; color: #333;")
        bed_row.addWidget(self._vsavg_label, 1)
        lay.addLayout(bed_row)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def set_profile(self, profile):
        self._profile = profile
        self._bedrock_idx = None
        self._populate_bedrock_combo()
        self._compute_vs_metrics()
        self._redraw()

    # ══════════════════════════════════════════════════════════════
    #  VS METRICS
    # ══════════════════════════════════════════════════════════════
    def _compute_vs_metrics(self):
        """Compute Vs30 and VsAvg from the current profile."""
        self._vs30 = None
        self._vs_avg = None
        if self._profile is None:
            return
        try:
            from ..core.vs_average import (
                vs_average_from_profile, compute_vs_average,
            )
            # Vs30
            result = vs_average_from_profile(self._profile, target_depth=30.0)
            self._vs30 = result

            # VsAvg to bedrock depth
            if self._bedrock_idx is not None:
                bedrock_depth = self._get_bedrock_depth()
                if bedrock_depth and bedrock_depth > 0:
                    layers = [(L.thickness, L.vs) for L in self._profile.layers]
                    self._vs_avg = compute_vs_average(
                        layers, target_depth=bedrock_depth,
                        use_halfspace=False)
        except ImportError:
            pass

        self._update_vsavg_label()

    def _get_bedrock_depth(self):
        """Return the depth of the selected bedrock interface."""
        if self._profile is None or self._bedrock_idx is None:
            return None
        finite = [L for L in self._profile.layers if not L.is_halfspace]
        z = 0.0
        for i, L in enumerate(finite):
            z += L.thickness
            if i == self._bedrock_idx:
                return z
        return z  # bottom of all finite layers

    def _populate_bedrock_combo(self):
        """Fill the bedrock interface combo with layer boundaries."""
        self._bedrock_combo.blockSignals(True)
        self._bedrock_combo.clear()
        if self._profile is None:
            self._bedrock_combo.blockSignals(False)
            return

        finite = [L for L in self._profile.layers if not L.is_halfspace]
        z = 0.0
        self._bedrock_combo.addItem("(auto — bottom of finite layers)", -1)
        for i, L in enumerate(finite):
            z += L.thickness
            self._bedrock_combo.addItem(
                f"Interface {i+1}: {z:.1f} m  (below Vs={L.vs:.0f})", i)

        # Default: last interface (bottom of finite layers)
        self._bedrock_combo.setCurrentIndex(self._bedrock_combo.count() - 1)
        self._bedrock_idx = len(finite) - 1
        self._bedrock_combo.blockSignals(False)

    def _on_bedrock_changed(self, combo_idx):
        if combo_idx <= 0:
            # Auto mode — use bottom of finite layers
            finite = [L for L in self._profile.layers if not L.is_halfspace]
            self._bedrock_idx = len(finite) - 1 if finite else None
        else:
            self._bedrock_idx = self._bedrock_combo.itemData(combo_idx)
        self._compute_vs_metrics()
        self._redraw()

    def _update_vsavg_label(self):
        parts = []
        if self._vs30 and self._chk_vs30.isChecked():
            ext = " (extrap.)" if self._vs30.extrapolated else ""
            parts.append(f"Vs30 = {self._vs30.vs_avg:.1f} m/s{ext}")
        if self._vs_avg and self._chk_vsavg.isChecked():
            parts.append(
                f"VsAvg(bedrock) = {self._vs_avg.vs_avg:.1f} m/s "
                f"(to {self._vs_avg.target_depth:.1f} m)")
        self._vsavg_label.setText("  |  ".join(parts) if parts else "")

    # ══════════════════════════════════════════════════════════════
    #  CLICK HANDLER
    # ══════════════════════════════════════════════════════════════
    def _on_click(self, event):
        """Click on profile to select bedrock interface."""
        if event.inaxes is None or self._profile is None:
            return
        if self._plot.toolbar.mode:
            return

        clicked_depth = event.ydata
        if clicked_depth is None:
            return

        # Find nearest interface
        finite = [L for L in self._profile.layers if not L.is_halfspace]
        z = 0.0
        best_idx = 0
        best_dist = float("inf")
        for i, L in enumerate(finite):
            z += L.thickness
            d = abs(z - clicked_depth)
            if d < best_dist:
                best_dist = d
                best_idx = i

        self._bedrock_combo.setCurrentIndex(best_idx + 1)  # +1 for "(auto)" item

    # ══════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════
    def _redraw(self):
        fig = self._plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if self._profile is None:
            ax.text(0.5, 0.5, "No profile loaded", ha="center", va="center",
                    color="gray", fontsize=14, transform=ax.transAxes)
            self._plot.refresh()
            return

        prof = self._profile
        depths, vs_vals = [], []
        z = 0.0
        finite = [L for L in prof.layers if not L.is_halfspace]
        hs = [L for L in prof.layers if L.is_halfspace]

        interface_depths = []  # depth at bottom of each finite layer
        for i, L in enumerate(finite):
            depths.append(z); vs_vals.append(L.vs)
            z += L.thickness
            depths.append(z); vs_vals.append(L.vs)
            interface_depths.append(z)
            if self._annotate.isChecked():
                mid = z - L.thickness / 2
                ax.annotate(f"Vs={L.vs:.0f}", xy=(L.vs, mid),
                            xytext=(10, 0), textcoords="offset points",
                            fontsize=7, color="teal", va="center")

        total = z
        if hs:
            hd = total * (self._hs_pct.value() / 100.0)
            depths.append(z); vs_vals.append(hs[0].vs)
            z += hd
            depths.append(z); vs_vals.append(hs[0].vs)
            if self._annotate.isChecked():
                ax.annotate(f"HS: Vs={hs[0].vs:.0f}", xy=(hs[0].vs, total + hd / 2),
                            xytext=(10, 0), textcoords="offset points",
                            fontsize=7, color="gray", va="center")

        ax.plot(vs_vals, depths, color="teal", lw=2)
        ax.fill_betweenx(depths, 0, vs_vals, alpha=0.08, color="teal")

        # Bedrock interface — user-selected (thick) vs bottom (thin dashed)
        bedrock_depth = self._get_bedrock_depth()
        if bedrock_depth and bedrock_depth < total:
            # User selected an intermediate interface
            ax.axhline(bedrock_depth, color="darkred", lw=2.0, ls="-",
                       alpha=0.8, label=f"Bedrock ({bedrock_depth:.1f} m)",
                       zorder=4)
            ax.axhline(total, color="red", lw=0.8, ls="--", alpha=0.4)
        else:
            ax.axhline(total, color="red", lw=1, ls="--", alpha=0.6,
                       label="Bedrock")

        # Vs30 line at 30 m
        if self._chk_vs30.isChecked() and self._vs30:
            ax.axhline(30.0, color="blue", lw=1.2, ls="-.", alpha=0.6,
                       label=f"30 m (Vs30={self._vs30.vs_avg:.0f} m/s)",
                       zorder=3)
            # Annotation box
            vs_range = max(vs_vals) if vs_vals else 500
            ax.annotate(
                f"Vs30 = {self._vs30.vs_avg:.1f} m/s",
                xy=(vs_range * 0.6, 30.0), xytext=(0, -14),
                textcoords="offset points", fontsize=9, color="blue",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="blue", alpha=0.85, lw=0.8))

        # VsAvg to bedrock annotation
        if (self._chk_vsavg.isChecked() and self._vs_avg
                and bedrock_depth and bedrock_depth > 0):
            vs_range = max(vs_vals) if vs_vals else 500
            ax.annotate(
                f"VsAvg = {self._vs_avg.vs_avg:.1f} m/s\n"
                f"(to {bedrock_depth:.1f} m)",
                xy=(vs_range * 0.5, bedrock_depth / 2),
                fontsize=9, color="darkred", fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0",
                          ec="darkred", alpha=0.9, lw=0.8))

        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        n_layers = len(finite)
        title = f"Velocity Profile — {n_layers} layers"
        ax.set_title(title)
        if self._grid.isChecked():
            ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        self._plot.refresh()
        self._update_vsavg_label()
