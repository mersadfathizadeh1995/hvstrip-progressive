"""HV Overlay View — Multi-step HV curve overlay from stripping results.

Right-panel canvas tab showing all step HV curves overlaid,
color-coded by stripping step, with peak markers.
"""
import os
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox,
)

from ..widgets.plot_widget import MatplotlibWidget


class HVOverlayView(QWidget):
    """Canvas view for multi-step HV overlay."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._strip_dir = None
        self._step_data = []  # List of {name, freqs, amps}
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self._plot = MatplotlibWidget(figsize=(14, 6))
        lay.addWidget(self._plot)

        opts = QHBoxLayout()
        self._log_x = QCheckBox("Log X"); self._log_x.setChecked(True)
        self._log_x.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._log_x)

        self._grid = QCheckBox("Grid"); self._grid.setChecked(True)
        self._grid.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._grid)

        self._peaks = QCheckBox("Show Peaks"); self._peaks.setChecked(True)
        self._peaks.stateChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._peaks)

        opts.addWidget(QLabel("Colormap:"))
        self._cmap = QComboBox()
        self._cmap.addItems(["cividis", "viridis", "plasma", "inferno", "tab10", "tab20"])
        self._cmap.currentTextChanged.connect(lambda _: self._redraw())
        opts.addWidget(self._cmap)
        opts.addStretch()
        lay.addLayout(opts)

        self._info = QLabel("")
        self._info.setStyleSheet("font-size: 11px; color: #555;")
        lay.addWidget(self._info)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def load_strip_dir(self, strip_dir):
        self._strip_dir = strip_dir
        self._step_data = self._collect_steps(strip_dir)
        self._info.setText(f"Loaded {len(self._step_data)} steps from {os.path.basename(strip_dir)}")
        self._redraw()

    # ══════════════════════════════════════════════════════════════
    #  DATA LOADING
    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def _collect_steps(strip_dir):
        from pathlib import Path
        steps = []
        base = Path(strip_dir)
        folders = sorted([d for d in base.iterdir()
                          if d.is_dir() and d.name.startswith("Step")],
                         key=lambda d: d.name)
        for folder in folders:
            csv_file = folder / "hv_curve.csv"
            if not csv_file.exists():
                for f in folder.glob("*.csv"):
                    if "hv" in f.name.lower():
                        csv_file = f
                        break
            if csv_file.exists():
                try:
                    data = np.loadtxt(str(csv_file), delimiter=",", skiprows=1)
                    if data.ndim == 2 and data.shape[1] >= 2:
                        steps.append({
                            "name": folder.name,
                            "freqs": data[:, 0],
                            "amps": data[:, 1],
                        })
                except Exception:
                    pass
        return steps

    # ══════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════
    def _redraw(self):
        fig = self._plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if not self._step_data:
            ax.text(0.5, 0.5, "No stripping data loaded\nRun a stripping workflow first",
                    ha="center", va="center", color="gray", fontsize=14,
                    transform=ax.transAxes)
            self._plot.refresh()
            return

        n = len(self._step_data)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(self._cmap.currentText(), max(n, 2))

        for i, step in enumerate(self._step_data):
            color = cmap(i / max(n - 1, 1))
            ax.plot(step["freqs"], step["amps"], color=color, lw=1.5,
                    label=step["name"], alpha=0.85)

            if self._peaks.isChecked():
                idx = np.argmax(step["amps"])
                ax.plot(step["freqs"][idx], step["amps"][idx],
                        "*", color=color, ms=10, zorder=5)

        if self._log_x.isChecked():
            ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Ratio")
        ax.set_title("HV Curves — Progressive Layer Stripping")
        if self._grid.isChecked():
            ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        fig.tight_layout()
        self._plot.refresh()
