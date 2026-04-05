"""Interactive Peak Picker Dialog — step-by-step manual peak selection.

Shows each stripping step's HV curve and lets the user click to select f0.
"""
import os
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSplitter, QTextEdit,
)

from ..widgets.plot_widget import MatplotlibWidget


class InteractivePeakPickerDialog(QDialog):
    """Step-through dialog for manually picking peaks on HV curves."""

    def __init__(self, result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Peak Selection")
        self.resize(900, 600)
        self._result = result or {}
        self._steps = []
        self._current = 0
        self._selected_peaks = {}
        self._mode = "f0"

        self._load_steps()
        self._build_ui()
        self._show_current()

    def _load_steps(self):
        """Load HV curve data from result dict."""
        result = self._result
        if isinstance(result, dict):
            # Try to extract step data from result
            if "steps" in result:
                self._steps = result["steps"]
            elif "result" in result and isinstance(result["result"], dict):
                if "steps" in result["result"]:
                    self._steps = result["result"]["steps"]
            # Try directory-based loading
            output_dir = result.get("output_dir", "")
            if output_dir and os.path.isdir(output_dir) and not self._steps:
                import glob
                csv_files = sorted(glob.glob(os.path.join(output_dir, "**/hv_curve*.csv"), recursive=True))
                for i, csv_path in enumerate(csv_files):
                    try:
                        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                        step_name = os.path.basename(os.path.dirname(csv_path))
                        self._steps.append({
                            "name": step_name or f"Step {i}",
                            "freq": data[:, 0].tolist(),
                            "amp": data[:, 1].tolist(),
                        })
                    except Exception:
                        continue

    def _build_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left — plot
        left = MatplotlibWidget(figsize=(8, 5))
        self._plot = left
        splitter.addWidget(left)

        # Right — info + controls
        right_w = QLabel()
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self._step_label = QLabel("Step 0 / 0")
        self._step_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self._step_label)

        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(120)
        right_layout.addWidget(self._info_text)

        # Mode toggle
        mode_row = QHBoxLayout()
        self._btn_f0 = QPushButton("Select f0")
        self._btn_f0.setCheckable(True); self._btn_f0.setChecked(True)
        self._btn_f0.setStyleSheet("background-color: #4CAF50; color: white; padding: 4px;")
        self._btn_f0.clicked.connect(lambda: self._set_mode("f0"))
        self._btn_sec = QPushButton("Select Secondary")
        self._btn_sec.setCheckable(True)
        self._btn_sec.setStyleSheet("background-color: #FF9800; color: white; padding: 4px;")
        self._btn_sec.clicked.connect(lambda: self._set_mode("secondary"))
        mode_row.addWidget(self._btn_f0)
        mode_row.addWidget(self._btn_sec)
        right_layout.addLayout(mode_row)

        self._peak_label = QLabel("Click on the HV curve to select peak")
        right_layout.addWidget(self._peak_label)

        # Navigation
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("← Previous")
        self._btn_prev.clicked.connect(self._prev_step)
        self._btn_next = QPushButton("Next →")
        self._btn_next.clicked.connect(self._next_step)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next)
        right_layout.addLayout(nav)

        right_layout.addStretch()

        # Accept/Cancel
        btn_row = QHBoxLayout()
        btn_accept = QPushButton("Accept All")
        btn_accept.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        btn_accept.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_accept)
        btn_row.addWidget(btn_cancel)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right_w)
        splitter.setSizes([600, 300])
        layout.addWidget(splitter)

        # Connect click
        self._plot.get_figure().canvas.mpl_connect("button_press_event", self._on_click)

    def _set_mode(self, mode):
        self._mode = mode
        self._btn_f0.setChecked(mode == "f0")
        self._btn_sec.setChecked(mode == "secondary")

    def _show_current(self):
        if not self._steps:
            self._step_label.setText("No steps available")
            return

        step = self._steps[self._current]
        self._step_label.setText(f"Step {self._current + 1} / {len(self._steps)}: {step.get('name', '')}")

        fig = self._plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        freq = np.array(step.get("freq", []))
        amp = np.array(step.get("amp", []))
        if len(freq) > 0:
            ax.plot(freq, amp, "b-", linewidth=1.5, label="H/V")
            ax.set_xscale("log")

        # Show selected peaks
        peaks = self._selected_peaks.get(self._current, {})
        if "f0" in peaks:
            f = peaks["f0"]["freq"]
            a = peaks["f0"]["amp"]
            ax.plot(f, a, "r*", markersize=15, label=f"f0={f:.2f} Hz")
        if "secondary" in peaks:
            for sp in peaks["secondary"]:
                ax.plot(sp["freq"], sp["amp"], "o", color="orange", markersize=8)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Amplitude")
        ax.set_title(step.get("name", f"Step {self._current + 1}"))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        self._plot.refresh()

        # Update info
        info = f"Name: {step.get('name', 'N/A')}\n"
        info += f"Points: {len(freq)}\n"
        if "f0" in peaks:
            info += f"f0: {peaks['f0']['freq']:.3f} Hz (amp: {peaks['f0']['amp']:.3f})\n"
        if "secondary" in peaks:
            for i, sp in enumerate(peaks["secondary"]):
                info += f"Secondary {i+1}: {sp['freq']:.3f} Hz\n"
        self._info_text.setPlainText(info)

        self._btn_prev.setEnabled(self._current > 0)
        self._btn_next.setEnabled(self._current < len(self._steps) - 1)

    def _on_click(self, event):
        if event.inaxes is None or not self._steps:
            return
        step = self._steps[self._current]
        freq = np.array(step.get("freq", []))
        amp = np.array(step.get("amp", []))
        if len(freq) == 0:
            return

        # Nearest point (log-distance)
        log_freq = np.log10(np.maximum(freq, 1e-6))
        log_click = np.log10(max(event.xdata, 1e-6))
        amp_range = max(amp) - min(amp) if max(amp) > min(amp) else 1
        freq_range = max(log_freq) - min(log_freq) if max(log_freq) > min(log_freq) else 1
        dist = ((log_freq - log_click) / freq_range) ** 2 + ((amp - event.ydata) / amp_range) ** 2
        idx = np.argmin(dist)

        if self._current not in self._selected_peaks:
            self._selected_peaks[self._current] = {}

        if self._mode == "f0":
            self._selected_peaks[self._current]["f0"] = {"freq": freq[idx], "amp": amp[idx]}
            self._peak_label.setText(f"f0 = {freq[idx]:.3f} Hz")
        else:
            if "secondary" not in self._selected_peaks[self._current]:
                self._selected_peaks[self._current]["secondary"] = []
            self._selected_peaks[self._current]["secondary"].append({"freq": freq[idx], "amp": amp[idx]})
            self._peak_label.setText(f"Secondary at {freq[idx]:.3f} Hz")

        self._show_current()

    def _prev_step(self):
        if self._current > 0:
            self._current -= 1
            self._show_current()

    def _next_step(self):
        if self._current < len(self._steps) - 1:
            self._current += 1
            self._show_current()

    def get_selected_peaks(self):
        return self._selected_peaks
