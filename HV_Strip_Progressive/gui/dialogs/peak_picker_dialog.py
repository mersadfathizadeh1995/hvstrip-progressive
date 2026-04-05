"""Peak Picker Dialog — Interactive click-to-pick peak selection per step.

Ported from the old PySide6 InteractivePeakPickerDialog to PyQt5.
"""
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QLabel, QPushButton, QMessageBox,
)

from ..widgets.plot_widget import MatplotlibWidget


class PeakPickerDialog(QDialog):
    """Interactive peak selection across stripping steps."""

    peaks_selected = pyqtSignal(dict)

    def __init__(self, result_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Peak Picker")
        self.resize(1100, 650)

        self._result = result_dict
        self._step_data = self._collect_steps(result_dict)
        self._current_idx = 0
        self._selected = {}   # {step_name: (freq, amp, idx)}
        self._undo_stack = []
        self._pick_mode = False

        self._build_ui()
        if self._step_data:
            self._load_step(0)

    def _collect_steps(self, result):
        steps = []
        step_results = result.get("step_results", {})
        strip_dir = result.get("strip_directory", "")

        for name in sorted(step_results.keys()):
            data = step_results[name]
            hv_csv = data.get("hv_csv")
            if not hv_csv:
                continue
            try:
                arr = np.loadtxt(str(hv_csv), delimiter=",", skiprows=1)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    steps.append({
                        "name": name,
                        "freqs": arr[:, 0],
                        "amps": arr[:, 1],
                        "model_file": data.get("model_file"),
                    })
            except Exception:
                pass
        return steps

    def _build_ui(self):
        lay = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # Left: step list
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(2, 2, 2, 2)
        self._step_list = QListWidget()
        for i, s in enumerate(self._step_data):
            self._step_list.addItem(s["name"])
        self._step_list.currentRowChanged.connect(self._on_step_select)
        left.setMinimumWidth(180)
        left.setMaximumWidth(250)
        left_lay.addWidget(QLabel("<b>Steps</b>"))
        left_lay.addWidget(self._step_list)
        splitter.addWidget(left)

        # Right: canvas + controls
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)

        self._info = QLabel("")
        self._info.setStyleSheet("font-size: 12px; padding: 4px;")
        right_lay.addWidget(self._info)

        self._canvas = MatplotlibWidget(figsize=(12, 5))
        self._canvas.canvas.mpl_connect("button_press_event", self._on_click)
        right_lay.addWidget(self._canvas)

        self._sel_label = QLabel("")
        self._sel_label.setStyleSheet("font-size: 11px; color: #333;")
        right_lay.addWidget(self._sel_label)

        # Control buttons
        ctrl = QHBoxLayout()
        self._btn_pick = QPushButton("Enable Pick Mode")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setStyleSheet(
            "QPushButton:checked { background-color: #4CAF50; color: white; }")
        self._btn_pick.toggled.connect(lambda on: setattr(self, '_pick_mode', on))
        ctrl.addWidget(self._btn_pick)

        self._btn_auto = QPushButton("Auto-Detect")
        self._btn_auto.clicked.connect(self._auto_detect)
        ctrl.addWidget(self._btn_auto)

        self._btn_undo = QPushButton("Undo")
        self._btn_undo.clicked.connect(self._undo)
        ctrl.addWidget(self._btn_undo)

        self._btn_prev = QPushButton("< Previous")
        self._btn_prev.clicked.connect(self._go_prev)
        ctrl.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next >")
        self._btn_next.clicked.connect(self._go_next)
        ctrl.addWidget(self._btn_next)

        ctrl.addStretch()

        self._btn_finish = QPushButton("Accept All")
        self._btn_finish.setStyleSheet(
            "background-color: #2E86AB; color: white; font-weight: bold; padding: 6px 16px;")
        self._btn_finish.clicked.connect(self._finish)
        ctrl.addWidget(self._btn_finish)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        ctrl.addWidget(btn_cancel)

        right_lay.addLayout(ctrl)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 900])

        lay.addWidget(splitter)

    def _load_step(self, idx):
        if idx < 0 or idx >= len(self._step_data):
            return
        self._current_idx = idx
        self._step_list.setCurrentRow(idx)
        step = self._step_data[idx]
        n_sel = sum(1 for s in self._step_data if s["name"] in self._selected)
        self._info.setText(
            f"Step {idx + 1} of {len(self._step_data)}: <b>{step['name']}</b> | "
            f"Peaks selected: {n_sel}/{len(self._step_data)}")
        self._plot_step(step)
        self._update_buttons()

    def _plot_step(self, step):
        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        ax.plot(step["freqs"], step["amps"], color="royalblue", lw=1.5)

        sel = self._selected.get(step["name"])
        if sel:
            f, a, _ = sel
            ax.plot(f, a, "*", color="red", ms=14, zorder=5)
            ax.axvline(f, color="red", ls="--", alpha=0.5, lw=0.8)
            ax.annotate(f"f0={f:.3f}", xy=(f, a), xytext=(10, 10),
                        textcoords="offset points", fontsize=9, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Ratio")
        ax.set_title(step["name"])
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        self._canvas.refresh()
        self._update_sel_label()

    def _on_click(self, event):
        if not self._pick_mode or event.inaxes is None:
            return
        if self._canvas.toolbar.mode:
            return
        step = self._step_data[self._current_idx]
        freqs, amps = step["freqs"], step["amps"]

        cx = event.xdata  # exact clicked frequency
        # Interpolate amplitude at the exact clicked frequency
        amp_interp = float(np.interp(cx, freqs, amps))
        idx = int(np.argmin(np.abs(freqs - cx)))

        old = self._selected.get(step["name"])
        self._undo_stack.append((step["name"], old))
        self._selected[step["name"]] = (float(cx), amp_interp, idx)

        self._plot_step(step)
        self._update_list_item(self._current_idx, True)

    def _auto_detect(self):
        step = self._step_data[self._current_idx]
        idx = int(np.argmax(step["amps"]))
        old = self._selected.get(step["name"])
        self._undo_stack.append((step["name"], old))
        self._selected[step["name"]] = (
            float(step["freqs"][idx]), float(step["amps"][idx]), idx)
        self._plot_step(step)
        self._update_list_item(self._current_idx, True)

    def _undo(self):
        if not self._undo_stack:
            return
        name, old = self._undo_stack.pop()
        if old is None:
            self._selected.pop(name, None)
        else:
            self._selected[name] = old
        # Find index and reload
        for i, s in enumerate(self._step_data):
            if s["name"] == name:
                self._load_step(i)
                self._update_list_item(i, name in self._selected)
                break

    def _go_prev(self):
        if self._current_idx > 0:
            self._load_step(self._current_idx - 1)

    def _go_next(self):
        if self._current_idx < len(self._step_data) - 1:
            self._load_step(self._current_idx + 1)

    def _on_step_select(self, row):
        if row >= 0:
            self._load_step(row)

    def _update_buttons(self):
        self._btn_prev.setEnabled(self._current_idx > 0)
        self._btn_next.setEnabled(self._current_idx < len(self._step_data) - 1)

    def _update_list_item(self, idx, selected):
        item = self._step_list.item(idx)
        if item:
            name = self._step_data[idx]["name"]
            prefix = "✓ " if selected else "  "
            item.setText(f"{prefix}{name}")

    def _update_sel_label(self):
        step = self._step_data[self._current_idx]
        sel = self._selected.get(step["name"])
        if sel:
            f, a, _ = sel
            self._sel_label.setText(f"Selected: f0 = {f:.4f} Hz (amp = {a:.3f})")
        else:
            self._sel_label.setText("No peak selected for this step")

    def _finish(self):
        # Auto-fill unselected with global max
        for i, step in enumerate(self._step_data):
            if step["name"] not in self._selected:
                idx = int(np.argmax(step["amps"]))
                self._selected[step["name"]] = (
                    float(step["freqs"][idx]), float(step["amps"][idx]), idx)
                self._update_list_item(i, True)
        self.peaks_selected.emit(self._selected)
        self.accept()

    def get_selected_peaks(self):
        return dict(self._selected)
