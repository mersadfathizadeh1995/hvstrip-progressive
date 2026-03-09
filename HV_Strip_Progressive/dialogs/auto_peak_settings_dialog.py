"""Auto Peak Detection Settings Dialog — configurable peak detection parameters.

Reusable dialog for configuring automatic peak detection: number of secondary
peaks, per-peak frequency ranges, min prominence, and min amplitude.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFormLayout, QScrollArea, QWidget, QDialogButtonBox,
)


class _PeakRangeRow(QWidget):
    """Single row: label + fmin + fmax spinboxes."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setFixedWidth(100)
        lay.addWidget(self._label)

        lay.addWidget(QLabel("Min:"))
        self.fmin = QDoubleSpinBox()
        self.fmin.setRange(0.001, 200.0)
        self.fmin.setValue(0.1)
        self.fmin.setDecimals(3)
        self.fmin.setSuffix(" Hz")
        lay.addWidget(self.fmin)

        lay.addWidget(QLabel("Max:"))
        self.fmax = QDoubleSpinBox()
        self.fmax.setRange(0.001, 200.0)
        self.fmax.setValue(50.0)
        self.fmax.setDecimals(3)
        self.fmax.setSuffix(" Hz")
        lay.addWidget(self.fmax)

        self.enabled = QCheckBox("Use")
        self.enabled.setChecked(False)
        lay.addWidget(self.enabled)

    def get_range(self):
        if self.enabled.isChecked():
            return {"min": self.fmin.value(), "max": self.fmax.value()}
        return None

    def set_range(self, r):
        if r:
            self.enabled.setChecked(True)
            self.fmin.setValue(r.get("min", 0.1))
            self.fmax.setValue(r.get("max", 50.0))
        else:
            self.enabled.setChecked(False)


class AutoPeakSettingsDialog(QDialog):
    """Dialog for configuring automatic peak detection parameters."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Peak Detection Settings")
        self.setMinimumWidth(460)
        self._config = config or {}
        self._range_rows = []
        self._build_ui()
        self._load_config(self._config)

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Detection parameters
        det_grp = QGroupBox("Detection Parameters")
        det_lay = QFormLayout()

        self._min_prominence = QDoubleSpinBox()
        self._min_prominence.setRange(0.0, 10.0)
        self._min_prominence.setValue(0.5)
        self._min_prominence.setSingleStep(0.1)
        self._min_prominence.setDecimals(2)
        det_lay.addRow("Min Prominence:", self._min_prominence)

        self._min_amplitude = QDoubleSpinBox()
        self._min_amplitude.setRange(0.0, 100.0)
        self._min_amplitude.setValue(2.0)
        self._min_amplitude.setSingleStep(0.5)
        self._min_amplitude.setDecimals(2)
        det_lay.addRow("Min Amplitude:", self._min_amplitude)

        self._n_secondary = QSpinBox()
        self._n_secondary.setRange(0, 10)
        self._n_secondary.setValue(2)
        self._n_secondary.valueChanged.connect(self._update_range_rows)
        det_lay.addRow("Secondary Peaks:", self._n_secondary)

        det_grp.setLayout(det_lay)
        main.addWidget(det_grp)

        # Frequency ranges
        range_grp = QGroupBox("Frequency Ranges (optional)")
        range_outer = QVBoxLayout()

        range_scroll = QScrollArea()
        range_scroll.setWidgetResizable(True)
        range_scroll.setMaximumHeight(200)
        self._range_container = QWidget()
        self._range_lay = QVBoxLayout(self._range_container)
        self._range_lay.setContentsMargins(2, 2, 2, 2)
        self._range_lay.setSpacing(4)
        range_scroll.setWidget(self._range_container)
        range_outer.addWidget(range_scroll)

        self._range_info = QLabel(
            "Enable 'Use' to constrain peak search within a frequency band.")
        self._range_info.setStyleSheet("color: gray; font-size: 10px;")
        range_outer.addWidget(self._range_info)

        range_grp.setLayout(range_outer)
        main.addWidget(range_grp)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

        self._update_range_rows(self._n_secondary.value())

    def _update_range_rows(self, n_sec):
        """Rebuild frequency range rows for primary + N secondary peaks."""
        # Clear existing rows
        for row in self._range_rows:
            self._range_lay.removeWidget(row)
            row.deleteLater()
        self._range_rows.clear()

        # Primary peak range
        primary = _PeakRangeRow("Primary Peak")
        self._range_lay.addWidget(primary)
        self._range_rows.append(primary)

        # Secondary peak ranges
        for i in range(n_sec):
            row = _PeakRangeRow(f"Secondary {i + 1}")
            self._range_lay.addWidget(row)
            self._range_rows.append(row)

    def _load_config(self, cfg):
        """Populate UI from config dict."""
        self._min_prominence.setValue(cfg.get("min_prominence", 0.5))
        self._min_amplitude.setValue(cfg.get("min_amplitude", 2.0))
        self._n_secondary.setValue(cfg.get("n_secondary", 2))

        ranges = cfg.get("ranges", [])
        for i, row in enumerate(self._range_rows):
            if i < len(ranges):
                row.set_range(ranges[i])

    def get_config(self):
        """Return config dict from current UI state."""
        ranges = []
        for row in self._range_rows:
            ranges.append(row.get_range())

        return {
            "min_prominence": self._min_prominence.value(),
            "min_amplitude": self._min_amplitude.value(),
            "n_secondary": self._n_secondary.value(),
            "ranges": ranges,
        }
