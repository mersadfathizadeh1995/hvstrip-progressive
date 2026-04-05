"""Auto Peak Detection Settings Dialog — multi-strategy peak detection config.

Three detection strategies presented as a waterfall menu:
1. Range-Constrained — user-defined per-peak frequency ranges (argmax within band)
2. Preset-Based     — predefined scipy parameter sets (Default/Forward/Conservative/Sharp)
3. Advanced         — full manual control of all peak_detection.py parameters
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QComboBox,
    QFormLayout, QScrollArea, QWidget, QDialogButtonBox,
    QStackedWidget, QFrame,
)


# ── Shared per-peak frequency-range row widget ──────────────────────

class _PeakRangeRow(QWidget):
    """Single row: label + fmin + fmax spinboxes + Use checkbox."""

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


# ── Scrollable frequency-range section (reused by all pages) ────────

class _RangesSection(QWidget):
    """N-secondary spin + scrollable range rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("Secondary Peaks:"))
        self._n_sec = QSpinBox()
        self._n_sec.setRange(0, 10)
        self._n_sec.setValue(2)
        self._n_sec.valueChanged.connect(self._rebuild_rows)
        row.addWidget(self._n_sec)
        row.addStretch()
        lay.addLayout(row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(180)
        self._container = QWidget()
        self._rows_lay = QVBoxLayout(self._container)
        self._rows_lay.setContentsMargins(2, 2, 2, 2)
        self._rows_lay.setSpacing(4)
        scroll.setWidget(self._container)
        lay.addWidget(scroll)

        info = QLabel("Enable 'Use' to constrain peak search within a frequency band.")
        info.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(info)

        self._range_rows = []
        self._rebuild_rows(self._n_sec.value())

    # public API
    @property
    def n_secondary(self):
        return self._n_sec.value()

    @n_secondary.setter
    def n_secondary(self, val):
        self._n_sec.setValue(val)

    def get_ranges(self):
        return [row.get_range() for row in self._range_rows]

    def set_ranges(self, ranges):
        for i, row in enumerate(self._range_rows):
            row.set_range(ranges[i] if i < len(ranges) else None)

    def _rebuild_rows(self, n_sec):
        for row in self._range_rows:
            self._rows_lay.removeWidget(row)
            row.deleteLater()
        self._range_rows.clear()

        primary = _PeakRangeRow("Primary Peak")
        self._rows_lay.addWidget(primary)
        self._range_rows.append(primary)

        for i in range(n_sec):
            row = _PeakRangeRow(f"Secondary {i + 1}")
            self._rows_lay.addWidget(row)
            self._range_rows.append(row)


# ── Main dialog ─────────────────────────────────────────────────────

STRATEGY_LABELS = [
    ("range_constrained", "Range-Constrained"),
    ("preset", "Preset-Based"),
    ("advanced", "Advanced"),
]


class AutoPeakSettingsDialog(QDialog):
    """Multi-strategy auto peak detection settings dialog."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Peak Detection Settings")
        self.setMinimumWidth(500)
        self._config = config or {}
        self._build_ui()
        self._load_config(self._config)

    # ── Build UI ────────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Strategy selector
        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Detection Strategy:"))
        self._strategy_combo = QComboBox()
        for _key, label in STRATEGY_LABELS:
            self._strategy_combo.addItem(label)
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strat_row.addWidget(self._strategy_combo, 1)
        main.addLayout(strat_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        main.addWidget(sep)

        # Stacked pages
        self._stack = QStackedWidget()
        self._page_range = self._build_range_page()
        self._page_preset = self._build_preset_page()
        self._page_advanced = self._build_advanced_page()
        self._stack.addWidget(self._page_range)
        self._stack.addWidget(self._page_preset)
        self._stack.addWidget(self._page_advanced)
        main.addWidget(self._stack)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

    def _on_strategy_changed(self, idx):
        self._stack.setCurrentIndex(idx)

    # ── Page 1: Range-Constrained ───────────────────────────────────

    def _build_range_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)

        info = QLabel(
            "Define frequency bands per peak. The highest amplitude within "
            "each enabled range is selected as the peak.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-style: italic; margin-bottom: 4px;")
        lay.addWidget(info)

        form = QFormLayout()
        self._rc_min_amp = QDoubleSpinBox()
        self._rc_min_amp.setRange(0.0, 100.0)
        self._rc_min_amp.setValue(2.0)
        self._rc_min_amp.setSingleStep(0.5)
        self._rc_min_amp.setDecimals(2)
        form.addRow("Min Amplitude:", self._rc_min_amp)
        lay.addLayout(form)

        self._rc_ranges = _RangesSection()
        lay.addWidget(self._rc_ranges)
        return page

    # ── Page 2: Preset-Based ────────────────────────────────────────

    def _build_preset_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)

        try:
            from .peak_detection import PRESET_DESCRIPTIONS
        except ImportError:
            PRESET_DESCRIPTIONS = {}

        form = QFormLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "default", "forward_modeling", "conservative", "forward_modeling_sharp"
        ])
        self._preset_combo.currentTextChanged.connect(self._update_preset_info)
        form.addRow("Preset:", self._preset_combo)

        self._preset_info = QLabel("")
        self._preset_info.setWordWrap(True)
        self._preset_info.setStyleSheet(
            "color: #444; font-size: 11px; background: #F8F8F0; "
            "padding: 6px; border-radius: 4px;")
        form.addRow("Details:", self._preset_info)

        self._pr_min_amp = QDoubleSpinBox()
        self._pr_min_amp.setRange(0.0, 100.0)
        self._pr_min_amp.setValue(1.5)
        self._pr_min_amp.setSingleStep(0.5)
        self._pr_min_amp.setDecimals(2)
        self._pr_min_amp.setToolTip("Override the preset's min amplitude (0 = use preset default)")
        form.addRow("Min Amplitude Override:", self._pr_min_amp)
        lay.addLayout(form)

        self._pr_ranges = _RangesSection()
        lay.addWidget(self._pr_ranges)

        self._update_preset_info(self._preset_combo.currentText())
        return page

    def _update_preset_info(self, preset_name):
        try:
            from .peak_detection import PRESET_DESCRIPTIONS, PEAK_DETECTION_PRESETS
            desc = PRESET_DESCRIPTIONS.get(preset_name, "")
            cfg = PEAK_DETECTION_PRESETS.get(preset_name, {})
            params = cfg.get("find_peaks_params", {})
            lines = [desc]
            lines.append(
                f"Method: {cfg.get('method','?')} | "
                f"Select: {cfg.get('select','?')} | "
                f"Prominence: {params.get('prominence','?')} | "
                f"Distance: {params.get('distance','?')}")
            if cfg.get("check_clarity_ratio"):
                lines.append(
                    f"Clarity check: ×{cfg.get('clarity_ratio_threshold', 1.5)}")
            self._preset_info.setText("\n".join(lines))
        except ImportError:
            self._preset_info.setText(preset_name)

    # ── Page 3: Advanced ────────────────────────────────────────────

    def _build_advanced_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)

        form = QFormLayout()
        form.setSpacing(3)

        self._adv_method = QComboBox()
        self._adv_method.addItems(["find_peaks", "max"])
        form.addRow("Method:", self._adv_method)

        self._adv_select = QComboBox()
        self._adv_select.addItems([
            "leftmost", "sharpest", "max", "leftmost_sharpest"])
        form.addRow("Selection Strategy:", self._adv_select)

        sep1 = QLabel("── scipy find_peaks params ──")
        sep1.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow(sep1)

        self._adv_prominence = QDoubleSpinBox()
        self._adv_prominence.setRange(0.0, 10.0)
        self._adv_prominence.setValue(0.2)
        self._adv_prominence.setSingleStep(0.05)
        self._adv_prominence.setDecimals(3)
        form.addRow("Prominence:", self._adv_prominence)

        self._adv_distance = QSpinBox()
        self._adv_distance.setRange(1, 50)
        self._adv_distance.setValue(3)
        form.addRow("Distance (bins):", self._adv_distance)

        self._adv_width = QDoubleSpinBox()
        self._adv_width.setRange(0.0, 50.0)
        self._adv_width.setValue(0.0)
        self._adv_width.setSingleStep(0.5)
        self._adv_width.setDecimals(1)
        self._adv_width.setToolTip("Min peak width in bins. 0 = disabled.")
        form.addRow("Width (bins):", self._adv_width)

        sep2 = QLabel("── Frequency & amplitude filters ──")
        sep2.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow(sep2)

        self._adv_freq_min = QDoubleSpinBox()
        self._adv_freq_min.setRange(0.0, 200.0)
        self._adv_freq_min.setValue(0.1)
        self._adv_freq_min.setDecimals(3)
        self._adv_freq_min.setSuffix(" Hz")
        form.addRow("Freq Min:", self._adv_freq_min)

        self._adv_freq_max = QDoubleSpinBox()
        self._adv_freq_max.setRange(0.0, 200.0)
        self._adv_freq_max.setValue(0.0)
        self._adv_freq_max.setDecimals(3)
        self._adv_freq_max.setSuffix(" Hz")
        self._adv_freq_max.setToolTip("0 = no upper limit")
        form.addRow("Freq Max:", self._adv_freq_max)

        self._adv_min_amp = QDoubleSpinBox()
        self._adv_min_amp.setRange(0.0, 100.0)
        self._adv_min_amp.setValue(1.5)
        self._adv_min_amp.setSingleStep(0.5)
        self._adv_min_amp.setDecimals(2)
        form.addRow("Min Amplitude:", self._adv_min_amp)

        self._adv_min_rel = QDoubleSpinBox()
        self._adv_min_rel.setRange(0.0, 1.0)
        self._adv_min_rel.setValue(0.15)
        self._adv_min_rel.setSingleStep(0.05)
        self._adv_min_rel.setDecimals(2)
        self._adv_min_rel.setToolTip("Fraction of global max amplitude")
        form.addRow("Min Relative Height:", self._adv_min_rel)

        self._adv_excl_n = QSpinBox()
        self._adv_excl_n.setRange(0, 20)
        self._adv_excl_n.setValue(1)
        self._adv_excl_n.setToolTip("Skip first N frequency bins")
        form.addRow("Exclude First N Bins:", self._adv_excl_n)

        sep3 = QLabel("── Clarity check ──")
        sep3.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow(sep3)

        self._adv_clarity = QCheckBox("Enable")
        self._adv_clarity.setToolTip(
            "Verify peak amplitude > threshold × amplitude at f0/2 and 2f0")
        form.addRow("Clarity Ratio:", self._adv_clarity)

        self._adv_clarity_thr = QDoubleSpinBox()
        self._adv_clarity_thr.setRange(1.0, 10.0)
        self._adv_clarity_thr.setValue(1.5)
        self._adv_clarity_thr.setSingleStep(0.1)
        self._adv_clarity_thr.setDecimals(1)
        form.addRow("Clarity Threshold:", self._adv_clarity_thr)

        lay.addLayout(form)

        self._adv_ranges = _RangesSection()
        lay.addWidget(self._adv_ranges)
        return page

    # ── Load / Get config ───────────────────────────────────────────

    def _load_config(self, cfg):
        """Populate UI from config dict. Handles old format (no strategy key)."""
        strategy = cfg.get("strategy", "range_constrained")
        strat_idx = next(
            (i for i, (k, _) in enumerate(STRATEGY_LABELS) if k == strategy), 0)
        self._strategy_combo.setCurrentIndex(strat_idx)

        # Range-constrained page
        self._rc_min_amp.setValue(cfg.get("min_amplitude", 2.0))
        rc_n = cfg.get("n_secondary", 2)
        self._rc_ranges.n_secondary = rc_n
        self._rc_ranges.set_ranges(cfg.get("ranges", []))

        # Preset page
        preset = cfg.get("preset", "default")
        idx = self._preset_combo.findText(preset)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self._pr_min_amp.setValue(cfg.get("min_amplitude", 1.5))
        pr_n = cfg.get("n_secondary", 2)
        self._pr_ranges.n_secondary = pr_n
        self._pr_ranges.set_ranges(cfg.get("ranges", []))

        # Advanced page
        method = cfg.get("method", "find_peaks")
        idx_m = self._adv_method.findText(method)
        if idx_m >= 0:
            self._adv_method.setCurrentIndex(idx_m)
        select = cfg.get("select", "leftmost")
        idx_s = self._adv_select.findText(select)
        if idx_s >= 0:
            self._adv_select.setCurrentIndex(idx_s)
        self._adv_prominence.setValue(cfg.get("prominence", 0.2))
        self._adv_distance.setValue(cfg.get("distance", 3))
        self._adv_width.setValue(cfg.get("width", 0.0))
        self._adv_freq_min.setValue(cfg.get("freq_min", 0.1))
        self._adv_freq_max.setValue(cfg.get("freq_max", 0.0))
        self._adv_min_amp.setValue(cfg.get("min_amplitude", 1.5))
        self._adv_min_rel.setValue(cfg.get("min_rel_height", 0.15))
        self._adv_excl_n.setValue(cfg.get("exclude_first_n", 1))
        self._adv_clarity.setChecked(cfg.get("check_clarity", False))
        self._adv_clarity_thr.setValue(cfg.get("clarity_threshold", 1.5))
        adv_n = cfg.get("n_secondary", 2)
        self._adv_ranges.n_secondary = adv_n
        self._adv_ranges.set_ranges(cfg.get("ranges", []))

        # Legacy: old config without strategy but with min_prominence →
        # map min_prominence to prominence for backward compat
        if "min_prominence" in cfg and "prominence" not in cfg:
            self._adv_prominence.setValue(cfg["min_prominence"])
            self._rc_min_amp.setValue(cfg.get("min_amplitude", 2.0))

    def get_config(self):
        """Return config dict from current UI state."""
        strat_key = STRATEGY_LABELS[self._strategy_combo.currentIndex()][0]

        if strat_key == "range_constrained":
            return {
                "strategy": "range_constrained",
                "min_amplitude": self._rc_min_amp.value(),
                "n_secondary": self._rc_ranges.n_secondary,
                "ranges": self._rc_ranges.get_ranges(),
            }

        if strat_key == "preset":
            return {
                "strategy": "preset",
                "preset": self._preset_combo.currentText(),
                "min_amplitude": self._pr_min_amp.value(),
                "n_secondary": self._pr_ranges.n_secondary,
                "ranges": self._pr_ranges.get_ranges(),
            }

        # advanced
        freq_max = self._adv_freq_max.value()
        return {
            "strategy": "advanced",
            "method": self._adv_method.currentText(),
            "select": self._adv_select.currentText(),
            "prominence": self._adv_prominence.value(),
            "distance": self._adv_distance.value(),
            "width": self._adv_width.value(),
            "freq_min": self._adv_freq_min.value(),
            "freq_max": freq_max if freq_max > 0 else None,
            "min_amplitude": self._adv_min_amp.value(),
            "min_rel_height": self._adv_min_rel.value(),
            "exclude_first_n": self._adv_excl_n.value(),
            "check_clarity": self._adv_clarity.isChecked(),
            "clarity_threshold": self._adv_clarity_thr.value(),
            "n_secondary": self._adv_ranges.n_secondary,
            "ranges": self._adv_ranges.get_ranges(),
        }
