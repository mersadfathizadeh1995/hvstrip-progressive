"""Editable soil-profile layer table widget.

Faithfully ports the original PySide6 LayerTableWidget to PyQt5.
"""
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QCheckBox, QAbstractItemView,
    QMessageBox,
)

# Column indices
COL_LAYER = 0
COL_THICK = 1
COL_VS    = 2
COL_VP    = 3
COL_NU    = 4
COL_VPMODE = 5
COL_DENSITY = 6
COL_HS    = 7
COL_SOIL  = 8
NUM_COLS  = 9
HEADERS = ["Layer", "Thickness", "Vs", "Vp", "Nu", "Vp Mode", "Density", "HS", "Soil Type"]

VP_MODES = ["Auto (from Vs)", "From Nu", "Manual Vp"]


def _suggest_nu(vs):
    if vs < 150: return 0.48
    if vs < 250: return 0.40
    if vs < 400: return 0.33
    if vs < 600: return 0.28
    if vs < 1000: return 0.25
    return 0.22


def _suggest_density(vs):
    if vs < 150: return 1600.0
    if vs < 250: return 1750.0
    if vs < 400: return 1900.0
    if vs < 600: return 2050.0
    if vs < 1000: return 2200.0
    return 2500.0


def _vp_from_vs_nu(vs, nu):
    return vs * np.sqrt((2.0 - 2.0 * nu) / (1.0 - 2.0 * nu))


def _soil_type(vs):
    if vs < 100: return "Very soft soil"
    if vs < 180: return "Soft soil"
    if vs < 250: return "Medium stiff soil"
    if vs < 360: return "Stiff soil"
    if vs < 500: return "Dense sand/gravel"
    if vs < 760: return "Soft/weathered rock"
    if vs < 1500: return "Rock"
    return "Hard rock"


class LayerTableWidget(QWidget):
    """Editable table for soil profile layers."""

    profile_changed = pyqtSignal()
    layer_selected  = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block_signals = False
        self._build_ui()
        self._new_default()

    # ── public API ──────────────────────────────────────────────
    def set_profile(self, profile):
        """Load a SoilProfile into the table."""
        self._block_signals = True
        self.table.setRowCount(0)
        for i, L in enumerate(profile.layers):
            self._add_row(
                thickness=L.thickness,
                vs=L.vs,
                vp=L.vp if L.vp else _vp_from_vs_nu(L.vs, _suggest_nu(L.vs)),
                nu=L.get_effective_nu() if hasattr(L, "get_effective_nu") else _suggest_nu(L.vs),
                density=L.density,
                is_hs=L.is_halfspace,
            )
        self._block_signals = False
        self._renumber()
        self.profile_changed.emit()

    def get_profile(self):
        """Return a SoilProfile built from current table data."""
        from core.soil_profile import SoilProfile, Layer
        layers = []
        for r in range(self.table.rowCount()):
            thick = self._float(r, COL_THICK, 0)
            vs    = self._float(r, COL_VS, 200)
            vp    = self._float(r, COL_VP, 400)
            density = self._float(r, COL_DENSITY, 2000)
            hs_cb = self.table.cellWidget(r, COL_HS)
            is_hs = hs_cb.isChecked() if hs_cb else False
            nu_val = self._float(r, COL_NU, 0.33)
            layers.append(Layer(
                thickness=0.0 if is_hs else thick,
                vs=vs, vp=vp, nu=nu_val, density=density,
                is_halfspace=is_hs,
            ))
        return SoilProfile(layers=layers, name="editor", description="Edited in GUI")

    # ── UI build ────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Table
        self.table = QTableWidget(0, NUM_COLS)
        self.table.setHorizontalHeaderLabels(HEADERS)
        hh = self.table.horizontalHeader()
        for c in range(NUM_COLS):
            hh.setSectionResizeMode(c, QHeaderView.Stretch if c in (COL_SOIL,) else QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.currentCellChanged.connect(lambda r, *_: self.layer_selected.emit(r))
        layout.addWidget(self.table)

        # Buttons
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Add Layer", self._add_layer),
            ("Remove", self._remove_layer),
            ("Move Up", self._move_up),
            ("Move Down", self._move_down),
            ("Auto-fill", self._auto_fill),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

    # ── row manipulation ────────────────────────────────────────
    def _add_row(self, thickness=10.0, vs=300.0, vp=None, nu=None, density=None, is_hs=False):
        if nu is None:
            nu = _suggest_nu(vs)
        if vp is None:
            vp = _vp_from_vs_nu(vs, nu)
        if density is None:
            density = _suggest_density(vs)

        r = self.table.rowCount()
        self.table.insertRow(r)

        # Layer number (read-only)
        item = QTableWidgetItem(str(r + 1))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, COL_LAYER, item)

        # Editable numeric cells
        for col, val in [(COL_THICK, thickness), (COL_VS, vs), (COL_VP, vp),
                         (COL_NU, nu), (COL_DENSITY, density)]:
            self.table.setItem(r, col, QTableWidgetItem(f"{val:.2f}" if isinstance(val, float) else str(val)))

        # Vp Mode combo
        combo = QComboBox()
        combo.addItems(VP_MODES)
        combo.currentIndexChanged.connect(lambda _: self._on_mode_changed(r))
        self.table.setCellWidget(r, COL_VPMODE, combo)

        # Half-space checkbox
        cb = QCheckBox()
        cb.setChecked(is_hs)
        cb.stateChanged.connect(lambda _: self._on_hs_changed(r))
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.addWidget(cb)
        hl.setAlignment(Qt.AlignCenter)
        hl.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(r, COL_HS, w)

        # Soil type (read-only)
        soil_item = QTableWidgetItem(_soil_type(vs))
        soil_item.setFlags(soil_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, COL_SOIL, soil_item)

    def _add_layer(self):
        self._block_signals = True
        self._add_row()
        self._block_signals = False
        self._renumber()
        self.profile_changed.emit()

    def _remove_layer(self):
        row = self.table.currentRow()
        if row < 0 or self.table.rowCount() <= 1:
            return
        self.table.removeRow(row)
        self._renumber()
        self.profile_changed.emit()

    def _move_up(self):
        r = self.table.currentRow()
        if r <= 0:
            return
        self._swap_rows(r, r - 1)
        self.table.setCurrentCell(r - 1, 0)
        self.profile_changed.emit()

    def _move_down(self):
        r = self.table.currentRow()
        if r < 0 or r >= self.table.rowCount() - 1:
            return
        self._swap_rows(r, r + 1)
        self.table.setCurrentCell(r + 1, 0)
        self.profile_changed.emit()

    def _auto_fill(self):
        """Fill Vp, Nu, Density from Vs using empirical relations."""
        self._block_signals = True
        for r in range(self.table.rowCount()):
            vs = self._float(r, COL_VS, 200)
            nu = _suggest_nu(vs)
            vp = _vp_from_vs_nu(vs, nu)
            density = _suggest_density(vs)
            self.table.item(r, COL_NU).setText(f"{nu:.3f}")
            self.table.item(r, COL_VP).setText(f"{vp:.1f}")
            self.table.item(r, COL_DENSITY).setText(f"{density:.0f}")
            soil_item = self.table.item(r, COL_SOIL)
            if soil_item:
                soil_item.setText(_soil_type(vs))
        self._block_signals = False
        self.profile_changed.emit()

    def _new_default(self):
        """Create default 3-layer profile."""
        self._block_signals = True
        self.table.setRowCount(0)
        self._add_row(thickness=5.0, vs=200.0)
        self._add_row(thickness=15.0, vs=400.0)
        self._add_row(thickness=0.0, vs=800.0, is_hs=True)
        self._block_signals = False
        self._renumber()

    # ── helpers ─────────────────────────────────────────────────
    def _float(self, row, col, default=0.0):
        item = self.table.item(row, col)
        if item is None:
            return default
        try:
            return float(item.text())
        except ValueError:
            return default

    def _renumber(self):
        self._block_signals = True
        for r in range(self.table.rowCount()):
            item = self.table.item(r, COL_LAYER)
            if item:
                item.setText(str(r + 1))
        self._block_signals = False

    def _swap_rows(self, r1, r2):
        self._block_signals = True
        for c in (COL_THICK, COL_VS, COL_VP, COL_NU, COL_DENSITY):
            t1 = self.table.item(r1, c).text() if self.table.item(r1, c) else ""
            t2 = self.table.item(r2, c).text() if self.table.item(r2, c) else ""
            self.table.item(r1, c).setText(t2)
            self.table.item(r2, c).setText(t1)
        # Swap HS checkboxes
        w1 = self.table.cellWidget(r1, COL_HS)
        w2 = self.table.cellWidget(r2, COL_HS)
        if w1 and w2:
            cb1 = w1.findChild(QCheckBox)
            cb2 = w2.findChild(QCheckBox)
            if cb1 and cb2:
                v1, v2 = cb1.isChecked(), cb2.isChecked()
                cb1.setChecked(v2)
                cb2.setChecked(v1)
        self._block_signals = False
        self._renumber()

    # ── event handlers ──────────────────────────────────────────
    def _on_cell_changed(self, row, col):
        if self._block_signals:
            return
        if col == COL_VS:
            vs = self._float(row, COL_VS, 200)
            soil_item = self.table.item(row, COL_SOIL)
            if soil_item:
                soil_item.setText(_soil_type(vs))
            combo = self.table.cellWidget(row, COL_VPMODE)
            if combo and combo.currentIndex() == 0:
                nu = _suggest_nu(vs)
                vp = _vp_from_vs_nu(vs, nu)
                self._block_signals = True
                self.table.item(row, COL_NU).setText(f"{nu:.3f}")
                self.table.item(row, COL_VP).setText(f"{vp:.1f}")
                self._block_signals = False
        self.profile_changed.emit()

    def _on_mode_changed(self, row):
        combo = self.table.cellWidget(row, COL_VPMODE)
        if combo is None:
            return
        mode = combo.currentIndex()
        vs = self._float(row, COL_VS, 200)
        self._block_signals = True
        if mode == 0:  # Auto
            nu = _suggest_nu(vs)
            vp = _vp_from_vs_nu(vs, nu)
            self.table.item(row, COL_NU).setText(f"{nu:.3f}")
            self.table.item(row, COL_VP).setText(f"{vp:.1f}")
        elif mode == 1:  # From Nu
            nu = self._float(row, COL_NU, 0.33)
            vp = _vp_from_vs_nu(vs, nu)
            self.table.item(row, COL_VP).setText(f"{vp:.1f}")
        self._block_signals = False
        self.profile_changed.emit()

    def _on_hs_changed(self, row):
        w = self.table.cellWidget(row, COL_HS)
        if w is None:
            return
        cb = w.findChild(QCheckBox)
        if cb is None:
            return
        if cb.isChecked():
            self._block_signals = True
            self.table.item(row, COL_THICK).setText("0.00")
            thick_item = self.table.item(row, COL_THICK)
            if thick_item:
                thick_item.setFlags(thick_item.flags() & ~Qt.ItemIsEditable)
            self._block_signals = False
        else:
            thick_item = self.table.item(row, COL_THICK)
            if thick_item:
                thick_item.setFlags(thick_item.flags() | Qt.ItemIsEditable)
        self.profile_changed.emit()
