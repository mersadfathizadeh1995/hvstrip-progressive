"""``LayerTable`` — the editable 9-column soil-layer table (Round 2).

Ported (copy-not-import) from the legacy ``gui/widgets/layer_table_widget.py``
with the load-bearing fixes:

* **PySide6**, house-styled (no ``setStyleSheet`` literals).
* **ONE derivation source** — the legacy widget carried its OWN empirical
  ν/density tables that DIVERGED from core's ``VelocityConverter``; this port
  takes a ``fill_provider(vs) -> {nu, vp, density, soil_type}`` callable
  (the AppState → api surface) and derives NOTHING locally.
* Works on plain **layer dicts** (the api shape), not ``SoilProfile``
  objects — the widget stays core-free.

Behaviour kept from the legacy widget: the Vp Mode column
(``Auto (from Vs)`` re-derives ν+Vp live on Vs edits · ``From Nu`` derives
Vp from the typed ν · ``Manual Vp``), the Auto-fill button, the half-space
row rule (thickness locked to 0), add/remove/move-up/move-down.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COL_LAYER = 0
COL_THICK = 1
COL_VS = 2
COL_VP = 3
COL_NU = 4
COL_VPMODE = 5
COL_DENSITY = 6
COL_HS = 7
COL_SOIL = 8
NUM_COLS = 9
HEADERS = ["Layer", "Thickness", "Vs", "Vp", "ν", "Vp Mode",
           "Density", "HS", "Soil Type"]

VP_MODES = ["Auto (from Vs)", "From Nu", "Manual Vp"]

#: ``fill_provider(vs, nu=None) -> {nu, vp, density, soil_type}`` — pass
#: ``nu`` to derive Vp from a user-typed Poisson's ratio.
FillProvider = Callable[..., Dict[str, Any]]


class LayerTable(QWidget):
    """Editable layer-dict table (see module docstring)."""

    layers_changed = Signal()
    layer_selected = Signal(int)

    def __init__(
        self,
        fill_provider: FillProvider,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._fill = fill_provider
        self._block = False
        self._build_ui()
        self.new_default()

    # ==================================================================
    #  Public API (dict in / dict out)
    # ==================================================================
    def set_layers(self, layers: List[Dict[str, Any]]) -> None:
        """Fill the table from api layer dicts.

        Partial profiles (Vs-only files) get ν/Vp/density DERIVED for
        display via the fill provider — the user can then edit or re-derive.
        """
        self._block = True
        try:
            self.table.setRowCount(0)
            for ly in layers:
                vs = float(ly.get("vs", 200.0))
                fill = self._fill(vs)
                nu = ly.get("nu")
                self._add_row(
                    thickness=float(ly.get("thickness", 0.0)),
                    vs=vs,
                    vp=float(ly["vp"]) if ly.get("vp") else fill["vp"],
                    nu=float(nu) if nu is not None else fill["nu"],
                    density=(float(ly["density"]) if ly.get("density")
                             else fill["density"]),
                    is_hs=bool(ly.get("is_halfspace")
                               or ly.get("thickness", 0) == 0),
                )
        finally:
            self._block = False
        self._renumber()
        self.layers_changed.emit()

    def get_layers(self) -> List[Dict[str, Any]]:
        """The table content as api layer dicts."""
        layers: List[Dict[str, Any]] = []
        for r in range(self.table.rowCount()):
            is_hs = self._hs_checked(r)
            layers.append({
                "thickness": 0.0 if is_hs else self._float(r, COL_THICK, 0.0),
                "vs": self._float(r, COL_VS, 200.0),
                "vp": self._float(r, COL_VP, 400.0),
                "nu": self._float(r, COL_NU, 0.33),
                "density": self._float(r, COL_DENSITY, 2000.0),
                "is_halfspace": is_hs,
            })
        return layers

    def new_default(self) -> None:
        """The legacy default 3-layer starter model (manual-editor path)."""
        self._block = True
        try:
            self.table.setRowCount(0)
            self._add_row(thickness=5.0, vs=200.0)
            self._add_row(thickness=15.0, vs=400.0)
            self._add_row(thickness=0.0, vs=800.0, is_hs=True)
        finally:
            self._block = False
        self._renumber()

    def auto_fill(self) -> None:
        """Fill ν, Vp, density + soil type from Vs on EVERY row."""
        self._block = True
        try:
            for r in range(self.table.rowCount()):
                vs = self._float(r, COL_VS, 200.0)
                fill = self._fill(vs)
                self.table.item(r, COL_NU).setText(f"{fill['nu']:.3f}")
                self.table.item(r, COL_VP).setText(f"{fill['vp']:.1f}")
                self.table.item(r, COL_DENSITY).setText(
                    f"{fill['density']:.0f}")
                self.table.item(r, COL_SOIL).setText(fill["soil_type"])
        finally:
            self._block = False
        self.layers_changed.emit()

    # ==================================================================
    #  UI build
    # ==================================================================
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.table = QTableWidget(0, NUM_COLS)
        self.table.setHorizontalHeaderLabels(HEADERS)
        hh = self.table.horizontalHeader()
        for c in range(NUM_COLS):
            hh.setSectionResizeMode(
                c, QHeaderView.Stretch if c == COL_SOIL
                else QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.currentCellChanged.connect(
            lambda r, *_: self.layer_selected.emit(r))
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        for label, slot in (
            ("＋ Layer", self._add_layer),
            ("− Remove", self._remove_layer),
            ("▲ Up", self._move_up),
            ("▼ Down", self._move_down),
        ):
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        auto_btn = QPushButton("Auto-fill from Vs")
        auto_btn.clicked.connect(self.auto_fill)
        btn_row.addWidget(auto_btn)
        layout.addLayout(btn_row)

    # ==================================================================
    #  Rows
    # ==================================================================
    def _add_row(self, thickness=10.0, vs=300.0, vp=None, nu=None,
                 density=None, is_hs=False) -> None:
        fill = self._fill(vs)
        nu = fill["nu"] if nu is None else nu
        vp = fill["vp"] if vp is None else vp
        density = fill["density"] if density is None else density

        r = self.table.rowCount()
        self.table.insertRow(r)

        item = QTableWidgetItem(str(r + 1))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, COL_LAYER, item)

        for col, val, spec in ((COL_THICK, thickness, "{:.2f}"),
                               (COL_VS, vs, "{:.1f}"),
                               (COL_VP, vp, "{:.1f}"),
                               (COL_NU, nu, "{:.3f}"),
                               (COL_DENSITY, density, "{:.0f}")):
            self.table.setItem(r, col, QTableWidgetItem(spec.format(val)))

        combo = QComboBox()
        combo.addItems(VP_MODES)
        combo.currentIndexChanged.connect(
            lambda _i, row=r: self._on_mode_changed(row))
        self.table.setCellWidget(r, COL_VPMODE, combo)

        cb = QCheckBox()
        cb.setChecked(is_hs)
        cb.stateChanged.connect(lambda _s, row=r: self._on_hs_changed(row))
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.addWidget(cb)
        hl.setAlignment(Qt.AlignCenter)
        hl.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(r, COL_HS, holder)

        soil_item = QTableWidgetItem(fill["soil_type"])
        soil_item.setFlags(soil_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, COL_SOIL, soil_item)
        if is_hs:
            self._lock_thickness(r, True)

    def _add_layer(self) -> None:
        self._block = True
        try:
            self._add_row()
        finally:
            self._block = False
        self._renumber()
        self.layers_changed.emit()

    def _remove_layer(self) -> None:
        row = self.table.currentRow()
        if row < 0 or self.table.rowCount() <= 1:
            return
        self.table.removeRow(row)
        self._renumber()
        self.layers_changed.emit()

    def _move_up(self) -> None:
        r = self.table.currentRow()
        if r <= 0:
            return
        self._swap_rows(r, r - 1)
        self.table.setCurrentCell(r - 1, 0)
        self.layers_changed.emit()

    def _move_down(self) -> None:
        r = self.table.currentRow()
        if r < 0 or r >= self.table.rowCount() - 1:
            return
        self._swap_rows(r, r + 1)
        self.table.setCurrentCell(r + 1, 0)
        self.layers_changed.emit()

    # ==================================================================
    #  Helpers
    # ==================================================================
    def _float(self, row: int, col: int, default: float = 0.0) -> float:
        item = self.table.item(row, col)
        if item is None:
            return default
        try:
            return float(item.text())
        except ValueError:
            return default

    def _hs_checked(self, row: int) -> bool:
        holder = self.table.cellWidget(row, COL_HS)
        cb = holder.findChild(QCheckBox) if holder else None
        return bool(cb and cb.isChecked())

    def _renumber(self) -> None:
        self._block = True
        try:
            for r in range(self.table.rowCount()):
                item = self.table.item(r, COL_LAYER)
                if item:
                    item.setText(str(r + 1))
        finally:
            self._block = False

    def _lock_thickness(self, row: int, locked: bool) -> None:
        item = self.table.item(row, COL_THICK)
        if item is None:
            return
        if locked:
            item.setText("0.00")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        else:
            item.setFlags(item.flags() | Qt.ItemIsEditable)

    def _swap_rows(self, r1: int, r2: int) -> None:
        self._block = True
        try:
            for c in (COL_THICK, COL_VS, COL_VP, COL_NU, COL_DENSITY,
                      COL_SOIL):
                i1, i2 = self.table.item(r1, c), self.table.item(r2, c)
                if i1 and i2:
                    t1, t2 = i1.text(), i2.text()
                    i1.setText(t2)
                    i2.setText(t1)
            hs1, hs2 = self._hs_checked(r1), self._hs_checked(r2)
            for row, val in ((r1, hs2), (r2, hs1)):
                holder = self.table.cellWidget(row, COL_HS)
                cb = holder.findChild(QCheckBox) if holder else None
                if cb:
                    cb.setChecked(val)
        finally:
            self._block = False
        self._renumber()

    # ==================================================================
    #  Live derivation handlers (legacy behaviour, api-sourced values)
    # ==================================================================
    def _on_cell_changed(self, row: int, col: int) -> None:
        if self._block:
            return
        if col == COL_VS:
            vs = self._float(row, COL_VS, 200.0)
            fill = self._fill(vs)
            self._block = True
            try:
                soil_item = self.table.item(row, COL_SOIL)
                if soil_item:
                    soil_item.setText(fill["soil_type"])
                combo = self.table.cellWidget(row, COL_VPMODE)
                if combo and combo.currentIndex() == 0:   # Auto (from Vs)
                    self.table.item(row, COL_NU).setText(f"{fill['nu']:.3f}")
                    self.table.item(row, COL_VP).setText(f"{fill['vp']:.1f}")
            finally:
                self._block = False
        self.layers_changed.emit()

    def _on_mode_changed(self, row: int) -> None:
        combo = self.table.cellWidget(row, COL_VPMODE)
        if combo is None:
            return
        mode = combo.currentIndex()
        vs = self._float(row, COL_VS, 200.0)
        self._block = True
        try:
            if mode == 0:          # Auto (from Vs)
                fill = self._fill(vs)
                self.table.item(row, COL_NU).setText(f"{fill['nu']:.3f}")
                self.table.item(row, COL_VP).setText(f"{fill['vp']:.1f}")
            elif mode == 1:        # From Nu — Vp from the TYPED ν
                nu = self._float(row, COL_NU, 0.33)
                fill = self._fill(vs, nu)
                self.table.item(row, COL_VP).setText(f"{fill['vp']:.1f}")
        finally:
            self._block = False
        self.layers_changed.emit()

    def _on_hs_changed(self, row: int) -> None:
        self._block = True
        try:
            self._lock_thickness(row, self._hs_checked(row))
        finally:
            self._block = False
        self.layers_changed.emit()


__all__ = ["LayerTable", "HEADERS", "VP_MODES"]
