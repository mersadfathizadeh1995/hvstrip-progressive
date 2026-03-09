"""
Profile editor widget — layer table + preview + validation.

Provides a full soil profile editor with:
- Editable layer table (thickness, Vs, Vp, density, Vp mode, halfspace)
- Auto-computed Nu, soil type, suggested Vp
- Add / Remove / Move Up / Move Down / Auto-fill
- Vs profile preview (graphical)
- New / Open / Save operations
- Poisson's ratio reference table
"""

import os
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QCheckBox, QPushButton, QLabel, QHeaderView, QGroupBox,
    QFileDialog, QMessageBox, QAbstractItemView, QSplitter, QTextEdit,
    QToolBar,
)
from PyQt5.QtGui import QColor

from ..widgets.plot_canvas import PlotCanvas


# ── Constants ────────────────────────────────────────────────────────

_COLUMNS = [
    ('Layer', 50), ('Thickness\n(m)', 90), ('Vs\n(m/s)', 80),
    ('Vp\n(m/s)', 80), ('Nu', 60), ('Vp Mode', 100),
    ('Density\n(kg/m³)', 100), ('Halfspace', 70), ('Soil Type', 130),
]
_VP_MODES = ['Auto (from Vs)', 'From Nu', 'Manual Vp']
_READ_ONLY_COLS = {0, 4, 8}  # Layer#, Nu, Soil Type


class ProfileEditor(QWidget):
    """Editable soil profile with layer table and graphical preview."""

    profile_changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._building = False  # suppress signals during table rebuild

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Toolbar ──────────────────────────────────────────────────
        tb = QToolBar()
        tb.addAction('New', self._on_new)
        tb.addAction('Open…', self._on_open)
        tb.addAction('Save…', self._on_save)
        tb.addSeparator()
        tb.addAction('Add Layer', self._on_add)
        tb.addAction('Remove', self._on_remove)
        tb.addAction('Move ▲', self._on_move_up)
        tb.addAction('Move ▼', self._on_move_down)
        tb.addSeparator()
        tb.addAction('Auto-fill', self._on_autofill)
        layout.addWidget(tb)

        # ── Splitter: table (left) + preview (right) ────────────────
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Table
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        hdr = self.table.horizontalHeader()
        for i, (_, w) in enumerate(_COLUMNS):
            hdr.resizeSection(i, w)
        hdr.setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellChanged.connect(self._on_cell_changed)
        splitter.addWidget(self.table)

        # Right panel: preview + validation
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.preview = PlotCanvas(figsize=(3, 5), dpi=80, toolbar=False)
        right_lay.addWidget(self.preview, 1)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(80)
        right_lay.addWidget(self.validation_text)

        splitter.addWidget(right)
        splitter.setSizes([500, 200])

        # Start with a default 2-layer profile
        self._new_default_profile()

    # ── Public API ───────────────────────────────────────────────────

    def load_profile(self, profile):
        """Populate table from a SoilProfile object."""
        self._building = True
        self.table.setRowCount(0)
        for i, layer in enumerate(profile.layers):
            self._insert_row(i, layer)
        self._building = False
        self._sync_to_state()

    def get_profile(self):
        """Build a SoilProfile from current table contents."""
        from ..core.soil_profile import SoilProfile, Layer
        layers = []
        for row in range(self.table.rowCount()):
            layer = self._row_to_layer(row)
            if layer is not None:
                layers.append(layer)
        return SoilProfile(layers=layers, name='Edited Profile')

    # ── Internal: table ↔ Layer conversion ───────────────────────────

    def _insert_row(self, row: int, layer=None):
        """Insert a row at *row* and populate from *layer*."""
        from ..core.soil_profile import Layer
        from ..core.velocity_utils import VelocityConverter as VC
        if layer is None:
            layer = Layer(thickness=5.0, vs=200.0, density=2000.0)

        self._building = True
        self.table.insertRow(row)

        # Col 0: Layer # (read-only)
        item = QTableWidgetItem(str(row + 1))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 0, item)

        # Col 1: Thickness
        val = '∞' if layer.is_halfspace else f'{layer.thickness:.2f}'
        self.table.setItem(row, 1, QTableWidgetItem(val))

        # Col 2: Vs
        self.table.setItem(row, 2, QTableWidgetItem(f'{layer.vs:.1f}'))

        # Col 3: Vp
        vp = layer.vp if layer.vp else layer.compute_vp()
        self.table.setItem(row, 3, QTableWidgetItem(f'{vp:.1f}'))

        # Col 4: Nu (read-only, computed)
        try:
            nu = VC.nu_from_vp_vs(vp, layer.vs)
        except Exception:
            nu = VC.suggest_nu(layer.vs)
        nu_item = QTableWidgetItem(f'{nu:.3f}')
        nu_item.setFlags(nu_item.flags() & ~Qt.ItemIsEditable)
        nu_item.setForeground(QColor(120, 120, 120))
        self.table.setItem(row, 4, nu_item)

        # Col 5: Vp Mode (combo)
        combo = QComboBox()
        combo.addItems(_VP_MODES)
        if layer.vp is not None and layer.nu is None:
            combo.setCurrentIndex(2)  # Manual
        elif layer.nu is not None:
            combo.setCurrentIndex(1)  # From Nu
        else:
            combo.setCurrentIndex(0)  # Auto
        combo.currentIndexChanged.connect(lambda: self._on_cell_changed(row, 5))
        self.table.setCellWidget(row, 5, combo)

        # Col 6: Density
        self.table.setItem(row, 6, QTableWidgetItem(f'{layer.density:.0f}'))

        # Col 7: Halfspace (checkbox)
        cb = QCheckBox()
        cb.setChecked(layer.is_halfspace)
        cb.stateChanged.connect(lambda: self._on_cell_changed(row, 7))
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.addWidget(cb)
        hbox.setAlignment(Qt.AlignCenter)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 7, container)

        # Col 8: Soil Type (read-only)
        st = VC.get_soil_type_description(layer.vs)
        st_item = QTableWidgetItem(st)
        st_item.setFlags(st_item.flags() & ~Qt.ItemIsEditable)
        st_item.setForeground(QColor(120, 120, 120))
        self.table.setItem(row, 8, st_item)

        self._building = False

    def _row_to_layer(self, row: int):
        """Extract a Layer from the given table row."""
        from ..core.soil_profile import Layer
        try:
            thick_text = self.table.item(row, 1).text().strip()
            thickness = 0.0 if thick_text in ('∞', 'inf', '0') else float(thick_text)

            vs = float(self.table.item(row, 2).text())
            vp = float(self.table.item(row, 3).text())
            density = float(self.table.item(row, 6).text())

            cb_container = self.table.cellWidget(row, 7)
            is_hs = False
            if cb_container:
                cb = cb_container.findChild(QCheckBox)
                if cb:
                    is_hs = cb.isChecked()

            combo = self.table.cellWidget(row, 5)
            mode = combo.currentIndex() if combo else 0

            nu = None
            vp_val = None
            if mode == 0:
                pass  # Auto — compute from Vs
            elif mode == 1:
                nu = float(self.table.item(row, 4).text())
            else:
                vp_val = vp

            return Layer(
                thickness=thickness if not is_hs else 0.0,
                vs=vs, vp=vp_val, nu=nu, density=density,
                is_halfspace=is_hs,
            )
        except (ValueError, AttributeError):
            return None

    def _reindex_rows(self):
        """Update Layer # column after add/remove/move."""
        self._building = True
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setText(str(row + 1))
        self._building = False

    def _update_computed_cells(self, row: int):
        """Recompute Nu, soil type, and optionally Vp for a row."""
        from ..core.velocity_utils import VelocityConverter as VC
        self._building = True
        try:
            vs = float(self.table.item(row, 2).text())
            vp = float(self.table.item(row, 3).text())

            # Nu
            try:
                nu = VC.nu_from_vp_vs(vp, vs)
            except Exception:
                nu = VC.suggest_nu(vs)
            self.table.item(row, 4).setText(f'{nu:.3f}')

            # Soil type
            self.table.item(row, 8).setText(VC.get_soil_type_description(vs))

            # Auto Vp mode
            combo = self.table.cellWidget(row, 5)
            if combo and combo.currentIndex() == 0:
                suggested_nu = VC.suggest_nu(vs)
                new_vp = VC.vp_from_vs_nu(vs, suggested_nu)
                self.table.item(row, 3).setText(f'{new_vp:.1f}')
                self.table.item(row, 4).setText(f'{suggested_nu:.3f}')
        except (ValueError, AttributeError):
            pass
        self._building = False

    # ── Sync to state ────────────────────────────────────────────────

    def _sync_to_state(self):
        """Build profile from table and push to state."""
        profile = self.get_profile()
        valid, errors = profile.validate()
        self.validation_text.setText(
            '✅ Valid' if valid else '❌ ' + '\n'.join(errors))
        self.state.set_profile(profile, self.state.active_profile_path)
        self.profile_changed.emit()
        self._draw_preview(profile)

    def _draw_preview(self, profile):
        """Draw Vs-depth step-function on the preview canvas."""
        self.preview.clear()
        ax = self.preview.add_subplot(111)
        if not profile.layers:
            self.preview.draw()
            return
        depths, vs_vals = [0.0], []
        for layer in profile.layers:
            if layer.is_halfspace:
                vs_vals.append(layer.vs)
                vs_vals.append(layer.vs)
                depths.append(depths[-1])
                hs_extra = max(sum(l.thickness for l in profile.layers
                                   if not l.is_halfspace) * 0.25, 5)
                depths.append(depths[-1] + hs_extra)
            else:
                vs_vals.append(layer.vs)
                vs_vals.append(layer.vs)
                depths.append(depths[-1])
                depths.append(depths[-1] + layer.thickness)

        if len(depths) == len(vs_vals) + 1:
            depths = depths[:-1]

        ax.step(vs_vals, depths[:len(vs_vals)], where='post', color='teal', lw=1.5)
        ax.invert_yaxis()
        ax.set_xlabel('Vs (m/s)', fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        self.preview.tight_layout()
        self.preview.draw()

    # ── Toolbar actions ──────────────────────────────────────────────

    def _new_default_profile(self):
        from ..core.soil_profile import SoilProfile, Layer
        profile = SoilProfile(layers=[
            Layer(thickness=5.0, vs=150.0, density=1800.0),
            Layer(thickness=10.0, vs=250.0, density=2000.0),
            Layer(thickness=0.0, vs=400.0, density=2200.0, is_halfspace=True),
        ], name='New Profile')
        self.load_profile(profile)

    def _on_new(self):
        self._new_default_profile()

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Model', '',
            'Model files (*.txt *.csv *.xlsx);;All files (*)')
        if not path:
            return
        try:
            from ..core.soil_profile import SoilProfile
            profile = SoilProfile.from_auto(path)
            self.state.active_profile_path = path
            self.load_profile(profile)
            self.state.status_message.emit(f'Loaded: {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load:\n{e}')

    def _on_save(self):
        path, filt = QFileDialog.getSaveFileName(
            self, 'Save Model', '',
            'HVf format (*.txt);;CSV (*.csv)')
        if not path:
            return
        try:
            profile = self.get_profile()
            if path.endswith('.csv'):
                profile.save_csv(path)
            else:
                profile.save_hvf(path)
            self.state.active_profile_path = path
            self.state.status_message.emit(f'Saved: {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save:\n{e}')

    def _on_add(self):
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount()
        else:
            row += 1
        self._insert_row(row)
        self._reindex_rows()
        self._sync_to_state()

    def _on_remove(self):
        row = self.table.currentRow()
        if row < 0 or self.table.rowCount() <= 1:
            return
        self.table.removeRow(row)
        self._reindex_rows()
        self._sync_to_state()

    def _on_move_up(self):
        row = self.table.currentRow()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self.table.setCurrentCell(row - 1, 0)
        self._sync_to_state()

    def _on_move_down(self):
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self.table.setCurrentCell(row + 1, 0)
        self._sync_to_state()

    def _swap_rows(self, a: int, b: int):
        """Swap data between two rows."""
        layer_a = self._row_to_layer(a)
        layer_b = self._row_to_layer(b)
        if layer_a and layer_b:
            self._building = True
            self.table.removeRow(b)
            self.table.removeRow(a)
            self._insert_row(a, layer_b)
            self._insert_row(b, layer_a)
            self._reindex_rows()
            self._building = False

    def _on_autofill(self):
        """Apply suggested Nu and density to all rows."""
        from ..core.velocity_utils import VelocityConverter as VC
        self._building = True
        for row in range(self.table.rowCount()):
            try:
                vs = float(self.table.item(row, 2).text())
                self.table.item(row, 6).setText(f'{VC.suggest_density(vs):.0f}')
                combo = self.table.cellWidget(row, 5)
                if combo and combo.currentIndex() == 0:
                    nu = VC.suggest_nu(vs)
                    vp = VC.vp_from_vs_nu(vs, nu)
                    self.table.item(row, 3).setText(f'{vp:.1f}')
                    self.table.item(row, 4).setText(f'{nu:.3f}')
            except (ValueError, AttributeError):
                pass
        self._building = False
        self._sync_to_state()

    # ── Cell change handler ──────────────────────────────────────────

    def _on_cell_changed(self, row: int, col: int):
        if self._building:
            return
        self._update_computed_cells(row)
        self._sync_to_state()
