"""
Layer table widget for editing soil profile layers.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QPushButton, QComboBox, QDoubleSpinBox,
    QCheckBox, QLabel, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush, QAction

from ...core.soil_profile import Layer, SoilProfile
from ...core.velocity_utils import VelocityConverter


class LayerTableWidget(QWidget):
    """
    Table widget for editing soil profile layers.
    
    Provides an editable table with columns for layer properties,
    automatic Vp calculation, and nu suggestions.
    """
    
    profile_changed = Signal()
    layer_selected = Signal(int)
    
    COLUMNS = [
        ("Layer", 50),
        ("Thickness (m)", 100),
        ("Vs (m/s)", 90),
        ("Vp (m/s)", 90),
        ("Nu", 70),
        ("Vp Mode", 90),
        ("Density (kg/m3)", 110),
        ("Half-space", 80),
        ("Soil Type", 150),
    ]
    
    COL_LAYER = 0
    COL_THICKNESS = 1
    COL_VS = 2
    COL_VP = 3
    COL_NU = 4
    COL_VP_MODE = 5
    COL_DENSITY = 6
    COL_HALFSPACE = 7
    COL_SOIL_TYPE = 8
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile = SoilProfile()
        self._updating = False
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([col[0] for col in self.COLUMNS])
        
        header = self.table.horizontalHeader()
        for i, (_, width) in enumerate(self.COLUMNS):
            header.resizeSection(i, width)
        header.setStretchLastSection(True)
        
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Layer")
        self.btn_add.clicked.connect(self._add_layer)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_remove = QPushButton("Remove Layer")
        self.btn_remove.clicked.connect(self._remove_layer)
        btn_layout.addWidget(self.btn_remove)
        
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_up.clicked.connect(self._move_layer_up)
        btn_layout.addWidget(self.btn_move_up)
        
        self.btn_move_down = QPushButton("Move Down")
        self.btn_move_down.clicked.connect(self._move_layer_down)
        btn_layout.addWidget(self.btn_move_down)
        
        btn_layout.addStretch()
        
        self.btn_suggest_all = QPushButton("Auto-fill Suggestions")
        self.btn_suggest_all.clicked.connect(self._suggest_all)
        btn_layout.addWidget(self.btn_suggest_all)
        
        layout.addLayout(btn_layout)
    
    def set_profile(self, profile: SoilProfile):
        """Set the soil profile to display and edit."""
        self._profile = profile
        self._refresh_table()
    
    def get_profile(self) -> SoilProfile:
        """Get the current soil profile."""
        return self._profile
    
    def _refresh_table(self):
        """Refresh the table from the profile data."""
        self._updating = True
        try:
            self.table.setRowCount(len(self._profile.layers))
            
            for row, layer in enumerate(self._profile.layers):
                self._set_row_data(row, layer)
        finally:
            self._updating = False
    
    def _set_row_data(self, row: int, layer: Layer):
        """Set data for a single row."""
        layer_item = QTableWidgetItem(str(row + 1))
        layer_item.setFlags(layer_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        layer_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, self.COL_LAYER, layer_item)
        
        thickness_item = QTableWidgetItem(f"{layer.thickness:.2f}")
        thickness_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if layer.is_halfspace:
            thickness_item.setFlags(thickness_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            thickness_item.setText("inf")
            thickness_item.setBackground(QBrush(QColor(240, 240, 240)))
        self.table.setItem(row, self.COL_THICKNESS, thickness_item)
        
        vs_item = QTableWidgetItem(f"{layer.vs:.1f}")
        vs_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, self.COL_VS, vs_item)
        
        vp_value = layer.compute_vp()
        vp_item = QTableWidgetItem(f"{vp_value:.1f}")
        vp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if layer.vp is None:
            vp_item.setForeground(QBrush(QColor(100, 100, 100)))
            vp_item.setFlags(vp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, self.COL_VP, vp_item)
        
        nu_value = layer.get_effective_nu()
        nu_item = QTableWidgetItem(f"{nu_value:.3f}")
        nu_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if layer.nu is None and layer.vp is None:
            nu_item.setForeground(QBrush(QColor(100, 100, 100)))
        self.table.setItem(row, self.COL_NU, nu_item)
        
        mode_combo = QComboBox()
        mode_combo.addItems(["Auto (from Vs)", "From Nu", "Manual Vp"])
        if layer.vp is not None:
            mode_combo.setCurrentIndex(2)
        elif layer.nu is not None:
            mode_combo.setCurrentIndex(1)
        else:
            mode_combo.setCurrentIndex(0)
        mode_combo.currentIndexChanged.connect(
            lambda idx, r=row: self._on_mode_changed(r, idx)
        )
        self.table.setCellWidget(row, self.COL_VP_MODE, mode_combo)
        
        density_item = QTableWidgetItem(f"{layer.density:.1f}")
        density_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, self.COL_DENSITY, density_item)
        
        hs_checkbox = QCheckBox()
        hs_checkbox.setChecked(layer.is_halfspace)
        hs_checkbox.stateChanged.connect(
            lambda state, r=row: self._on_halfspace_changed(r, state)
        )
        hs_widget = QWidget()
        hs_layout = QHBoxLayout(hs_widget)
        hs_layout.setContentsMargins(0, 0, 0, 0)
        hs_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hs_layout.addWidget(hs_checkbox)
        self.table.setCellWidget(row, self.COL_HALFSPACE, hs_widget)
        
        soil_type = layer.get_soil_type()
        soil_item = QTableWidgetItem(soil_type)
        soil_item.setFlags(soil_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        soil_item.setForeground(QBrush(QColor(80, 80, 80)))
        self.table.setItem(row, self.COL_SOIL_TYPE, soil_item)
    
    def _on_cell_changed(self, row: int, col: int):
        """Handle cell value changes."""
        if self._updating or row >= len(self._profile.layers):
            return
        
        layer = self._profile.layers[row]
        item = self.table.item(row, col)
        if item is None:
            return
        
        try:
            text = item.text().strip()
            
            if col == self.COL_THICKNESS:
                if text.lower() != "inf":
                    layer.thickness = float(text)
            elif col == self.COL_VS:
                layer.vs = float(text)
                self._update_computed_values(row)
            elif col == self.COL_VP:
                if layer.vp is not None:
                    layer.vp = float(text)
            elif col == self.COL_NU:
                layer.nu = float(text)
                self._update_computed_values(row)
            elif col == self.COL_DENSITY:
                layer.density = float(text)
            
            self.profile_changed.emit()
        except ValueError:
            self._refresh_table()
    
    def _on_mode_changed(self, row: int, mode_index: int):
        """Handle Vp mode changes."""
        if row >= len(self._profile.layers):
            return
        
        layer = self._profile.layers[row]
        
        if mode_index == 0:
            layer.vp = None
            layer.nu = None
        elif mode_index == 1:
            layer.vp = None
            if layer.nu is None:
                layer.nu = VelocityConverter.suggest_nu(layer.vs)
        elif mode_index == 2:
            layer.nu = None
            if layer.vp is None:
                layer.vp = layer.compute_vp()
        
        self._updating = True
        self._set_row_data(row, layer)
        self._updating = False
        self.profile_changed.emit()
    
    def _on_halfspace_changed(self, row: int, state: int):
        """Handle half-space checkbox changes."""
        if row >= len(self._profile.layers):
            return
        
        layer = self._profile.layers[row]
        layer.is_halfspace = state == Qt.CheckState.Checked.value
        
        if layer.is_halfspace:
            layer.thickness = 0
        elif layer.thickness == 0:
            layer.thickness = 10.0
        
        self._updating = True
        self._set_row_data(row, layer)
        self._updating = False
        self.profile_changed.emit()
    
    def _update_computed_values(self, row: int):
        """Update computed values for a row after Vs or Nu changes."""
        if row >= len(self._profile.layers):
            return
        
        layer = self._profile.layers[row]
        
        self._updating = True
        
        vp_value = layer.compute_vp()
        vp_item = self.table.item(row, self.COL_VP)
        if vp_item:
            vp_item.setText(f"{vp_value:.1f}")
        
        nu_value = layer.get_effective_nu()
        nu_item = self.table.item(row, self.COL_NU)
        if nu_item:
            nu_item.setText(f"{nu_value:.3f}")
        
        soil_item = self.table.item(row, self.COL_SOIL_TYPE)
        if soil_item:
            soil_item.setText(layer.get_soil_type())
        
        self._updating = False
    
    def _add_layer(self):
        """Add a new layer."""
        current_row = self.table.currentRow()
        insert_idx = current_row + 1 if current_row >= 0 else len(self._profile.layers)
        
        new_layer = Layer(
            thickness=5.0,
            vs=300.0,
            density=VelocityConverter.suggest_density(300.0)
        )
        
        self._profile.layers.insert(insert_idx, new_layer)
        
        if len(self._profile.layers) == 1:
            new_layer.is_halfspace = True
            new_layer.thickness = 0
        
        self._refresh_table()
        self.profile_changed.emit()
    
    def _remove_layer(self):
        """Remove the selected layer."""
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self._profile.layers):
            return
        
        if len(self._profile.layers) <= 1:
            QMessageBox.warning(
                self, "Cannot Remove",
                "Profile must have at least one layer."
            )
            return
        
        self._profile.layers.pop(current_row)
        self._refresh_table()
        self.profile_changed.emit()
    
    def _move_layer_up(self):
        """Move the selected layer up."""
        current_row = self.table.currentRow()
        if current_row <= 0:
            return
        
        self._profile.move_layer(current_row, current_row - 1)
        self._refresh_table()
        self.table.selectRow(current_row - 1)
        self.profile_changed.emit()
    
    def _move_layer_down(self):
        """Move the selected layer down."""
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self._profile.layers) - 1:
            return
        
        self._profile.move_layer(current_row, current_row + 1)
        self._refresh_table()
        self.table.selectRow(current_row + 1)
        self.profile_changed.emit()
    
    def _suggest_all(self):
        """Apply suggested values to all layers."""
        for layer in self._profile.layers:
            if layer.vp is None and layer.nu is None:
                layer.nu = VelocityConverter.suggest_nu(layer.vs)
            if layer.density == 2000.0:
                layer.density = VelocityConverter.suggest_density(layer.vs)
        
        self._refresh_table()
        self.profile_changed.emit()
    
    def _on_selection_changed(self):
        """Handle row selection changes."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.layer_selected.emit(current_row)
    
    def _show_context_menu(self, pos):
        """Show context menu."""
        menu = QMenu(self)
        
        add_action = QAction("Add Layer", self)
        add_action.triggered.connect(self._add_layer)
        menu.addAction(add_action)
        
        remove_action = QAction("Remove Layer", self)
        remove_action.triggered.connect(self._remove_layer)
        menu.addAction(remove_action)
        
        menu.addSeparator()
        
        suggest_action = QAction("Apply Suggested Values", self)
        suggest_action.triggered.connect(self._suggest_all)
        menu.addAction(suggest_action)
        
        menu.exec(self.table.viewport().mapToGlobal(pos))
