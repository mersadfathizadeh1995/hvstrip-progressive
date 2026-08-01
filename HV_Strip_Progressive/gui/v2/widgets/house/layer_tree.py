"""``LayerTree`` — the right-dock layer/runs tree over the :class:`LayerModel`.

Copied (copy-not-import) from the bedrock ``LayerPanel`` and retargeted to HV
Invert's stage-aware :class:`LayerModel`.  Groups → leaves; the checkbox drives
visibility; selecting a row drives the Properties panel.  Rebuilds on
``layers_rebuilt`` and shows only available layers.  GUI-only — reads and
writes the view-model, never the session.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.state.layer_model import LayerModel, LayerNode


class LayerTree(QWidget):
    """Layer-visibility tree; emits :attr:`layer_selected` (key or "")."""

    layer_selected = Signal(str)

    def __init__(
        self, layer_model: LayerModel, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._model = layer_model
        self._updating = False
        self._items: Dict[str, QTreeWidgetItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setProperty("role", "layerTree")
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self._tree, 1)

        self._empty = QLabel("No layers yet — advance the workflow.")
        self._empty.setProperty("role", "muted")
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)

        layer_model.layers_rebuilt.connect(self.rebuild)
        layer_model.visibility_changed.connect(self._on_model_visibility)
        self.rebuild()

    def _on_model_visibility(self, key: str) -> None:
        """Keep checkboxes in sync when visibility changes elsewhere
        (e.g. the Properties panel's group Show/Hide-all)."""
        if self._updating:
            return
        item = self._items.get(key)
        if item is not None:
            self._updating = True
            item.setCheckState(
                0, Qt.Checked if self._model.is_visible(key) else Qt.Unchecked)
            self._updating = False

    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        self._updating = True
        self._tree.clear()
        self._items.clear()
        groups = self._model.groups()
        for group in groups:
            gi = self._add_node(self._tree, group)
            for child in group.children:
                self._add_node(gi, child)
        self._tree.expandAll()
        self._updating = False
        self._empty.setVisible(not groups)
        self._tree.setVisible(bool(groups))

    def _add_node(self, parent, node: LayerNode) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent)
        item.setText(0, node.label)
        item.setData(0, Qt.UserRole, node.key)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(
            0, Qt.Checked if self._model.is_visible(node.key)
            else Qt.Unchecked
        )
        self._items[node.key] = item
        return item

    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTreeWidgetItem, _col: int) -> None:
        if self._updating:
            return
        key = item.data(0, Qt.UserRole)
        if not key:
            return
        visible = item.checkState(0) == Qt.Checked
        self._model.set_visible(key, visible)
        if item.childCount():
            self._updating = True
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(
                    0, Qt.Checked if visible else Qt.Unchecked
                )
                ck = child.data(0, Qt.UserRole)
                if ck:
                    self._model.set_visible(ck, visible)
            self._updating = False

    def _on_current_changed(self, current, _previous) -> None:
        key = current.data(0, Qt.UserRole) if current is not None else ""
        self.layer_selected.emit(key or "")

    # ------------------------------------------------------------------
    def select(self, key: str) -> None:
        item = self._items.get(key)
        if item is not None:
            self._tree.setCurrentItem(item)

    def set_checked(self, key: str, checked: bool) -> None:
        item = self._items.get(key)
        if item is not None:
            item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)


__all__ = ["LayerTree"]
