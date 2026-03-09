"""
Collapsible group-box widget.

A QGroupBox with a clickable title that toggles visibility of its
contents.  Used throughout the left/right panels for compactness.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QWidget, QSizePolicy,
)


class CollapsibleGroup(QGroupBox):
    """Group box whose content area can be collapsed by clicking the title."""

    toggled_signal = pyqtSignal(bool)  # True = expanded

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(not collapsed)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(4)

        outer = QVBoxLayout()
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self._content)
        super().setLayout(outer)

        self._content.setVisible(not collapsed)
        super().toggled.connect(self._on_toggle)

    # Public API

    def content_layout(self) -> QVBoxLayout:
        """Return the layout to add child widgets into."""
        return self._layout

    def add_widget(self, widget: QWidget):
        """Convenience: add a widget to the content area."""
        self._layout.addWidget(widget)

    def add_layout(self, layout):
        """Convenience: add a sub-layout to the content area."""
        self._layout.addLayout(layout)

    def is_expanded(self) -> bool:
        return self.isChecked()

    def set_expanded(self, expanded: bool):
        self.setChecked(expanded)

    # Internal

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        self.toggled_signal.emit(checked)
