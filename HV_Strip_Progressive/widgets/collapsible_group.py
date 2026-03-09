"""
Collapsible Group Box
=====================

A custom collapsible container widget with toggle header (▼/►).
Used to reduce visual clutter in settings/config panels.
Ported from bedrock_mapping package.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QToolButton, QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt


class CollapsibleGroupBox(QWidget):
    """A collapsible container with a toggle header.

    Usage::

        group = CollapsibleGroupBox("📊 Settings", collapsed=True)
        content = QVBoxLayout()
        content.addWidget(QLabel("hello"))
        group.setContentLayout(content)
    """

    def __init__(self, title: str = "", collapsed: bool = False,
                 parent=None):
        super().__init__(parent)

        self._toggle = QToolButton(self)
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setArrowType(
            Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle.toggled.connect(self._on_toggled)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._line = QFrame(self)
        self._line.setFrameShape(QFrame.HLine)
        self._line.setFrameShadow(QFrame.Sunken)

        self._content = QWidget(self)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 0, 0, 4)
        self._content.setVisible(not collapsed)

        main = QVBoxLayout(self)
        main.setSpacing(0)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self._toggle)
        main.addWidget(self._line)
        main.addWidget(self._content)

    def _on_toggled(self, checked: bool):
        self._toggle.setArrowType(
            Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)

    def setContentLayout(self, layout):
        """Replace the content layout."""
        QWidget().setLayout(self._content.layout())
        self._content.setLayout(layout)
        layout.setContentsMargins(8, 4, 4, 4)

    def addWidget(self, widget):
        self._content_layout.addWidget(widget)

    def addLayout(self, layout):
        self._content_layout.addLayout(layout)

    def setCollapsed(self, collapsed: bool):
        self._toggle.setChecked(not collapsed)

    def isCollapsed(self) -> bool:
        return not self._toggle.isChecked()

    def title(self) -> str:
        return self._toggle.text()

    def setTitle(self, title: str):
        self._toggle.setText(title)
