"""Log View — Color-coded console output with timestamps.

Right-panel canvas tab showing all analysis messages with
auto-scroll, timestamps, and color coding.
"""
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QCheckBox,
)

from ..widgets.style_constants import MONOSPACE_PREVIEW, EMOJI


class LogView(QWidget):
    """Canvas view for analysis log output."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._auto_scroll = True
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        # Header
        hdr_row = QHBoxLayout()
        hdr = QLabel(f"<b>{EMOJI['info']} Analysis Log</b>")
        hdr.setStyleSheet("font-size: 13px; padding: 2px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self._chk_scroll = QCheckBox("Auto-scroll")
        self._chk_scroll.setChecked(True)
        self._chk_scroll.toggled.connect(self._set_auto_scroll)
        hdr_row.addWidget(self._chk_scroll)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(60)
        btn_clear.clicked.connect(self._clear)
        hdr_row.addWidget(btn_clear)
        lay.addLayout(hdr_row)

        # Log text
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(MONOSPACE_PREVIEW)
        lay.addWidget(self._text)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def append(self, msg):
        """Append a timestamped message to the log."""
        ts = datetime.now().strftime("%H:%M:%S")

        # Color-code based on content
        if "error" in msg.lower() or "ERROR" in msg:
            color = "#E63946"
        elif "success" in msg.lower() or "completed" in msg.lower() or msg.startswith("OK"):
            color = "#27AE60"
        elif "warning" in msg.lower():
            color = "#FF9800"
        else:
            color = "#333"

        self._text.append(
            f"<span style='color:#888;'>[{ts}]</span> "
            f"<span style='color:{color};'>{msg}</span>")

        if self._auto_scroll:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _clear(self):
        self._text.clear()

    def _set_auto_scroll(self, on):
        self._auto_scroll = on
