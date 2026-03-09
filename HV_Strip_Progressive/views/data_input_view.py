"""Data Input View — canvas tab for single-profile data loading.

Minimal wrapper around FormatInputStack. Fills the canvas with
format selector + Vs preview. Used by Forward Single and Strip Single.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

from ..widgets.format_input_stack import FormatInputStack


class DataInputView(QWidget):
    """Canvas view for loading a single soil profile via format dropdown."""

    profile_loaded = pyqtSignal(object, str)  # (SoilProfile, path)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # The main format input widget — fills the canvas
        self._format_input = FormatInputStack()
        self._format_input.profile_loaded.connect(self._on_loaded)
        lay.addWidget(self._format_input, 1)

    def _on_loaded(self, profile, path):
        self.profile_loaded.emit(profile, path)
        if self._mw:
            from pathlib import Path
            name = Path(path).name if path != "editor" else "Editor profile"
            self._mw.log(f"Profile loaded: {name}")

    # Public API
    def get_profile(self):
        return self._format_input.get_profile()

    def get_path(self):
        return self._format_input.get_path()

    def load_profile(self, path):
        self._format_input.load_profile(path)
