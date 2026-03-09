"""Pytest conftest — force PyQt5 for matplotlib before any test imports."""
import os
os.environ["QT_API"] = "pyqt5"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass

# Force PyQt5 to load before PySide6
import PyQt5.QtCore   # noqa: F401
import PyQt5.QtWidgets  # noqa: F401
