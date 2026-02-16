"""
GUI module for HVSR Progressive Layer Stripping.

A comprehensive graphical user interface built with PySide6 and QFluentWidgets.
"""

import sys
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


__all__ = ['MainWindow', 'run_gui']
__version__ = '1.0.0'
