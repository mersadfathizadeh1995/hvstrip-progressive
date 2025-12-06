"""
HVSTRIP-Progressive GUI Application
Main application entry point using PySide6 and QFluentWidgets
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from qfluentwidgets import setTheme, Theme, FluentIcon

from .main_window import MainWindow


class HVStripApp(QApplication):
    """Main application class for HVSTRIP-Progressive GUI"""

    def __init__(self, argv):
        super().__init__(argv)

        # Set application metadata
        self.setApplicationName("HVSTRIP-Progressive")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("HVSTRIP")

        # Enable high DPI scaling
        self.setAttribute(Qt.AA_EnableHighDpiScaling)
        self.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # Set default theme
        setTheme(Theme.AUTO)

        # Create and show main window
        self.main_window = MainWindow()
        self.main_window.show()


def main():
    """Main entry point for the GUI application"""
    app = HVStripApp(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
