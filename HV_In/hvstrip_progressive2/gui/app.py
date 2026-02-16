"""
HVSR Progressive Layer Stripping - GUI Application Entry Point

A comprehensive graphical interface for progressive layer stripping analysis
using PySide6 and QFluentWidgets.

Author: HVSR-Diffuse Development Team
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QLocale
from qfluentwidgets import setTheme, Theme, FluentTranslator

from .main_window import MainWindow


def main():
    """Main application entry point."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("HVSR Progressive Layer Stripping")
    app.setOrganizationName("HVSR-Diffuse")
    app.setApplicationVersion("1.0.0")

    # Set locale for translations
    locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
    translator = FluentTranslator(locale)
    app.installTranslator(translator)

    # Set theme (Auto will follow system theme)
    setTheme(Theme.AUTO)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
