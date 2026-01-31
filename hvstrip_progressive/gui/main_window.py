"""
Main Window for HVSR Progressive Layer Stripping GUI

Reorganized with simplified 4-tab navigation:
1. Home - Workflow and batch processing
2. Profile Editor - Create/edit soil profiles  
3. Forward Modeling - Compute HV curves
4. Visualization - Generate figures
"""

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtWidgets import QWidget, QStackedWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtGui import QIcon
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition,
    FluentIcon as FIF, TitleLabel, setFont,
    isDarkTheme, MSFluentWindow, SplashScreen
)

from .pages.home_page import HomePage
from .pages.forward_modeling_page import ForwardModelingPage
from .pages.visualization_page import VisualizationPage
from .pages.settings_page import SettingsPage


class MainWindow(MSFluentWindow):
    """Main application window with simplified navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVSR Progressive Layer Stripping Analysis")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        self._create_pages()
        self._connect_signals()
        self.initNavigation()

        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(102, 102))
        self.show()
        QTimer.singleShot(500, self.splashScreen.finish)

    def _create_pages(self):
        """Create all application pages."""
        self.homePage = HomePage(self)
        self.forwardModelingPage = ForwardModelingPage(self)
        self.visualizationPage = VisualizationPage(self)
        self.settingsPage = SettingsPage(self)

    def _connect_signals(self):
        """Connect inter-page signals."""
        pass

    def initNavigation(self):
        """Initialize simplified navigation interface."""
        self.addSubInterface(
            self.homePage,
            FIF.HOME,
            'Home',
            FIF.HOME
        )

        self.addSubInterface(
            self.forwardModelingPage,
            FIF.CALORIES,
            'HV Forward',
            FIF.CALORIES
        )

        self.addSubInterface(
            self.visualizationPage,
            FIF.PHOTO,
            'Figures',
            FIF.PHOTO
        )

        self.addSubInterface(
            self.settingsPage,
            FIF.SETTING,
            'Settings',
            FIF.SETTING,
            NavigationItemPosition.BOTTOM
        )

    def closeEvent(self, event):
        """Handle window close event."""
        super().closeEvent(event)
