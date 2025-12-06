"""
Main Window with Navigation
Uses QFluentWidgets navigation interface
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon,
    NavigationItemPosition,
    MSFluentWindow,
    SplashScreen
)

from .pages.home_page import HomePage
from .pages.settings_page import SettingsPage
from .pages.results_page import ResultsPage


class MainWindow(MSFluentWindow):
    """Main application window with navigation sidebar"""

    def __init__(self):
        super().__init__()

        # Create pages
        self.home_page = HomePage(self)
        self.settings_page = SettingsPage(self)
        self.results_page = ResultsPage(self)

        # Initialize UI
        self.init_navigation()
        self.init_window()

        # Create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(128, 128))
        self.show()

    def init_navigation(self):
        """Initialize navigation interface"""

        # Add pages to navigation
        self.addSubInterface(
            self.home_page,
            FluentIcon.HOME,
            "Home",
            position=NavigationItemPosition.TOP
        )

        self.addSubInterface(
            self.settings_page,
            FluentIcon.SETTING,
            "Settings",
            position=NavigationItemPosition.TOP
        )

        self.addSubInterface(
            self.results_page,
            FluentIcon.FOLDER,
            "Results",
            position=NavigationItemPosition.TOP
        )

        # Add separator and bottom items
        self.navigationInterface.addSeparator(NavigationItemPosition.BOTTOM)

        # Add about/help item
        self.navigationInterface.addItem(
            routeKey="about",
            icon=FluentIcon.INFO,
            text="About",
            onClick=self.show_about,
            selectable=False,
            position=NavigationItemPosition.BOTTOM
        )

    def init_window(self):
        """Initialize window properties"""
        self.resize(1200, 800)
        self.setWindowTitle("HVSTRIP-Progressive - Layer Stripping Analysis")

        # Center window on screen
        desktop = self.screen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

    def show_about(self):
        """Show about dialog"""
        from qfluentwidgets import MessageBox

        about_text = """
        <h3>HVSTRIP-Progressive</h3>
        <p>Progressive Layer Stripping Analysis Tool</p>
        <p>Version 1.0.0</p>
        <br>
        <p>Progressive layer stripping analysis of Horizontal-to-Vertical
        Spectral Ratio (HVSR) data using diffuse-field theory.</p>
        <br>
        <p>Identifies which subsurface interfaces control HVSR peaks through
        systematic layer removal and forward modeling.</p>
        """

        MessageBox(
            "About",
            about_text,
            self
        ).exec()

    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())
