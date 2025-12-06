"""
Main Window for HVSR Progressive Layer Stripping GUI

Provides navigation interface with sidebar and page management.
"""

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtWidgets import QWidget, QStackedWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtGui import QIcon
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition,
    FluentIcon as FIF, TitleLabel, setFont,
    isDarkTheme, MSFluentWindow, SplashScreen
)

from .pages.workflow_page import WorkflowPage
from .pages.strip_page import StripPage
from .pages.forward_page import ForwardPage
from .pages.postprocess_page import PostprocessPage
from .pages.report_page import ReportPage
from .pages.batch_page import BatchPage
from .pages.analysis_page import AnalysisPage
from .pages.settings_page import SettingsPage


class MainWindow(MSFluentWindow):
    """Main application window with fluent design."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVSR Progressive Layer Stripping Analysis")
        self.resize(1200, 800)

        # Create pages
        self.workflowPage = WorkflowPage(self)
        self.stripPage = StripPage(self)
        self.forwardPage = ForwardPage(self)
        self.postprocessPage = PostprocessPage(self)
        self.reportPage = ReportPage(self)
        self.batchPage = BatchPage(self)
        self.analysisPage = AnalysisPage(self)
        self.settingsPage = SettingsPage(self)

        # Initialize navigation
        self.initNavigation()

        # Create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(102, 102))
        self.show()

        # Close splash screen after a short delay to show the main window
        QTimer.singleShot(500, self.splashScreen.finish)

    def initNavigation(self):
        """Initialize navigation interface."""
        # Add workflow page (main functionality)
        self.addSubInterface(
            self.workflowPage,
            FIF.PLAY,
            'Complete Workflow',
            FIF.PLAY
        )

        # Add individual component pages
        self.addSubInterface(
            self.stripPage,
            FIF.CUT,
            'Layer Stripping',
            FIF.CUT
        )

        self.addSubInterface(
            self.forwardPage,
            FIF.CALORIES,
            'HV Forward',
            FIF.CALORIES
        )

        self.addSubInterface(
            self.postprocessPage,
            FIF.PIE_SINGLE,
            'Post-processing',
            FIF.PIE_SINGLE
        )

        self.addSubInterface(
            self.reportPage,
            FIF.DOCUMENT,
            'Report Generation',
            FIF.DOCUMENT
        )

        # Add batch and analysis pages
        self.addSubInterface(
            self.batchPage,
            FIF.LAYOUT,
            'Batch Processing',
            FIF.LAYOUT
        )

        self.addSubInterface(
            self.analysisPage,
            FIF.SEARCH,
            'Advanced Analysis',
            FIF.SEARCH
        )

        # Add settings at bottom
        self.addSubInterface(
            self.settingsPage,
            FIF.SETTING,
            'Settings',
            FIF.SETTING,
            NavigationItemPosition.BOTTOM
        )

    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up any running processes if needed
        super().closeEvent(event)
