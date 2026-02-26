"""
Main Window for HVSR Progressive Layer Stripping GUI

Reorganized with simplified 4-tab navigation:
1. Home - Workflow and batch processing
2. Profile Editor - Create/edit soil profiles  
3. Forward Modeling - Compute HV curves
4. Visualization - Generate figures
"""

from pathlib import Path

import yaml
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

_SETTINGS_DIR = Path.home() / ".hvstrip"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.yaml"


class MainWindow(MSFluentWindow):
    """Main application window with simplified navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVSR Progressive Layer Stripping Analysis")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        self._saved_config = self._load_settings()
        self._create_pages()
        self._apply_saved_config()
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
        self.settingsPage.settingsSaved.connect(self._on_settings_saved)

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

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def _load_settings() -> dict:
        """Load persisted settings from disk."""
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _save_settings(config: dict):
        """Persist settings to disk."""
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def _apply_saved_config(self):
        """Push persisted config into pages."""
        if not self._saved_config:
            return
        self.settingsPage.config.update(self._saved_config)
        self.settingsPage.updateUIFromConfig()
        engine_cfg = self._saved_config.get("engine_settings", {})
        self.homePage._engine_settings_config = engine_cfg
        self.forwardModelingPage._engine_settings_config = engine_cfg

    def _on_settings_saved(self):
        """Handle settings saved signal — persist and propagate."""
        cfg = self.settingsPage.getConfig()
        self._save_settings(cfg)
        engine_cfg = cfg.get("engine_settings", {})
        self.homePage._engine_settings_config = engine_cfg
        self.forwardModelingPage._engine_settings_config = engine_cfg

    def closeEvent(self, event):
        """Handle window close — auto-save current settings."""
        try:
            cfg = self.settingsPage.getConfig()
            self._save_settings(cfg)
        except Exception:
            pass
        super().closeEvent(event)
