"""HV Strip Progressive — Main Window.

4-tab navigation matching the original PySide6 GUI structure:
  Home | HV Forward | Figures | Settings
"""
import os
import yaml
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QLabel, QWidget, QVBoxLayout,
)
from PyQt5.QtGui import QIcon


_SETTINGS_DIR = Path.home() / ".hvstrip"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.yaml"


def _get_default_config():
    return {
        "engine": {"name": "diffuse_field"},
        "hv_forward": {
            "exe_path": "",
            "fmin": 0.5,
            "fmax": 20.0,
            "nf": 500,
        },
        "plot": {
            "dpi": 150,
            "x_axis_scale": "log",
            "y_axis_scale": "linear",
        },
        "peak_detection": {
            "preset": "default",
            "method": "find_peaks",
            "select": "leftmost",
        },
        "dual_resonance": {
            "separation_ratio_threshold": 1.2,
            "separation_shift_threshold": 0.3,
        },
        "engine_settings": {
            "diffuse_field": {},
            "ellipticity": {},
            "sh_wave": {},
        },
    }


class HVStripWindow(QMainWindow):
    """Main window for HV Strip Progressive analysis."""

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HVSR Progressive Layer Stripping Analysis")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        self._config = _get_default_config()
        self._load_settings()

        self._build_ui()
        self._wire_pages()
        self._apply_saved_config()

    # ── UI construction ─────────────────────────────────────────
    def _build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create pages (lazy — imported when first built)
        self._home_page = self._make_placeholder("Home — Stripping & Batch")
        self._forward_page = self._make_placeholder("HV Forward Modeling")
        self._figures_page = self._make_placeholder("Figures & Visualization")
        self._settings_page = self._make_placeholder("Settings")

        self.tabs.addTab(self._home_page, "🏠 Home")
        self.tabs.addTab(self._forward_page, "📊 HV Forward")
        self.tabs.addTab(self._figures_page, "📈 Figures")
        self.tabs.addTab(self._settings_page, "⚙ Settings")

        # Status bar
        self._status_engine = QLabel("Engine: diffuse_field")
        self._status_msg = QLabel("Ready")
        sb = QStatusBar()
        sb.addWidget(self._status_msg, 1)
        sb.addPermanentWidget(self._status_engine)
        self.setStatusBar(sb)

    def _wire_pages(self):
        """Replace placeholder tabs with real pages."""
        try:
            from .pages.forward_modeling_page import ForwardModelingPage
            page = ForwardModelingPage(main_window=self)
            page.apply_config(self._config)
            self.set_forward_page(page)
            self._forward_page = page
        except Exception as e:
            print(f"[HVStrip] Forward page init error: {e}")

        try:
            from .pages.home_page import HomePage
            page = HomePage(main_window=self)
            page.apply_config(self._config)
            self.set_home_page(page)
            self._home_page = page
        except Exception as e:
            print(f"[HVStrip] Home page init error: {e}")

        try:
            from .pages.visualization_page import VisualizationPage
            page = VisualizationPage(main_window=self)
            page.apply_config(self._config)
            self.set_figures_page(page)
            self._figures_page = page
        except Exception as e:
            print(f"[HVStrip] Figures page init error: {e}")

        try:
            from .pages.settings_page import SettingsPage
            page = SettingsPage(main_window=self)
            page.apply_config(self._config)
            self.set_settings_page(page)
            self._settings_page = page
        except Exception as e:
            print(f"[HVStrip] Settings page init error: {e}")

    def _make_placeholder(self, text):
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: gray; font-size: 18px;")
        layout.addWidget(lbl)
        return w

    # ── page replacement (called once real pages are created) ───
    def set_home_page(self, page):
        idx = self.tabs.indexOf(self._home_page)
        self.tabs.removeTab(idx)
        self._home_page = page
        self.tabs.insertTab(idx, page, "🏠 Home")
        self.tabs.setCurrentIndex(idx)

    def set_forward_page(self, page):
        idx = self.tabs.indexOf(self._forward_page)
        self.tabs.removeTab(idx)
        self._forward_page = page
        self.tabs.insertTab(idx, page, "📊 HV Forward")

    def set_figures_page(self, page):
        idx = self.tabs.indexOf(self._figures_page)
        self.tabs.removeTab(idx)
        self._figures_page = page
        self.tabs.insertTab(idx, page, "📈 Figures")

    def set_settings_page(self, page):
        idx = self.tabs.indexOf(self._settings_page)
        self.tabs.removeTab(idx)
        self._settings_page = page
        self.tabs.insertTab(idx, page, "⚙ Settings")

    # ── config accessors ────────────────────────────────────────
    @property
    def config(self):
        return self._config

    def get_engine_name(self):
        engine = self._config.get("engine", {})
        if isinstance(engine, dict):
            return engine.get("name", "diffuse_field")
        return str(engine) if engine else "diffuse_field"

    def get_engine_settings(self):
        return self._config.get("engine_settings", {})

    def update_config(self, cfg):
        """Deep-merge *cfg* into current config and save."""
        self._deep_merge(self._config, cfg)
        self._save_settings()

    # ── settings persistence ────────────────────────────────────
    def _load_settings(self):
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    self._deep_merge(self._config, data)
                    # Fix engine field if it's a dict
                    eng = self._config.get("engine", "diffuse_field")
                    if isinstance(eng, dict):
                        self._config["engine"] = eng
                    elif isinstance(eng, str):
                        self._config["engine"] = {"name": eng}
            except Exception:
                pass

    def _save_settings(self):
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(_SETTINGS_FILE, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception:
            pass

    def _apply_saved_config(self):
        engine_name = self.get_engine_name()
        self._status_engine.setText(f"Engine: {engine_name}")

    @staticmethod
    def _deep_merge(base, update):
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                HVStripWindow._deep_merge(base[k], v)
            else:
                base[k] = v

    # ── events ──────────────────────────────────────────────────
    def closeEvent(self, event):
        self._save_settings()
        self.closed.emit()
        super().closeEvent(event)

    def set_status(self, msg):
        self._status_msg.setText(msg)
