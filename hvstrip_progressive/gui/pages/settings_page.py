"""
Settings Page

Configure application settings and preferences.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, SpinBox, DoubleSpinBox, ComboBox,
    FluentIcon as FIF, InfoBar, InfoBarPosition,
    PrimaryPushButton, TransparentToolButton
, ScrollArea)

# Import from parent package
from ...utils.config import load_config, save_config


class SettingsPage(QWidget):
    """Settings page."""

    settingsSaved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self.config = self.getDefaultConfig()
        self.initUI()

    def initUI(self):
        """Initialize user interface."""
        # Main layout
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        scrollArea = ScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Content widget
        contentWidget = QWidget()
        layout = QVBoxLayout(contentWidget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        titleLabel = TitleLabel("Settings", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Configure application settings and default parameters.\n"
            "Settings can be saved to and loaded from YAML configuration files.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # HVf Configuration Card
        hvfCard = self.createHvfCard()
        layout.addWidget(hvfCard)

        # Frequency Configuration Card
        freqCard = self.createFrequencyCard()
        layout.addWidget(freqCard)

        # Plot Configuration Card
        plotCard = self.createPlotCard()
        layout.addWidget(plotCard)

        # Peak Detection Card
        peakCard = self.createPeakDetectionCard()
        layout.addWidget(peakCard)

        # Configuration File Card
        configCard = self.createConfigFileCard()
        layout.addWidget(configCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        resetButton = PushButton(FIF.SYNC, "Reset to Defaults", self)
        resetButton.clicked.connect(self.resetToDefaults)
        buttonLayout.addWidget(resetButton)

        saveButton = PrimaryPushButton(FIF.SAVE, "Save Settings", self)
        saveButton.clicked.connect(self.saveSettings)
        buttonLayout.addWidget(saveButton)

        layout.addLayout(buttonLayout)

        # Set scroll area content
        scrollArea.setWidget(contentWidget)
        mainLayout.addWidget(scrollArea)
        layout.addStretch()

    def createHvfCard(self):
        """Create HVf configuration card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("HVf Executable Configuration", card)
        cardLayout.addWidget(cardTitle)

        # HVf executable path
        exeLayout = QHBoxLayout()
        exeLabel = BodyLabel("HVf Executable:", card)
        exeLabel.setFixedWidth(150)
        exeLayout.addWidget(exeLabel)

        self.exeEdit = LineEdit(card)
        self.exeEdit.setPlaceholderText("Path to HVf executable (leave blank for auto-detect)")
        self.exeEdit.setText(self.config.get('hv_forward', {}).get('exe_path', ''))
        exeLayout.addWidget(self.exeEdit)

        exeButton = TransparentToolButton(FIF.FOLDER, card)
        exeButton.clicked.connect(self.selectExeFile)
        exeLayout.addWidget(exeButton)

        cardLayout.addLayout(exeLayout)

        return card

    def createFrequencyCard(self):
        """Create frequency configuration card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Default Frequency Range", card)
        cardLayout.addWidget(cardTitle)

        # Frequency grid layout
        gridLayout = QHBoxLayout()

        # Minimum frequency
        fminLayout = QVBoxLayout()
        fminLabel = BodyLabel("Min Frequency (Hz):", card)
        fminLayout.addWidget(fminLabel)
        self.fminSpin = DoubleSpinBox(card)
        self.fminSpin.setRange(0.01, 100.0)
        self.fminSpin.setValue(self.config.get('hv_forward', {}).get('fmin', 0.2))
        self.fminSpin.setDecimals(2)
        fminLayout.addWidget(self.fminSpin)
        gridLayout.addLayout(fminLayout)

        # Maximum frequency
        fmaxLayout = QVBoxLayout()
        fmaxLabel = BodyLabel("Max Frequency (Hz):", card)
        fmaxLayout.addWidget(fmaxLabel)
        self.fmaxSpin = DoubleSpinBox(card)
        self.fmaxSpin.setRange(0.1, 200.0)
        self.fmaxSpin.setValue(self.config.get('hv_forward', {}).get('fmax', 20.0))
        self.fmaxSpin.setDecimals(1)
        fmaxLayout.addWidget(self.fmaxSpin)
        gridLayout.addLayout(fmaxLayout)

        # Number of frequency points
        nfLayout = QVBoxLayout()
        nfLabel = BodyLabel("Frequency Points:", card)
        nfLayout.addWidget(nfLabel)
        self.nfSpin = SpinBox(card)
        self.nfSpin.setRange(10, 500)
        self.nfSpin.setValue(self.config.get('hv_forward', {}).get('nf', 71))
        nfLayout.addWidget(self.nfSpin)
        gridLayout.addLayout(nfLayout)

        cardLayout.addLayout(gridLayout)

        return card

    def createPlotCard(self):
        """Create plot settings card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Plot Settings", card)
        cardLayout.addWidget(cardTitle)

        # Settings grid
        gridLayout = QHBoxLayout()

        # DPI
        dpiLayout = QVBoxLayout()
        dpiLabel = BodyLabel("Default DPI:", card)
        dpiLayout.addWidget(dpiLabel)
        self.dpiSpin = SpinBox(card)
        self.dpiSpin.setRange(72, 600)
        self.dpiSpin.setValue(self.config.get('plot', {}).get('dpi', 150))
        dpiLayout.addWidget(self.dpiSpin)
        gridLayout.addLayout(dpiLayout)

        # X-axis scale
        xscaleLayout = QVBoxLayout()
        xscaleLabel = BodyLabel("X-axis Scale:", card)
        xscaleLayout.addWidget(xscaleLabel)
        self.xscaleCombo = ComboBox(card)
        self.xscaleCombo.addItems(["log", "linear"])
        current_xscale = self.config.get('plot', {}).get('x_axis_scale', 'log')
        self.xscaleCombo.setCurrentText(current_xscale)
        xscaleLayout.addWidget(self.xscaleCombo)
        gridLayout.addLayout(xscaleLayout)

        # Y-axis scale
        yscaleLayout = QVBoxLayout()
        yscaleLabel = BodyLabel("Y-axis Scale:", card)
        yscaleLayout.addWidget(yscaleLabel)
        self.yscaleCombo = ComboBox(card)
        self.yscaleCombo.addItems(["linear", "log"])
        current_yscale = self.config.get('plot', {}).get('y_axis_scale', 'linear')
        self.yscaleCombo.setCurrentText(current_yscale)
        yscaleLayout.addWidget(self.yscaleCombo)
        gridLayout.addLayout(yscaleLayout)

        cardLayout.addLayout(gridLayout)

        return card

    def createPeakDetectionCard(self):
        """Create peak detection settings card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Peak Detection", card)
        cardLayout.addWidget(cardTitle)

        # Preset selection
        presetLayout = QHBoxLayout()
        presetLabel = BodyLabel("Preset:", card)
        presetLabel.setFixedWidth(150)
        presetLayout.addWidget(presetLabel)

        self.peakPresetCombo = ComboBox(card)
        self.peakPresetCombo.addItems(["default", "forward_modeling", "forward_modeling_sharp", "conservative", "custom"])
        current_preset = self.config.get('peak_detection', {}).get('preset', 'default')
        self.peakPresetCombo.setCurrentText(current_preset)
        self.peakPresetCombo.currentTextChanged.connect(self._onPresetChanged)
        presetLayout.addWidget(self.peakPresetCombo)

        cardLayout.addLayout(presetLayout)

        # Peak detection method (for custom mode)
        methodLayout = QHBoxLayout()
        methodLabel = BodyLabel("Detection Method:", card)
        methodLabel.setFixedWidth(150)
        methodLayout.addWidget(methodLabel)

        self.peakMethodCombo = ComboBox(card)
        self.peakMethodCombo.addItems(["max", "find_peaks", "manual"])
        current_method = self.config.get('peak_detection', {}).get('method', 'find_peaks')
        self.peakMethodCombo.setCurrentText(current_method)
        methodLayout.addWidget(self.peakMethodCombo)

        cardLayout.addLayout(methodLayout)

        # Peak selection mode
        selectLayout = QHBoxLayout()
        selectLabel = BodyLabel("Peak Selection:", card)
        selectLabel.setFixedWidth(150)
        selectLayout.addWidget(selectLabel)

        self.peakSelectCombo = ComboBox(card)
        self.peakSelectCombo.addItems(["leftmost", "max", "sharpest", "leftmost_sharpest"])
        current_select = self.config.get('peak_detection', {}).get('select', 'leftmost')
        self.peakSelectCombo.setCurrentText(current_select)
        selectLayout.addWidget(self.peakSelectCombo)

        cardLayout.addLayout(selectLayout)

        # Preset descriptions
        self.presetDescLabel = BodyLabel("", card)
        self.presetDescLabel.setWordWrap(True)
        cardLayout.addWidget(self.presetDescLabel)
        self._updatePresetDescription()

        # Custom settings visibility
        self._onPresetChanged(current_preset)

        return card

    def _onPresetChanged(self, preset: str):
        """Handle preset selection change."""
        is_custom = preset == "custom"
        self.peakMethodCombo.setEnabled(is_custom)
        self.peakSelectCombo.setEnabled(is_custom)
        self._updatePresetDescription()

    def _updatePresetDescription(self):
        """Update preset description label."""
        preset = self.peakPresetCombo.currentText()
        descriptions = {
            "default": "Default: find_peaks with leftmost selection, prominence=0.2, "
                       "freq_min=0.5 Hz, min_rel_height=25%",
            "forward_modeling": "Forward Modeling: Optimized for synthetic HVSR curves. "
                                "Lower thresholds (prominence=0.1), clarity ratio check enabled, "
                                "min_amplitude=1.5",
            "forward_modeling_sharp": "Forward Modeling (Sharp): Like forward_modeling but selects "
                                      "leftmost among sharp peaks (prominence >= 50% of max)",
            "conservative": "Conservative: Stricter criteria (prominence=0.5), selects highest "
                            "amplitude peak, min_amplitude=2.0, min_rel_height=30%",
            "custom": "Custom: Configure all parameters manually using the options below."
        }
        self.presetDescLabel.setText(descriptions.get(preset, ""))

    def createConfigFileCard(self):
        """Create configuration file card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Configuration File", card)
        cardLayout.addWidget(cardTitle)

        # Load/Save buttons
        buttonLayout = QHBoxLayout()

        loadButton = PushButton(FIF.FOLDER, "Load from File", card)
        loadButton.clicked.connect(self.loadConfigFile)
        buttonLayout.addWidget(loadButton)

        saveFileButton = PushButton(FIF.SAVE, "Save to File", card)
        saveFileButton.clicked.connect(self.saveConfigFile)
        buttonLayout.addWidget(saveFileButton)

        buttonLayout.addStretch()

        cardLayout.addLayout(buttonLayout)

        # Current config file
        self.configFileLabel = BodyLabel("No configuration file loaded", card)
        cardLayout.addWidget(self.configFileLabel)

        return card

    def selectExeFile(self):
        """Select HVf executable file."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HVf Executable",
            "",
            "Executable Files (*.exe HVf HVf_Serial);;All Files (*)"
        )
        if file:
            self.exeEdit.setText(file)

    def loadConfigFile(self):
        """Load configuration from file."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "YAML Files (*.yaml *.yml);;JSON Files (*.json);;All Files (*)"
        )
        if file:
            try:
                self.config = load_config(file)
                self.updateUIFromConfig()
                self.configFileLabel.setText(f"Loaded: {Path(file).name}")
                InfoBar.success(
                    "Success",
                    "Configuration loaded successfully",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=2000
                )
            except Exception as e:
                InfoBar.error(
                    "Error",
                    f"Failed to load configuration: {e}",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000
                )

    def saveConfigFile(self):
        """Save configuration to file."""
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "YAML Files (*.yaml);;JSON Files (*.json)"
        )
        if file:
            try:
                self.updateConfigFromUI()

                # Determine format from extension
                file_path = Path(file)
                format = 'yaml' if file_path.suffix.lower() in ['.yaml', '.yml'] else 'json'

                save_config(self.config, file, format=format)
                self.configFileLabel.setText(f"Saved: {file_path.name}")
                InfoBar.success(
                    "Success",
                    "Configuration saved successfully",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=2000
                )
            except Exception as e:
                InfoBar.error(
                    "Error",
                    f"Failed to save configuration: {e}",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000
                )

    def resetToDefaults(self):
        """Reset to default configuration."""
        self.config = self.getDefaultConfig()
        self.updateUIFromConfig()
        self.configFileLabel.setText("Reset to default settings")
        InfoBar.info(
            "Reset",
            "Settings reset to defaults",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=2000
        )

    def saveSettings(self):
        """Save current settings."""
        self.updateConfigFromUI()
        self.settingsSaved.emit()
        InfoBar.success(
            "Success",
            "Settings saved successfully",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=2000
        )

    def getDefaultConfig(self):
        """Get default configuration."""
        return {
            'hv_forward': {
                'exe_path': '',
                'fmin': 0.2,
                'fmax': 20.0,
                'nf': 71
            },
            'plot': {
                'dpi': 150,
                'x_axis_scale': 'log',
                'y_axis_scale': 'linear'
            },
            'peak_detection': {
                'preset': 'default',
                'method': 'find_peaks',
                'select': 'leftmost'
            }
        }

    def updateUIFromConfig(self):
        """Update UI widgets from configuration."""
        # HVf settings
        self.exeEdit.setText(self.config.get('hv_forward', {}).get('exe_path', ''))

        # Frequency settings
        self.fminSpin.setValue(self.config.get('hv_forward', {}).get('fmin', 0.2))
        self.fmaxSpin.setValue(self.config.get('hv_forward', {}).get('fmax', 20.0))
        self.nfSpin.setValue(self.config.get('hv_forward', {}).get('nf', 71))

        # Plot settings
        self.dpiSpin.setValue(self.config.get('plot', {}).get('dpi', 150))
        self.xscaleCombo.setCurrentText(self.config.get('plot', {}).get('x_axis_scale', 'log'))
        self.yscaleCombo.setCurrentText(self.config.get('plot', {}).get('y_axis_scale', 'linear'))

        # Peak detection
        peak_cfg = self.config.get('peak_detection', {})
        self.peakPresetCombo.setCurrentText(peak_cfg.get('preset', 'default'))
        self.peakMethodCombo.setCurrentText(peak_cfg.get('method', 'find_peaks'))
        self.peakSelectCombo.setCurrentText(peak_cfg.get('select', 'leftmost'))
        self._onPresetChanged(self.peakPresetCombo.currentText())

    def updateConfigFromUI(self):
        """Update configuration from UI widgets."""
        self.config['hv_forward'] = {
            'exe_path': self.exeEdit.text(),
            'fmin': self.fminSpin.value(),
            'fmax': self.fmaxSpin.value(),
            'nf': self.nfSpin.value()
        }

        self.config['plot'] = {
            'dpi': self.dpiSpin.value(),
            'x_axis_scale': self.xscaleCombo.currentText(),
            'y_axis_scale': self.yscaleCombo.currentText()
        }

        preset = self.peakPresetCombo.currentText()
        self.config['peak_detection'] = {
            'preset': preset,
            'method': self.peakMethodCombo.currentText(),
            'select': self.peakSelectCombo.currentText()
        }

    def getConfig(self):
        """Get current configuration."""
        self.updateConfigFromUI()
        return self.config
