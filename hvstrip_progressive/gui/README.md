# HVSTRIP-Progressive GUI

Modern graphical user interface for HVSTRIP-Progressive built with PySide6 and QFluentWidgets.

## Overview

The HVSTRIP-Progressive GUI provides an intuitive, component-based interface for progressive layer stripping analysis of HVSR data. It features a modern Fluent Design interface with multi-page navigation, comprehensive settings management, and real-time workflow monitoring.

## Features

- **Multi-page Interface**: Navigate between Home, Settings, and Results pages
- **Component-Based Architecture**: Modular, reusable UI components for easy expansion
- **Real-time Progress Tracking**: Monitor workflow execution with live progress updates
- **Configuration Management**: Save and load analysis configurations
- **Results Viewer**: Browse and preview generated plots and reports
- **Background Processing**: Non-blocking workflow execution using Qt threads
- **Comprehensive Settings**: Access all analysis parameters through organized panels

## Installation

### Prerequisites

- Python 3.8 or higher
- PySide6
- QFluentWidgets

### Install Dependencies

```bash
pip install -r hvstrip_progressive/gui/requirements.txt
```

Or install individually:

```bash
pip install PySide6>=6.5.0
pip install PyQt-Fluent-Widgets>=1.4.0
```

## Running the GUI

### From Python

```python
from hvstrip_progressive.gui import main

main()
```

### From Command Line

```bash
python -m hvstrip_progressive.gui.app
```

### Create a Launcher Script

Create a file `run_gui.py` in your project root:

```python
#!/usr/bin/env python3
from hvstrip_progressive.gui import main

if __name__ == "__main__":
    main()
```

Then run:

```bash
python run_gui.py
```

## User Guide

### Home Page - Workflow Execution

1. **Select Input Files**:
   - Click "Browse" next to "Velocity Model File" to select your HVf format model
   - Click "Browse" next to "HVf.exe Path" to locate the HVf executable
   - Click "Browse" next to "Output Directory" to choose where results will be saved

2. **Choose Processing Mode**:
   - **Complete Workflow**: Run all steps (strip, forward, postprocess, report)
   - **Strip Only**: Only perform layer stripping
   - **Forward Only**: Only compute HV curves (for existing models)
   - **Postprocess Only**: Only generate plots (for existing HV curves)

3. **Run Workflow**:
   - Click "Run Workflow" to start processing
   - Monitor progress in the progress bar and log output
   - Click "Stop" to cancel a running workflow

4. **Save/Load Configurations**:
   - Click "Save Config" to save current settings for later use
   - Click "Load Config" to load previously saved configurations

### Settings Page - Advanced Configuration

Configure all analysis parameters organized into four main sections:

#### Frequency Settings
- **Basic Parameters**:
  - Minimum Frequency (Hz): Starting frequency for HV computation
  - Maximum Frequency (Hz): Ending frequency for HV computation
  - Number of Points: Frequency sampling resolution

- **Advanced Parameters** (Expandable):
  - Adaptive Scanning: Auto-expand frequency range if peaks near edges
  - HVf Parameters: nmr, nml, nks for the forward solver
  - Frequency Limits: Hard limits for adaptive scanning

#### Peak Detection Settings
- **Method**: Choose detection algorithm (Global Max, Find Peaks, Manual)
- **Selection**: For Find Peaks - choose fundamental or maximum amplitude
- **Constraints**: Prominence, distance, frequency range, height thresholds

#### Visualization Settings
- **HV Curve Options**:
  - Axis scales (log/linear)
  - Smoothing parameters
  - Frequency bands display
  - Figure dimensions and DPI

- **Vs Profile Options**:
  - Show/hide profile plot
  - Annotation preferences
  - Figure sizing

- **Output Options**:
  - Save individual plots
  - Save combined figures

#### Report Generation Options
- **Data Reports**: CSV summaries, JSON metadata, text reports
- **Visualizations**: HV overlay, peak evolution, interface analysis, waterfall, publication figures
- **Output Format**: PNG, PDF, or both

### Results Page - View Analysis Outputs

1. **Browse Directory**:
   - Click "Browse Output Directory" to select a workflow output folder

2. **View Files**:
   - All generated files are listed in the left panel
   - Click on any file to select it

3. **Preview Images**:
   - PNG/JPG images are automatically previewed in the right panel
   - Zoom and pan as needed

4. **Open Files**:
   - Click "Open File" to open selected file with default application
   - Click "Open Folder" to open the directory in file explorer

## Architecture

### Project Structure

```
hvstrip_progressive/gui/
├── app.py                      # Application entry point
├── main_window.py              # Main window with navigation
├── components/                 # Reusable UI components
│   ├── input_panel.py          # File/path selection
│   ├── frequency_panel.py      # Frequency settings
│   ├── peak_detection_panel.py # Peak detection options
│   ├── visualization_panel.py  # Visualization settings
│   └── report_panel.py         # Report generation options
├── pages/                      # Main application pages
│   ├── home_page.py            # Workflow execution
│   ├── settings_page.py        # Advanced settings
│   └── results_page.py         # Results viewer
├── workers/                    # Background processing
│   └── workflow_worker.py      # Qt worker threads
├── utils/                      # GUI utilities
│   └── config_manager.py       # Save/load configurations
└── resources/                  # Icons, styles, etc.
```

### Component-Based Design

The GUI is built with reusable components that can be easily extended:

- **Panels**: Self-contained UI sections with their own logic
- **Pages**: Top-level views composed of multiple components
- **Workers**: Background threads for long-running operations
- **Utilities**: Helper classes for configuration, validation, etc.

### Adding New Components

1. Create a new file in `components/` directory
2. Inherit from `CardWidget` or appropriate base class
3. Implement `get_settings()` and `set_settings()` methods
4. Add signals for change notifications
5. Import and use in relevant pages

Example:

```python
from qfluentwidgets import CardWidget, Signal

class MyCustomPanel(CardWidget):
    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def get_settings(self):
        return {'my_param': self.my_widget.value()}

    def set_settings(self, settings):
        if 'my_param' in settings:
            self.my_widget.setValue(settings['my_param'])
```

### Adding New Pages

1. Create a new file in `pages/` directory
2. Inherit from `ScrollArea`
3. Compose page using existing components
4. Add to `main_window.py` navigation

## Configuration File Format

Configurations are saved as JSON files with the following structure:

```json
{
  "inputs": {
    "model_file": "/path/to/model.txt",
    "exe_path": "/path/to/HVf.exe",
    "output_dir": "/path/to/output"
  },
  "mode": "Complete Workflow",
  "settings": {
    "frequency": { ... },
    "peak_detection": { ... },
    "visualization": { ... },
    "reports": { ... }
  }
}
```

## Troubleshooting

### GUI doesn't start

- Ensure PySide6 and QFluentWidgets are installed
- Check Python version (3.8+ required)
- Try running with: `python -m hvstrip_progressive.gui.app`

### Workflow fails immediately

- Verify HVf.exe path is correct
- Check model file format is valid
- Ensure output directory is writable

### Settings not saving

- Check write permissions in Documents/HVSTRIP-Configs directory
- Verify JSON configuration file is valid

### Images not previewing

- Ensure image files are PNG/JPG format
- Check file paths don't contain special characters

## Development

### Requirements for Development

```bash
pip install -r requirements.txt
pip install pytest pytest-qt  # For testing
```

### Running Tests

```bash
pytest hvstrip_progressive/gui/tests/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document all public methods
- Keep components modular and reusable

## License

Same as the main HVSTRIP-Progressive package.

## Support

For issues, questions, or contributions, please refer to the main project repository.
