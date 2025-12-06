# HVSR Progressive Layer Stripping - GUI Application

A comprehensive graphical user interface for progressive layer stripping analysis using diffuse-field theory for HVSR data.

## Features

### Complete Workflow
- **Automated Pipeline**: Run the complete analysis workflow with a single click
- **Layer Stripping**: Progressive removal of deepest finite layers
- **HV Forward Modeling**: Compute theoretical HVSR curves using HVf
- **Post-Processing**: Generate publication-ready plots and summaries
- **Real-time Progress**: Monitor workflow execution with detailed logging

### Individual Components

#### 1. Layer Stripping
- Strip layers from velocity model
- Create sequence of peeled models
- Preview generated step folders

#### 2. HV Forward Modeling
- Compute HVSR curves for individual models
- Configure frequency range (fmin, fmax, nf)
- Auto-detect or manually specify HVf executable
- Display peak frequency and amplitude
- Export results to CSV

#### 3. Post-Processing
- Generate HV curve plots with customizable scales
- Create Vs profile visualizations
- Configure smoothing and plot appearance
- Export publication-quality figures (PNG, PDF)

#### 4. Report Generation
- Create comprehensive analysis reports
- Multiple visualization types:
  - HV curves overlay
  - Peak evolution analysis
  - Interface analysis
  - Waterfall plots
  - Publication-ready figures
- Export data summaries (CSV, JSON)
- Generate text reports

#### 5. Batch Processing
- Process multiple soil profiles automatically
- Pattern-based file selection (wildcards supported)
- Progress tracking for each profile
- Batch summary with success/failure statistics
- Optional comprehensive visualization reports

#### 6. Advanced Analysis
- Statistical analysis across stripping steps
- Controlling interface detection
- Layer contribution analysis
- Export analysis data to CSV
- Correlation analysis

### Settings & Configuration
- Configure HVf executable path
- Set default frequency ranges
- Customize plot appearance (DPI, scales, colors)
- Configure peak detection methods
- Save/load configurations (YAML, JSON)
- Reset to default settings

## Installation

### Requirements

The GUI is part of the `hvstrip_progressive` package. All dependencies are included in the package requirements.

### Setup

1. Install the hvstrip_progressive package with GUI dependencies:
```bash
cd "new package"
pip install -e .
pip install -r hvstrip_progressive/gui/requirements.txt
```

2. Run the GUI application:
```bash
# Option 1: Using the launcher script (Linux/Mac)
cd hvstrip_progressive/gui
./run_gui.sh

# Option 2: Using the launcher script (Windows)
cd hvstrip_progressive\gui
run_gui.bat

# Option 3: Run as Python module
python -m hvstrip_progressive.gui.app
```

## Usage

### Quick Start

1. **Launch the application**:
   ```bash
   # From the gui directory
   cd hvstrip_progressive/gui
   ./run_gui.sh  # Linux/Mac
   # or
   run_gui.bat   # Windows

   # Or run as module from anywhere
   python -m hvstrip_progressive.gui.app
   ```

2. **Complete Workflow** (Recommended for first-time users):
   - Navigate to "Complete Workflow" page
   - Select your velocity model file
   - Choose output directory
   - Configure frequency range (optional)
   - Click "Run Complete Workflow"
   - Monitor progress in real-time

3. **View Results**:
   - Check the output directory for:
     - `strip/` - Stripped models and individual step results
     - Each step folder contains:
       - Model file
       - HV curve CSV
       - HV curve plot
       - Vs profile plot
       - Summary CSV

### Individual Component Usage

#### Layer Stripping Only
1. Go to "Layer Stripping" page
2. Select model file and output directory
3. Click "Strip Layers"
4. View generated step folders in output/strip/

#### HV Forward Modeling Only
1. Go to "HV Forward" page
2. Select model file
3. Configure frequency range
4. Click "Compute HV Curve"
5. Results displayed with peak information

#### Post-Processing Only
1. Go to "Post-processing" page
2. Select HV CSV and model files
3. Configure plot settings
4. Click "Generate Plots"
5. View generated plots in output directory

#### Report Generation
1. Go to "Report Generation" page
2. Select strip directory (from previous workflow)
3. Optionally specify output directory
4. Click "Generate Report"
5. Access comprehensive analysis reports

#### Batch Processing
1. Go to "Batch Processing" page
2. Select directory containing multiple profiles
3. Configure file pattern (e.g., "*.txt", "profile_*.txt")
4. Set output directory
5. Optionally enable report generation
6. Click "Run Batch Processing"
7. Monitor progress for each profile

#### Advanced Analysis
1. Go to "Advanced Analysis" page
2. Select strip directory
3. Choose analysis options:
   - Statistical summary
   - Controlling interface detection
   - Layer contribution analysis
4. Click "Run Analysis"
5. View results and export data

### Configuration

#### Settings Page
- **HVf Executable**: Path to HVf executable (auto-detected on Linux/Windows)
- **Frequency Defaults**: Default fmin, fmax, nf values
- **Plot Settings**: DPI, axis scales, colors
- **Peak Detection**: Choose method (max, find_peaks, manual)

#### Save/Load Configuration
- Save current settings to YAML or JSON
- Load previously saved configurations
- Reset to default settings

## File Formats

### Input Files

#### Velocity Model (*.txt)
```
4
5.0  500  250  1800
10.0 800  400  2000
15.0 1200 600  2200
0.0  2000 1000 2400
```
Format: N (number of layers), then N rows of: thickness(m) Vp(m/s) Vs(m/s) density(kg/m³)
Last row must have thickness = 0 (half-space)

### Output Files

#### HV Curve CSV
```
Frequency_Hz,HVSR_Amplitude
0.200000,1.234567
0.300000,1.456789
...
```

#### Summary CSV
Contains peak frequencies, amplitudes, model parameters, etc.

## Troubleshooting

### HVf Executable Not Found
- **Linux**: Ensure `bin/exe_Linux/HVf` or `bin/exe_Linux/HVf_Serial` exists and is executable
- **Windows**: Ensure `bin/exe_Win/HVf.exe` exists
- Or manually specify path in Settings

### Import Errors
- Ensure hvstrip_progressive package is installed: `pip install -e .` in package directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Qt Platform Plugin Error
- Linux: Install Qt dependencies: `sudo apt-get install libxcb-xinerama0`
- Set environment variable: `export QT_QPA_PLATFORM=xcb`

### Memory Issues (Large Batch Processing)
- Process profiles in smaller batches
- Reduce frequency point count (nf parameter)
- Close other applications

## Architecture

```
hvstrip_progressive/
├── gui/                    # GUI module (this package)
│   ├── __init__.py
│   ├── app.py              # Application entry point
│   ├── main_window.py      # Main window with navigation
│   ├── pages/              # Individual page modules
│   │   ├── __init__.py
│   │   ├── workflow_page.py    # Complete workflow
│   │   ├── strip_page.py       # Layer stripping
│   │   ├── forward_page.py     # HV forward modeling
│   │   ├── postprocess_page.py # Post-processing
│   │   ├── report_page.py      # Report generation
│   │   ├── batch_page.py       # Batch processing
│   │   ├── analysis_page.py    # Advanced analysis
│   │   └── settings_page.py    # Settings & configuration
│   ├── README.md
│   ├── requirements.txt
│   ├── run_gui.sh          # Linux/Mac launcher
│   └── run_gui.bat         # Windows launcher
├── core/                   # Core analysis modules
├── utils/                  # Utility functions
└── ...
```

## Key Technologies

- **PySide6**: Qt6 Python bindings for cross-platform GUI
- **QFluentWidgets**: Modern Fluent Design components
- **QThread**: Non-blocking background operations
- **hvstrip_progressive**: Core analysis package

## Development

### Adding New Pages

1. Create new page file in `pages/` directory
2. Inherit from `QWidget`
3. Implement UI in `initUI()` method
4. Use QThread workers for long operations
5. Add to `main_window.py` navigation
6. Update `pages/__init__.py`

### Customizing Appearance

- Edit Settings page to add new preferences
- Modify QFluentWidgets theme in `app.py`
- Customize colors in individual page files

## License

MIT License - See LICENSE file for details

## Authors

HVSR-Diffuse Development Team

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hvstrip_progressive_gui,
  title = {HVSR Progressive Layer Stripping - GUI Application},
  author = {HVSR-Diffuse Development Team},
  year = {2024},
  version = {1.0.0}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: [repository URL]
- Email: [contact email]
- Documentation: [docs URL]

## Changelog

### Version 1.0.0 (2024-12-06)
- Initial release
- Complete workflow implementation
- All individual component pages
- Batch processing support
- Advanced analysis tools
- Settings and configuration management
- Publication-ready visualization
- Comprehensive documentation
