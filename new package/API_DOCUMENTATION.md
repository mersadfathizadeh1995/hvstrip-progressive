# HVSR Progressive Layer Stripping - API Documentation

Comprehensive API reference for the hvstrip-progressive package.

## Table of Contents

1. [Core Modules](#core-modules)
2. [CLI Commands](#cli-commands)
3. [Configuration](#configuration)
4. [Data Structures](#data-structures)
5. [Utilities](#utilities)

---

## Core Modules

### `hvstrip_progressive.core.stripper`

Layer stripping functionality for creating peeled soil profile models.

#### `write_peel_sequence(model_path, output_base)`

Creates a sequence of stripped models by progressively removing deepest layers.

**Parameters:**
- `model_path` (str): Path to initial HVf-format model file
- `output_base` (str): Base directory for output

**Returns:**
- `Path`: Path to created strip directory

**Example:**
```python
from hvstrip_progressive.core.stripper import write_peel_sequence

strip_dir = write_peel_sequence(
    model_path="model.txt",
    output_base="output/"
)
print(f"Strip directory: {strip_dir}")
```

**Output Structure:**
```
output/
└── strip/
    ├── Step0_N-layer/
    │   └── model_Step0_N-layer.txt
    ├── Step1_(N-1)-layer/
    │   └── model_Step1_(N-1)-layer.txt
    └── ...
```

#### `generate_peel_sequence(rows)`

Generate sequence of peeled model rows.

**Parameters:**
- `rows` (List[List[float]]): Initial model rows [thk, vp, vs, rho]

**Returns:**
- `List[List[List[float]]]`: List of step models

---

### `hvstrip_progressive.core.hv_forward`

HV forward computation using HVf executable.

#### `compute_hv_curve(model_path, config=None)`

Compute theoretical HVSR curve for a velocity model.

**Parameters:**
- `model_path` (str): Path to velocity model file
- `config` (Dict, optional): Configuration dictionary

**Returns:**
- `Tuple[List[float], List[float]]`: (frequencies, amplitudes)

**Configuration:**
```python
config = {
    "exe_path": "/path/to/HVf",  # Auto-detected if None
    "fmin": 0.2,                  # Minimum frequency (Hz)
    "fmax": 20.0,                 # Maximum frequency (Hz)
    "nf": 71,                     # Number of frequency points
    "nmr": 10,                    # Max Rayleigh modes
    "nml": 10,                    # Max Love modes
    "nks": 10                     # K values for integrals
}
```

**Example:**
```python
from hvstrip_progressive.core.hv_forward import compute_hv_curve

freqs, amps = compute_hv_curve(
    model_path="model.txt",
    config={"fmin": 0.1, "fmax": 50, "nf": 100}
)

# Find peak
peak_idx = amps.index(max(amps))
print(f"Peak: {max(amps):.2f} at {freqs[peak_idx]:.2f} Hz")
```

**Raises:**
- `FileNotFoundError`: If HVf executable not found
- `RuntimeError`: If HVf execution fails

---

### `hvstrip_progressive.core.batch_workflow`

Complete workflow orchestration.

#### `run_complete_workflow(initial_model_path, output_base_dir, workflow_config=None)`

Run complete progressive layer stripping workflow.

**Parameters:**
- `initial_model_path` (str): Path to initial velocity model
- `output_base_dir` (str): Base directory for outputs
- `workflow_config` (Dict, optional): Workflow configuration

**Returns:**
- `Dict`: Results dictionary with keys:
  - `success` (bool): Whether workflow succeeded
  - `strip_directory` (Path): Strip output directory
  - `step_folders` (List[Path]): List of step folders
  - `step_results` (Dict): Results for each step
  - `summary` (Dict): Summary statistics
  - `total_time` (float): Total execution time

**Example:**
```python
from hvstrip_progressive.core.batch_workflow import run_complete_workflow

results = run_complete_workflow(
    initial_model_path="model.txt",
    output_base_dir="output/",
    workflow_config={
        "hv_forward": {
            "fmin": 0.2,
            "fmax": 20.0,
            "nf": 71
        }
    }
)

if results['success']:
    print(f"✓ Processed {results['summary']['total_steps']} steps")
    print(f"✓ Success rate: {results['summary']['completion_rate']:.1f}%")
    print(f"✓ Total time: {results['total_time']:.2f}s")
```

#### `find_step_folders(strip_directory)`

Find all StepX_Y-layer folders in strip directory.

**Parameters:**
- `strip_directory` (Path): Strip directory path

**Returns:**
- `List[Path]`: Sorted list of step folders

---

### `hvstrip_progressive.core.hv_postprocess`

Post-processing and visualization.

#### `process(hv_csv_path, model_path, output_dir, config=None)`

Generate plots and summaries from HVSR data.

**Parameters:**
- `hv_csv_path` (str): Path to H/V curve CSV file
- `model_path` (str): Path to velocity model file
- `output_dir` (str): Output directory
- `config` (Dict, optional): Post-processing configuration

**Returns:**
- `Dict`: Dictionary with paths to generated files

**Configuration:**
```python
config = {
    "peak_detection": {
        "method": "find_peaks",    # "max", "find_peaks", or "manual"
        "freq_min": 0.5,            # Minimum peak frequency
        "min_rel_height": 0.25      # Minimum relative height
    },
    "hv_plot": {
        "x_axis_scale": "log",      # "log" or "linear"
        "y_axis_scale": "log",      # "log" or "linear"
        "dpi": 300,                 # Plot resolution
        "smoothing": {
            "enable": True,
            "window_length": 7,
            "poly_order": 3
        }
    },
    "vs_plot": {
        "show": True,
        "annotate_deepest": True,
        "dpi": 300
    }
}
```

**Example:**
```python
from hvstrip_progressive.core.hv_postprocess import process

results = process(
    hv_csv_path="hv_curve.csv",
    model_path="model.txt",
    output_dir="plots/",
    config={"hv_plot": {"dpi": 300}}
)

print(f"Generated plots:")
for key, path in results.items():
    if 'png' in key:
        print(f"  - {path}")
```

---

### `hvstrip_progressive.core.advanced_analysis`

Advanced statistical and interface analysis.

#### `class StrippingAnalyzer`

Analyzer for progressive layer stripping results.

**Constructor:**
```python
analyzer = StrippingAnalyzer(strip_directory)
```

**Parameters:**
- `strip_directory` (Path): Path to strip directory

**Methods:**

##### `compute_statistics()`

Compute comprehensive statistics across stripping steps.

**Returns:**
- `Dict`: Statistics dictionary with keys:
  - `n_steps`: Number of steps
  - `peak_frequencies`: List of peak frequencies
  - `peak_amplitudes`: List of peak amplitudes
  - `peak_freq_mean`: Mean peak frequency
  - `peak_freq_std`: Std dev of peak frequency
  - `peak_freq_changes`: Step-wise frequency changes
  - `max_freq_change_step`: Step with maximum change

**Example:**
```python
stats = analyzer.compute_statistics()

print(f"Mean peak frequency: {stats['peak_freq_mean']:.2f} Hz")
print(f"Std deviation: {stats['peak_freq_std']:.2f} Hz")
print(f"Total frequency change: {stats['peak_freq_change_total']:.2f} Hz")
```

##### `detect_controlling_interfaces(threshold_percentile=75)`

Detect controlling interfaces based on H/V changes.

**Parameters:**
- `threshold_percentile` (float): Percentile for significance threshold

**Returns:**
- `List[Dict]`: List of interface dictionaries with keys:
  - `step`: Step number
  - `from_layers`: Initial layer count
  - `to_layers`: Final layer count
  - `freq_change`: Frequency change (Hz)
  - `freq_change_pct`: Frequency change (%)
  - `amp_change`: Amplitude change
  - `amp_change_pct`: Amplitude change (%)
  - `correlation`: Curve correlation
  - `significance_score`: Overall significance
  - `is_controlling`: Whether interface is controlling

**Example:**
```python
controlling = analyzer.detect_controlling_interfaces(threshold_percentile=80)

for ci in controlling:
    if ci['is_controlling']:
        print(f"Controlling interface at step {ci['step']}")
        print(f"  Frequency change: {ci['freq_change']:.2f} Hz ({ci['freq_change_pct']:.1f}%)")
        print(f"  Amplitude change: {ci['amp_change']:.2f} ({ci['amp_change_pct']:.1f}%)")
        print(f"  Significance score: {ci['significance_score']:.1f}")
```

##### `analyze_layer_contributions()`

Analyze contribution of each layer to H/V curve.

**Returns:**
- `pandas.DataFrame`: Contributions dataframe with columns:
  - `step`: Step number
  - `removed_layer`: Layer being removed
  - `from_n_layers`: Initial layer count
  - `to_n_layers`: Final layer count
  - `freq_shift_hz`: Frequency shift
  - `amp_change`: Amplitude change
  - `spectral_energy_change`: Spectral energy change
  - `original_peak_freq`: Original peak frequency
  - `new_peak_freq`: New peak frequency

**Example:**
```python
contributions = analyzer.analyze_layer_contributions()

# Find most influential layers
most_influential = contributions.nlargest(3, 'spectral_energy_change')
print(most_influential[['removed_layer', 'freq_shift_hz', 'spectral_energy_change']])
```

##### `generate_analysis_report(output_path)`

Generate comprehensive text report.

**Parameters:**
- `output_path` (Path): Output file path

**Example:**
```python
analyzer.generate_analysis_report("analysis_report.txt")
```

##### `export_data_csv(output_path)`

Export stripping data to CSV.

**Parameters:**
- `output_path` (Path): Output CSV file path

**Example:**
```python
analyzer.export_data_csv("stripping_data.csv")
```

#### `analyze_strip_directory(strip_dir, output_dir=None)`

Convenience function for complete analysis.

**Parameters:**
- `strip_dir` (Path): Strip directory path
- `output_dir` (Path, optional): Output directory for reports

**Returns:**
- `Dict`: Analysis results with keys:
  - `statistics`: Statistics dictionary
  - `controlling_interfaces`: List of interfaces
  - `layer_contributions`: Contributions DataFrame
  - `analyzer`: StrippingAnalyzer instance

**Example:**
```python
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory

results = analyze_strip_directory(
    strip_dir="output/strip",
    output_dir="output/analysis"
)

# Access results
stats = results['statistics']
controlling = results['controlling_interfaces']
contributions = results['layer_contributions']
```

---

## CLI Commands

### `hvstrip-progressive workflow`

Run complete workflow for single profile.

**Usage:**
```bash
hvstrip-progressive workflow MODEL_FILE OUTPUT_DIR [OPTIONS]
```

**Arguments:**
- `MODEL_FILE`: Path to velocity model file
- `OUTPUT_DIR`: Output directory

**Options:**
- `--exe-path TEXT`: Path to HVf executable (auto-detected if not specified)
- `--fmin FLOAT`: Minimum frequency in Hz (default: 0.2)
- `--fmax FLOAT`: Maximum frequency in Hz (default: 20.0)
- `--nf INTEGER`: Number of frequency points (default: 71)

**Example:**
```bash
hvstrip-progressive workflow model.txt output/ --fmin 0.1 --fmax 50 --nf 100
```

---

### `hvstrip-progressive batch`

Process multiple profiles in batch mode.

**Usage:**
```bash
hvstrip-progressive batch PROFILES_DIR OUTPUT_DIR [OPTIONS]
```

**Arguments:**
- `PROFILES_DIR`: Directory containing profile files
- `OUTPUT_DIR`: Base output directory

**Options:**
- `--pattern TEXT`: File pattern for profiles (default: *.txt)
- `--exe-path TEXT`: Path to HVf executable
- `--generate-report`: Generate visualization report

**Example:**
```bash
hvstrip-progressive batch profiles/ batch_output/ --pattern "profile_*.txt" --generate-report
```

---

### `hvstrip-progressive strip`

Strip layers from velocity model.

**Usage:**
```bash
hvstrip-progressive strip MODEL_FILE OUTPUT_DIR
```

---

### `hvstrip-progressive forward`

Compute HVSR curve for model.

**Usage:**
```bash
hvstrip-progressive forward MODEL_FILE OUTPUT_FILE [OPTIONS]
```

**Options:**
- `--fmin`, `--fmax`, `--nf`: Frequency parameters
- `--exe-path`: HVf executable path

---

### `hvstrip-progressive postprocess`

Generate plots from HVSR data.

**Usage:**
```bash
hvstrip-progressive postprocess HV_CSV MODEL_FILE OUTPUT_DIR [OPTIONS]
```

**Options:**
- `--x-scale`: X-axis scale (log/linear)
- `--y-scale`: Y-axis scale (log/linear)
- `--smoothing/--no-smoothing`: Enable/disable smoothing
- `--dpi INTEGER`: Plot resolution

---

### `hvstrip-progressive examples`

Show usage examples.

**Usage:**
```bash
hvstrip-progressive examples
```

---

## Configuration

### Default Workflow Configuration

```python
DEFAULT_WORKFLOW_CONFIG = {
    "stripper": {
        "output_folder_name": "strip"
    },
    "hv_forward": {
        "exe_path": "<auto-detected>",
        "fmin": 0.2,
        "fmax": 20.0,
        "nf": 71,
        "nmr": 10,
        "nml": 10,
        "nks": 10,
        "adaptive": {
            "enable": True,
            "max_passes": 2,
            "edge_margin_frac": 0.05,
            "fmax_expand_factor": 2.0,
            "fmin_shrink_factor": 0.5,
            "fmax_limit": 60.0,
            "fmin_limit": 0.05
        }
    },
    "hv_postprocess": {
        "peak_detection": {
            "method": "find_peaks",
            "select": "leftmost",
            "freq_min": 0.5,
            "min_rel_height": 0.25,
            "exclude_first_n": 1
        },
        "hv_plot": {
            "x_axis_scale": "log",
            "y_axis_scale": "log",
            "dpi": 200,
            "smoothing": {
                "enable": True,
                "window_length": 9,
                "poly_order": 3
            }
        }
    }
}
```

---

## Data Structures

### Model File Format

HVf-format text file:

```
N
thickness_1 vp_1 vs_1 density_1
thickness_2 vp_2 vs_2 density_2
...
0 vp_N vs_N density_N
```

**Example:**
```
3
10 727 366 1610
24 1474 742 1921
0 1545 778 1943
```

### H/V Curve CSV Format

```csv
Frequency_Hz,HVSR_Amplitude
0.200000,1.523400
0.481013,1.678200
...
```

---

## Utilities

### Validation

```python
from hvstrip_progressive.utils.validation import validate_model_file

is_valid, message = validate_model_file("model.txt")
if not is_valid:
    print(f"Invalid model: {message}")
```

### Configuration Management

```python
from hvstrip_progressive.utils.config import load_config, save_config

# Load configuration from file
config = load_config("config.yaml")

# Modify and save
config['hv_forward']['fmax'] = 50.0
save_config(config, "config.yaml")
```

---

## Error Handling

### Common Exceptions

```python
try:
    results = run_complete_workflow("model.txt", "output/")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Workflow error: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
```

### Debugging

Enable verbose output:

```bash
hvstrip-progressive --verbose workflow model.txt output/
```

---

## Version Information

```python
import hvstrip_progressive
print(hvstrip_progressive.__version__)  # "1.0.0"
```

---

For usage examples, see [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)

