# HVSR Progressive Layer Stripping - Usage Examples

Comprehensive guide to using the hvstrip-progressive package for HVSR analysis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Single Profile Analysis](#single-profile-analysis)
3. [Batch Processing](#batch-processing)
4. [Advanced Analysis](#advanced-analysis)
5. [Programmatic Usage](#programmatic-usage)
6. [Customization Options](#customization-options)

---

## Quick Start

### Installation

```bash
cd "new package"
pip install -e .
```

### Basic Usage

Process a single soil profile:

```bash
hvstrip-progressive workflow profile.txt output/
```

Process multiple profiles:

```bash
hvstrip-progressive batch profiles_dir/ batch_output/
```

---

## Single Profile Analysis

### Complete Workflow

Run the full analysis (stripping + HV computation + visualization):

```bash
hvstrip-progressive workflow model.txt output/
```

**What it does:**
1. Strips layers progressively
2. Computes H/V curves for each step
3. Generates plots and summaries
4. Creates visualization reports

**Output structure:**
```
output/
├── strip/
│   ├── Step0_N-layer/
│   │   ├── model_Step0_N-layer.txt
│   │   ├── hv_curve.csv
│   │   ├── hv_curve.png
│   │   ├── vs_profile.png
│   │   └── step_summary.csv
│   ├── Step1_(N-1)-layer/
│   │   └── ...
│   └── ...
```

### Custom Frequency Range

Adjust the frequency range for analysis:

```bash
hvstrip-progressive workflow model.txt output/ --fmin 0.1 --fmax 50 --nf 100
```

Parameters:
- `--fmin`: Minimum frequency (Hz)
- `--fmax`: Maximum frequency (Hz)
- `--nf`: Number of frequency points

### Individual Components

#### 1. Layer Stripping Only

```bash
hvstrip-progressive strip model.txt output/
```

Creates stripped models without computing H/V curves.

#### 2. HV Forward Modeling Only

```bash
hvstrip-progressive forward model.txt hv_curve.csv
```

Computes H/V curve for a single model and saves to CSV.

#### 3. Post-Processing Only

```bash
hvstrip-progressive postprocess hv_curve.csv model.txt plots/
```

Generates plots from existing H/V data.

---

## Batch Processing

### Basic Batch Mode

Process all profiles in a directory:

```bash
hvstrip-progressive batch profiles/ batch_output/
```

### Custom File Pattern

Process only specific files:

```bash
hvstrip-progressive batch profiles/ output/ --pattern "site_*.txt"
```

### With Visualization Report

Generate comprehensive cross-profile visualizations:

```bash
hvstrip-progressive batch profiles/ output/ --generate-report
```

### Batch Processing Output

```
batch_output/
├── profile_01/
│   └── strip/
│       ├── Step0_3-layer/
│       ├── Step1_2-layer/
│       └── Step2_1-layer/
├── profile_02/
│   └── strip/
│       └── ...
├── ...
```

### Performance Metrics

Example batch performance (12 profiles):
- Layer stripping: ~0.01s per profile
- HV computation: ~0.05s per profile per step
- Post-processing: ~1.2s per profile
- **Total: ~1.3s per profile**

---

## Advanced Analysis

### Using the Advanced Analysis Module

```python
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory

# Analyze stripping results
results = analyze_strip_directory(
    strip_dir="output/strip",
    output_dir="output/analysis"
)

# Access results
stats = results['statistics']
controlling = results['controlling_interfaces']
contributions = results['layer_contributions']
```

### Detecting Controlling Interfaces

```python
from hvstrip_progressive.core.advanced_analysis import StrippingAnalyzer

analyzer = StrippingAnalyzer("output/strip")

# Detect interfaces with significant H/V changes
controlling_interfaces = analyzer.detect_controlling_interfaces(
    threshold_percentile=75
)

# Print results
for ci in controlling_interfaces:
    if ci['is_controlling']:
        print(f"Step {ci['step']}: Significance {ci['significance_score']:.1f}")
        print(f"  Freq change: {ci['freq_change']:.2f} Hz")
        print(f"  Amp change: {ci['amp_change']:.2f}")
```

### Layer Contribution Analysis

```python
# Analyze contribution of each layer
contributions = analyzer.analyze_layer_contributions()

# Find most influential layers
most_influential = contributions.nlargest(3, 'spectral_energy_change')
print(most_influential[['removed_layer', 'freq_shift_hz', 'spectral_energy_change']])
```

### Statistical Analysis

```python
# Compute comprehensive statistics
stats = analyzer.compute_statistics()

print(f"Peak frequency range: {stats['peak_freq_range']}")
print(f"Peak amplitude range: {stats['peak_amp_range']}")
print(f"Maximum frequency change at step: {stats['max_freq_change_step']}")
```

---

## Programmatic Usage

### Python API

#### Complete Workflow

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
    print(f"Processed {results['summary']['total_steps']} steps")
    print(f"Success rate: {results['summary']['completion_rate']:.1f}%")
```

#### Layer Stripping

```python
from hvstrip_progressive.core.stripper import write_peel_sequence

strip_dir = write_peel_sequence(
    model_path="model.txt",
    output_base="output/"
)

print(f"Created stripped models in: {strip_dir}")
```

#### HV Forward Modeling

```python
from hvstrip_progressive.core.hv_forward import compute_hv_curve

freqs, amps = compute_hv_curve(
    model_path="model.txt",
    config={
        "fmin": 0.2,
        "fmax": 20.0,
        "nf": 71
    }
)

# Find peak
peak_idx = amps.index(max(amps))
print(f"Peak: {max(amps):.2f} at {freqs[peak_idx]:.2f} Hz")
```

#### Batch Processing Script

```python
from pathlib import Path
from hvstrip_progressive.core.batch_workflow import run_complete_workflow

profiles_dir = Path("profiles")
output_dir = Path("batch_output")

for profile_file in profiles_dir.glob("*.txt"):
    profile_name = profile_file.stem
    profile_output = output_dir / profile_name

    print(f"Processing {profile_name}...")

    try:
        result = run_complete_workflow(
            str(profile_file),
            str(profile_output)
        )

        print(f"  ✓ Success: {result['summary']['total_steps']} steps")

    except Exception as e:
        print(f"  ✗ Error: {e}")
```

---

## Customization Options

### HV Forward Configuration

```python
config = {
    "hv_forward": {
        "exe_path": "/path/to/HVf",  # Auto-detected if not specified
        "fmin": 0.1,                  # Minimum frequency (Hz)
        "fmax": 50.0,                 # Maximum frequency (Hz)
        "nf": 100,                    # Number of frequency points
        "nmr": 10,                    # Max Rayleigh modes
        "nml": 10,                    # Max Love modes
        "nks": 10,                    # K values for integrals

        # Adaptive frequency scanning
        "adaptive": {
            "enable": True,
            "max_passes": 2,
            "edge_margin_frac": 0.05,
            "fmax_expand_factor": 2.0,
            "fmin_shrink_factor": 0.5
        }
    }
}
```

### Post-Processing Configuration

```python
config = {
    "hv_postprocess": {
        "peak_detection": {
            "method": "find_peaks",  # "max", "find_peaks", or "manual"
            "freq_min": 0.5,          # Minimum peak frequency
            "min_rel_height": 0.25    # Minimum relative peak height
        },

        "hv_plot": {
            "x_axis_scale": "log",    # "log" or "linear"
            "y_axis_scale": "log",    # "log" or "linear"
            "smoothing": {
                "enable": True,
                "window_length": 7,
                "poly_order": 3
            },
            "dpi": 300                # Plot resolution
        },

        "vs_plot": {
            "show": True,
            "annotate_deepest": True,
            "annotate_f0": True,
            "dpi": 300
        }
    }
}
```

### Controlling Interface Detection

```python
# Adjust sensitivity
controlling = analyzer.detect_controlling_interfaces(
    threshold_percentile=80  # Higher = more selective
)

# Filter by frequency change
significant_freq_changes = [
    ci for ci in controlling
    if ci['freq_change'] > 1.0  # More than 1 Hz change
]

# Filter by amplitude change
significant_amp_changes = [
    ci for ci in controlling
    if ci['amp_change_pct'] > 20  # More than 20% change
]
```

---

## Complete End-to-End Example

```python
#!/usr/bin/env python3
"""
Complete HVSR analysis workflow example.
"""
from pathlib import Path
from hvstrip_progressive.core.batch_workflow import run_complete_workflow
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory

# 1. Process profile with full workflow
print("Step 1: Processing soil profile...")
result = run_complete_workflow(
    initial_model_path="profile.txt",
    output_base_dir="analysis_output"
)

if not result['success']:
    print("Workflow failed!")
    exit(1)

# 2. Run advanced analysis
print("\nStep 2: Running advanced analysis...")
strip_dir = result['strip_directory']
analysis_output = strip_dir.parent / "advanced_analysis"

analysis_results = analyze_strip_directory(
    strip_dir=strip_dir,
    output_dir=analysis_output
)

# 3. Display key results
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

stats = analysis_results['statistics']
print(f"\nStatistics:")
print(f"  Steps: {stats['n_steps']}")
print(f"  Peak freq: {stats['peak_freq_mean']:.2f} ± {stats['peak_freq_std']:.2f} Hz")
print(f"  Peak amp: {stats['peak_amp_mean']:.2f} ± {stats['peak_amp_std']:.2f}")

controlling = analysis_results['controlling_interfaces']
n_controlling = sum(1 for c in controlling if c.get('is_controlling', False))
print(f"\nControlling Interfaces: {n_controlling} detected")

for ci in controlling:
    if ci.get('is_controlling', False):
        print(f"  Step {ci['step']}: "
              f"Freq Δ={ci['freq_change']:.2f} Hz, "
              f"Amp Δ={ci['amp_change']:.2f}")

print(f"\n✓ All outputs saved in: {result['output_directory']}")
print(f"✓ Analysis reports in: {analysis_output}")
```

---

## Tips and Best Practices

### Performance Optimization

1. **Use batch mode** for multiple profiles (parallelization potential)
2. **Adjust frequency range** to your specific needs (fewer points = faster)
3. **Disable post-processing plots** if you only need data:
   ```python
   config = {"hv_postprocess": {"output": {"save_separate": False}}}
   ```

### Quality Control

1. **Check HV curve quality**: Ensure smooth curves without artifacts
2. **Verify peak detection**: Manual inspection recommended for complex curves
3. **Review controlling interfaces**: Cross-check with geological information

### Troubleshooting

**Issue: "HVf executable not found"**
- Solution: Package auto-detects, but you can specify manually:
  ```bash
  hvstrip-progressive workflow model.txt output/ --exe-path /path/to/HVf
  ```

**Issue: "No peaks detected"**
- Solution: Adjust peak detection parameters:
  ```python
  config = {"hv_postprocess": {"peak_detection": {"method": "max"}}}
  ```

**Issue: "Frequency range too narrow"**
- Solution: Expand frequency range:
  ```bash
  hvstrip-progressive workflow model.txt output/ --fmin 0.05 --fmax 100
  ```

---

## Getting Help

```bash
# General help
hvstrip-progressive --help

# Command-specific help
hvstrip-progressive workflow --help
hvstrip-progressive batch --help

# Show examples
hvstrip-progressive examples
```

---

For more information, see:
- [API Documentation](API_DOCUMENTATION.md)
- [Tutorial Notebook](tutorial.ipynb)
- Package README

