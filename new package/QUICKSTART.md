# HVSR Progressive Layer Stripping - Quick Start Guide

Get started with hvstrip-progressive in 5 minutes!

## What is Progressive Layer Stripping?

Progressive layer stripping is a technique to identify controlling interfaces in soil profiles by:
1. Computing H/V curves for the initial profile
2. Systematically removing the deepest layer
3. Computing H/V curves for each stripped model
4. Analyzing changes to identify influential layers

**Controlling interfaces** are layer boundaries that significantly affect the observed H/V curve. They typically correspond to strong impedance contrasts in the subsurface.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Package

```bash
cd "new package"
pip install -e .
```

This installs:
- Command-line interface (`hvstrip-progressive`)
- Python API modules
- Example data and executables

---

## Quick Start: Single Profile

### Step 1: Prepare Your Model File

Create a text file with your velocity model in HVf format:

**Example:** `my_profile.txt`
```
4
10  500  250  1800
15  800  400  1900
20  1200 600  2000
0   1500 750  2100
```

Format: `N` (number of layers), then N lines of `thickness vp vs density`
- Last layer has `thickness=0` (halfspace)
- Units: thickness (m), vp/vs (m/s), density (kg/m³)

### Step 2: Run Analysis

```bash
hvstrip-progressive workflow my_profile.txt results/
```

This command:
- ✓ Strips layers progressively
- ✓ Computes H/V curves for each step
- ✓ Generates plots and summaries
- ✓ Takes ~1-2 seconds per profile

### Step 3: View Results

```
results/
└── strip/
    ├── Step0_3-layer/          # Original model
    │   ├── model_Step0_3-layer.txt
    │   ├── hv_curve.csv        # H/V data
    │   ├── hv_curve.png        # H/V plot
    │   ├── vs_profile.png      # Velocity profile
    │   └── step_summary.csv    # Peak info
    ├── Step1_2-layer/          # After removing deepest finite layer
    ├── Step2_1-layer/          # After removing another layer
    └── ...
```

### Step 4: Analyze Results

```python
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory

# Run advanced analysis
results = analyze_strip_directory(
    strip_dir="results/strip",
    output_dir="results/analysis"
)

# Check controlling interfaces
for ci in results['controlling_interfaces']:
    if ci['is_controlling']:
        print(f"Controlling interface at step {ci['step']}")
        print(f"  Frequency change: {ci['freq_change']:.2f} Hz")
```

---

## Quick Start: Multiple Profiles

### Batch Processing

Process all profiles in a directory:

```bash
hvstrip-progressive batch soil_profiles/ batch_results/
```

**Benefits:**
- Process dozens of profiles automatically
- Consistent analysis parameters
- Summary statistics across all profiles
- ~1.3s per profile on average

### With Visualization

```bash
hvstrip-progressive batch soil_profiles/ batch_results/ --generate-report
```

Creates cross-profile comparison plots and statistics.

---

## Example Workflow: Step by Step

### 1. Create Test Data

Use the included example profiles:

```bash
cd "new package"
ls hvstrip_progressive/Example/profiles/
# profile_01.txt  profile_02.txt  ...  profile_12.txt
```

### 2. Single Profile Analysis

```bash
hvstrip-progressive workflow \
    hvstrip_progressive/Example/profiles/profile_01.txt \
    test_output/
```

**Output:**
```
🚀 HVSR Progressive Layer Stripping - Complete Workflow
📁 Initial model: .../profile_01.txt
📁 Output directory: test_output
⚙️  HVf executable: <auto-detected>

🔄 STEP 1/3: Layer Stripping
✅ Layer stripping completed in 0.00s
📊 Generated 2 stripped models

⚡ STEP 2/3: HV Forward Modeling
✅ HV forward modeling completed in 0.04s
📊 Successfully processed 2/2 models

📊 STEP 3/3: Post-Processing & Visualization
✅ Post-processing completed in 1.27s

🎉 WORKFLOW COMPLETED SUCCESSFULLY!
⏱️  Total time: 1.31s
```

### 3. Batch Processing

```bash
hvstrip-progressive batch \
    hvstrip_progressive/Example/profiles/ \
    batch_output/
```

**Output:**
```
🚀 BATCH PROCESSING MODE
📊 Found 12 profiles to process

[1/12] Processing profile_01...
✅ Workflow completed successfully!

[2/12] Processing profile_02...
✅ Workflow completed successfully!

...

📊 BATCH PROCESSING SUMMARY
Total profiles: 12
Successful: 12
Failed: 0
Total time: 15.23s
Average time per profile: 1.27s
```

### 4. Advanced Analysis

```python
from hvstrip_progressive.core.advanced_analysis import StrippingAnalyzer

# Load results
analyzer = StrippingAnalyzer("test_output/strip")

# Compute statistics
stats = analyzer.compute_statistics()
print(f"Peak frequency range: {stats['peak_freq_range']}")

# Detect controlling interfaces
controlling = analyzer.detect_controlling_interfaces()
print(f"Found {sum(1 for c in controlling if c['is_controlling'])} controlling interfaces")

# Generate report
analyzer.generate_analysis_report("test_output/analysis_report.txt")
```

---

## Common Tasks

### Custom Frequency Range

```bash
hvstrip-progressive workflow model.txt output/ \
    --fmin 0.1 \
    --fmax 50 \
    --nf 100
```

### High-Resolution Plots

```bash
hvstrip-progressive postprocess hv_curve.csv model.txt plots/ \
    --dpi 300 \
    --x-scale log \
    --y-scale log
```

### Find Controlling Interfaces

```python
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory

results = analyze_strip_directory("output/strip", "output/analysis")

# Print controlling interfaces
for ci in results['controlling_interfaces']:
    if ci['is_controlling']:
        print(f"Step {ci['step']}: {ci['from_layers']} → {ci['to_layers']} layers")
        print(f"  Significance: {ci['significance_score']:.1f}")
        print(f"  Freq change: {ci['freq_change']:.2f} Hz ({ci['freq_change_pct']:.1f}%)")
        print(f"  Amp change: {ci['amp_change']:.2f} ({ci['amp_change_pct']:.1f}%)")
```

---

## Understanding the Output

### H/V Curve Plot

The H/V curve shows the ratio of horizontal to vertical motion as a function of frequency:

- **Peak frequency (f₀)**: Fundamental resonance frequency
- **Peak amplitude**: H/V ratio at resonance
- **Curve shape**: Information about velocity structure

### Velocity Profile Plot

Shows the shear wave velocity (Vs) structure:

- Layers are shown as horizontal bars
- Depth increases downward
- Annotations show key depths and velocities

### Step Summary CSV

Contains:
- Peak frequency and amplitude
- Number of layers
- Model parameters
- Quality metrics

### Controlling Interface Report

Identifies:
- Steps with significant H/V changes
- Magnitude of frequency/amplitude shifts
- Significance scores
- Layer contributions

---

## Tips for Success

### Model Preparation

1. **Ensure proper format**: Check that last layer has thickness=0
2. **Realistic velocities**: Vp > Vs, both increase with depth typically
3. **Appropriate density**: Usually 1600-2500 kg/m³ for soils

### Analysis Parameters

1. **Frequency range**: Should encompass expected resonance (typically 0.2-20 Hz)
2. **Frequency points**: 71 is usually sufficient, increase for higher resolution
3. **Peak detection**: Use `find_peaks` for complex curves, `max` for simple ones

### Interpretation

1. **Large frequency shifts**: Indicate controlling interfaces
2. **Amplitude changes**: Show impedance contrast strength
3. **Correlation < 0.9**: Suggests significant structural change
4. **Multiple controlling interfaces**: Common in layered deposits

---

## Troubleshooting

### Problem: No peak detected

**Solution:** Adjust peak detection:
```python
config = {
    "hv_postprocess": {
        "peak_detection": {
            "method": "max",  # Use simple maximum
            "freq_min": None   # Remove constraints
        }
    }
}
```

### Problem: Peak at frequency boundary

**Solution:** Expand frequency range:
```bash
hvstrip-progressive workflow model.txt output/ --fmin 0.05 --fmax 50
```

### Problem: Computational errors

**Solution:** Check model validity:
- Last layer must have thickness = 0
- Velocities must be positive
- Vp > Vs (typically Vp ≈ √3 × Vs)

---

## Next Steps

1. **Read full documentation:**
   - [Usage Examples](USAGE_EXAMPLES.md) - Comprehensive examples
   - [API Documentation](API_DOCUMENTATION.md) - Full API reference

2. **Explore advanced features:**
   - Statistical analysis of results
   - Controlling interface detection algorithms
   - Custom visualization options

3. **Run on your data:**
   - Prepare your velocity models
   - Run batch processing
   - Analyze controlling interfaces

4. **Get help:**
   ```bash
   hvstrip-progressive --help
   hvstrip-progressive examples
   ```

---

## Example: Complete Analysis Script

Save as `analyze_profile.py`:

```python
#!/usr/bin/env python3
"""Complete analysis workflow example."""

from pathlib import Path
from hvstrip_progressive.core.batch_workflow import run_complete_workflow
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory

# 1. Run workflow
print("Running workflow...")
result = run_complete_workflow(
    initial_model_path="my_profile.txt",
    output_base_dir="results/"
)

if not result['success']:
    print("❌ Workflow failed!")
    exit(1)

print(f"✅ Workflow completed in {result['total_time']:.2f}s")

# 2. Advanced analysis
print("\nRunning advanced analysis...")
analysis = analyze_strip_directory(
    strip_dir=result['strip_directory'],
    output_dir=result['output_directory'] / "analysis"
)

# 3. Print results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

stats = analysis['statistics']
print(f"\nStatistics:")
print(f"  Steps processed: {stats['n_steps']}")
print(f"  Peak freq: {stats['peak_freq_mean']:.2f} ± {stats['peak_freq_std']:.2f} Hz")

controlling = analysis['controlling_interfaces']
n_controlling = sum(1 for c in controlling if c.get('is_controlling', False))

print(f"\nControlling Interfaces: {n_controlling}")
for ci in controlling:
    if ci.get('is_controlling', False):
        print(f"  Step {ci['step']}: Freq Δ={ci['freq_change']:.2f} Hz")

print(f"\n✅ All results saved in: results/")
```

Run it:
```bash
python3 analyze_profile.py
```

---

## Getting Help

- View examples: `hvstrip-progressive examples`
- Command help: `hvstrip-progressive workflow --help`
- Issues: https://github.com/yourusername/hvstrip-progressive/issues

---

**Ready to analyze your HVSR data? Start with `hvstrip-progressive workflow`!**

