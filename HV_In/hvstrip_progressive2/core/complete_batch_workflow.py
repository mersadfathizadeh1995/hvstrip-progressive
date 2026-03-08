"""
Complete Batch Workflow with Post-Processing
=============================================

A comprehensive workflow that:
1. Runs HVSR stripping on all profiles (parallel)
2. Extracts frequencies correctly from step summaries
3. Generates resonance separation figures for examples
4. Creates aggregate statistics
5. Produces publication-quality figures and tables

Author: Mersad Fathizadeh
Date: December 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np

# Ensure package is available
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hvstrip_progressive.core.batch_workflow import run_complete_workflow


def extract_frequencies_from_profile(
    profile_output_dir: Path,
    min_ratio: float = 1.5,
    min_f0: float = 0.3
) -> Tuple[float, float, dict]:
    """
    Extract f0 and f1 from a processed profile directory.
    
    f0 = FIRST significant peak from original profile (deep resonance)
    f1 = Peak from stripped model (shallow resonance)
    
    Args:
        profile_output_dir: Path to the profile output directory
        min_ratio: Minimum f1/f0 ratio for valid dual-resonance (default 1.5)
        min_f0: Minimum frequency for f0 detection (default 0.3 Hz)
        
    Returns:
        (f0, f1, metadata_dict)
        metadata includes 'valid_dual_resonance' flag
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    f0, f1 = 0.0, 0.0
    metadata = {'valid_dual_resonance': True}
    
    strip_dir = profile_output_dir / "strip"
    if not strip_dir.exists():
        return f0, f1, metadata
    
    # Find all step folders (Step0_*, Step1_*, Step2_*)
    step_folders = sorted(strip_dir.glob("Step*"))
    
    if not step_folders:
        return f0, f1, metadata
    
    # Step0 (original, full profile) -> f0 (deep resonance)
    # IMPORTANT: Use FIRST significant peak, not global max!
    step0_folders = [f for f in step_folders if f.name.startswith("Step0")]
    if step0_folders:
        hv_file = step0_folders[0] / "hv_curve.csv"
        if hv_file.exists():
            try:
                hv_df = pd.read_csv(hv_file)
                freqs = hv_df['Frequency_Hz'].values
                amps = hv_df['HVSR_Amplitude'].values
                
                # Smooth slightly
                amps_smooth = gaussian_filter1d(amps, sigma=1)
                
                # Find peaks
                min_prominence = 0.3 * (np.max(amps_smooth) - np.min(amps_smooth))
                peaks, _ = find_peaks(amps_smooth, 
                                      prominence=min_prominence * 0.3,
                                      distance=3)
                
                if len(peaks) >= 1:
                    # f0 is the FIRST peak (lowest frequency = deep resonance)
                    f0_idx = peaks[0]
                    f0 = float(freqs[f0_idx])
                    metadata['f0_amplitude'] = float(amps[f0_idx])
                else:
                    # Fallback to global max
                    f0_idx = np.argmax(amps)
                    f0 = float(freqs[f0_idx])
                    metadata['f0_amplitude'] = float(amps[f0_idx])
                    
            except Exception as e:
                # Fallback to step_summary.csv
                summary_file = step0_folders[0] / "step_summary.csv"
                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file)
                        if not df.empty:
                            f0 = float(df.iloc[0].get('Peak_Frequency_Hz', 0))
                            metadata['f0_amplitude'] = float(df.iloc[0].get('Peak_Amplitude', 0))
                    except:
                        pass
        
        # Get layer count from summary
        summary_file = step0_folders[0] / "step_summary.csv"
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                if not df.empty:
                    metadata['n_layers_original'] = int(df.iloc[0].get('N_Finite_Layers', 0))
            except:
                pass
    
    # Last step (most stripped) -> f1 (shallow resonance)
    # For the stripped model, global max IS correct since the deep layer is removed
    last_step_folder = step_folders[-1]
    summary_file = last_step_folder / "step_summary.csv"
    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            if not df.empty:
                f1 = float(df.iloc[0].get('Peak_Frequency_Hz', 0))
                metadata['f1_amplitude'] = float(df.iloc[0].get('Peak_Amplitude', 0))
                metadata['n_layers_stripped'] = int(df.iloc[0].get('N_Finite_Layers', 0))
        except Exception:
            pass
    
    # =========================================================================
    # VALIDATION: Check if this is a valid dual-resonance case
    # =========================================================================
    metadata['freq_ratio'] = f1 / f0 if f0 > 0 else 0
    
    # Invalid cases:
    # 1. f0 too low (below physical minimum)
    # 2. f1/f0 ratio too close to 1 (not truly separated resonances)
    # 3. f0 >= f1 (should not happen - deep resonance must be lower frequency)
    
    invalid_reasons = []
    
    if f0 < min_f0:
        invalid_reasons.append(f"f0={f0:.2f}Hz < min={min_f0}Hz")
    
    if f0 > 0 and f1 > 0:
        ratio = f1 / f0
        if ratio < min_ratio:
            invalid_reasons.append(f"ratio={ratio:.2f} < min={min_ratio}")
        if f0 >= f1:
            invalid_reasons.append(f"f0={f0:.2f} >= f1={f1:.2f}")
    
    if invalid_reasons:
        metadata['valid_dual_resonance'] = False
        metadata['invalid_reasons'] = invalid_reasons
    else:
        metadata['valid_dual_resonance'] = True
    
    return f0, f1, metadata


def process_single_profile(
    profile_path: Path,
    output_dir: Path,
    config: dict
) -> dict:
    """Process a single profile and extract results."""
    start_time = time.time()
    profile_name = profile_path.stem
    
    try:
        # Run the stripping workflow
        result = run_complete_workflow(
            str(profile_path),
            str(output_dir),
            config
        )
        
        # Extract frequencies from output
        f0, f1, metadata = extract_frequencies_from_profile(output_dir)
        
        elapsed = time.time() - start_time
        
        return {
            'profile_name': profile_name,
            'profile_path': str(profile_path),
            'output_dir': str(output_dir),
            'success': result.get('success', False),
            'f0_hz': f0,
            'f1_hz': f1,
            'freq_ratio': f1 / f0 if f0 > 0 else 0,
            'f0_amplitude': metadata.get('f0_amplitude', 0),
            'f1_amplitude': metadata.get('f1_amplitude', 0),
            'n_layers_original': metadata.get('n_layers_original', 0),
            'n_layers_stripped': metadata.get('n_layers_stripped', 0),
            'time_seconds': elapsed,
            'error': ''
        }
        
    except Exception as e:
        return {
            'profile_name': profile_name,
            'profile_path': str(profile_path),
            'output_dir': str(output_dir),
            'success': False,
            'f0_hz': 0,
            'f1_hz': 0,
            'freq_ratio': 0,
            'time_seconds': time.time() - start_time,
            'error': str(e)[:200]
        }


def generate_resonance_separation_figure(
    profile_output_dir: Path,
    output_path: Path
) -> bool:
    """
    Generate a publication-quality resonance separation figure.
    Uses the special_plots module for proper visualization.
    """
    try:
        from hvstrip_progressive.visualization.special_plots import generate_resonance_separation_figure as special_fig
        
        strip_dir = profile_output_dir / "strip"
        if not strip_dir.exists():
            return False
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        special_fig(str(strip_dir), str(output_path))
        return True
        
    except Exception as e:
        print(f"Error generating figure: {e}")
        return False


def generate_study_statistics(results_df: pd.DataFrame, output_dir: Path, min_ratio: float = 1.5) -> dict:
    """Generate comprehensive study statistics from batch results."""
    
    stats = {}
    
    # Filter successful results with valid frequencies
    valid = results_df[(results_df['success'] == True) & 
                       (results_df['f0_hz'] > 0) & 
                       (results_df['f1_hz'] > 0)].copy()
    
    # Additional filter: check valid_dual_resonance if available
    if 'valid_dual_resonance' in valid.columns:
        n_before = len(valid)
        valid = valid[valid['valid_dual_resonance'] == True]
        n_excluded = n_before - len(valid)
        stats['n_excluded_invalid'] = n_excluded
    else:
        # Apply ratio filter manually if column doesn't exist
        n_before = len(valid)
        valid = valid[valid['freq_ratio'] >= min_ratio]
        n_excluded = n_before - len(valid)
        stats['n_excluded_invalid'] = n_excluded
    
    if len(valid) == 0:
        return stats
    
    # Basic statistics
    stats['n_total'] = len(results_df)
    stats['n_successful'] = len(results_df[results_df['success'] == True])
    stats['n_valid_dual_resonance'] = len(valid)
    stats['n_valid_frequencies'] = len(valid)  # Keep for backward compatibility
    
    # f0 statistics (deep resonance)
    stats['f0_mean'] = valid['f0_hz'].mean()
    stats['f0_std'] = valid['f0_hz'].std()
    stats['f0_min'] = valid['f0_hz'].min()
    stats['f0_max'] = valid['f0_hz'].max()
    
    # f1 statistics (shallow resonance)
    stats['f1_mean'] = valid['f1_hz'].mean()
    stats['f1_std'] = valid['f1_hz'].std()
    stats['f1_min'] = valid['f1_hz'].min()
    stats['f1_max'] = valid['f1_hz'].max()
    
    # Frequency ratio
    stats['ratio_mean'] = valid['freq_ratio'].mean()
    stats['ratio_std'] = valid['freq_ratio'].std()
    stats['ratio_min'] = valid['freq_ratio'].min()
    stats['ratio_max'] = valid['freq_ratio'].max()
    
    # Save statistics
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as text
    with open(output_dir / 'study_statistics.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DUAL RESONANCE STUDY - STATISTICAL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total profiles analyzed: {stats['n_total']}\n")
        f.write(f"Successfully processed: {stats['n_successful']}\n")
        if 'n_excluded_invalid' in stats:
            f.write(f"Excluded (invalid ratio): {stats['n_excluded_invalid']}\n")
        f.write(f"Valid dual-resonance profiles: {stats['n_valid_dual_resonance']}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("Deep Resonance (f0):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean: {stats['f0_mean']:.3f} Hz\n")
        f.write(f"  Std:  {stats['f0_std']:.3f} Hz\n")
        f.write(f"  Range: {stats['f0_min']:.3f} - {stats['f0_max']:.3f} Hz\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("Shallow Resonance (f1):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean: {stats['f1_mean']:.3f} Hz\n")
        f.write(f"  Std:  {stats['f1_std']:.3f} Hz\n")
        f.write(f"  Range: {stats['f1_min']:.3f} - {stats['f1_max']:.3f} Hz\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("Frequency Ratio (f1/f0):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean: {stats['ratio_mean']:.2f}\n")
        f.write(f"  Std:  {stats['ratio_std']:.2f}\n")
        f.write(f"  Range: {stats['ratio_min']:.2f} - {stats['ratio_max']:.2f}\n")
    
    return stats


def generate_publication_figures(results_df: pd.DataFrame, output_dir: Path, min_ratio: float = 1.5) -> List[str]:
    """Generate publication-quality figures from batch results."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Set publication-quality defaults
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    # Filter valid data
    valid = results_df[(results_df['success'] == True) & 
                       (results_df['f0_hz'] > 0) & 
                       (results_df['f1_hz'] > 0)].copy()
    
    # Additional filter: exclude profiles with invalid ratio
    if 'valid_dual_resonance' in valid.columns:
        valid = valid[valid['valid_dual_resonance'] == True]
    else:
        valid = valid[valid['freq_ratio'] >= min_ratio]
    
    if len(valid) < 2:
        return generated
    
    # =========================================================================
    # Figure 1: Frequency Distribution
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # f0 histogram
    axes[0].hist(valid['f0_hz'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(valid['f0_hz'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid["f0_hz"].mean():.2f} Hz')
    axes[0].set_xlabel('Deep Resonance f₀ (Hz)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('(a) Deep Resonance Distribution', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # f1 histogram
    axes[1].hist(valid['f1_hz'], bins=20, color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(valid['f1_hz'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid["f1_hz"].mean():.2f} Hz')
    axes[1].set_xlabel('Shallow Resonance f₁ (Hz)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('(b) Shallow Resonance Distribution', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Ratio histogram
    axes[2].hist(valid['freq_ratio'], bins=20, color='seagreen', edgecolor='white', alpha=0.8)
    axes[2].axvline(valid['freq_ratio'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid["freq_ratio"].mean():.2f}')
    axes[2].set_xlabel('Frequency Ratio f₁/f₀', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title('(c) Peak Separation Ratio', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = figures_dir / "Fig1_frequency_distributions.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated.append(str(fig_path))
    
    # =========================================================================
    # Figure 2: f1 vs f0 Scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    
    scatter = ax.scatter(valid['f0_hz'], valid['f1_hz'], 
                        c=valid['freq_ratio'], cmap='viridis',
                        s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add ratio lines
    f0_range = np.linspace(valid['f0_hz'].min() * 0.8, valid['f0_hz'].max() * 1.2, 100)
    for ratio, style, color in [(2, ':', 'gray'), (3, '--', 'orange'), 
                                 (4, '--', 'red'), (5, ':', 'gray')]:
        ax.plot(f0_range, f0_range * ratio, style, color=color, 
               alpha=0.7, label=f'{ratio}:1')
    
    ax.set_xlabel('Deep Resonance f₀ (Hz)', fontsize=12)
    ax.set_ylabel('Shallow Resonance f₁ (Hz)', fontsize=12)
    ax.set_title('Relationship Between Deep and Shallow Resonance', fontsize=13)
    ax.legend(title='Ratio', loc='upper left')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('f₁/f₀ Ratio')
    
    plt.tight_layout()
    fig_path = figures_dir / "Fig2_f1_vs_f0_scatter.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated.append(str(fig_path))
    
    # =========================================================================
    # Figure 3: Comprehensive Summary
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    
    # Box plots
    ax1 = fig.add_subplot(2, 2, 1)
    bp = ax1.boxplot([valid['f0_hz'], valid['f1_hz']], 
                     labels=['f₀ (Deep)', 'f₁ (Shallow)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax1.set_ylabel('Frequency (Hz)', fontsize=11)
    ax1.set_title('(a) Resonance Frequency Comparison', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    ax1.annotate(f'{valid["f0_hz"].mean():.2f} Hz', 
                xy=(1, valid["f0_hz"].mean()), xytext=(1.3, valid["f0_hz"].mean()),
                fontsize=10, color='steelblue')
    ax1.annotate(f'{valid["f1_hz"].mean():.2f} Hz', 
                xy=(2, valid["f1_hz"].mean()), xytext=(2.3, valid["f1_hz"].mean()),
                fontsize=10, color='coral')
    
    # Scatter with regression
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(valid['f0_hz'], valid['f1_hz'], alpha=0.6, s=40)
    
    # Linear regression
    from scipy import stats as scipy_stats
    slope, intercept, r_value, _, _ = scipy_stats.linregress(valid['f0_hz'], valid['f1_hz'])
    line_x = np.linspace(valid['f0_hz'].min(), valid['f0_hz'].max(), 100)
    line_y = slope * line_x + intercept
    ax2.plot(line_x, line_y, 'r--', linewidth=2, 
            label=f'R² = {r_value**2:.3f}')
    
    ax2.set_xlabel('f₀ (Hz)', fontsize=11)
    ax2.set_ylabel('f₁ (Hz)', fontsize=11)
    ax2.set_title('(b) Correlation Analysis', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ratio distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(valid['freq_ratio'], bins=25, color='seagreen', 
            edgecolor='white', alpha=0.8, density=True)
    
    # Fit normal distribution
    mu, std = valid['freq_ratio'].mean(), valid['freq_ratio'].std()
    x = np.linspace(valid['freq_ratio'].min(), valid['freq_ratio'].max(), 100)
    ax3.plot(x, scipy_stats.norm.pdf(x, mu, std), 'r-', linewidth=2,
            label=f'μ={mu:.2f}, σ={std:.2f}')
    
    ax3.set_xlabel('Frequency Ratio f₁/f₀', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('(c) Ratio Distribution', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    STUDY SUMMARY
    {'='*40}
    
    Profiles Analyzed: {len(valid)}
    
    Deep Resonance (f₀):
      Mean: {valid['f0_hz'].mean():.3f} ± {valid['f0_hz'].std():.3f} Hz
      Range: {valid['f0_hz'].min():.3f} - {valid['f0_hz'].max():.3f} Hz
    
    Shallow Resonance (f₁):
      Mean: {valid['f1_hz'].mean():.3f} ± {valid['f1_hz'].std():.3f} Hz
      Range: {valid['f1_hz'].min():.3f} - {valid['f1_hz'].max():.3f} Hz
    
    Frequency Ratio (f₁/f₀):
      Mean: {valid['freq_ratio'].mean():.2f} ± {valid['freq_ratio'].std():.2f}
      Range: {valid['freq_ratio'].min():.2f} - {valid['freq_ratio'].max():.2f}
    
    Correlation: R² = {r_value**2:.3f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('(d) Summary Statistics', fontsize=12)
    
    plt.suptitle('Dual Resonance Study - Comprehensive Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = figures_dir / "Fig3_comprehensive_summary.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated.append(str(fig_path))
    
    # =========================================================================
    # Table 1: Statistics Table as Image
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table_data = [
        ['Parameter', 'Mean ± Std', 'Range', 'Unit'],
        ['Deep resonance (f₀)', 
         f'{valid["f0_hz"].mean():.3f} ± {valid["f0_hz"].std():.3f}',
         f'{valid["f0_hz"].min():.3f} - {valid["f0_hz"].max():.3f}', 'Hz'],
        ['Shallow resonance (f₁)', 
         f'{valid["f1_hz"].mean():.3f} ± {valid["f1_hz"].std():.3f}',
         f'{valid["f1_hz"].min():.3f} - {valid["f1_hz"].max():.3f}', 'Hz'],
        ['Frequency ratio (f₁/f₀)', 
         f'{valid["freq_ratio"].mean():.2f} ± {valid["freq_ratio"].std():.2f}',
         f'{valid["freq_ratio"].min():.2f} - {valid["freq_ratio"].max():.2f}', '-'],
        ['Sample size', f'{len(valid)}', '-', 'profiles'],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', cellLoc='center',
                    colColours=['lightsteelblue']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    ax.set_title('Table 1: Summary Statistics of Dual Resonance Characteristics',
                fontsize=13, fontweight='bold', pad=20)
    
    fig_path = figures_dir / "Table1_statistics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated.append(str(fig_path))
    
    return generated


def run_complete_batch_analysis(
    input_dir: str,
    output_dir: str,
    n_workers: int = 10,
    n_example_figures: int = 5,
    fmin: float = 0.1,
    fmax: float = 30.0,
    progress_callback=None
) -> dict:
    """
    Run the complete batch analysis workflow.
    
    Args:
        input_dir: Directory containing profile .txt files
        output_dir: Output directory for results
        n_workers: Number of parallel workers
        n_example_figures: Number of example resonance separation figures to generate
        fmin, fmax: Frequency range for HVSR
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        Dictionary with results summary and file paths
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PHASE 1: Collect profile files
    # =========================================================================
    if progress_callback:
        progress_callback(0, 100, "Scanning for profiles...")
    
    profile_files = []
    
    # Check for txt subfolder
    txt_dir = input_path / "txt" if (input_path / "txt").exists() else input_path
    profile_files = list(txt_dir.glob("*.txt"))
    
    if not profile_files:
        return {'success': False, 'error': 'No profile files found'}
    
    total = len(profile_files)
    print(f"\n{'='*60}")
    print(f"COMPLETE BATCH WORKFLOW")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Profiles: {total}")
    print(f"Workers: {n_workers}")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # PHASE 2: Process all profiles (parallel)
    # =========================================================================
    if progress_callback:
        progress_callback(0, total, "Processing profiles...")
    
    print("PHASE 1: Processing profiles...")
    
    config = {
        "hv_forward": {
            "fmin": fmin,
            "fmax": fmax,
            "nf": 100
        }
    }
    
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for profile_path in profile_files:
            profile_name = profile_path.stem
            profile_output = output_path / "profiles" / profile_name
            
            future = executor.submit(
                process_single_profile,
                profile_path, profile_output, config
            )
            futures[future] = profile_name
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total, f"Processed {result['profile_name']}")
            
            if completed % 10 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")
    
    # Save batch results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "batch_results.csv", index=False)
    
    successful = len(results_df[results_df['success'] == True])
    valid = len(results_df[(results_df['f0_hz'] > 0) & (results_df['f1_hz'] > 0)])
    
    print(f"\n  Completed: {completed}")
    print(f"  Successful: {successful}")
    print(f"  Valid frequencies: {valid}")
    
    # =========================================================================
    # PHASE 3: Generate example resonance separation figures
    # =========================================================================
    print(f"\nPHASE 2: Generating {n_example_figures} example figures...")
    
    if progress_callback:
        progress_callback(0, n_example_figures, "Generating example figures...")
    
    examples_dir = output_path / "example_figures"
    examples_dir.mkdir(exist_ok=True)
    
    # Select profiles with valid frequencies
    valid_profiles = results_df[(results_df['f0_hz'] > 0) & (results_df['f1_hz'] > 0)]
    
    if len(valid_profiles) > 0:
        # Sample evenly across the range
        sample_indices = np.linspace(0, len(valid_profiles)-1, 
                                     min(n_example_figures, len(valid_profiles)), 
                                     dtype=int)
        sample_profiles = valid_profiles.iloc[sample_indices]
        
        for i, (_, row) in enumerate(sample_profiles.iterrows()):
            profile_output = Path(row['output_dir'])
            fig_path = examples_dir / f"example_{i+1}_{row['profile_name']}.png"
            
            success = generate_resonance_separation_figure(profile_output, fig_path)
            
            if progress_callback:
                progress_callback(i+1, n_example_figures, f"Generated {fig_path.name}")
            
            print(f"  [{i+1}/{n_example_figures}] {fig_path.name}: {'OK' if success else 'FAILED'}")
    
    # =========================================================================
    # PHASE 4: Generate statistics
    # =========================================================================
    print("\nPHASE 3: Computing statistics...")
    
    if progress_callback:
        progress_callback(0, 1, "Computing statistics...")
    
    stats_dir = output_path / "statistics"
    stats = generate_study_statistics(results_df, stats_dir)
    
    if stats:
        print(f"  f₀: {stats['f0_mean']:.3f} ± {stats['f0_std']:.3f} Hz")
        print(f"  f₁: {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f} Hz")
        print(f"  Ratio: {stats['ratio_mean']:.2f} ± {stats['ratio_std']:.2f}")
    
    # =========================================================================
    # PHASE 5: Generate publication figures
    # =========================================================================
    print("\nPHASE 4: Generating publication figures...")
    
    if progress_callback:
        progress_callback(0, 4, "Generating publication figures...")
    
    figures = generate_publication_figures(results_df, output_path)
    
    for fig in figures:
        print(f"  Created: {Path(fig).name}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"\nGenerated files:")
    print(f"  - batch_results.csv")
    print(f"  - statistics/study_statistics.txt")
    print(f"  - example_figures/ ({n_example_figures} figures)")
    print(f"  - figures/ ({len(figures)} publication figures)")
    print(f"{'='*60}\n")
    
    return {
        'success': True,
        'n_profiles': total,
        'n_successful': successful,
        'n_valid': valid,
        'statistics': stats,
        'figures': figures,
        'output_dir': str(output_path)
    }


if __name__ == "__main__":
    # Test with the high_contrast data
    result = run_complete_batch_analysis(
        input_dir=r"D:\Github_Papers\Two_Resonance\Project_Basin\profiles_expanded\high_contrast",
        output_dir=r"D:\Github_Papers\Two_Resonance\Project_Basin\Test\high_contrast_complete",
        n_workers=8,
        n_example_figures=5
    )
    print(result)
