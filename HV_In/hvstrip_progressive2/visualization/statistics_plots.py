"""
Statistical Visualization Module for Two Resonance Analysis.

Creates publication-ready statistical plots from batch analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional
from scipy import stats


def create_frequency_distribution_plot(results_df: pd.DataFrame, 
                                       output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create histogram showing distribution of f0 (deep) and f1 (shallow) frequencies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter successful results
    df = results_df[results_df['success'] == True].copy()
    
    # f0 distribution
    ax1 = axes[0]
    ax1.hist(df['f0_original_Hz'], bins=15, color='#2E86AB', alpha=0.7, 
             edgecolor='black', linewidth=1.2)
    ax1.axvline(df['f0_original_Hz'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["f0_original_Hz"].mean():.2f} Hz')
    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Deep Resonance ($f_0$) Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # f1 distribution
    ax2 = axes[1]
    ax2.hist(df['f1_shallow_Hz'], bins=15, color='#E63946', alpha=0.7,
             edgecolor='black', linewidth=1.2)
    ax2.axvline(df['f1_shallow_Hz'].mean(), color='darkred', linestyle='--',
                linewidth=2, label=f'Mean: {df["f1_shallow_Hz"].mean():.2f} Hz')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Shallow Resonance ($f_1$) Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    
    return fig


def create_theoretical_validation_plot(results_df: pd.DataFrame,
                                       output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create scatter plot comparing theoretical vs measured frequencies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df = results_df[results_df['success'] == True].copy()
    
    # f0: Theoretical vs Measured
    ax1 = axes[0]
    ax1.scatter(df['f0_theoretical_Hz'], df['f0_original_Hz'], 
                s=80, c='#2E86AB', alpha=0.7, edgecolors='black', linewidth=1)
    
    # 1:1 line
    max_val = max(df['f0_theoretical_Hz'].max(), df['f0_original_Hz'].max())
    min_val = min(df['f0_theoretical_Hz'].min(), df['f0_original_Hz'].min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')
    
    # Regression line
    slope, intercept, r_value, _, _ = stats.linregress(df['f0_theoretical_Hz'], df['f0_original_Hz'])
    x_reg = np.linspace(min_val, max_val, 100)
    ax1.plot(x_reg, slope * x_reg + intercept, 'r-', linewidth=2, 
             label=f'R² = {r_value**2:.3f}')
    
    ax1.set_xlabel('Theoretical $f_0$ (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Measured $f_0$ (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Deep Resonance Validation', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # f1: Theoretical vs Measured  
    ax2 = axes[1]
    ax2.scatter(df['f1_theoretical_Hz'], df['f1_shallow_Hz'],
                s=80, c='#E63946', alpha=0.7, edgecolors='black', linewidth=1)
    
    max_val = max(df['f1_theoretical_Hz'].max(), df['f1_shallow_Hz'].max())
    min_val = min(df['f1_theoretical_Hz'].min(), df['f1_shallow_Hz'].min())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')
    
    slope, intercept, r_value, _, _ = stats.linregress(df['f1_theoretical_Hz'], df['f1_shallow_Hz'])
    x_reg = np.linspace(min_val, max_val, 100)
    ax2.plot(x_reg, slope * x_reg + intercept, 'darkred', linewidth=2,
             label=f'R² = {r_value**2:.3f}')
    
    ax2.set_xlabel('Theoretical $f_1$ (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Measured $f_1$ (Hz)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Shallow Resonance Validation', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    
    return fig


def create_frequency_ratio_plot(results_df: pd.DataFrame,
                                output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create plot showing frequency ratio (f1/f0) distribution and success rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df = results_df[results_df['success'] == True].copy()
    
    # Ratio histogram
    ax1 = axes[0]
    ax1.hist(df['freq_ratio'], bins=15, color='#457B9D', alpha=0.7,
             edgecolor='black', linewidth=1.2)
    ax1.axvline(1.0, color='gray', linestyle=':', linewidth=2, label='f₁ = f₀')
    ax1.axvline(1.2, color='green', linestyle='--', linewidth=2, label='Separation threshold')
    ax1.axvline(df['freq_ratio'].mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {df["freq_ratio"].mean():.2f}')
    ax1.set_xlabel('Frequency Ratio ($f_1$/$f_0$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Frequency Ratio Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Success rate pie chart
    ax2 = axes[1]
    success_count = df['separation_success'].sum()
    fail_count = len(df) - success_count
    
    colors = ['#2E86AB', '#E63946']
    labels = [f'Successful\n({success_count})', f'Unclear\n({fail_count})']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax2.pie([success_count, fail_count], 
                                        explode=explode, labels=labels,
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('(b) Separation Success Rate', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    
    return fig


def create_comprehensive_summary_plot(results_df: pd.DataFrame, stats: Dict,
                                      output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a comprehensive summary figure for publication.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    df = results_df[results_df['success'] == True].copy()
    
    # (a) f0 vs f1 scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df['f0_original_Hz'], df['f1_shallow_Hz'], 
                s=60, c='#2E86AB', alpha=0.7, edgecolors='black')
    ax1.set_xlabel('$f_0$ (Hz)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('$f_1$ (Hz)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Deep vs Shallow Frequency', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # (b) Frequency ratio histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['freq_ratio'], bins=12, color='#457B9D', alpha=0.7, edgecolor='black')
    ax2.axvline(df['freq_ratio'].mean(), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('$f_1$/$f_0$ Ratio', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Frequency Ratio Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # (c) Max frequency shift
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['max_freq_shift_Hz'], bins=12, color='#E63946', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Max Δf (Hz)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Maximum Frequency Shift', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # (d) Theoretical validation - f0
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['f0_theoretical_Hz'], df['f0_original_Hz'], s=50, c='#2E86AB', alpha=0.7)
    max_val = max(df['f0_theoretical_Hz'].max(), df['f0_original_Hz'].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
    ax4.set_xlabel('Theoretical $f_0$ (Hz)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Measured $f_0$ (Hz)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) $f_0$ Validation', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # (e) Theoretical validation - f1
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df['f1_theoretical_Hz'], df['f1_shallow_Hz'], s=50, c='#E63946', alpha=0.7)
    max_val = max(df['f1_theoretical_Hz'].max(), df['f1_shallow_Hz'].max())
    ax5.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
    ax5.set_xlabel('Theoretical $f_1$ (Hz)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Measured $f_1$ (Hz)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) $f_1$ Validation', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # (f) Controlling step distribution
    ax6 = fig.add_subplot(gs[1, 2])
    step_counts = df['controlling_step'].value_counts().sort_index()
    ax6.bar(step_counts.index, step_counts.values, color='#A8DADC', edgecolor='black')
    ax6.set_xlabel('Stripping Step', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax6.set_title('(f) Controlling Layer Step', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # (g-h-i) Statistics summary table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Create summary text
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                           BATCH ANALYSIS SUMMARY                                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════╣
    ║  Total Profiles: {stats.get('n_profiles', len(df)):>6}          Successful: {stats.get('n_successful', len(df)):>6} ({stats.get('success_rate', 100):.1f}%)                         ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════╣
    ║  FREQUENCY STATISTICS                                                                     ║
    ║  ─────────────────────────────────────────────────────────────────────────────────────── ║
    ║  Deep Resonance (f₀):      {stats.get('f0_mean', df['f0_original_Hz'].mean()):.2f} ± {stats.get('f0_std', df['f0_original_Hz'].std()):.2f} Hz   (range: {stats.get('f0_min', df['f0_original_Hz'].min()):.2f} - {stats.get('f0_max', df['f0_original_Hz'].max()):.2f} Hz)        ║
    ║  Shallow Resonance (f₁):   {stats.get('f1_mean', df['f1_shallow_Hz'].mean()):.2f} ± {stats.get('f1_std', df['f1_shallow_Hz'].std()):.2f} Hz   (range: {stats.get('f1_min', df['f1_shallow_Hz'].min()):.2f} - {stats.get('f1_max', df['f1_shallow_Hz'].max()):.2f} Hz)        ║
    ║  Frequency Ratio (f₁/f₀):  {stats.get('freq_ratio_mean', df['freq_ratio'].mean()):.2f} ± {stats.get('freq_ratio_std', df['freq_ratio'].std()):.2f}                                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════╣
    ║  VALIDATION                                                                               ║
    ║  ─────────────────────────────────────────────────────────────────────────────────────── ║
    ║  f₀ Theoretical Correlation (R²): {stats.get('f0_theoretical_correlation', 0)**2:.3f}                                                   ║
    ║  f₁ Theoretical Correlation (R²): {stats.get('f1_theoretical_correlation', 0)**2:.3f}                                                   ║
    ║  Separation Success Rate:         {stats.get('separation_success_rate', 0):.1f}%                                                    ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_table.text(0.5, 0.5, summary_text, transform=ax_table.transAxes,
                  fontsize=10, family='monospace', ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#343A40', linewidth=2))
    
    plt.suptitle('Two Resonance Separation Analysis - Statistical Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    
    return fig


def generate_all_statistics_plots(results_csv: str, stats_json: str, 
                                  output_dir: str) -> Dict[str, Path]:
    """
    Generate all statistical plots from batch results.
    
    Args:
        results_csv: Path to batch_results.csv
        stats_json: Path to batch_statistics.json
        output_dir: Output directory for plots
        
    Returns:
        Dict mapping plot names to paths
    """
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(results_csv)
    with open(stats_json, 'r') as f:
        stats = json.load(f)
    
    print(f"\n{'='*50}")
    print("Generating Statistical Plots")
    print(f"{'='*50}")
    
    outputs = {}
    
    # 1. Frequency distribution
    path = output_path / "frequency_distribution.png"
    create_frequency_distribution_plot(df, path)
    outputs['frequency_distribution'] = path
    plt.close()
    
    # 2. Theoretical validation
    path = output_path / "theoretical_validation.png"
    create_theoretical_validation_plot(df, path)
    outputs['theoretical_validation'] = path
    plt.close()
    
    # 3. Frequency ratio
    path = output_path / "frequency_ratio.png"
    create_frequency_ratio_plot(df, path)
    outputs['frequency_ratio'] = path
    plt.close()
    
    # 4. Comprehensive summary
    path = output_path / "comprehensive_summary.png"
    create_comprehensive_summary_plot(df, stats, path)
    outputs['comprehensive_summary'] = path
    plt.close()
    
    print(f"\n[OK] Generated {len(outputs)} statistical plots")
    print(f"  Output directory: {output_dir}")
    
    return outputs


__all__ = [
    "create_frequency_distribution_plot",
    "create_theoretical_validation_plot", 
    "create_frequency_ratio_plot",
    "create_comprehensive_summary_plot",
    "generate_all_statistics_plots"
]
