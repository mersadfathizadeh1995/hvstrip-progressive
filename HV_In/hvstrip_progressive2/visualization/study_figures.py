"""
Study Figures Module - Publication-ready figures for Dual Resonance Study.

Creates figures that present FINDINGS about the phenomenon, not method validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Tuple
from scipy import stats as scipy_stats


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def create_frequency_characteristics_figure(df: pd.DataFrame, 
                                            output_path: str) -> str:
    """
    Figure 1: Frequency Characteristics of Dual-Resonance Sites
    
    Shows the distribution and relationship of f0 (deep) and f1 (shallow).
    """
    set_publication_style()
    
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    f0 = df['f0_original_Hz'].values
    f1 = df['f1_shallow_Hz'].values
    
    # (a) Frequency distributions
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 6, 20)
    ax1.hist(f0, bins=bins, alpha=0.7, color='#2E86AB', label=f'$f_0$ (deep)', edgecolor='white')
    ax1.hist(f1, bins=bins, alpha=0.7, color='#E63946', label=f'$f_1$ (shallow)', edgecolor='white')
    ax1.axvline(np.mean(f0), color='#2E86AB', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(f1), color='#E63946', linestyle='--', linewidth=2)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Resonance Frequency Distribution')
    ax1.legend(loc='upper right')
    
    # (b) f1 vs f0 scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(f0, f1, s=60, c='#2E86AB', edgecolors='white', linewidth=0.5, alpha=0.8)
    
    # Add 1:1 and ratio lines
    freq_range = np.array([0.5, 2.0])
    ax2.plot(freq_range, freq_range, 'k--', alpha=0.3, label='1:1')
    ax2.plot(freq_range, 2*freq_range, 'g--', alpha=0.5, label='2:1')
    ax2.plot(freq_range, 3*freq_range, 'r--', alpha=0.5, label='3:1')
    ax2.plot(freq_range, 4*freq_range, 'm--', alpha=0.5, label='4:1')
    
    ax2.set_xlabel('Deep Resonance $f_0$ (Hz)')
    ax2.set_ylabel('Shallow Resonance $f_1$ (Hz)')
    ax2.set_title('(b) $f_1$ vs $f_0$ Relationship')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim(0.8, 1.8)
    ax2.set_ylim(2, 6)
    
    # (c) Frequency ratio distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ratios = f1 / f0
    ax3.hist(ratios, bins=15, color='#457B9D', edgecolor='white', alpha=0.8)
    ax3.axvline(np.mean(ratios), color='#E63946', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(ratios):.2f}')
    ax3.set_xlabel('Frequency Ratio $f_1/f_0$')
    ax3.set_ylabel('Count')
    ax3.set_title('(c) Peak Separation Ratio')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"[OK] Created: {output_path}")
    return output_path


def create_depth_frequency_figure(df: pd.DataFrame, 
                                  output_path: str) -> str:
    """
    Figure 2: Frequency-Depth Relationships
    
    Shows how resonance frequencies relate to basin structure.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    f0 = df['f0_original_Hz'].values
    f1 = df['f1_shallow_Hz'].values
    depths = df['total_depth_m'].values
    f0_theo = df['f0_theoretical_Hz'].values
    f1_theo = df['f1_theoretical_Hz'].values
    
    # (a) f0 vs Total Depth
    ax1 = axes[0]
    ax1.scatter(depths, f0, s=60, c='#2E86AB', edgecolors='white', linewidth=0.5, alpha=0.8)
    
    # Regression line
    slope, intercept, r, _, _ = scipy_stats.linregress(depths, f0)
    x_line = np.array([depths.min(), depths.max()])
    ax1.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2,
             label=f'$R^2$ = {r**2:.3f}')
    
    ax1.set_xlabel('Total Basin Depth (m)')
    ax1.set_ylabel('Deep Resonance $f_0$ (Hz)')
    ax1.set_title('(a) Deep Resonance vs Basin Depth')
    ax1.legend(loc='upper right')
    
    # (b) Measured vs Theoretical
    ax2 = axes[1]
    ax2.scatter(f0_theo, f0, s=60, c='#2E86AB', edgecolors='white', linewidth=0.5, 
                alpha=0.8, label='$f_0$ (deep)')
    ax2.scatter(f1_theo, f1, s=60, c='#E63946', edgecolors='white', linewidth=0.5, 
                alpha=0.8, label='$f_1$ (shallow)')
    
    # 1:1 line
    all_f = np.concatenate([f0_theo, f1_theo, f0, f1])
    f_range = np.array([all_f.min() * 0.9, all_f.max() * 1.1])
    ax2.plot(f_range, f_range, 'k--', alpha=0.5, label='1:1')
    
    # R2 values
    r0 = np.corrcoef(f0_theo, f0)[0, 1]
    r1 = np.corrcoef(f1_theo, f1)[0, 1]
    
    ax2.set_xlabel('Theoretical Frequency (Hz)')
    ax2.set_ylabel('Measured Frequency (Hz)')
    ax2.set_title(f'(b) Theoretical Validation\n$R^2_{{f_0}}$={r0**2:.2f}, $R^2_{{f_1}}$={r1**2:.2f}')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"[OK] Created: {output_path}")
    return output_path


def create_comprehensive_study_figure(df: pd.DataFrame, 
                                      output_path: str) -> str:
    """
    Figure 3: Comprehensive Study Summary (4-panel figure)
    
    Main figure for the paper showing all key findings.
    """
    set_publication_style()
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    f0 = df['f0_original_Hz'].values
    f1 = df['f1_shallow_Hz'].values
    depths = df['total_depth_m'].values
    ratios = f1 / f0
    
    # (a) Frequency distributions with box plots
    ax1 = fig.add_subplot(gs[0, 0])
    positions = [1, 2]
    bp = ax1.boxplot([f0, f1], positions=positions, widths=0.6, patch_artist=True)
    colors = ['#2E86AB', '#E63946']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['$f_0$ (Deep)', '$f_1$ (Shallow)'])
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('(a) Resonance Frequency Ranges')
    
    # Add mean annotations
    ax1.annotate(f'{np.mean(f0):.2f} Hz', xy=(1, np.mean(f0)), xytext=(1.3, np.mean(f0)),
                fontsize=9, va='center')
    ax1.annotate(f'{np.mean(f1):.2f} Hz', xy=(2, np.mean(f1)), xytext=(2.3, np.mean(f1)),
                fontsize=9, va='center')
    
    # (b) f1/f0 ratio vs depth
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(depths, ratios, s=60, c=ratios, cmap='RdYlBu_r', 
                          edgecolors='white', linewidth=0.5, vmin=2, vmax=5)
    plt.colorbar(scatter, ax=ax2, label='$f_1/f_0$')
    ax2.axhline(np.mean(ratios), color='red', linestyle='--', alpha=0.7,
                label=f'Mean = {np.mean(ratios):.2f}')
    ax2.set_xlabel('Total Depth (m)')
    ax2.set_ylabel('Frequency Ratio $f_1/f_0$')
    ax2.set_title('(b) Peak Separation vs Basin Depth')
    ax2.legend(loc='upper right')
    
    # (c) Resonance frequency vs depth (both peaks)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(depths, f0, s=60, c='#2E86AB', edgecolors='white', linewidth=0.5,
                alpha=0.8, label='$f_0$ (deep)')
    ax3.scatter(depths, f1, s=60, c='#E63946', edgecolors='white', linewidth=0.5,
                alpha=0.8, label='$f_1$ (shallow)')
    
    # Fit lines
    slope0, int0, r0, _, _ = scipy_stats.linregress(depths, f0)
    slope1, int1, r1, _, _ = scipy_stats.linregress(depths, f1)
    x_line = np.array([depths.min(), depths.max()])
    ax3.plot(x_line, slope0 * x_line + int0, '#2E86AB', linestyle='--', linewidth=1.5)
    ax3.plot(x_line, slope1 * x_line + int1, '#E63946', linestyle='--', linewidth=1.5)
    
    ax3.set_xlabel('Total Depth (m)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title(f'(c) Resonance Frequencies vs Depth')
    ax3.legend(loc='upper right')
    
    # (d) Summary statistics table as text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""
    STUDY SUMMARY
    {'='*40}
    
    Sites Analyzed: {len(df)}
    
    Deep Resonance ($f_0$):
      Mean: {np.mean(f0):.2f} +/- {np.std(f0):.2f} Hz
      Range: {f0.min():.2f} - {f0.max():.2f} Hz
    
    Shallow Resonance ($f_1$):
      Mean: {np.mean(f1):.2f} +/- {np.std(f1):.2f} Hz
      Range: {f1.min():.2f} - {f1.max():.2f} Hz
    
    Frequency Ratio ($f_1$/$f_0$):
      Mean: {np.mean(ratios):.2f} +/- {np.std(ratios):.2f}
      Range: {ratios.min():.2f} - {ratios.max():.2f}
    
    Depth Range: {depths.min():.1f} - {depths.max():.1f} m
    
    $f_0$ vs Depth: $R^2$ = {r0**2:.3f}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('(d) Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"[OK] Created: {output_path}")
    return output_path


def create_statistics_table(df: pd.DataFrame, output_path: str) -> str:
    """
    Create a formatted statistics table as an image.
    """
    set_publication_style()
    
    f0 = df['f0_original_Hz'].values
    f1 = df['f1_shallow_Hz'].values
    depths = df['total_depth_m'].values
    ratios = f1 / f0
    f0_theo = df['f0_theoretical_Hz'].values
    f1_theo = df['f1_theoretical_Hz'].values
    
    # Compute correlations
    r0_theo = np.corrcoef(f0_theo, f0)[0, 1] ** 2
    r1_theo = np.corrcoef(f1_theo, f1)[0, 1] ** 2
    _, _, r0_depth, _, _ = scipy_stats.linregress(depths, f0)
    
    # Table data
    table_data = [
        ['Parameter', 'Mean +/- Std', 'Range', 'Unit'],
        ['Deep resonance ($f_0$)', f'{np.mean(f0):.2f} +/- {np.std(f0):.2f}', 
         f'{f0.min():.2f} - {f0.max():.2f}', 'Hz'],
        ['Shallow resonance ($f_1$)', f'{np.mean(f1):.2f} +/- {np.std(f1):.2f}',
         f'{f1.min():.2f} - {f1.max():.2f}', 'Hz'],
        ['Frequency ratio ($f_1$/$f_0$)', f'{np.mean(ratios):.2f} +/- {np.std(ratios):.2f}',
         f'{ratios.min():.2f} - {ratios.max():.2f}', '-'],
        ['Total basin depth', f'{np.mean(depths):.1f} +/- {np.std(depths):.1f}',
         f'{depths.min():.1f} - {depths.max():.1f}', 'm'],
        ['', '', '', ''],
        ['Correlation', '$R^2$', '', ''],
        ['$f_0$ vs theoretical', f'{r0_theo:.3f}', '', ''],
        ['$f_1$ vs theoretical', f'{r1_theo:.3f}', '', ''],
        ['$f_0$ vs depth', f'{r0_depth**2:.3f}', '', ''],
    ]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style section header
    table[(6, 0)].set_facecolor('#E8E8E8')
    table[(6, 1)].set_facecolor('#E8E8E8')
    
    plt.title('Table 1: Summary Statistics of Dual Resonance Characteristics', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"[OK] Created: {output_path}")
    return output_path


def generate_all_study_figures(results_csv: str, output_dir: str) -> dict:
    """
    Generate all figures for the dual resonance study paper.
    
    Returns dict with paths to all generated figures.
    """
    df = pd.read_csv(results_csv)
    df = df[df['success'] == True].copy()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING STUDY FIGURES")
    print("=" * 60)
    
    figures = {}
    
    # Figure 1: Frequency characteristics
    figures['fig1_frequency_characteristics'] = create_frequency_characteristics_figure(
        df, str(output_path / 'Fig1_frequency_characteristics.png')
    )
    
    # Figure 2: Depth-frequency relationships
    figures['fig2_depth_frequency'] = create_depth_frequency_figure(
        df, str(output_path / 'Fig2_depth_frequency.png')
    )
    
    # Figure 3: Comprehensive summary
    figures['fig3_comprehensive'] = create_comprehensive_study_figure(
        df, str(output_path / 'Fig3_comprehensive_summary.png')
    )
    
    # Table 1: Statistics table
    figures['table1_statistics'] = create_statistics_table(
        df, str(output_path / 'Table1_statistics.png')
    )
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)
    
    return figures


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_all_study_figures(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python study_figures.py <results_csv> <output_dir>")
