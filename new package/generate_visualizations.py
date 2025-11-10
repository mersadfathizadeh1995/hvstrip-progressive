"""
Comprehensive visualization generator for HVSR progressive layer stripping analysis.

Generates:
1. Individual profile plots showing all stripping steps
2. Comparative plots across all profiles
3. Summary statistics and reports
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Add package to path
pkg_dir = Path(__file__).parent
sys.path.insert(0, str(pkg_dir))

from hvstrip_progressive.core import hv_forward, stripper


def compute_profile_with_steps(profile_path, output_dir):
    """
    Compute H/V curves for a profile and all its stripping steps.

    Returns:
        dict with 'original' and 'steps' data
    """
    profile_name = profile_path.stem
    print(f"  Processing {profile_name}...")

    results = {
        'profile_name': profile_name,
        'profile_path': str(profile_path),
        'original': {},
        'steps': []
    }

    # Compute original H/V
    freqs, amps = hv_forward.compute_hv_curve(str(profile_path))
    results['original'] = {
        'freqs': np.array(freqs),
        'amps': np.array(amps),
        'peak_idx': np.argmax(amps),
        'peak_freq': freqs[np.argmax(amps)],
        'peak_amp': max(amps)
    }

    # Create stripped models and compute H/V for each
    profile_output_dir = output_dir / profile_name
    strip_dir = stripper.write_peel_sequence(str(profile_path), str(profile_output_dir))

    # Process each step
    step_dirs = sorted([d for d in strip_dir.iterdir() if d.is_dir()])

    for step_dir in step_dirs:
        step_name = step_dir.name
        model_files = list(step_dir.glob("*.txt"))

        if not model_files:
            continue

        model_file = model_files[0]

        try:
            step_freqs, step_amps = hv_forward.compute_hv_curve(str(model_file))

            # Extract layer count from step name (e.g., "Step0_7-layer" -> 7)
            layer_count = int(step_name.split('_')[1].split('-')[0])

            results['steps'].append({
                'step_name': step_name,
                'layer_count': layer_count,
                'freqs': np.array(step_freqs),
                'amps': np.array(step_amps),
                'peak_idx': np.argmax(step_amps),
                'peak_freq': step_freqs[np.argmax(step_amps)],
                'peak_amp': max(step_amps)
            })
        except Exception as e:
            print(f"    Warning: Failed to process {step_name}: {e}")

    return results


def plot_single_profile_analysis(profile_data, output_path):
    """
    Create a comprehensive plot for a single profile showing all stripping steps.
    """
    profile_name = profile_data['profile_name']

    # Set up the plot with publication-quality style
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Main plot: All H/V curves overlaid
    ax_main = fig.add_subplot(gs[0:2, :])

    # Color scheme
    n_steps = len(profile_data['steps'])
    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

    # Plot each step
    for i, step_data in enumerate(profile_data['steps']):
        label = f"{step_data['layer_count']}-layer"
        ax_main.semilogx(step_data['freqs'], step_data['amps'],
                        color=colors[i], linewidth=2, alpha=0.7, label=label)

        # Mark peak
        ax_main.plot(step_data['peak_freq'], step_data['peak_amp'],
                    'o', color=colors[i], markersize=8, markeredgecolor='black',
                    markeredgewidth=1)

    ax_main.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('H/V Amplitude', fontsize=12, fontweight='bold')
    ax_main.set_title(f'Progressive Layer Stripping Analysis: {profile_name}',
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, which='both')
    ax_main.legend(loc='upper right', ncol=2, framealpha=0.9)

    # Bottom left: Peak frequency evolution
    ax_freq = fig.add_subplot(gs[2, 0])
    layer_counts = [s['layer_count'] for s in profile_data['steps']]
    peak_freqs = [s['peak_freq'] for s in profile_data['steps']]

    ax_freq.plot(layer_counts, peak_freqs, 'o-', color='darkblue',
                linewidth=2, markersize=8, markerfacecolor='lightblue',
                markeredgecolor='darkblue', markeredgewidth=2)
    ax_freq.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
    ax_freq.set_ylabel('Peak Frequency (Hz)', fontsize=11, fontweight='bold')
    ax_freq.set_title('(a) Peak Frequency Evolution', fontsize=11)
    ax_freq.grid(True, alpha=0.3)
    ax_freq.invert_xaxis()  # Show layer removal progression

    # Bottom right: Peak amplitude evolution
    ax_amp = fig.add_subplot(gs[2, 1])
    peak_amps = [s['peak_amp'] for s in profile_data['steps']]

    ax_amp.plot(layer_counts, peak_amps, 'o-', color='darkred',
               linewidth=2, markersize=8, markerfacecolor='lightcoral',
               markeredgecolor='darkred', markeredgewidth=2)
    ax_amp.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
    ax_amp.set_ylabel('Peak Amplitude', fontsize=11, fontweight='bold')
    ax_amp.set_title('(b) Peak Amplitude Evolution', fontsize=11)
    ax_amp.grid(True, alpha=0.3)
    ax_amp.invert_xaxis()  # Show layer removal progression

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


def plot_all_profiles_comparison(all_profiles_data, output_path):
    """
    Create comparative plots across all profiles.
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    n_profiles = len(all_profiles_data)
    colors = plt.cm.tab20(np.linspace(0, 1, n_profiles))

    # Panel 1: Original H/V curves comparison
    ax1 = fig.add_subplot(gs[0, :])
    for i, profile_data in enumerate(all_profiles_data):
        orig = profile_data['original']
        label = profile_data['profile_name']
        ax1.semilogx(orig['freqs'], orig['amps'],
                    color=colors[i], linewidth=2, alpha=0.7, label=label)

    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('H/V Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Original H/V Curves - All Profiles', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', ncol=3, fontsize=9, framealpha=0.9)

    # Panel 2: Peak frequencies comparison
    ax2 = fig.add_subplot(gs[1, 0])
    profile_names = [p['profile_name'] for p in all_profiles_data]
    peak_freqs = [p['original']['peak_freq'] for p in all_profiles_data]

    bars = ax2.bar(range(len(profile_names)), peak_freqs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(profile_names)))
    ax2.set_xticklabels(profile_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Peak Frequency (Hz)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Peak Frequencies by Profile', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, peak_freqs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Panel 3: Peak amplitudes comparison
    ax3 = fig.add_subplot(gs[1, 1])
    peak_amps = [p['original']['peak_amp'] for p in all_profiles_data]

    bars = ax3.bar(range(len(profile_names)), peak_amps, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(profile_names)))
    ax3.set_xticklabels(profile_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Peak Amplitude', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Peak Amplitudes by Profile', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, peak_amps)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Panel 4: Layer count distribution
    ax4 = fig.add_subplot(gs[2, 0])
    layer_counts = [len(p['steps']) for p in all_profiles_data]

    ax4.hist(layer_counts, bins=range(min(layer_counts), max(layer_counts)+2),
            color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Layer Count Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Summary statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Calculate statistics
    avg_peak_freq = np.mean(peak_freqs)
    std_peak_freq = np.std(peak_freqs)
    avg_peak_amp = np.mean(peak_amps)
    std_peak_amp = np.std(peak_amps)

    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}

    Total Profiles: {n_profiles}

    Peak Frequency:
      Mean: {avg_peak_freq:.2f} Hz
      Std Dev: {std_peak_freq:.2f} Hz
      Range: {min(peak_freqs):.2f} - {max(peak_freqs):.2f} Hz

    Peak Amplitude:
      Mean: {avg_peak_amp:.2f}
      Std Dev: {std_peak_amp:.2f}
      Range: {min(peak_amps):.2f} - {max(peak_amps):.2f}

    Layer Counts:
      Range: {min(layer_counts)} - {max(layer_counts)} layers
      Mean: {np.mean(layer_counts):.1f} layers
    """

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Cross-Profile Comparison Analysis', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


def generate_summary_report(all_profiles_data, output_path):
    """
    Generate a text summary report of the analysis.
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HVSR PROGRESSIVE LAYER STRIPPING ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total Profiles Analyzed: {len(all_profiles_data)}\n")
        f.write(f"Analysis Date: {Path(output_path).stat().st_mtime}\n\n")

        f.write("-"*70 + "\n")
        f.write("INDIVIDUAL PROFILE RESULTS\n")
        f.write("-"*70 + "\n\n")

        for profile_data in all_profiles_data:
            name = profile_data['profile_name']
            orig = profile_data['original']
            steps = profile_data['steps']

            f.write(f"Profile: {name}\n")
            f.write(f"  Initial Layers: {steps[0]['layer_count'] if steps else 'N/A'}\n")
            f.write(f"  Original Peak: {orig['peak_amp']:.3f} @ {orig['peak_freq']:.2f} Hz\n")
            f.write(f"  Stripping Steps: {len(steps)}\n")

            if steps:
                f.write(f"\n  Layer Stripping Progression:\n")
                for step in steps:
                    f.write(f"    {step['layer_count']}-layer: "
                           f"Peak={step['peak_amp']:.3f} @ {step['peak_freq']:.2f} Hz\n")

            f.write("\n")

        # Summary statistics
        f.write("-"*70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n\n")

        peak_freqs = [p['original']['peak_freq'] for p in all_profiles_data]
        peak_amps = [p['original']['peak_amp'] for p in all_profiles_data]

        f.write(f"Peak Frequency Statistics:\n")
        f.write(f"  Mean: {np.mean(peak_freqs):.2f} Hz\n")
        f.write(f"  Std Dev: {np.std(peak_freqs):.2f} Hz\n")
        f.write(f"  Range: {min(peak_freqs):.2f} - {max(peak_freqs):.2f} Hz\n\n")

        f.write(f"Peak Amplitude Statistics:\n")
        f.write(f"  Mean: {np.mean(peak_amps):.2f}\n")
        f.write(f"  Std Dev: {np.std(peak_amps):.2f}\n")
        f.write(f"  Range: {min(peak_amps):.2f} - {max(peak_amps):.2f}\n\n")

        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")


def main():
    """Main visualization generation workflow."""
    print("="*70)
    print("HVSR Progressive Layer Stripping - Visualization Generator")
    print("="*70)

    # Setup paths
    profiles_dir = pkg_dir / "hvstrip_progressive" / "Example" / "profiles"
    work_dir = pkg_dir / "visualization_output"
    work_dir.mkdir(exist_ok=True)

    plots_dir = work_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if not profiles_dir.exists():
        print(f"\nError: Profiles directory not found: {profiles_dir}")
        print("Please run parse_soil_profiles.py first!")
        return 1

    # Get all profile files
    profile_files = sorted(profiles_dir.glob("profile_*.txt"))

    if not profile_files:
        print(f"\nError: No profile files found in {profiles_dir}")
        return 1

    print(f"\nFound {len(profile_files)} profiles")
    print(f"Output directory: {work_dir}\n")

    # Process all profiles
    print("Step 1: Computing H/V curves for all profiles and stripping steps...")
    print("-"*70)

    all_profiles_data = []
    for profile_file in profile_files:
        profile_data = compute_profile_with_steps(profile_file, work_dir)
        all_profiles_data.append(profile_data)

    print(f"\n✓ Processed {len(all_profiles_data)} profiles\n")

    # Generate individual profile plots
    print("Step 2: Generating individual profile visualizations...")
    print("-"*70)

    for profile_data in all_profiles_data:
        profile_name = profile_data['profile_name']
        output_path = plots_dir / f"{profile_name}_analysis.png"
        plot_single_profile_analysis(profile_data, output_path)
        print(f"  ✓ Created {output_path.name}")

    print(f"\n✓ Generated {len(all_profiles_data)} individual plots\n")

    # Generate comparison plot
    print("Step 3: Generating cross-profile comparison...")
    print("-"*70)

    comparison_path = plots_dir / "all_profiles_comparison.png"
    plot_all_profiles_comparison(all_profiles_data, comparison_path)
    print(f"  ✓ Created {comparison_path.name}\n")

    # Generate summary report
    print("Step 4: Generating summary report...")
    print("-"*70)

    report_path = work_dir / "analysis_summary.txt"
    generate_summary_report(all_profiles_data, report_path)
    print(f"  ✓ Created {report_path.name}\n")

    # Summary
    print("="*70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated Files:")
    print(f"  Individual plots: {len(all_profiles_data)} files in {plots_dir}")
    print(f"  Comparison plot: {comparison_path}")
    print(f"  Summary report: {report_path}")
    print(f"\nAll outputs saved to: {work_dir}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
