"""
Complete end-to-end HVSR analysis workflow script.

This script demonstrates the full capability of the hvstrip-progressive package:
1. Batch processing of multiple soil profiles
2. H/V curve computation with adaptive frequency scanning
3. Comprehensive visualization and reporting
4. Statistical analysis and comparisons

Usage:
    python3 run_full_analysis.py [profiles_dir] [output_dir]
"""
import sys
from pathlib import Path
import time

# Add package to path
pkg_dir = Path(__file__).parent
sys.path.insert(0, str(pkg_dir))


def run_full_analysis(profiles_dir, output_base_dir):
    """Run complete end-to-end analysis workflow."""
    from hvstrip_progressive.core.batch_workflow import run_complete_workflow

    print("="*80)
    print("HVSR PROGRESSIVE LAYER STRIPPING - COMPLETE ANALYSIS WORKFLOW")
    print("="*80)
    print(f"📁 Profiles directory: {profiles_dir}")
    print(f"📁 Output base directory: {output_base_dir}")
    print()

    # Create output directory
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Find all profile files
    profiles_dir = Path(profiles_dir)
    profile_files = sorted(profiles_dir.glob("profile_*.txt"))

    if not profile_files:
        print(f"❌ No profile files found in {profiles_dir}")
        return 1

    print(f"Found {len(profile_files)} profiles to process\n")

    # =========================================================================
    # PHASE 1: BATCH PROCESSING
    # =========================================================================
    print("="*80)
    print("PHASE 1: BATCH PROCESSING OF ALL PROFILES")
    print("="*80)
    print()

    batch_results = []
    total_start = time.time()

    for i, profile_file in enumerate(profile_files, 1):
        profile_name = profile_file.stem
        profile_output = output_base_dir / profile_name

        print(f"\n[{i}/{len(profile_files)}] Processing {profile_name}")
        print("-"*80)

        try:
            result = run_complete_workflow(
                str(profile_file),
                str(profile_output)
            )

            batch_results.append({
                'name': profile_name,
                'success': result.get('success', False),
                'time': result.get('total_time', 0),
                'steps': result.get('summary', {}).get('total_steps', 0),
                'output_dir': profile_output
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            batch_results.append({
                'name': profile_name,
                'success': False,
                'error': str(e)
            })

    batch_time = time.time() - total_start
    successful_batch = sum(1 for r in batch_results if r['success'])

    print("\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    print(f"Profiles processed: {len(profile_files)}")
    print(f"Successful: {successful_batch}")
    print(f"Failed: {len(profile_files) - successful_batch}")
    print(f"Total time: {batch_time:.2f}s")
    print(f"Average time: {batch_time/len(profile_files):.2f}s per profile")

    # =========================================================================
    # PHASE 2: VISUALIZATION AND REPORTING
    # =========================================================================
    if successful_batch > 0:
        print("\n" + "="*80)
        print("PHASE 2: GENERATING VISUALIZATIONS AND REPORTS")
        print("="*80)
        print()

        viz_start = time.time()

        # Run visualization script if available
        viz_output = output_base_dir / "visualization_output"
        viz_output.mkdir(exist_ok=True)

        print("📊 Generating comparative visualizations...")
        print(f"   Output directory: {viz_output}")

        try:
            # Import visualization modules
            import numpy as np
            import matplotlib.pyplot as plt
            from hvstrip_progressive.core import hv_forward

            # Collect data from all successful profiles
            all_profiles_data = []

            for result in batch_results:
                if not result['success']:
                    continue

                profile_name = result['name']
                profile_file = profiles_dir / f"{profile_name}.txt"

                print(f"   Processing {profile_name}...")

                # Compute H/V for original profile
                freqs, amps = hv_forward.compute_hv_curve(str(profile_file))

                all_profiles_data.append({
                    'name': profile_name,
                    'freqs': np.array(freqs),
                    'amps': np.array(amps),
                    'peak_freq': freqs[np.argmax(amps)],
                    'peak_amp': max(amps)
                })

            # Create summary comparison plot
            print("\n   Creating summary comparison plot...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: All H/V curves
            colors = plt.cm.tab20(np.linspace(0, 1, len(all_profiles_data)))
            for i, data in enumerate(all_profiles_data):
                ax1.semilogx(data['freqs'], data['amps'],
                           color=colors[i], linewidth=2, alpha=0.7,
                           label=data['name'])

            ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('H/V Amplitude', fontsize=12, fontweight='bold')
            ax1.set_title('All Profiles H/V Curves Comparison', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, which='both')
            ax1.legend(loc='upper left', ncol=2, fontsize=8)

            # Plot 2: Peak comparison
            names = [d['name'] for d in all_profiles_data]
            peak_freqs = [d['peak_freq'] for d in all_profiles_data]
            peak_amps = [d['peak_amp'] for d in all_profiles_data]

            x = np.arange(len(names))
            width = 0.35

            bars1 = ax2.bar(x - width/2, peak_freqs, width, label='Peak Frequency (Hz)',
                          color='steelblue', alpha=0.7, edgecolor='black')
            ax2_twin = ax2.twinx()
            bars2 = ax2_twin.bar(x + width/2, peak_amps, width, label='Peak Amplitude',
                               color='coral', alpha=0.7, edgecolor='black')

            ax2.set_xlabel('Profile', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Peak Frequency (Hz)', fontsize=11, fontweight='bold', color='steelblue')
            ax2_twin.set_ylabel('Peak Amplitude', fontsize=11, fontweight='bold', color='coral')
            ax2.set_title('Peak Characteristics Comparison', fontsize=11, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='steelblue')
            ax2_twin.tick_params(axis='y', labelcolor='coral')
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            summary_plot = viz_output / "all_profiles_summary.png"
            plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"   ✅ Saved: {summary_plot.name}")

            # Generate text summary
            summary_file = viz_output / "analysis_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("HVSR PROGRESSIVE LAYER STRIPPING - ANALYSIS SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total profiles analyzed: {successful_batch}\n")
                f.write(f"Total processing time: {batch_time:.2f}s\n\n")

                f.write("-"*80 + "\n")
                f.write("INDIVIDUAL PROFILE RESULTS\n")
                f.write("-"*80 + "\n\n")

                for data in all_profiles_data:
                    f.write(f"Profile: {data['name']}\n")
                    f.write(f"  Peak Frequency: {data['peak_freq']:.2f} Hz\n")
                    f.write(f"  Peak Amplitude: {data['peak_amp']:.2f}\n\n")

                f.write("-"*80 + "\n")
                f.write("STATISTICAL SUMMARY\n")
                f.write("-"*80 + "\n\n")

                all_peak_freqs = [d['peak_freq'] for d in all_profiles_data]
                all_peak_amps = [d['peak_amp'] for d in all_profiles_data]

                f.write(f"Peak Frequency Statistics:\n")
                f.write(f"  Mean: {np.mean(all_peak_freqs):.2f} Hz\n")
                f.write(f"  Std Dev: {np.std(all_peak_freqs):.2f} Hz\n")
                f.write(f"  Range: {min(all_peak_freqs):.2f} - {max(all_peak_freqs):.2f} Hz\n\n")

                f.write(f"Peak Amplitude Statistics:\n")
                f.write(f"  Mean: {np.mean(all_peak_amps):.2f}\n")
                f.write(f"  Std Dev: {np.std(all_peak_amps):.2f}\n")
                f.write(f"  Range: {min(all_peak_amps):.2f} - {max(all_peak_amps):.2f}\n\n")

                f.write("="*80 + "\n")

            print(f"   ✅ Saved: {summary_file.name}")

            viz_time = time.time() - viz_start
            print(f"\n✅ Visualization phase completed in {viz_time:.2f}s")

        except Exception as e:
            print(f"⚠️  Visualization phase encountered errors: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - total_start

    print("\n" + "="*80)
    print("🎉 COMPLETE ANALYSIS WORKFLOW FINISHED!")
    print("="*80)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"   ├─ Batch processing: {batch_time:.2f}s")
    if successful_batch > 0:
        print(f"   └─ Visualization: {viz_time:.2f}s")
    print(f"\n📁 All outputs saved in: {output_base_dir}")
    print("\n📋 Generated outputs:")
    print(f"   • Individual profile results: {successful_batch} directories")
    print(f"   • Visualization summary: {viz_output}")
    print("="*80)

    return 0


def main():
    """Main entry point."""
    # Default paths
    default_profiles = pkg_dir / "hvstrip_progressive" / "Example" / "profiles"
    default_output = pkg_dir / "full_analysis_output"

    # Parse command line arguments
    if len(sys.argv) > 1:
        profiles_dir = Path(sys.argv[1])
    else:
        profiles_dir = default_profiles

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = default_output

    # Validate profiles directory
    if not profiles_dir.exists():
        print(f"❌ Error: Profiles directory not found: {profiles_dir}")
        print(f"\nUsage: python3 {Path(__file__).name} [profiles_dir] [output_dir]")
        return 1

    # Run the analysis
    return run_full_analysis(profiles_dir, output_dir)


if __name__ == "__main__":
    sys.exit(main())
