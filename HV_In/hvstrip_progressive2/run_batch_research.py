"""
Batch Research Runner - Complete workflow for Two Resonance Separation study.

This script runs the complete analysis on multiple profiles and generates
all statistical outputs for publication.

Usage:
    python run_batch_research.py PROFILES_DIR OUTPUT_DIR [--fmin 0.1] [--fmax 30]
    
Example:
    python run_batch_research.py ./profiles/txt ./results --fmin 0.1 --fmax 30
"""

import sys
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from hvstrip_progressive.core.research_workflow import run_batch_analysis
from hvstrip_progressive.visualization.statistics_plots import generate_all_statistics_plots
from hvstrip_progressive.visualization.special_plots import generate_resonance_separation_figure


def run_complete_research(profiles_dir: str, output_dir: str, 
                         fmin: float = 0.1, fmax: float = 30.0,
                         n_examples: int = 3) -> dict:
    """
    Run the complete research workflow.
    
    Args:
        profiles_dir: Directory containing .txt profile files
        output_dir: Base output directory
        fmin: Minimum frequency for HVSR calculation
        fmax: Maximum frequency for HVSR calculation
        n_examples: Number of example figures to generate
        
    Returns:
        Dict with all results and paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("TWO RESONANCE SEPARATION - COMPLETE RESEARCH WORKFLOW")
    print("="*70)
    
    # Workflow configuration
    workflow_config = {
        "hv_forward": {
            "fmin": fmin,
            "fmax": fmax,
            "nf": 100  # Higher resolution
        },
        "hv_postprocess": {
            "hv_plot": {
                "y_axis_scale": "linear",
                "smoothing": {"enable": True, "window_length": 7, "poly_order": 3}
            }
        }
    }
    
    # Step 1: Run batch analysis
    print("\n" + "-"*70)
    print("STEP 1: Running Batch Analysis")
    print("-"*70)
    
    batch_result = run_batch_analysis(
        profiles_dir=profiles_dir,
        output_dir=str(output_path / "individual_results"),
        workflow_config=workflow_config
    )
    
    if not batch_result['success']:
        print(f"❌ Batch analysis failed: {batch_result.get('error', 'Unknown error')}")
        return batch_result
    
    # Step 2: Generate statistical plots
    print("\n" + "-"*70)
    print("STEP 2: Generating Statistical Plots")
    print("-"*70)
    
    stats_output = output_path / "statistics"
    stats_output.mkdir(exist_ok=True)
    
    plot_paths = generate_all_statistics_plots(
        results_csv=batch_result['results_file'],
        stats_json=batch_result['stats_file'],
        output_dir=str(stats_output)
    )
    
    # Step 3: Generate example resonance separation figures
    print("\n" + "-"*70)
    print(f"STEP 3: Generating {n_examples} Example Figures")
    print("-"*70)
    
    examples_output = output_path / "example_figures"
    examples_output.mkdir(exist_ok=True)
    
    # Select profiles with best separation for examples
    successful_results = [r for r in batch_result['results'] if r.success]
    # Sort by frequency ratio (best separation first)
    successful_results.sort(key=lambda x: x.freq_ratio, reverse=True)
    
    example_paths = []
    for i, result in enumerate(successful_results[:n_examples]):
        strip_dir = Path(batch_result['output_dir']) / result.profile_name / "strip"
        if strip_dir.exists():
            output_fig = examples_output / f"example_{i+1}_{result.profile_name}.png"
            try:
                generate_resonance_separation_figure(str(strip_dir), str(output_fig))
                example_paths.append(str(output_fig))
                print(f"  ✓ Example {i+1}: {result.profile_name} (ratio={result.freq_ratio:.2f})")
            except Exception as e:
                print(f"  ✗ Failed for {result.profile_name}: {e}")
    
    # Step 4: Copy best results to main output
    print("\n" + "-"*70)
    print("STEP 4: Organizing Final Outputs")
    print("-"*70)
    
    # Copy CSV and JSON to main output
    import shutil
    shutil.copy(batch_result['results_file'], output_path / "batch_results.csv")
    shutil.copy(batch_result['stats_file'], output_path / "batch_statistics.json")
    
    # Generate summary report
    summary_path = output_path / "RESEARCH_SUMMARY.txt"
    generate_summary_report(batch_result, summary_path)
    
    print(f"\n✓ Results CSV: {output_path / 'batch_results.csv'}")
    print(f"✓ Statistics JSON: {output_path / 'batch_statistics.json'}")
    print(f"✓ Summary Report: {summary_path}")
    print(f"✓ Statistical Plots: {stats_output}")
    print(f"✓ Example Figures: {examples_output}")
    
    print("\n" + "="*70)
    print("RESEARCH WORKFLOW COMPLETE")
    print("="*70)
    
    return {
        "success": True,
        "batch_result": batch_result,
        "plot_paths": plot_paths,
        "example_paths": example_paths,
        "output_dir": str(output_path)
    }


def generate_summary_report(batch_result: dict, output_path: Path):
    """Generate a text summary report."""
    stats = batch_result['statistics']
    
    report = f"""
================================================================================
TWO RESONANCE SEPARATION ANALYSIS - SUMMARY REPORT
================================================================================
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
--------------------------------------------------------------------------------
Total Profiles Analyzed:    {stats.n_profiles}
Successfully Processed:     {stats.n_successful} ({stats.success_rate:.1f}%)

FREQUENCY STATISTICS
--------------------------------------------------------------------------------
Deep Resonance (f₀):
  Mean:     {stats.f0_mean:.3f} Hz
  Std Dev:  {stats.f0_std:.3f} Hz
  Range:    {stats.f0_min:.3f} - {stats.f0_max:.3f} Hz

Shallow Resonance (f₁):
  Mean:     {stats.f1_mean:.3f} Hz
  Std Dev:  {stats.f1_std:.3f} Hz
  Range:    {stats.f1_min:.3f} - {stats.f1_max:.3f} Hz

Frequency Ratio (f₁/f₀):
  Mean:     {stats.freq_ratio_mean:.3f}
  Std Dev:  {stats.freq_ratio_std:.3f}

VALIDATION METRICS
--------------------------------------------------------------------------------
f₀ Theoretical Correlation:  R = {stats.f0_theoretical_correlation:.3f}
f₁ Theoretical Correlation:  R = {stats.f1_theoretical_correlation:.3f}

METHOD PERFORMANCE
--------------------------------------------------------------------------------
Separation Success Rate:     {stats.separation_success_rate:.1f}%
(Profiles where f₁/f₀ > 1.2 and clear frequency shift observed)

INTERPRETATION
--------------------------------------------------------------------------------
The progressive layer stripping method successfully separated deep and shallow
resonance peaks in {stats.separation_success_rate:.1f}% of analyzed profiles.

The mean frequency ratio of {stats.freq_ratio_mean:.2f} indicates that shallow
resonance frequencies are on average {stats.freq_ratio_mean:.1f}x higher than
deep resonance frequencies, consistent with the theoretical expectation that
shallow layers produce higher fundamental frequencies.

================================================================================
"""
    
    with open(output_path, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Two Resonance Separation research workflow"
    )
    parser.add_argument("profiles_dir", help="Directory containing .txt profile files")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--fmin", type=float, default=0.1, help="Min frequency (Hz)")
    parser.add_argument("--fmax", type=float, default=30.0, help="Max frequency (Hz)")
    parser.add_argument("--examples", type=int, default=3, help="Number of example figures")
    
    args = parser.parse_args()
    
    run_complete_research(
        profiles_dir=args.profiles_dir,
        output_dir=args.output_dir,
        fmin=args.fmin,
        fmax=args.fmax,
        n_examples=args.examples
    )


if __name__ == "__main__":
    main()
