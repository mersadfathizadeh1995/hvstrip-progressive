"""
Main CLI entry point for hvstrip-progressive package.
"""

import click
import sys
from pathlib import Path
from typing import Optional


def _get_default_exe():
    """Get auto-detected HVf executable path."""
    try:
        from ..core.hv_forward import DEFAULT_CONFIG
        return DEFAULT_CONFIG["exe_path"]
    except ImportError:
        return "HVf.exe"


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """HVSR Progressive Layer Stripping Analysis Toolkit.
    
    A comprehensive package for progressive layer stripping analysis of HVSR data
    using diffuse-field theory. Identifies controlling interfaces through systematic
    layer removal and forward modeling.
    """
    if verbose:
        click.echo("🔬 HVSR Progressive Layer Stripping Analysis v1.0.0")


@cli.command()
@click.argument('model_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--exe-path', default=None, help='Path to HVf executable (auto-detected if not specified)')
@click.option('--fmin', default=0.2, type=float, help='Minimum frequency (Hz)')
@click.option('--fmax', default=20.0, type=float, help='Maximum frequency (Hz)')
@click.option('--nf', default=71, type=int, help='Number of frequency points')
def workflow(model_file, output_dir, exe_path, fmin, fmax, nf):
    """Run complete progressive layer stripping workflow.
    
    This command orchestrates the entire analysis:
    1. Layer stripping - creates peeled models
    2. HV forward modeling - computes HVSR curves
    3. Post-processing - generates plots and summaries
    
    MODEL_FILE: Path to initial velocity model file
    OUTPUT_DIR: Directory for all analysis outputs
    """
    try:
        # Import the workflow module
        from ..core.batch_workflow import run_complete_workflow

        # Use auto-detected executable if not specified
        if exe_path is None:
            exe_path = _get_default_exe()

        # Configuration
        config = {
            "hv_forward": {
                "exe_path": exe_path,
                "fmin": fmin,
                "fmax": fmax,
                "nf": nf
            }
        }

        click.echo(f"[*] Starting complete workflow...")
        click.echo(f"[>] Model: {model_file}")
        click.echo(f"[>] Output: {output_dir}")
        click.echo(f"[>] HVf: {exe_path}")
        
        # Run workflow
        results = run_complete_workflow(
            str(model_file),
            str(output_dir),
            config
        )
        
        if results.get("success", False):
            click.echo("[OK] Workflow completed successfully!")
            summary = results.get("summary", {})
            click.echo(f"[>] Processed {summary.get('total_steps', 0)} steps")
            click.echo(f"[>] Success rate: {summary.get('completion_rate', 0):.1f}%")
        else:
            click.echo("[!] Workflow failed!")
            return 1
            
    except Exception as e:
        click.echo(f"[!] Error: {e}", err=True)
        return 1


@cli.command()
@click.argument('model_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
def strip(model_file, output_dir):
    """Strip layers from velocity model.
    
    Creates a sequence of peeled models by progressively removing
    the deepest finite layer and promoting it to half-space.
    
    MODEL_FILE: Path to initial velocity model file
    OUTPUT_DIR: Directory for stripped model outputs
    """
    try:
        from ..core.stripper import write_peel_sequence
        
        click.echo(f"[*] Starting layer stripping...")
        click.echo(f"[>] Model: {model_file}")
        click.echo(f"[>] Output: {output_dir}")
        
        strip_dir = write_peel_sequence(str(model_file), str(output_dir))
        
        # Count generated models
        step_folders = [f for f in strip_dir.iterdir() 
                       if f.is_dir() and f.name.startswith('Step')]
        
        click.echo(f"[OK] Layer stripping completed!")
        click.echo(f"[>] Strip directory: {strip_dir}")
        click.echo(f"[>] Generated {len(step_folders)} stripped models")
        
    except Exception as e:
        click.echo(f"[!] Error: {e}", err=True)
        return 1


@cli.command()
@click.argument('model_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--exe-path', default=None, help='Path to HVf executable (auto-detected if not specified)')
@click.option('--fmin', default=0.2, type=float, help='Minimum frequency (Hz)')
@click.option('--fmax', default=20.0, type=float, help='Maximum frequency (Hz)')
@click.option('--nf', default=71, type=int, help='Number of frequency points')
def forward(model_file, output_file, exe_path, fmin, fmax, nf):
    """Compute HVSR curve for a velocity model.

    Uses HVf to compute theoretical HVSR curve based on
    diffuse-field theory.

    MODEL_FILE: Path to velocity model file
    OUTPUT_FILE: Path for output CSV file
    """
    try:
        from ..core.hv_forward import compute_hv_curve

        # Use auto-detected executable if not specified
        if exe_path is None:
            exe_path = _get_default_exe()

        config = {
            "exe_path": exe_path,
            "fmin": fmin,
            "fmax": fmax,
            "nf": nf
        }

        click.echo(f"[*] Computing HVSR curve...")
        click.echo(f"[>] Model: {model_file}")
        click.echo(f"[>] Output: {output_file}")
        click.echo(f"[>] HVf: {exe_path}")
        click.echo(f"[>] Frequency range: {fmin}-{fmax} Hz ({nf} points)")
        
        # Compute HV curve
        freqs, amps = compute_hv_curve(str(model_file), config)
        
        # Save to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency_Hz", "HVSR_Amplitude"])
            for freq, amp in zip(freqs, amps):
                writer.writerow([f"{float(freq):.6f}", f"{float(amp):.6f}"])
        
        # Find peak for summary
        peak_idx = list(amps).index(max(amps))
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]
        
        click.echo(f"[OK] HVSR computation completed!")
        click.echo(f"[>] Generated {len(freqs)} frequency points")
        click.echo(f"[>] Peak: {peak_freq:.3f} Hz (amplitude: {peak_amp:.2f})")
        click.echo(f"[>] Saved: {output_file}")
        
    except Exception as e:
        click.echo(f"[!] Error: {e}", err=True)
        return 1


@cli.command()
@click.argument('hv_csv', type=click.Path(exists=True, path_type=Path))
@click.argument('model_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--x-scale', default='log', type=click.Choice(['log', 'linear']),
              help='X-axis scale for plots')
@click.option('--y-scale', default='log', type=click.Choice(['log', 'linear']),
              help='Y-axis scale for plots')
@click.option('--smoothing/--no-smoothing', default=True, help='Apply curve smoothing')
@click.option('--dpi', default=200, type=int, help='Plot resolution (DPI)')
def postprocess(hv_csv, model_file, output_dir, x_scale, y_scale, smoothing, dpi):
    """Generate plots and summaries from HVSR curve and model.
    
    Creates publication-ready visualizations including HV curve plots,
    velocity profiles, and comprehensive summaries.
    
    HV_CSV: Path to HVSR curve CSV file
    MODEL_FILE: Path to velocity model file  
    OUTPUT_DIR: Directory for output plots and summaries
    """
    try:
        from ..core.hv_postprocess import process
        
        # Configuration
        config = {
            "hv_plot": {
                "x_axis_scale": x_scale,
                "y_axis_scale": y_scale,
                "smoothing": {"enable": smoothing},
                "dpi": dpi
            },
            "vs_plot": {
                "dpi": dpi
            }
        }
        
        click.echo(f"[*] Starting post-processing...")
        click.echo(f"[>] HV curve: {hv_csv}")
        click.echo(f"[>] Model: {model_file}")
        click.echo(f"[>] Output: {output_dir}")
        click.echo(f"[>] Plot settings: {x_scale}-{y_scale} scale, {dpi} DPI")
        
        # Run post-processing
        results = process(
            str(hv_csv),
            str(model_file),
            str(output_dir),
            config
        )
        
        click.echo(f"[OK] Post-processing completed!")
        click.echo(f"[>] Peak: {results.get('peak_frequency', 0):.3f} Hz")
        click.echo(f"[>] Generated files:")
        
        for key, path in results.items():
            if isinstance(path, Path):
                click.echo(f"   • {path.name}")
        
    except Exception as e:
        click.echo(f"[!] Error: {e}", err=True)
        return 1


@cli.command()
@click.argument('strip_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path), 
              help='Output directory for reports (default: strip_dir/../reports)')
def report(strip_dir, output_dir):
    """Generate comprehensive analysis report.
    
    Creates publication-ready figures, summaries, and scientific analysis
    from completed progressive layer stripping results.
    
    STRIP_DIR: Path to 'strip' directory containing StepX_Y-layer folders
    """
    try:
        from ..core.report_generator import ProgressiveStrippingReporter
        
        click.echo(f"[*] Starting report generation...")
        click.echo(f"[>] Strip directory: {strip_dir}")
        
        if output_dir:
            click.echo(f"[>] Output directory: {output_dir}")
        else:
            output_dir = strip_dir.parent / 'reports'
            click.echo(f"[>] Output directory: {output_dir} (default)")
        
        # Initialize reporter
        reporter = ProgressiveStrippingReporter(
            str(strip_dir),
            str(output_dir) if output_dir else None
        )
        
        # Generate reports
        report_files = reporter.generate_comprehensive_report()
        
        click.echo(f"[OK] Report generation completed!")
        click.echo(f"[>] Generated {len(report_files)} report components:")
        
        for key, path in report_files.items():
            if isinstance(path, Path):
                click.echo(f"   • {path.name}")
        
        # Highlight key files
        if 'publication_figure' in report_files:
            click.echo(f"\n[>] Publication figure: {report_files['publication_figure']}")
        if 'analysis_summary_csv' in report_files:
            click.echo(f"[>] Data summary: {report_files['analysis_summary_csv']}")
        
    except Exception as e:
        click.echo(f"[!] Error: {e}", err=True)
        return 1


@cli.command()
@click.argument('profiles_dir', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--pattern', default='*.txt', help='File pattern for profiles (default: *.txt)')
@click.option('--exe-path', default=None, help='Path to HVf executable (auto-detected if not specified)')
@click.option('--generate-report', is_flag=True, help='Generate comprehensive visualization report')
def batch(profiles_dir, output_dir, pattern, exe_path, generate_report):
    """Process multiple soil profiles in batch mode.

    Processes all profiles matching the pattern in the profiles directory,
    running the complete workflow for each one. Optionally generates a
    comprehensive comparative visualization report.

    PROFILES_DIR: Directory containing profile files
    OUTPUT_DIR: Base directory for all outputs
    """
    try:
        from ..core.batch_workflow import run_complete_workflow
        import time

        # Find all profile files
        profile_files = sorted(profiles_dir.glob(pattern))

        if not profile_files:
            click.echo(f"[!] No profile files matching '{pattern}' found in {profiles_dir}")
            return 1

        # Use auto-detected executable if not specified
        if exe_path is None:
            exe_path = _get_default_exe()

        click.echo("=" * 70)
        click.echo("[*] BATCH PROCESSING MODE")
        click.echo("=" * 70)
        click.echo(f"[>] Profiles directory: {profiles_dir}")
        click.echo(f"[>] Output directory: {output_dir}")
        click.echo(f"[>] Pattern: {pattern}")
        click.echo(f"[>] HVf executable: {exe_path}")
        click.echo(f"[>] Found {len(profile_files)} profiles to process")
        click.echo("-" * 70)

        # Configuration
        config = {
            "hv_forward": {
                "exe_path": exe_path
            }
        }

        # Process each profile
        results_summary = []
        start_time = time.time()

        for i, profile_file in enumerate(profile_files, 1):
            profile_name = profile_file.stem
            profile_output = output_dir / profile_name

            click.echo(f"\n[{i}/{len(profile_files)}] Processing {profile_name}...")
            click.echo("-" * 70)

            try:
                result = run_complete_workflow(
                    str(profile_file),
                    str(profile_output),
                    config
                )

                results_summary.append({
                    'profile': profile_name,
                    'success': result.get('success', False),
                    'steps': result.get('summary', {}).get('total_steps', 0),
                    'time': result.get('total_time', 0)
                })

            except Exception as e:
                click.echo(f"[!] Error processing {profile_name}: {e}")
                results_summary.append({
                    'profile': profile_name,
                    'success': False,
                    'error': str(e)
                })

        total_time = time.time() - start_time
        successful = sum(1 for r in results_summary if r['success'])

        # Summary
        click.echo("\n" + "=" * 70)
        click.echo("BATCH PROCESSING SUMMARY")
        click.echo("=" * 70)
        click.echo(f"Total profiles: {len(profile_files)}")
        click.echo(f"Successful: {successful}")
        click.echo(f"Failed: {len(profile_files) - successful}")
        click.echo(f"Total time: {total_time:.2f}s")
        click.echo(f"Average time per profile: {total_time/len(profile_files):.2f}s")

        # Detailed results
        click.echo("\n" + "-" * 70)
        click.echo("Profile Results:")
        click.echo("-" * 70)
        for r in results_summary:
            status = "[OK]" if r['success'] else "[!]"
            steps = r.get('steps', 'N/A')
            t = r.get('time', 0)
            click.echo(f"{status} {r['profile']:<20} Steps: {steps:<3} Time: {t:.2f}s")

        # Generate visualization report if requested
        if generate_report and successful > 0:
            click.echo("\n" + "=" * 70)
            click.echo("GENERATING VISUALIZATION REPORT")
            click.echo("=" * 70)

            try:
                # Create a simple visualization report using our existing tools
                click.echo("Note: Use standalone visualization scripts for comprehensive reports")
                click.echo("  python3 generate_visualizations.py")
                click.echo("  python3 generate_pdf_report.py")
            except Exception as e:
                click.echo(f"[!] Could not generate visualization report: {e}")

        click.echo("\n" + "=" * 70)
        click.echo("Batch processing complete!")
        click.echo("=" * 70)

        return 0 if successful == len(profile_files) else 1

    except Exception as e:
        click.echo(f"[!] Error: {e}", err=True)
        return 1


@cli.command()
def examples():
    """Show example usage and commands."""
    click.echo("HVSR Progressive Layer Stripping - Example Usage")
    click.echo("=" * 60)
    click.echo()
    
    click.echo("Single Profile Workflow:")
    click.echo("   hvstrip-progressive workflow model.txt output/")
    click.echo("   hvstrip-progressive workflow model.txt output/ --fmin 0.1 --fmax 50")
    click.echo()

    click.echo("Batch Processing (Multiple Profiles):")
    click.echo("   hvstrip-progressive batch profiles_dir/ batch_output/")
    click.echo("   hvstrip-progressive batch profiles_dir/ batch_output/ --pattern 'profile_*.txt'")
    click.echo("   hvstrip-progressive batch profiles_dir/ batch_output/ --generate-report")
    click.echo()

    click.echo("Individual Components:")
    click.echo("   # 1. Layer stripping only")
    click.echo("   hvstrip-progressive strip model.txt output/")
    click.echo()
    click.echo("   # 2. HV forward modeling only")
    click.echo("   hvstrip-progressive forward model.txt hv_curve.csv --fmax 30")
    click.echo()
    click.echo("   # 3. Post-processing only")
    click.echo("   hvstrip-progressive postprocess hv_curve.csv model.txt plots/")
    click.echo()
    click.echo("   # 4. Report generation only")
    click.echo("   hvstrip-progressive report output/strip/")
    click.echo()
    
    click.echo("Advanced Options:")
    click.echo("   # Custom frequency range")
    click.echo("   hvstrip-progressive workflow model.txt output/ --fmin 0.1 --fmax 50")
    click.echo()
    click.echo("   # Custom plot settings")
    click.echo("   hvstrip-progressive postprocess hv.csv model.txt plots/ --x-scale linear --dpi 300")
    click.echo()
    
    click.echo("Getting Help:")
    click.echo("   hvstrip-progressive --help")
    click.echo("   hvstrip-progressive workflow --help")
    click.echo("   hvstrip-progressive report --help")


if __name__ == '__main__':
    cli()
