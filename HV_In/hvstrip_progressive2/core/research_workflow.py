"""
Research Workflow Module for Two Resonance Separation Analysis.

This module provides a complete workflow for batch processing soil profiles,
collecting statistics, and generating publication-ready outputs.

Author: Mersad Fathizadeh
"""

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .batch_workflow import run_complete_workflow
from .hv_postprocess import read_hv_csv, read_model, detect_peak, DEFAULT_CONFIG


@dataclass
class ProfileResult:
    """Results from analyzing a single profile."""
    profile_name: str
    profile_path: str
    success: bool
    
    # Layer info
    n_layers: int
    total_depth: float
    layer_thicknesses: List[float]
    layer_vs: List[float]
    
    # Original model results
    f0_original: float  # Peak frequency of full model
    a0_original: float  # Peak amplitude of full model
    f0_theoretical: float  # Vs_avg / (4 * H_total)
    
    # After stripping (shallow resonance)
    f1_shallow: float  # Peak after removing deepest layer
    a1_shallow: float
    f1_theoretical: float  # Vs_top / (4 * H_top)
    
    # Frequency changes per stripping step
    freq_per_step: List[float]
    amp_per_step: List[float]
    
    # Derived metrics
    freq_ratio: float  # f1/f0 - should be > 1 if separation works
    max_freq_shift: float  # Largest Δf between steps
    controlling_step: int  # Step with largest frequency shift
    separation_success: bool  # True if clear separation achieved
    
    # Error message if failed
    error_message: str = ""


@dataclass  
class BatchStatistics:
    """Aggregate statistics from batch analysis."""
    n_profiles: int
    n_successful: int
    success_rate: float
    
    # Frequency statistics
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    
    f1_mean: float
    f1_std: float
    f1_min: float
    f1_max: float
    
    # Ratio statistics
    freq_ratio_mean: float
    freq_ratio_std: float
    
    # Theoretical correlation
    f0_theoretical_correlation: float
    f1_theoretical_correlation: float
    
    # Separation success rate
    separation_success_rate: float
    
    # Step-wise statistics
    mean_freq_shift_per_step: List[float]
    std_freq_shift_per_step: List[float]


def calculate_theoretical_frequency(layers: List[Dict]) -> Tuple[float, float]:
    """
    Calculate theoretical fundamental frequency using f = Vs_avg / (4*H).
    
    Returns:
        (f0_full, f1_shallow): Theoretical frequencies for full and top layer
    """
    # Full model: average Vs over total depth
    total_thickness = sum(l['thickness'] for l in layers if l['thickness'] > 0)
    if total_thickness <= 0:
        return 0.0, 0.0
    
    # Time-averaged Vs
    travel_time = sum(l['thickness'] / l['vs'] for l in layers if l['thickness'] > 0)
    vs_avg = total_thickness / travel_time if travel_time > 0 else 0
    
    f0_full = vs_avg / (4 * total_thickness) if total_thickness > 0 else 0
    
    # Top layer only
    if layers and layers[0]['thickness'] > 0:
        f1_shallow = layers[0]['vs'] / (4 * layers[0]['thickness'])
    else:
        f1_shallow = 0.0
    
    return f0_full, f1_shallow


def analyze_single_profile(profile_path: Path, output_dir: Path, 
                          workflow_config: Optional[Dict] = None) -> ProfileResult:
    """
    Run complete analysis on a single profile and extract statistics.
    """
    profile_name = profile_path.stem
    
    # Default result for failures
    default_result = ProfileResult(
        profile_name=profile_name,
        profile_path=str(profile_path),
        success=False,
        n_layers=0,
        total_depth=0,
        layer_thicknesses=[],
        layer_vs=[],
        f0_original=0,
        a0_original=0,
        f0_theoretical=0,
        f1_shallow=0,
        a1_shallow=0,
        f1_theoretical=0,
        freq_per_step=[],
        amp_per_step=[],
        freq_ratio=0,
        max_freq_shift=0,
        controlling_step=0,
        separation_success=False,
        error_message=""
    )
    
    try:
        # Run the workflow
        result = run_complete_workflow(
            initial_model_path=str(profile_path),
            output_base_dir=str(output_dir),
            workflow_config=workflow_config
        )
        
        if not result['success']:
            default_result.error_message = "Workflow failed"
            return default_result
        
        strip_dir = Path(result['strip_directory'])
        
        # Find all step folders
        step_folders = sorted(strip_dir.glob("Step*_*-layer"))
        if not step_folders:
            default_result.error_message = "No step folders found"
            return default_result
        
        # Read original model
        step0_dir = step_folders[0]
        model_file = step0_dir / f"model_{step0_dir.name}.txt"
        if not model_file.exists():
            default_result.error_message = f"Model file not found: {model_file}"
            return default_result
            
        model = read_model(model_file)
        layers = model['layers']
        
        # Extract layer info
        layer_thicknesses = [l['thickness'] for l in layers]
        layer_vs = [l['vs'] for l in layers]
        total_depth = sum(t for t in layer_thicknesses if t > 0)
        
        # Calculate theoretical frequencies
        f0_theo, f1_theo = calculate_theoretical_frequency(layers)
        
        # Collect frequencies and amplitudes for each step
        freq_per_step = []
        amp_per_step = []
        
        for step_folder in step_folders:
            hv_csv = step_folder / "hv_curve.csv"
            if hv_csv.exists():
                freqs, amps = read_hv_csv(hv_csv)
                f_peak, a_peak, _ = detect_peak(freqs, amps, DEFAULT_CONFIG)
                freq_per_step.append(f_peak)
                amp_per_step.append(a_peak)
        
        if len(freq_per_step) < 2:
            default_result.error_message = "Not enough steps for analysis"
            return default_result
        
        # Original (Step 0) results
        f0_original = freq_per_step[0]
        a0_original = amp_per_step[0]
        
        # After first strip (Step 1) - shallow resonance
        f1_shallow = freq_per_step[1] if len(freq_per_step) > 1 else freq_per_step[0]
        a1_shallow = amp_per_step[1] if len(amp_per_step) > 1 else amp_per_step[0]
        
        # Calculate frequency shifts between steps
        freq_shifts = []
        for i in range(1, len(freq_per_step)):
            shift = abs(freq_per_step[i] - freq_per_step[i-1])
            freq_shifts.append(shift)
        
        max_freq_shift = max(freq_shifts) if freq_shifts else 0
        controlling_step = freq_shifts.index(max_freq_shift) + 1 if freq_shifts else 0
        
        # Frequency ratio (shallow/deep)
        freq_ratio = f1_shallow / f0_original if f0_original > 0 else 0
        
        # Separation success: f1 should be higher than f0 (shallow resonates at higher freq)
        # and the ratio should be meaningful (> 1.2 for clear separation)
        separation_success = freq_ratio > 1.2 and max_freq_shift > 0.3
        
        return ProfileResult(
            profile_name=profile_name,
            profile_path=str(profile_path),
            success=True,
            n_layers=len(layers),
            total_depth=total_depth,
            layer_thicknesses=layer_thicknesses,
            layer_vs=layer_vs,
            f0_original=f0_original,
            a0_original=a0_original,
            f0_theoretical=f0_theo,
            f1_shallow=f1_shallow,
            a1_shallow=a1_shallow,
            f1_theoretical=f1_theo,
            freq_per_step=freq_per_step,
            amp_per_step=amp_per_step,
            freq_ratio=freq_ratio,
            max_freq_shift=max_freq_shift,
            controlling_step=controlling_step,
            separation_success=separation_success,
            error_message=""
        )
        
    except Exception as e:
        default_result.error_message = str(e)
        return default_result


def compute_batch_statistics(results: List[ProfileResult]) -> BatchStatistics:
    """Compute aggregate statistics from batch results."""
    successful = [r for r in results if r.success]
    n_successful = len(successful)
    
    if n_successful == 0:
        return BatchStatistics(
            n_profiles=len(results),
            n_successful=0,
            success_rate=0,
            f0_mean=0, f0_std=0, f0_min=0, f0_max=0,
            f1_mean=0, f1_std=0, f1_min=0, f1_max=0,
            freq_ratio_mean=0, freq_ratio_std=0,
            f0_theoretical_correlation=0,
            f1_theoretical_correlation=0,
            separation_success_rate=0,
            mean_freq_shift_per_step=[],
            std_freq_shift_per_step=[]
        )
    
    # Extract arrays
    f0_values = np.array([r.f0_original for r in successful])
    f1_values = np.array([r.f1_shallow for r in successful])
    f0_theo = np.array([r.f0_theoretical for r in successful])
    f1_theo = np.array([r.f1_theoretical for r in successful])
    ratios = np.array([r.freq_ratio for r in successful])
    
    # Correlations
    f0_corr = np.corrcoef(f0_values, f0_theo)[0, 1] if len(f0_values) > 1 else 0
    f1_corr = np.corrcoef(f1_values, f1_theo)[0, 1] if len(f1_values) > 1 else 0
    
    # Handle NaN correlations
    f0_corr = f0_corr if not np.isnan(f0_corr) else 0
    f1_corr = f1_corr if not np.isnan(f1_corr) else 0
    
    # Step-wise frequency shifts
    max_steps = max(len(r.freq_per_step) for r in successful)
    freq_shifts_per_step = [[] for _ in range(max_steps - 1)]
    
    for r in successful:
        for i in range(1, len(r.freq_per_step)):
            if i - 1 < len(freq_shifts_per_step):
                shift = r.freq_per_step[i] - r.freq_per_step[i-1]
                freq_shifts_per_step[i-1].append(shift)
    
    mean_shifts = [np.mean(s) if s else 0 for s in freq_shifts_per_step]
    std_shifts = [np.std(s) if s else 0 for s in freq_shifts_per_step]
    
    separation_count = sum(1 for r in successful if r.separation_success)
    
    return BatchStatistics(
        n_profiles=len(results),
        n_successful=n_successful,
        success_rate=n_successful / len(results) * 100 if results else 0,
        f0_mean=float(np.mean(f0_values)),
        f0_std=float(np.std(f0_values)),
        f0_min=float(np.min(f0_values)),
        f0_max=float(np.max(f0_values)),
        f1_mean=float(np.mean(f1_values)),
        f1_std=float(np.std(f1_values)),
        f1_min=float(np.min(f1_values)),
        f1_max=float(np.max(f1_values)),
        freq_ratio_mean=float(np.mean(ratios)),
        freq_ratio_std=float(np.std(ratios)),
        f0_theoretical_correlation=float(f0_corr),
        f1_theoretical_correlation=float(f1_corr),
        separation_success_rate=separation_count / n_successful * 100 if n_successful else 0,
        mean_freq_shift_per_step=mean_shifts,
        std_freq_shift_per_step=std_shifts
    )


def run_batch_analysis(profiles_dir: str, output_dir: str,
                      workflow_config: Optional[Dict] = None,
                      progress_callback=None) -> Dict:
    """
    Run batch analysis on all profiles in a directory.
    
    Args:
        profiles_dir: Directory containing .txt profile files
        output_dir: Base output directory
        workflow_config: Optional workflow configuration
        progress_callback: Optional callback(current, total, profile_name)
        
    Returns:
        Dict with results, statistics, and output paths
    """
    profiles_path = Path(profiles_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all profile files
    profile_files = sorted(profiles_path.glob("*.txt"))
    if not profile_files:
        return {"success": False, "error": "No .txt files found in profiles directory"}
    
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS - Two Resonance Separation")
    print(f"{'='*60}")
    print(f"Profiles directory: {profiles_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of profiles: {len(profile_files)}")
    print(f"{'='*60}\n")
    
    # Process each profile
    results: List[ProfileResult] = []
    
    for i, profile_file in enumerate(profile_files):
        print(f"\n[{i+1}/{len(profile_files)}] Processing: {profile_file.name}")
        
        if progress_callback:
            progress_callback(i + 1, len(profile_files), profile_file.name)
        
        profile_output = output_path / profile_file.stem
        result = analyze_single_profile(profile_file, profile_output, workflow_config)
        results.append(result)
        
        if result.success:
            print(f"  [OK] f0={result.f0_original:.2f} Hz, f1={result.f1_shallow:.2f} Hz, "
                  f"ratio={result.freq_ratio:.2f}")
        else:
            print(f"  [X] Failed: {result.error_message}")
    
    # Compute statistics
    print(f"\n{'='*60}")
    print("Computing batch statistics...")
    statistics = compute_batch_statistics(results)
    
    # Save results
    results_file = output_path / "batch_results.csv"
    stats_file = output_path / "batch_statistics.json"
    
    # Save individual results to CSV
    save_results_csv(results, results_file)
    
    # Save statistics to JSON
    with open(stats_file, 'w') as f:
        json.dump(asdict(statistics), f, indent=2)
    
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {statistics.n_profiles} profiles")
    print(f"Successful: {statistics.n_successful} ({statistics.success_rate:.1f}%)")
    print(f"\nFrequency Statistics:")
    print(f"  f0 (deep):    {statistics.f0_mean:.2f} +/- {statistics.f0_std:.2f} Hz")
    print(f"  f1 (shallow): {statistics.f1_mean:.2f} +/- {statistics.f1_std:.2f} Hz")
    print(f"  Ratio f1/f0:  {statistics.freq_ratio_mean:.2f} +/- {statistics.freq_ratio_std:.2f}")
    print(f"\nSeparation Success Rate: {statistics.separation_success_rate:.1f}%")
    print(f"\nResults saved to: {results_file}")
    print(f"Statistics saved to: {stats_file}")
    print(f"{'='*60}\n")
    
    return {
        "success": True,
        "results": results,
        "statistics": statistics,
        "results_file": str(results_file),
        "stats_file": str(stats_file),
        "output_dir": str(output_path)
    }


def save_results_csv(results: List[ProfileResult], output_path: Path):
    """Save results to CSV file."""
    rows = []
    for r in results:
        rows.append({
            "profile_name": r.profile_name,
            "success": r.success,
            "n_layers": r.n_layers,
            "total_depth_m": r.total_depth,
            "f0_original_Hz": r.f0_original,
            "a0_original": r.a0_original,
            "f0_theoretical_Hz": r.f0_theoretical,
            "f1_shallow_Hz": r.f1_shallow,
            "a1_shallow": r.a1_shallow,
            "f1_theoretical_Hz": r.f1_theoretical,
            "freq_ratio": r.freq_ratio,
            "max_freq_shift_Hz": r.max_freq_shift,
            "controlling_step": r.controlling_step,
            "separation_success": r.separation_success,
            "error": r.error_message
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


# Export
__all__ = [
    "ProfileResult",
    "BatchStatistics", 
    "analyze_single_profile",
    "compute_batch_statistics",
    "run_batch_analysis",
    "calculate_theoretical_frequency"
]
