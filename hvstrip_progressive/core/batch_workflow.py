"""
Batch Workflow Module
====================

Orchestrates the complete progressive layer stripping analysis workflow:
1. Layer stripping (stripper.py) - Creates peeled model files
2. HV forward modeling (hv_forward.py) - Computes HV curves for each model
3. Post-processing (hv_postprocess.py) - Generates plots and summaries

This module coordinates all three core modules in the correct sequential order.

Author: HVSR-Diffuse Development Team
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import time


# Default configuration for the complete workflow
DEFAULT_WORKFLOW_CONFIG = {
    # Stripper configuration
    "stripper": {
        "output_folder_name": "strip"
    },
    
    # HV Forward modeling configuration
    "hv_forward": {
        "exe_path": "HVf.exe",
        "fmin": 0.2,
        "fmax": 20.0,
        "nf": 71,
        "nmr": 10,
        "nml": 10,
        "nks": 10,
        # Adaptive frequency scanning to avoid boundary misses
        "adaptive": {
            "enable": True,
            "max_passes": 2,
            "edge_margin_frac": 0.05,   # consider within 5% of bounds as edge
            "fmax_expand_factor": 2.0,
            "fmin_shrink_factor": 0.5,
            "fmax_limit": 60.0,
            "fmin_limit": 0.05,
        },
    },
    
    # Post-processing configuration
    "hv_postprocess": {
        "peak_detection": {
            "method": "find_peaks",
            "select": "leftmost",
            "find_peaks_params": {"prominence": 0.2, "distance": 3},
            "freq_min": 0.5,          # guard against edge picks at 0.2 Hz
            "min_rel_height": 0.25,   # keep peaks with >=25% of global max
            "exclude_first_n": 1      # ignore the very first frequency bin
        },
        "hv_plot": {
            "x_axis_scale": "log",
            "y_axis_scale": "log",
            "y_compression": 1.5,
            "smoothing": {
                "enable": True,
                "window_length": 9,
                "poly_order": 3
            },
            "show_bands": True,
            "freq_window_left": 0.3,
            "freq_window_right": 3.0,
            "figure_width": 12,
            "figure_height": 6,
            "dpi": 200,
        },
        "vs_plot": {
            "show": True,
            "annotate_deepest": True,
            "annotate_max_vs": True,
            "annotate_f0": True,
            "figure_width": 6,
            "figure_height": 8,
            "dpi": 200,
        },
        "output": {
            "save_separate": True,
            "save_combined": True,
            "hv_filename": "hv_curve.png",
            "vs_filename": "vs_profile.png",
            "combined_filename": "combined_figure.png",
            "summary_filename": "step_summary.csv",
        }
    }
}


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def find_step_folders(strip_directory: Path) -> List[Path]:
    """Find all StepX_Y-layer folders in the strip directory."""
    step_folders = []
    if strip_directory.exists():
        for item in strip_directory.iterdir():
            if item.is_dir() and item.name.startswith("Step") and "_" in item.name and "-layer" in item.name:
                step_folders.append(item)
    
    # Sort by step number
    step_folders.sort(key=lambda x: int(x.name.split("_")[0].replace("Step", "")))
    return step_folders


def save_hv_csv(csv_path: Path, freqs, amps):
    """Save HV curve data to CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency_Hz", "HVSR_Amplitude"])
        for freq, amp in zip(freqs, amps):
            writer.writerow([f"{float(freq):.6f}", f"{float(amp):.6f}"])
    
    
def _compute_hv_curve_adaptive(model_path: str, cfg: Dict, compute_func):
    """Run compute_func(model_path, cfg) with optional adaptive frequency range.

    Expands fmax or shrinks fmin when the detected global max lies near edges.
    """
    # Base run
    local_cfg = dict(cfg)
    adaptive = dict(cfg.get("adaptive", {})) if isinstance(cfg.get("adaptive"), dict) else {}
    if not adaptive.get("enable", False):
        return compute_func(model_path, local_cfg), local_cfg

    passes = 0
    max_passes = int(adaptive.get("max_passes", 2) or 0)
    edge_margin = float(adaptive.get("edge_margin_frac", 0.05) or 0.05)
    fmax_limit = float(adaptive.get("fmax_limit", 60.0) or 60.0)
    fmin_limit = float(adaptive.get("fmin_limit", 0.05) or 0.05)
    fmax_expand = float(adaptive.get("fmax_expand_factor", 2.0) or 2.0)
    fmin_shrink = float(adaptive.get("fmin_shrink_factor", 0.5) or 0.5)

    while True:
        freqs, amps = compute_func(model_path, local_cfg)
        if not freqs or not amps:
            return (freqs, amps), local_cfg
        # Global max index
        try:
            amax = max(amps)
            i_max = list(amps).index(amax)
        except Exception:
            return (freqs, amps), local_cfg

        fmin = float(local_cfg.get("fmin", 0.2))
        fmax = float(local_cfg.get("fmax", 20.0))
        f_peak = float(list(freqs)[i_max])
        n = len(freqs)
        # Edge proximity: near first/last few bins or near bounds by margin
        near_low_idx = i_max <= max(1, int(0.02 * n))
        near_high_idx = i_max >= n - 1 - max(1, int(0.02 * n))
        near_low_f = f_peak <= fmin * (1.0 + edge_margin)
        near_high_f = f_peak >= fmax * (1.0 - edge_margin)

        adjust = None
        if (near_high_idx or near_high_f) and fmax * fmax_expand <= fmax_limit + 1e-9:
            # Expand upper range
            local_cfg["fmax"] = min(fmax * fmax_expand, fmax_limit)
            adjust = "expand_fmax"
        elif (near_low_idx or near_low_f) and fmin * fmin_shrink >= fmin_limit - 1e-9:
            # Shrink lower bound
            local_cfg["fmin"] = max(fmin * fmin_shrink, fmin_limit)
            adjust = "shrink_fmin"

        if adjust is None or passes >= max_passes:
            return (freqs, amps), local_cfg
        passes += 1

def run_complete_workflow(initial_model_path: str, output_base_dir: str, 
                         workflow_config: Optional[Dict] = None) -> Dict[str, any]:
    """
    Run the complete progressive layer stripping workflow.
    
    Parameters:
    -----------
    initial_model_path : str
        Path to the initial velocity model file
    output_base_dir : str
        Base directory for all outputs
    workflow_config : dict, optional
        Configuration dictionary for all workflow steps
        
    Returns:
    --------
    dict
        Dictionary with results and paths from the complete workflow
    """
    # Setup paths
    initial_model_path = Path(initial_model_path)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge configuration
    config = DEFAULT_WORKFLOW_CONFIG.copy()
    if workflow_config:
        config = deep_update(config, workflow_config)
    
    # Validate initial model
    if not initial_model_path.exists():
        raise FileNotFoundError(f"Initial model file not found: {initial_model_path}")
    
    print("=" * 80)
    print("🚀 HVSR Progressive Layer Stripping - Complete Workflow")
    print("=" * 80)
    print(f"📁 Initial model: {initial_model_path}")
    print(f"📁 Output directory: {output_base_dir}")
    print(f"⚙️  HVf executable: {config['hv_forward']['exe_path']}")
    print("-" * 80)
    
    # Results dictionary
    results = {
        "initial_model": initial_model_path,
        "output_directory": output_base_dir,
        "step_results": {},
        "workflow_config": config
    }
    
    try:
        # =================================================================
        # STEP 1: LAYER STRIPPING
        # =================================================================
        print("\n🔄 STEP 1/3: Layer Stripping")
        print("-" * 40)
        
        start_time = time.time()
        
        # Import stripper module
        try:
            from .stripper import write_peel_sequence
        except ImportError:
            # Fallback for direct execution
            from stripper import write_peel_sequence
        
        # Run layer stripping
        strip_output_dir = write_peel_sequence(
            str(initial_model_path),
            str(output_base_dir)
        )
        
        strip_time = time.time() - start_time
        print(f"✅ Layer stripping completed in {strip_time:.2f}s")
        print(f"📂 Strip directory: {strip_output_dir}")
        
        # Find all step folders
        step_folders = find_step_folders(Path(strip_output_dir))
        print(f"📊 Generated {len(step_folders)} stripped models")
        
        results["strip_directory"] = Path(strip_output_dir)
        results["step_folders"] = step_folders
        
        # =================================================================
        # STEP 2: HV FORWARD MODELING
        # =================================================================
        print(f"\n⚡ STEP 2/3: HV Forward Modeling")
        print("-" * 40)
        
        start_time = time.time()
        
        # Import hv_forward module
        try:
            from .hv_forward import compute_hv_curve
        except ImportError:
            from hv_forward import compute_hv_curve
        
        hv_forward_config = config["hv_forward"]
        successful_hv = 0
        
        for i, step_folder in enumerate(step_folders, 1):
            step_name = step_folder.name
            print(f"  🔸 Processing {step_name} ({i}/{len(step_folders)})")
            
            # Find model file in step folder
            model_files = list(step_folder.glob("model_*.txt"))
            if not model_files:
                print(f"    ❌ No model file found in {step_folder}")
                continue
            
            model_file = model_files[0]
            
            try:
                # Prepare per-step HVf config (optionally pre-extend for very shallow)
                step_cfg = dict(hv_forward_config)
                try:
                    parts = step_name.split("_")
                    n_layers = int(parts[1].split("-")[0]) if len(parts) > 1 else None
                except Exception:
                    n_layers = None
                if n_layers is not None and n_layers <= 2:
                    step_cfg["fmax"] = max(float(hv_forward_config.get("fmax", 20.0)) * 2.0, 40.0)

                # Compute with adaptive scanning if enabled
                (freqs, amps), used_cfg = _compute_hv_curve_adaptive(str(model_file), step_cfg, compute_hv_curve)
                
                # Save HV curve in same folder as model
                hv_csv_path = step_folder / "hv_curve.csv"
                save_hv_csv(hv_csv_path, freqs, amps)
                
                print(f"    ✅ HV curve saved: {hv_csv_path.name}")
                successful_hv += 1
                
                # Store results
                results["step_results"][step_name] = {
                    "model_file": model_file,
                    "hv_csv": hv_csv_path,
                    "n_frequencies": len(freqs),
                    "peak_amplitude": float(max(amps)),
                    "peak_frequency": float(freqs[list(amps).index(max(amps))])
                }
                
            except Exception as e:
                print(f"    ❌ Error computing HV curve: {e}")
                continue
        
        hv_time = time.time() - start_time
        print(f"✅ HV forward modeling completed in {hv_time:.2f}s")
        print(f"📊 Successfully processed {successful_hv}/{len(step_folders)} models")
        
        # =================================================================
        # STEP 3: POST-PROCESSING
        # =================================================================
        print(f"\n📊 STEP 3/3: Post-Processing & Visualization")
        print("-" * 40)
        
        start_time = time.time()
        
        # Import hv_postprocess module
        try:
            from .hv_postprocess import process
        except ImportError:
            from hv_postprocess import process
        
        postprocess_config = config["hv_postprocess"]
        successful_post = 0
        
        for i, step_folder in enumerate(step_folders, 1):
            step_name = step_folder.name
            print(f"  🔸 Post-processing {step_name} ({i}/{len(step_folders)})")
            
            # Check if we have both model and HV curve files
            hv_csv = step_folder / "hv_curve.csv"
            model_files = list(step_folder.glob("model_*.txt"))
            
            if not hv_csv.exists():
                print(f"    ⚠️  No HV curve file found, skipping")
                continue
            
            if not model_files:
                print(f"    ⚠️  No model file found, skipping")
                continue
            
            model_file = model_files[0]
            
            try:
                # Run post-processing
                post_results = process(
                    str(hv_csv),
                    str(model_file),
                    str(step_folder),  # Output to same folder
                    postprocess_config
                )
                
                print(f"    ✅ Generated {len([k for k in post_results.keys() if 'png' in k])} plots")
                successful_post += 1
                
                # Update results
                if step_name in results["step_results"]:
                    results["step_results"][step_name].update(post_results)
                
            except Exception as e:
                print(f"    ❌ Error in post-processing: {e}")
                continue
        
        post_time = time.time() - start_time
        print(f"✅ Post-processing completed in {post_time:.2f}s")
        print(f"📊 Successfully processed {successful_post}/{len(step_folders)} models")
        
        # =================================================================
        # WORKFLOW SUMMARY
        # =================================================================
        total_time = strip_time + hv_time + post_time
        
        print("\n" + "=" * 80)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"   ├─ Layer stripping: {strip_time:.2f}s")
        print(f"   ├─ HV forward: {hv_time:.2f}s")
        print(f"   └─ Post-processing: {post_time:.2f}s")
        print(f"📊 Processed {len(step_folders)} stripped models")
        print(f"📁 All outputs saved in: {output_base_dir}")
        
        # Summary of generated files
        print(f"\n📋 Generated files per step:")
        for step_name, step_data in results["step_results"].items():
            print(f"   📂 {step_name}:")
            print(f"      ├─ model_*.txt")
            print(f"      ├─ hv_curve.csv")
            if 'hv_curve_png' in step_data:
                print(f"      ├─ hv_curve.png")
            if 'vs_profile_png' in step_data:
                print(f"      ├─ vs_profile.png")
            if 'combined_png' in step_data:
                print(f"      ├─ combined_figure.png")
            if 'summary_csv' in step_data:
                print(f"      └─ summary.csv")
        
        print("\n✨ Ready for analysis!")
        print("=" * 80)
        
        results["success"] = True
        results["total_time"] = total_time
        results["summary"] = {
            "total_steps": len(step_folders),
            "successful_hv": successful_hv,
            "successful_post": successful_post,
            "completion_rate": successful_post / len(step_folders) * 100 if step_folders else 0
        }
        
    except Exception as e:
        print(f"\n❌ WORKFLOW FAILED: {e}")
        results["success"] = False
        results["error"] = str(e)
        raise
    
    return results


__all__ = [
    "run_complete_workflow",
    "DEFAULT_WORKFLOW_CONFIG",
    "find_step_folders",
    "save_hv_csv"
]
