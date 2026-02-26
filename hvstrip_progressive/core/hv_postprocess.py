"""HV Post-Processing Module - Publication-ready HVSR analysis and visualization."""

import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from .peak_detection import (
    PEAK_DETECTION_PRESETS,
    get_peak_detection_preset,
    detect_peak,
    find_all_peaks,
)


DEFAULT_CONFIG = {
    "peak_detection": {
        "preset": "default",  # "default", "forward_modeling", "conservative", or "custom"
        "method": "max",  # "max", "find_peaks", or "manual"
        "select": "max",   # when using find_peaks: "max" or "leftmost"
        "manual_frequency": None,
        "find_peaks_params": {"prominence": 0.5, "distance": 5},
        # Optional frequency constraints for peak selection
        "freq_min": None,
        "freq_max": None,
        # Additional guards to avoid boundary/artefact picks
        "min_rel_height": 0.0,   # fraction of global max (e.g., 0.25)
        "exclude_first_n": 0,    # exclude the first N bins (e.g., 1)
        # New parameters for improved detection
        "min_amplitude": None,   # absolute minimum peak amplitude (e.g., 2.0)
        "check_clarity_ratio": False,  # check if peak > threshold * amp at f0/2 and 2*f0
        "clarity_ratio_threshold": 1.5,  # multiplier for clarity check
    },
    "hv_plot": {
        "x_axis_scale": "log",
        "y_axis_scale": "log",  # "linear" or "log"
        "freq_window_mode": "relative",  # "relative" or "absolute"
        "freq_window_left": 0.3,  # multiplier for relative mode
        "freq_window_right": 3.0,
        "abs_freq_min": 0.1,  # for absolute mode
        "abs_freq_max": 50.0,
        "y_compression": 1.5,
        "smoothing": {"enable": True, "window_length": 7, "poly_order": 3},
        "show_bands": True,
        "figure_width": 12,
        "figure_height": 6,
        "dpi": 150,
    },
    "vs_plot": {
        "show": True,
        "annotate_deepest": True,
        "annotate_max_vs": True,
        "annotate_f0": True,
        "figure_width": 6,
        "figure_height": 8,
        "dpi": 150,
    },
    "output": {
        "save_separate": True,
        "save_combined": True,
        "hv_filename": "hv_curve.png",
        "vs_filename": "vs_profile.png",
        "combined_filename": "combined_figure.png",
        "summary_filename": "summary.csv",
    }
}


def read_hv_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


def read_model(path: Path) -> Dict:
    with open(path, 'r') as f:
        lines = f.readlines()
    n_layers = int(lines[0].strip())
    layers = []
    for i in range(1, n_layers + 1):
        parts = lines[i].strip().split()
        layers.append({
            'thickness': float(parts[0]),
            'vp': float(parts[1]),
            'vs': float(parts[2]),
            'rho': float(parts[3])
        })
    
    # Extract step info from path if present
    step_info = None
    path_str = str(path)
    if 'Step' in path_str:
        match = re.search(r'Step(\d+)_(\d+)-layer', path_str)
        if match:
            step_info = {
                'step_number': int(match.group(1)),
                'n_finite_layers': int(match.group(2))
            }
    
    return {'n_layers': n_layers, 'layers': layers, 'step_info': step_info}



def apply_smoothing(amps: np.ndarray, config: Dict) -> np.ndarray:
    """Apply optional smoothing."""
    smooth_cfg = config.get('hv_plot', {}).get('smoothing', {})
    if not smooth_cfg.get('enable', False):
        return amps
    
    window = smooth_cfg.get('window_length', 7)
    poly = smooth_cfg.get('poly_order', 3)
    
    if window > len(amps):
        window = len(amps) if len(amps) % 2 == 1 else len(amps) - 1
    if window < poly + 2:
        return amps
    
    try:
        smoothed = savgol_filter(amps, window, poly)
        return np.maximum(smoothed, amps * 0.9)
    except:
        return amps


def deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively update nested dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def process(hv_csv_path: str, model_txt_path: str, output_dir: str, 
           plot_config: Optional[Dict] = None) -> Dict[str, Path]:
    """Main processing function - creates plots and summaries from HV curve and model."""
    hv_csv_path = Path(hv_csv_path)
    model_txt_path = Path(model_txt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge configs
    config = DEFAULT_CONFIG.copy()
    if plot_config:
        config = deep_update(config, plot_config)
    
    # Read data
    freqs, amps = read_hv_csv(hv_csv_path)
    model = read_model(model_txt_path)
    
    # Detect peak
    f0, a0, idx = detect_peak(freqs, amps, config)
    
    # Apply smoothing
    amps_plot = apply_smoothing(amps, config)
    
    # Create outputs dict
    outputs = {'peak_frequency': f0, 'peak_amplitude': a0, 'peak_index': idx}
    
    # Create HV plot
    hv_cfg = config.get('hv_plot', {})
    fig, ax = plt.subplots(figsize=(hv_cfg.get('figure_width', 12), 
                                    hv_cfg.get('figure_height', 6)))
    
    # Plot curve
    ax.plot(freqs, amps_plot, linewidth=2, color='#2E86AB', label='H/V Curve')
    ax.fill_between(freqs, amps_plot, alpha=0.15, color='#2E86AB')
    ax.scatter(f0, a0, s=100, color='#E63946', zorder=5, 
              edgecolors='white', linewidth=2, label=f'Peak: {f0:.2f} Hz')
    
    # Add frequency bands
    if hv_cfg.get('show_bands', True):
        mode = hv_cfg.get('freq_window_mode', 'relative')
        if mode == 'relative':
            band_left = f0 * hv_cfg.get('freq_window_left', 0.3)
            band_right = f0 * hv_cfg.get('freq_window_right', 3.0)
        else:
            band_left = hv_cfg.get('abs_freq_min', 0.1)
            band_right = hv_cfg.get('abs_freq_max', 50.0)
        ax.axvspan(band_left, f0, alpha=0.1, color='gray')
        ax.axvspan(f0, band_right, alpha=0.1, color='gray')
    
    # Smart annotation placement
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Determine if peak is in left or right half of plot
    if hv_cfg.get('x_axis_scale', 'log') == 'log':
        log_center = np.sqrt(x_min * x_max)  # geometric center for log scale
        peak_in_left = f0 < log_center
        x_text = f0 * 2.0 if peak_in_left else f0 * 0.5
    else:
        linear_center = (x_min + x_max) / 2
        peak_in_left = f0 < linear_center
        x_text = f0 + (x_max - x_min) * 0.1 if peak_in_left else f0 - (x_max - x_min) * 0.1
    
    # Place annotation in upper part but not too high
    y_text = y_min + (y_max - y_min) * 0.75
    
    # Annotate with smart positioning
    ax.annotate(f'f₀ = {f0:.2f} Hz\nA = {a0:.1f}',
                xy=(f0, a0), xytext=(x_text, y_text),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
                fontsize=12, 
                ha='left' if peak_in_left else 'right',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#E63946', alpha=0.9))
    
    # Set scales
    if hv_cfg.get('x_axis_scale', 'log') == 'log':
        ax.set_xscale('log')
        ticks = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 20]
        ticks = [t for t in ticks if t >= np.min(freqs)*0.9 and t <= np.max(freqs)*1.1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks])
    
    # Set y-axis scale
    if hv_cfg.get('y_axis_scale', 'linear') == 'log':
        ax.set_yscale('log')
    
    # Apply y-compression
    y_comp = hv_cfg.get('y_compression', 1.0)
    if y_comp < 1.0:
        y_min, y_max = np.min(amps_plot), np.max(amps_plot)
        y_center = (y_max + y_min) / 2
        y_span = (y_max - y_min) * y_comp
        ax.set_ylim(y_center - y_span/2, y_center + y_span/2)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('H/V Amplitude Ratio', fontsize=12)
    ax.set_title('HVSR Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    hv_path = output_dir / config['output'].get('hv_filename', 'hv_curve.png')
    fig.savefig(hv_path, dpi=hv_cfg.get('dpi', 150), bbox_inches='tight')
    outputs['hv_curve_png'] = hv_path
    plt.close(fig)
    
    # Create Vs profile if enabled
    if config['vs_plot'].get('show', True):
        vs_cfg = config['vs_plot']
        fig, ax = plt.subplots(figsize=(vs_cfg.get('figure_width', 6),
                                        vs_cfg.get('figure_height', 8)))
        
        layers = model['layers']
        depths = [0]
        vs_values = []
        current_depth = 0
        
        for layer in layers:
            if layer['thickness'] > 0:
                vs_values.append(layer['vs'])
                current_depth += layer['thickness']
                depths.append(current_depth)
            else:
                vs_values.append(layer['vs'])
                depths.append(current_depth * 1.5)
        
        # Plot step profile
        for i in range(len(vs_values)):
            if i < len(vs_values) - 1:
                ax.fill_betweenx([depths[i], depths[i+1]], 0, vs_values[i], 
                               alpha=0.3, color='#A8DADC')
                ax.plot([vs_values[i], vs_values[i]], [depths[i], depths[i+1]], 
                       linewidth=2, color='#2E86AB')
                if i > 0:
                    ax.plot([vs_values[i-1], vs_values[i]], [depths[i], depths[i]], 
                           linewidth=1, color='#2E86AB', linestyle='--')
            else:
                ax.fill_betweenx([depths[i], depths[i+1]], 0, vs_values[i], 
                               alpha=0.2, color='lightgray')
                ax.plot([vs_values[i], vs_values[i]], [depths[i], depths[i+1]], 
                       linewidth=2, color='gray', linestyle=':')
        
        # Annotations
        if vs_cfg.get('annotate_deepest', True) and len(layers) > 1:
            deepest_idx = len(layers) - 2
            if deepest_idx >= 0:
                deepest_depth = sum(l['thickness'] for l in layers[:deepest_idx+1])
                ax.annotate(f'Deepest: {layers[deepest_idx]["vs"]:.0f} m/s',
                           xy=(layers[deepest_idx]['vs'], deepest_depth),
                           xytext=(layers[deepest_idx]['vs']*1.2, deepest_depth),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, color='red')
        
        if vs_cfg.get('annotate_f0', True) and len(layers) > 1:
            deepest_depth = sum(l['thickness'] for l in layers[:-1])
            ax.text(np.min([l['vs'] for l in layers])*1.1, deepest_depth*0.95,
                   f'f₀ = {f0:.2f} Hz', fontsize=12, color='green',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green"))
        
        ax.set_xlabel('Vs (m/s)', fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_title('Vs Profile', fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        vs_path = output_dir / config['output'].get('vs_filename', 'vs_profile.png')
        fig.savefig(vs_path, dpi=vs_cfg.get('dpi', 150), bbox_inches='tight')
        outputs['vs_profile_png'] = vs_path
        plt.close(fig)
    
    # Write summary CSV
    csv_path = output_dir / config['output'].get('summary_filename', 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Step', 'N_Finite_Layers', 'Peak_Frequency_Hz', 'Peak_Amplitude']
        max_layers = len(model['layers']) - 1
        for i in range(max_layers):
            header.extend([f'L{i+1}_Thickness', f'L{i+1}_Vp', f'L{i+1}_Vs', f'L{i+1}_Rho'])
        header.extend(['HS_Vp', 'HS_Vs', 'HS_Rho'])
        writer.writerow(header)
        
        # Data row
        row = []
        if model.get('step_info'):
            row.append(f"Step{model['step_info']['step_number']}")
            row.append(model['step_info']['n_finite_layers'])
        else:
            row.append('N/A')
            row.append(len(model['layers']) - 1)
        
        row.extend([f'{f0:.6f}', f'{a0:.6f}'])
        
        for layer in model['layers'][:-1]:
            row.extend([f"{layer['thickness']:.2f}", f"{layer['vp']:.2f}",
                       f"{layer['vs']:.2f}", f"{layer['rho']:.2f}"])
        
        hs = model['layers'][-1]
        row.extend([f"{hs['vp']:.2f}", f"{hs['vs']:.2f}", f"{hs['rho']:.2f}"])
        
        writer.writerow(row)
    
    outputs['summary_csv'] = csv_path
    
    return outputs


__all__ = [
    "process",
    "detect_peak",
    "find_all_peaks",
    "get_peak_detection_preset",
    "PEAK_DETECTION_PRESETS",
    "apply_smoothing",
    "read_hv_csv",
    "read_model",
    "DEFAULT_CONFIG",
]
