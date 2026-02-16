"""
Specialized plots for HVSR analysis, including the Two Resonance Separation demonstration.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def generate_resonance_separation_figure(strip_dir, output_path):
    """
    Generates a publication-quality figure demonstrating the separation
    of deep and shallow resonances.
    """
    strip_path = Path(strip_dir)
    
    # 1. Load Data
    step0_dirs = list(strip_path.glob("Step0_*"))
    if not step0_dirs: return
    step0_dir = step0_dirs[0]
    hv_csv_0 = step0_dir / "hv_curve.csv"
    if not hv_csv_0.exists(): return
    df0 = pd.read_csv(hv_csv_0)
    
    # Select the specific step where ONLY the deep layer is removed (Step 1)
    step1_dirs = list(strip_path.glob("Step1_*"))
    
    if step1_dirs:
        step_final_dir = step1_dirs[0]
        print(f"Selecting {step_final_dir.name} (Deepest layer removed) for visualization.")
    else:
        # Fallback to whatever is available if Step 1 is missing
        step_final_dirs = sorted(list(strip_path.glob("Step*")))
        if len(step_final_dirs) < 2: return
        step_final_dir = step_final_dirs[-1]

    hv_csv_final = step_final_dir / "hv_curve.csv"
    if not hv_csv_final.exists(): return
    df_final = pd.read_csv(hv_csv_final)

    # Read Models to get Vs/H info for theoretical calc
    def read_model_params(path):
        with open(path) as f: lines = f.readlines()
        layers = []
        try:
            n_layers = int(lines[0])
            for line in lines[1:n_layers+1]:
                parts = [float(x) for x in line.split()]
                if len(parts) >= 3:
                    layers.append({'thick': parts[0], 'vp': parts[1], 'vs': parts[2], 'rho': parts[3]})
        except: pass
        return layers

    model0_layers = read_model_params(step0_dir / f"model_{step0_dir.name}.txt")
    
    # Calculate Theoretical f1 (Fundamental of top layer) for guidance
    f1_theo = 0
    if model0_layers and model0_layers[0]['thick'] > 0:
        f1_theo = model0_layers[0]['vs'] / (4 * model0_layers[0]['thick'])
    
    # 2. Setup Figure
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.3)
    ax_hv = plt.subplot(gs[0])
    ax_vs = plt.subplot(gs[1])

    # 3. Plot HV Curves
    l1, = ax_hv.semilogx(df0['Frequency_Hz'], df0['HVSR_Amplitude'], 
                   color='black', linewidth=3, label='Original Basin Model', zorder=3)
    
    l2, = ax_hv.semilogx(df_final['Frequency_Hz'], df_final['HVSR_Amplitude'], 
                   color='#E63946', linewidth=2.5, linestyle='--',
                   label='Stripped Model (Deep Layer Removed)', zorder=4)

    # --- PEAK SELECTION LOGIC ---
    
    # 1. Deep Peak (f0) - FIRST significant peak, not global max!
    # Use scipy.find_peaks to detect all peaks, then take the first one
    freqs = df0['Frequency_Hz'].values
    amps = df0['HVSR_Amplitude'].values
    
    # Smooth the curve slightly to reduce noise
    amps_smooth = gaussian_filter1d(amps, sigma=1)
    
    # Find all peaks with minimum prominence and height
    min_prominence = 0.3 * (np.max(amps_smooth) - np.min(amps_smooth))
    min_height = np.mean(amps_smooth) + 0.2 * np.std(amps_smooth)
    
    peaks, properties = find_peaks(amps_smooth, 
                                   prominence=min_prominence * 0.3,
                                   height=min_height * 0.5,
                                   distance=3)
    
    if len(peaks) >= 1:
        # f0 is the FIRST significant peak (lowest frequency = deep resonance)
        f0_idx = peaks[0]
        f0 = freqs[f0_idx]
        a0 = amps[f0_idx]  # Use original amplitude, not smoothed
        print(f"Detected f0 (first peak): {f0:.2f} Hz, Amp: {a0:.2f}")
        
        # If there are multiple peaks, the global max is likely f1
        if len(peaks) >= 2:
            global_max_idx = np.argmax(amps)
            # Check if global max is different from first peak
            if abs(freqs[global_max_idx] - f0) > 0.5:
                print(f"Note: Global max at {freqs[global_max_idx]:.2f} Hz (likely f1)")
    else:
        # Fallback to global max if no peaks found
        f0_idx = np.argmax(amps)
        f0 = freqs[f0_idx]
        a0 = amps[f0_idx]
        print(f"Fallback: using global max as f0: {f0:.2f} Hz")

    # 2. Shallow Peak (f1) - FROM CORE PACKAGE SUMMARY (stripped model)
    summary_file = step_final_dir / "step_summary.csv"
    
    # Default fallback
    f_shallow = f0 * 3
    a_shallow = 1.0
    
    if summary_file.exists():
        try:
            summary_df = pd.read_csv(summary_file)
            if not summary_df.empty:
                f_shallow = float(summary_df.iloc[0]['Peak_Frequency_Hz'])
                a_shallow = float(summary_df.iloc[0]['Peak_Amplitude'])
                print(f"Using core package peak for f1: {f_shallow:.2f} Hz, Amp: {a_shallow:.2f}")
        except Exception as e:
            print(f"Error reading summary file: {e}")
    
    # Ensure we point to the actual curve at this frequency
    idx_snap = (np.abs(df_final['Frequency_Hz'] - f_shallow)).argmin()
    
    # Annotation Limits

    # Annotation Limits

    # Annotation Limits
    max_amp = max(a0, a_shallow, np.max(df0['HVSR_Amplitude']))
    ax_hv.set_ylim(0, max_amp * 1.35)

    # Deep Peak Annotation
    ax_hv.annotate(f'Deep Resonance ($f_0$)\n{f0:.2f} Hz (Removed)', 
                   xy=(f0, a0), 
                   xytext=(f0 * 0.45, max_amp * 1.15),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=11, fontweight='bold', ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

    # Shallow Peak Annotation
    ax_hv.annotate(f'Shallow Resonance ($f_1$)\n{f_shallow:.2f} Hz (Persists)', 
                   xy=(f_shallow, a_shallow), 
                   xytext=(f_shallow * 1.6, max_amp * 1.15),
                   arrowprops=dict(facecolor='#E63946', shrink=0.05, width=1.5),
                   fontsize=11, fontweight='bold', color='#E63946', ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E63946", alpha=0.9))
    
    # Mark peaks
    ax_hv.plot(f0, a0, 'o', color='black', markersize=7, zorder=5)
    ax_hv.plot(f_shallow, a_shallow, 'o', color='#E63946', markersize=7, zorder=5)

    ax_hv.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
    ax_hv.set_ylabel('H/V Amplitude', fontsize=13, fontweight='bold')
    ax_hv.set_title('(a) Resonance Mode Separation', fontsize=15, fontweight='bold', pad=15)
    ax_hv.grid(True, which='both', alpha=0.3)
    ax_hv.legend(fontsize=11, loc='lower left', frameon=True, framealpha=0.95)
    ax_hv.set_xlim(0.2, 20.0)

    # 4. Improved Velocity Profile Plot
    def read_model_full(path):
        with open(path) as f: lines = f.readlines()
        layers = []
        depth = 0
        try:
            n_layers = int(lines[0])
            for line in lines[1:n_layers+1]:
                parts = [float(x) for x in line.split()]
                if len(parts) < 3: continue
                thick, vs = parts[0], parts[2]
                if thick == 0: thick = max(depth * 0.25, 30)
                layers.append({'top': depth, 'bot': depth+thick, 'vs': vs, 'thick': thick})
                depth += thick
        except Exception: return []
        return layers

    model0 = read_model_full(step0_dir / f"model_{step0_dir.name}.txt")
    
    # Count how many finite layers are in the stripped model (excluding halfspace)
    # Step1_2-layer means 2 finite layers are kept.
    try:
        parts = step_final_dir.name.split('_')
        kept_layers_str = parts[1].split('-')[0]
        n_kept_finite = int(kept_layers_str)
    except:
        n_kept_finite = 1 # Fallback
    
    if model0:
        max_vs = max(l['vs'] for l in model0)
        max_depth = model0[-1]['bot']
        
        for i, layer in enumerate(model0):
            # The stripped model keeps the top 'n_kept_finite' layers.
            # layer indices 0 to n_kept_finite-1 are kept.
            is_retained = i < n_kept_finite
            
            color = '#E63946' if is_retained else '#B0B0B0'
            alpha = 0.6 if is_retained else 0.4
            
            # Draw Block
            rect = Rectangle((0, layer['top']), layer['vs'], layer['thick'], 
                           facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
            ax_vs.add_patch(rect)
            
            # Label OUTSIDE to the right
            # Leader line strategy
            y_center = (layer['top'] + layer['bot']) / 2
            
            label_txt = f"$V_s$={int(layer['vs'])}"
            
            # Dynamic label positioning to avoid vertical overlap
            # We assume layers are ordered by depth
            
            ax_vs.annotate(label_txt, 
                          xy=(layer['vs'], y_center), 
                          xytext=(max_vs * 1.15, y_center),
                          ha='left', va='center', fontsize=11, fontweight='bold',
                          color='black',
                          arrowprops=dict(arrowstyle='-', color='gray', linewidth=0.5))

        # Explicit "Removed" Annotation for the deep layer(s)
        if n_kept_finite < len(model0):
            removed_top = model0[n_kept_finite]['top']
            # If the last layer is halfspace, we treat its visualization bottom as the removal bottom
            removed_bot = model0[-1]['bot']
            
            # Draw Bracket on the far right
            bracket_x = max_vs * 1.6
            
            mid_y = (removed_top + removed_bot) / 2
            
            ax_vs.annotate('REMOVED', 
                          xy=(bracket_x, mid_y),
                          xytext=(bracket_x + max_vs*0.1, mid_y),
                          rotation=90, va='center', ha='center', fontweight='bold', color='#666666', fontsize=12)
            
            # Bracket lines
            ax_vs.plot([bracket_x, bracket_x], [removed_top, removed_bot], color='#666666', linewidth=2)
            ax_vs.plot([bracket_x - max_vs*0.05, bracket_x], [removed_top, removed_top], color='#666666', linewidth=2)
            ax_vs.plot([bracket_x - max_vs*0.05, bracket_x], [removed_bot, removed_bot], color='#666666', linewidth=2)

        ax_vs.invert_yaxis()
        ax_vs.set_xlabel('Shear Wave Velocity, $V_s$ (m/s)', fontsize=13, fontweight='bold')
        ax_vs.set_ylabel('Depth (m)', fontsize=13, fontweight='bold')
        ax_vs.set_title('(b) Progressive Stripping', fontsize=15, fontweight='bold', pad=15)
        ax_vs.grid(True, alpha=0.3, which='major', linestyle='--')
        
        ax_vs.set_xlim(0, max_vs * 2.0) # More room for bracket and labels
        ax_vs.set_ylim(max_depth, 0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Generated resonance separation figure: {output_path}")